#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/llm/dataset_compiler.py

Kanonik D3-istisna çift-kayıt derleyicisi ve ikili veri doğrulama kütüphanesi (T-0134 / G3b).

Marangozluk ve gelecekteki dikey uzmanlık modelleri (Bahçıvan, Berber vb.) için:
  1. `tokenize_specialization_jsonl`:
     - Satır başına 2 kayıt üretebilen D3-istisna SFT yapısını jenerikleştirir:
       (a) Ham kayıt: json.dumps(d)
       (b) Normalize SFT komut çifti: {"instruction": system_prompt, "input": q, "output": d["output"]}
     - Geriye dönük tam uyumluluk: `system_prompt` verilmezse varsayılan "Ahşap uzmanı olarak cevapla." kullanılır.
  2. `bin_dekod_dogrula`:
     - Üretilen .bin dosyalarını memmap ile okuyarak <BOS>, <EOS>, <OUTPUT>, </OUTPUT>
       özel jetonlarının kayıt sayısı ile eşitliğini ve token bütünlüğünü denetler.
  3. `compile_specialization_dataset`:
     - JSONL kütüklerini ve isteğe bağlı anti-forgetting çıpalarını birleştirip .bin + .meta.json yazar
       ve ardından `bin_dekod_dogrula` ile otomatik doğrular.
"""

import os
import sys
import json
import random
from typing import Dict, Tuple, List, Optional, Any
import numpy as np

from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.frozen_guard import is_frozen_path


def check_output_path(output_bin: str, allow_frozen_write: bool = False) -> None:
    """Çıktı yolunun donmuş olup olmadığını denetler. allow_frozen_write=False ise RuntimeError fırlatır."""
    if is_frozen_path(output_bin) and not allow_frozen_write:
        raise RuntimeError(
            f"Donmus yola yazma engellendi: {output_bin} (allow_frozen_write=False)"
        )


def tokenize_specialization_jsonl(
    filepath: str,
    tokenizer: KristalTokenizer,
    system_prompt: str = "Ahşap uzmanı olarak cevapla.",
    domain_name: Optional[str] = None
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    D3-istisna çift-kayıt derleyicisi (jenerik).

    Ahşap ve diğer dikey alan uzmanlık kütüklerini satır satır okur:
      1. Ham kayıt: JSON formatında token dizisine çevrilir (len > 2 ise tutulur).
      2. Normalize SFT komut çifti:
         - Soru 'input' alanından veya 'instruction' içindeki ':' sonrasından çıkarılır (q).
         - 'instruction': system_prompt
         - 'input': q
         - 'output': d['output']
         olarak normalize edilir ve token dizisine çevrilir (len > 2 ise tutulur).

    9 sayaçlı tam muhasebe ve D4 özdeşlik denetimi döner:
      - raw_lines, sampled, json_errors, encode_errors, normalize_encode_errors,
        lines_ok, short_skipped, kept, missing
    """
    stats: Dict[str, Any] = {
        "missing": False,
        "raw_lines": 0,
        "sampled": 0,
        "json_errors": 0,
        "encode_errors": 0,
        "normalize_encode_errors": 0,
        "lines_ok": 0,
        "short_skipped": 0,
        "kept": 0,
        "domain_name": domain_name,
        "system_prompt": system_prompt,
    }
    records: List[List[int]] = []

    if not os.path.exists(filepath):
        stats["missing"] = True
        return records, stats

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    stats["raw_lines"] = len(lines)
    stats["sampled"] = len(lines)
    logged_errors = 0

    for line_idx, line in enumerate(lines, 1):
        try:
            d = json.loads(line)
        except Exception as e:
            stats["json_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_specialization] {line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        try:
            raw_s = json.dumps(d, ensure_ascii=False)
            tids = tokenizer.encode(raw_s)
        except Exception as e:
            stats["encode_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_specialization] {line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        stats["lines_ok"] += 1

        # 1. Ham kayıt
        if len(tids) > 2:
            records.append(tids)
            stats["kept"] += 1
        else:
            stats["short_skipped"] += 1

        # 2. SFT Standart Komut Çifti (system_prompt ile normalize)
        inst = d.get("instruction", "")
        q = d.get("input", "")
        if not q and ":" in inst:
            q = inst.split(":", 1)[1].strip()
        if q:
            norm_d = {
                "instruction": system_prompt,
                "input": q,
                "output": d.get("output", "")
            }
            try:
                norm_tids = tokenizer.encode(json.dumps(norm_d, ensure_ascii=False))
                if len(norm_tids) > 2:
                    records.append(norm_tids)
                    stats["kept"] += 1
                else:
                    stats["short_skipped"] += 1
            except Exception as e:
                stats["normalize_encode_errors"] += 1
                if logged_errors < 5:
                    sys.stderr.write(f"[tokenize_specialization-sft] {line_idx} {type(e).__name__}: {e}\n")
                    logged_errors += 1

    # D4 Özdeşlik denetimi (satır/hata/ok üçlüsü)
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"], (
        f"Özdeşlik hatası ({filepath}): {stats['sampled']} != "
        f"{stats['json_errors']} + {stats['encode_errors']} + {stats['lines_ok']}"
    )
    # Ek invariant: satır başına en fazla 2 kayıt üretilebilir
    assert stats["kept"] <= 2 * stats["lines_ok"], (
        f"Kayıt tavan aşımı: kept ({stats['kept']}) > 2 * lines_ok ({stats['lines_ok']})"
    )

    return records, stats


def tokenize_standard_jsonl(
    filepath: str,
    tokenizer: KristalTokenizer,
    max_samples: Optional[int] = None,
    min_tokens: int = 2
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Tekil SFT / Çıpa JSONL kütüğünü satır satır okur ve 8 sayaçlı tam muhasebe istatistiği döner."""
    stats = {
        "missing": False,
        "raw_lines": 0,
        "sampled": 0,
        "json_errors": 0,
        "encode_errors": 0,
        "lines_ok": 0,
        "short_skipped": 0,
        "kept": 0,
    }
    records: List[List[int]] = []

    if not os.path.exists(filepath):
        stats["missing"] = True
        return records, stats

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    stats["raw_lines"] = len(lines)

    if max_samples and len(lines) > max_samples:
        random.seed(42)
        lines = random.sample(lines, max_samples)

    stats["sampled"] = len(lines)
    logged_errors = 0

    for line_idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
        except Exception as e:
            stats["json_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_standard_jsonl] {os.path.basename(filepath)}:{line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        try:
            raw_prompt = json.dumps(item, ensure_ascii=False)
            token_ids = tokenizer.encode(raw_prompt)
        except Exception as e:
            stats["encode_errors"] += 1
            if logged_errors < 5:
                sys.stderr.write(f"[tokenize_standard_jsonl] {os.path.basename(filepath)}:{line_idx} {type(e).__name__}: {e}\n")
                logged_errors += 1
            continue

        stats["lines_ok"] += 1

        if len(token_ids) <= min_tokens:
            stats["short_skipped"] += 1
        else:
            records.append(token_ids)
            stats["kept"] += 1

    # D4 Özdeşlik denetimi
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"], (
        f"Özdeşlik hatası ({filepath}): {stats['sampled']} != "
        f"{stats['json_errors']} + {stats['encode_errors']} + {stats['lines_ok']}"
    )

    return records, stats


def bin_dekod_dogrula(
    bin_path: str,
    meta_path: str,
    vocab: Vocabulary
) -> Dict[str, Any]:
    """
    Derlenen .bin dosyasını ve meta kütüğünü bağımsız memmap okuyucusuyla denetler.

    İnvariantlar:
      1. <BOS> sayısı == total_records
      2. <EOS> sayısı == total_records
      3. <OUTPUT> sayısı == total_records
      4. </OUTPUT> sayısı == total_records
      5. Toplam jeton sayısı == meta['total_tokens']

    Herhangi bir sapma durumunda AssertionError fırlatır; başarılı olursa özet sözlüğü döner.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary dosya bulunamadı: {bin_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta dosya bulunamadı: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total_tokens = len(mm)

    bos_id = vocab.stoi.get("<BOS>")
    eos_id = vocab.stoi.get("<EOS>")
    output_id = vocab.stoi.get("<OUTPUT>")
    output_end_id = vocab.stoi.get("</OUTPUT>")

    bos_count = int(np.sum(mm == bos_id))
    eos_count = int(np.sum(mm == eos_id))
    output_count = int(np.sum(mm == output_id))
    output_end_count = int(np.sum(mm == output_end_id))

    expected_records = meta.get("total_records") or meta.get("record_count")

    assert bos_count == expected_records, f"<BOS> ({bos_count}) != meta records ({expected_records})"
    assert eos_count == expected_records, f"<EOS> ({eos_count}) != meta records ({expected_records})"
    assert output_count == expected_records, f"<OUTPUT> ({output_count}) != meta records ({expected_records})"
    assert output_end_count == expected_records, f"</OUTPUT> ({output_end_count}) != meta records ({expected_records})"
    assert total_tokens == meta.get("total_tokens"), f"Token boyutu ({total_tokens}) != meta ({meta.get('total_tokens')})"

    return {
        "ok": True,
        "total_tokens": total_tokens,
        "total_records": expected_records,
        "bos_count": bos_count,
        "eos_count": eos_count,
        "output_count": output_count,
        "output_end_count": output_end_count,
        "avg_tokens_per_record": float(total_tokens / expected_records) if expected_records else 0.0
    }
