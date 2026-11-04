#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.2 — EVRE-5 AKIL YÜRÜTME (CoT) VERİ DERLEME (İLK DERLEME)
===========================================================
Girdi  : data/pedagogy/cot_vault.jsonl  (pedagojik süpervizör CoT kasası)
Çıktı  : data/train_reasoning_cot.bin  (uint16, düz jeton akışı) + .meta.json

Neden yeni bir betik (P2.2 notu):
  v1.5.0'daki izlenen 'scripts/prepare_reasoning_dataset.py' (commit 52a7875) BAYAT
  bir sözlikle (data/vocab.json, 31.357) derliyordu ve meta'sında train.py
  sınır kapısının zorunlu alanları (sozluk_giris / sozluk_sha256) yoktu. Donmuş
  anka sözüğü (33.114) ve P2.3 pencere-uyumluluğu için bu betik kanonik hattı
  (src/llm/dataset_compiler.tokenize_standard_jsonl deseni) kullanır.

Kaynak şeması (ölçülen, 91 satırın TAMAMI):
  {timestamp, query, instruction, thought_trace, final_answer, source}
  91 satır → 22 içerik-benzersiz kayıt (tek sorgu 70× yinelenir).

SFT kayıt kalıbı (kaynaktan, UYDURMA YOK — v1.5.0 format_reasoning_sample kalıbı):
  instruction = vault 'instruction'  (pedagojik istem yüzü)
  input       = vault 'query'
  output      = "<DUSUNCE> {thought_trace} </DUSUNCE> {final_answer}"
  (<DUSUNCE>/</DUSUNCE> donmuş anka sözüğünde 31338/31339'dur; CONTROL_TOKENS üyesi.)

Kanonik tokenizasyon (pencere eğitimiyle uyumlu, T-0013):
  tokenizer.encode(json.dumps(sft_kaydi))  →
  <BOS> <INSTRUCTION> … </INSTRUCTION> <INPUT> … </INPUT> <OUTPUT> <DUSUNCE> …
  </DUSUNCE> … </OUTPUT> <EOS>      (encode başa <BOS>, sona <EOS> koyar)

Kalite kapıları (ölçülür, raporlanır; kalite > nicelik — İLK derleme):
  1. İçerik tekilleştirme: (query, instruction, thought, answer) birebir tekilleştirme.
  2. Etiket-artefakt temizliği: sızan literal 'DUSUNCE' öneki ve <BILGI_KARTI>
     etiketleri ayıklanır (extract_cot_and_card'ın kendi ayıklama deseni).
  3. Yapısal kart kapısı: final_answer ADIM listesi değil KART olmalı —
     numaralı/madde işaretli adım kalıbıyla başlayan, markdown '**' içeren,
     yarıda kesilmiş (noktalama ile bitmeyen) veya < 40 karakter kartlar DÜŞÜRÜLÜR.
  4. UNK eşiği: kayıt başına UNK oranı --unk-esik (varsayılan %10) üzerindeyse
     DÜŞÜRÜLÜR; ölçüm her durumda raporlanır (P2.6 hedefi <3% referanstır).
"""

import os
import sys
import json
import re
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.dataset_compiler import bin_dekod_dogrula

# Donmuş sabitler — P2.2 derlemesi anka_r1_33114 sözlüğüyle yapılır (kapı: max id < 33114)
VOCAB_PATH = os.path.join(KOK, "data/rebuild/vocab_anka_r1_33114.json")
LEXICON_PATH = os.path.join(KOK, "data/lexicon/roots.tsv")
VAULT_PATH = os.path.join(KOK, "data/pedagogy/cot_vault.jsonl")
OUTPUT_BIN = os.path.join(KOK, "data/train_reasoning_cot.bin")
BLOCK_SIZE = 128          # train.py meta otomatik-algılama anahtarı
MIN_THOUGHT_LEN = 15      # v1.5.0 eşiği korunur
MIN_CARD_LEN = 40
UNK_ESIK_PCT = 10.0       # felaket eşiği; ölçüm her halükârda raporlanır


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Kaynak okuma + sanitizasyon (kalıp kaynaktan: extract_cot_and_card / v1.5.0)
# ----------------------------------------------------------------------------
_RE_DUSUNCE_LITERAL = re.compile(r"^\s*DUSUNCE\s*:?\s*\n?", re.IGNORECASE)
_RE_BILGI_KARTI_TAG = re.compile(r"</?B[İI]LG[İI][_ ]?KART[İI]>", re.IGNORECASE)
_RE_ADIM_BASLANGICI = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])", re.S)


def sanitize_thought(t: str) -> str:
    """Sızan literal 'DUSUNCE' satır önekini ayıkla (etiket-artefakt, içerik değil)."""
    t = _RE_DUSUNCE_LITERAL.sub("", (t or "").strip())
    return t.strip()


def sanitize_answer(a: str) -> str:
    """Sızan <BILGI_KARTI>/</BILGI_KARTI> etiketlerini ayıkla (kaynağın kendi deseni)."""
    a = _RE_BILGI_KARTI_TAG.sub("", (a or "").strip())
    return a.strip()


def kart_kalitesi(a: str) -> Tuple[bool, str]:
    """final_answer ADIM listesi değil KART mı? (yapısal kapı)"""
    if len(a) < MIN_CARD_LEN:
        return False, f"kart_kisa(<{MIN_CARD_LEN})"
    if _RE_ADIM_BASLANGICI.match(a) or a.startswith("**"):
        return False, "kart_adim_listesi"
    if "**" in a:
        return False, "kart_markdown"
    if not a.rstrip().endswith((".", "!", "?")):
        return False, "kart_yarida_kesilmis"
    return True, "temiz"


def load_vault(
    vault_path: str,
    dedup: bool,
    limit: Optional[int],
) -> Tuple[List[dict], Dict[str, int]]:
    """Kasayı oku: satır → sanitize → (isteğe bağlı) içerik-tekilleştirme."""
    ham_satir = 0
    json_hata = 0
    recs: List[dict] = []
    seen: set = set()
    for line in open(vault_path, "r", encoding="utf-8"):
        line_s = line.strip()
        if not line_s:
            continue
        ham_satir += 1
        try:
            r = json.loads(line_s)
        except Exception:
            json_hata += 1
            continue
        t = sanitize_thought(r.get("thought_trace", ""))
        a = sanitize_answer(r.get("final_answer", ""))
        rec = {
            "query": (r.get("query") or "").strip(),
            "instruction": (r.get("instruction") or "").strip(),
            "thought": t,
            "answer": a,
            "source": (r.get("source") or "").strip(),
            "timestamp": (r.get("timestamp") or "").strip(),
        }
        if dedup:
            key = hashlib.sha256(json.dumps(
                [rec["query"], rec["instruction"], rec["thought"], rec["answer"]],
                ensure_ascii=False).encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
        recs.append(rec)
    if limit is not None:
        recs = recs[:limit]
    stats = {
        "ham_satir": ham_satir,
        "json_hata": json_hata,
        "kapidan_sonra": len(recs),
    }
    return recs, stats


def build_tokenizer(vocab_path: str, lexicon_path: str):
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    # literal_entity_mode=False: kanonik pencere hattı (assemble_kulliyat_v2 / T-0209 deseni)
    tokenizer = KristalTokenizer(compiler, vocab)
    return vocab, compiler, tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P2.2: cot_vault.jsonl → data/train_reasoning_cot.bin (Evre-5 ilk derleme)")
    ap.add_argument("--vault", default=VAULT_PATH)
    ap.add_argument("--output", default=OUTPUT_BIN)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--lexicon", default=LEXICON_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="İşlenecek en fazla kayıt sayısı (tekilleştirme sonrası)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="İçerik-tekilleştirmeyi kapat (91 ham satırın tamamı işlenir)")
    ap.add_argument("--kalite-kapisi-yok", action="store_true",
                    help="Yapısal kart kapısını kapat (kirli kartlar da derlenir — teşhis için)")
    ap.add_argument("--unk-esik", type=float, default=UNK_ESIK_PCT,
                    help="Kayıt başına UNK oranı eşiği (%%; üzerindeki kayıt düşürülür)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Dosya YAZMA; tüm ölçüm ve doğrulamaları raporla")
    args = ap.parse_args()

    print("=" * 72)
    print(" P2.2 — EVRE-5 AKIL YÜRÜTME (CoT) VERİ DERLEME")
    print("=" * 72)
    print(f"  Kasa        : {args.vault}")
    print(f"  Çıktı       : {args.output}")
    print(f"  Sözlük      : {args.vocab}")
    print(f"  Kip         : {'DRY-RUN (dosya yazılmaz)' if args.dry_run else 'ÜRETİM'}")

    vocab, compiler, tokenizer = build_tokenizer(args.vocab, args.lexicon)
    vocab_size = len(vocab.stoi)
    unk_id = vocab.stoi.get("<UNK>", 0)
    bos_id = vocab.stoi.get("<BOS>")
    eos_id = vocab.stoi.get("<EOS>")
    out_id = vocab.stoi.get("<OUTPUT>")
    out_end_id = vocab.stoi.get("</OUTPUT>")
    dus_id = vocab.stoi.get("<DUSUNCE>")
    dus_end_id = vocab.stoi.get("</DUSUNCE>")
    print(f"\n[1] Sözlük yüklendi: {vocab_size} jeton (UNK id={unk_id}, "
          f"<DUSUNCE> id={dus_id}, </DUSUNCE> id={dus_end_id})")

    # ---- Kaynak yükle + kapılar ------------------------------------------------
    recs, src_stats = load_vault(args.vault, dedup=not args.no_dedup, limit=args.limit)
    print(f"[2] Kasa muhasebesi: ham_satir={src_stats['ham_satir']} "
          f"json_hata={src_stats['json_hata']} "
          f"{'icerik_benzersiz' if not args.no_dedup else 'ham'}={len(recs)}")

    kept: List[dict] = []
    drop_sebepler: Dict[str, int] = {}
    for i, r in enumerate(recs):
        if len(r["thought"]) < MIN_THOUGHT_LEN:
            drop_sebepler["dusunce_kisa"] = drop_sebepler.get("dusunce_kisa", 0) + 1
            continue
        if not args.kalite_kapisi_yok:
            ok, sebep = kart_kalitesi(r["answer"])
            if not ok:
                drop_sebepler[sebep] = drop_sebepler.get(sebep, 0) + 1
                continue
        kept.append(r)
    print(f"[3] Yapısal kapı: kabul={len(kept)} "
          f"elenen={sum(drop_sebepler.values())} {dict(drop_sebepler)}")

    if not kept:
        print("DURDURULDU: kapı sonrası geçerli kayıt kalmadı. "
              "--kalite-kapisi-yok ile teşhis koşabilirsiniz.")
        return 1

    # ---- SFT kayıtları (kalıbı kaynaktan: v1.5.0 format_reasoning_sample) ------
    sft_records = []
    for r in kept:
        out_text = f"<DUSUNCE> {r['thought']} </DUSUNCE> {r['answer']}".strip()
        sft_records.append({
            "instruction": r["instruction"] or "Soruyu derinlemesine akıl yürüterek açıkla.",
            "input": r["query"],
            "output": out_text,
        })

    # ---- Kanonik tokenizasyon --------------------------------------------------
    seqs: List[List[int]] = []
    unk_drop = 0
    rec_stats: List[dict] = []
    for rec in sft_records:
        # kanonik yol: encode(json.dumps(item)) → _normalize_text → yapısal zarf
        ids = tokenizer.encode(json.dumps(rec, ensure_ascii=False))
        n_unk = sum(1 for t in ids if t == unk_id)
        unk_pct = 100.0 * n_unk / len(ids)
        if unk_pct > args.unk_esik:
            unk_drop += 1
            rec_stats.append({"unk_pct": unk_pct, "kabul": False, "sebep": "unk_esigi"})
            continue
        seqs.append(ids)
        rec_stats.append({
            "unk_pct": round(unk_pct, 3), "kabul": True,
            "uzunluk": len(ids), "maks_id": max(ids),
        })

    total_tokens = sum(len(s) for s in seqs)
    if not seqs:
        print("DURDURULDU: UNK eşiği sonrası kayıt kalmadı. --unk-esik ile eşiği gözden geçirin.")
        return 1

    all_ids = np.array([t for s in seqs for t in s], dtype=np.uint16)
    maks_id = int(all_ids.max()) if len(all_ids) else 0
    n_unk = int((all_ids == unk_id).sum())
    bos_cnt = int((all_ids == bos_id).sum())
    eos_cnt = int((all_ids == eos_id).sum())
    out_cnt = int((all_ids == out_id).sum())
    out_end_cnt = int((all_ids == out_end_id).sum())
    dus_cnt = int((all_ids == dus_id).sum())
    dus_end_cnt = int((all_ids == dus_end_id).sum())
    unk_pct = 100.0 * n_unk / total_tokens

    # ---- SINIR KAPISI (train.py:284 deseni — max jeton id < vocab_size) ---------
    if maks_id >= vocab_size:
        print(f"DURDURULDU: maks jeton id={maks_id} >= vocab_size={vocab_size}. "
              f"Derleme yanlış sözükle yapılmış.")
        return 1
    # Bölümleme değişmezleri (bin_dekod_dogrula ile aynı)
    assert bos_cnt == len(seqs), f"<BOS> ({bos_cnt}) != kayıt ({len(seqs)})"
    assert eos_cnt == len(seqs), f"<EOS> ({eos_cnt}) != kayıt ({len(seqs)})"
    assert out_cnt == len(seqs), f"<OUTPUT> ({out_cnt}) != kayıt ({len(seqs)})"
    assert out_end_cnt == len(seqs), f"</OUTPUT> ({out_end_cnt}) != kayıt ({len(seqs)})"
    assert dus_cnt == len(seqs) and dus_end_cnt == len(seqs), "<DUSUNCE> bölümlemesi bozuk"

    print(f"[4] Tokenizasyon: kayıt={len(seqs)} token={total_tokens:,} "
          f"ort={total_tokens / len(seqs):,.1f} token/kayıt "
          f"(min={min(len(s) for s in seqs)}, maks={max(len(s) for s in seqs)})")
    print(f"    SINIR KAPISI: maks jeton id={maks_id} < vocab_size={vocab_size} (GÜVENLİ)")
    print(f"    UNK: {n_unk} jeton = %{unk_pct:.3f} (eşik %{args.unk_esik}; "
          f"UNK eşiğinden düşen: {unk_drop})")
    for i, st in enumerate(rec_stats):
        print(f"    - kayıt {i + 1}: uzunluk={st.get('uzunluk')} "
              f"unk=%{st.get('unk_pct')} {'OK' if st['kabul'] else 'DÜŞÜRÜLDÜ'}")

    # ---- Örnek geri-çözüm (decompile_sentence) ----------------------------------
    print("\n[5] Örnek geri-çözüm (ilk 2 kayıt, MorphemeDecompiler):")
    dec = MorphemeDecompiler(compiler, vocab)
    for i, s in enumerate(seqs[:2]):
        tags = tokenizer.decode(s)
        metin = dec.decompile_sentence(tags)
        print(f"  --- kayıt {i + 1} (jeton={len(s)}) ---")
        print(f"  {metin[:600]}")

    if args.dry_run:
        print("\n[DRY-RUN] Dosya YAZILMADI. Üretim için --dry-run olmadan koşun.")
        return 0

    # ---- İkili çıktı + meta -----------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_ids.tofile(args.output)

    boundaries = []
    off = 0
    for s in seqs:
        boundaries.append({"offset": off, "length": len(s)})
        off += len(s)

    meta = {
        "dataset_name": "train_reasoning_cot",
        "evre": "Evre-5 akıl yürütme derlemesi (P2.2, İLK derleme)",
        "ureten_betik": "scripts/prepare_reasoning_dataset_p22.py",
        "ureten_gorev": "P2.2",
        "damga_utc": datetime.now(timezone.utc).isoformat(),
        "block_size": BLOCK_SIZE,
        "total_records": len(seqs),
        "record_count": len(seqs),
        "total_tokens": int(total_tokens),
        "sozluk": os.path.relpath(args.vocab, KOK),
        "sozluk_giris": vocab_size,
        "sozluk_sha256": sha256_file(args.vocab),
        "lexicon": os.path.relpath(args.lexicon, KOK),
        "lexicon_sha256": sha256_file(args.lexicon),
        "literal_entity_mode": False,
        "kaynak": {
            "vault": os.path.relpath(args.vault, KOK),
            "vault_sha256": sha256_file(args.vault),
            "ham_satir": src_stats["ham_satir"],
            "json_hata": src_stats["json_hata"],
            "icerik_benzersiz": src_stats["kapidan_sonra"],
            "kabul_edilen": len(seqs),
            "yapisal_elenen": dict(drop_sebepler),
            "unk_esigi_pct": args.unk_esik,
            "unk_elenen": unk_drop,
        },
        "cikti_istatistik": {
            "min_jeton": min(len(s) for s in seqs),
            "maks_jeton": max(len(s) for s in seqs),
            "ort_jeton": round(total_tokens / len(seqs), 2),
            "maks_jeton_id": maks_id,
            "unk_adet": n_unk,
            "unk_oran_pct": round(unk_pct, 4),
            "dusunce_bolum": {"<DUSUNCE>": dus_cnt, "</DUSUNCE>": dus_end_cnt},
        },
        "boundaries": boundaries,
    }
    meta_path = args.output + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[6] Çıktı yazıldı: {args.output} "
          f"({os.path.getsize(args.output) / 1024:.2f} KB) + {meta_path}")

    # ---- Bağımsız doğrulama: dataset_compiler.bin_dekod_dogrula -----------------
    dogrulama = bin_dekod_dogrula(args.output, meta_path, vocab)
    print(f"[7] bin_dekod_dogrula: OK — kayıt={dogrulama['total_records']} "
          f"token={dogrulama['total_tokens']:,} "
          f"BOS={dogrulama['bos_count']} EOS={dogrulama['eos_count']} "
          f"<OUTPUT>={dogrulama['output_count']} </OUTPUT>={dogrulama['output_end_count']}")

    print("=" * 72)
    print(f" P2.2 TAMAM — {len(seqs)} kayıt / {total_tokens:,} jeton → {args.output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())