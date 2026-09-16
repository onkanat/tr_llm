"""tests/test_f2_producers.py

T-0042 — F2 veri üreticileri (chat_balanced, carpenter_specialization, balanced_sft)
için donmuş çıktı yolu kapısı, muhasebeli tokenizasyon ve D4 özdeşlik testleri.
"""

import os
import sys
import json
import pytest

from scripts.prepare_chat_balanced_dataset import (
    check_output_path as check_chat,
    tokenize_jsonl as tok_chat,
)
from scripts.prepare_carpenter_specialization_dataset import (
    check_output_path as check_carp,
    tokenize_jsonl as tok_carp,
)
from scripts.prepare_balanced_sft_dataset import (
    check_output_path as check_sft,
    tokenize_jsonl as tok_sft,
)


class DummyTokenizer:
    """Yalnızca encode metoduna sahip minimal test tokenizer'ı."""
    def encode(self, text: str) -> list:
        # 'error' kelimesi içeriyorsa hata fırlatır (encode_error testi için)
        if "ENCODE_FAIL" in text:
            raise ValueError("Bilinmeyen morfolojik kodlama")
        # Kısa metin için 1 token, normal metin için 10 token üretir
        if "KISA" in text:
            return [1]
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def four_line_jsonl(tmp_path):
    """4 satırlık test JSONL kütüğü:
    Satır 1: Geçerli - uzun (kept)
    Satır 2: Bozuk JSON (json_error)
    Satır 3: Geçerli - kısa (short_skipped, len <= 2)
    Satır 4: Geçerli - uzun (kept)
    """
    jsonl_file = tmp_path / "sample_accounting.jsonl"
    lines = [
        json.dumps({"prompt": "Bu birinci geçerli uzun satırdır."}, ensure_ascii=False),
        "{BOZUK_JSON: 123",  # Bozuk satır
        json.dumps({"prompt": "KISA"}, ensure_ascii=False),  # Kısa satır (len=1)
        json.dumps({"prompt": "Bu dördüncü geçerli uzun satırdır."}, ensure_ascii=False),
    ]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(jsonl_file)


# ==============================================================================
# 1. Donmuş Çıktı Kapısı Testleri (check_output_path)
# ==============================================================================

@pytest.mark.parametrize(
    "check_fn, frozen_path",
    [
        (check_chat, "data/train_chat_balanced.bin"),
        (check_carp, "data/train_carpenter_specialization.bin"),
        (check_sft, "data/train_balanced_sft.bin"),
    ]
)
def test_check_output_path_frozen_behavior(check_fn, frozen_path):
    # 1. Donmuş hedef + allow_frozen_write=False -> RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        check_fn(frozen_path, allow_frozen_write=False)
    assert "Donmus yola yazma engellendi" in str(exc_info.value)

    # 2. Donmuş hedef + allow_frozen_write=True -> İstisna YOK
    check_fn(frozen_path, allow_frozen_write=True)

    # 3. Pozitif kontrol: Donmuş olmayan hedef + allow_frozen_write=False -> İstisna YOK
    # Not: data/*.bin deseni fnmatch gereği data/eval/*.bin yolunu da yakaladığından,
    # donmamış dizin örneği olarak scratch/ dizini veya tmp_path kullanılır.
    check_fn("scratch/gecici_probe.bin", allow_frozen_write=False)


# ==============================================================================
# 2. tokenize_jsonl Muhasebe ve D4 Özdeşlik Testleri
# ==============================================================================

@pytest.mark.parametrize(
    "tokenize_fn",
    [
        tok_chat,
        tok_carp,
        tok_sft,
    ]
)
def test_tokenize_jsonl_accounting_and_identity(tokenize_fn, four_line_jsonl):
    dummy_tok = DummyTokenizer()
    records, stats = tokenize_fn(four_line_jsonl, dummy_tok, min_tokens=2)

    # Beklenen 7 sayaç kontrolü
    assert stats["missing"] is False
    assert stats["raw_lines"] == 4
    assert stats["sampled"] == 4
    assert stats["json_errors"] == 1
    assert stats["encode_errors"] == 0
    assert stats["lines_ok"] == 3
    assert stats["short_skipped"] == 1
    assert stats["kept"] == 2
    assert len(records) == 2

    # D4 Özdeşlik denetimi
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"]


# ==============================================================================
# 3. T-0043: Uçtan Uca main() Kapı Bağlantı Testi (A)
# ==============================================================================

import ast
import inspect
from scripts import (
    prepare_chat_balanced_dataset as chat_module,
    prepare_carpenter_specialization_dataset as carp_module,
    prepare_balanced_sft_dataset as sft_module,
)
from scripts.prepare_carpenter_specialization_dataset import tokenize_carpenter_jsonl


@pytest.mark.parametrize(
    "mod, expected_frozen_bin",
    [
        (chat_module, "data/train_chat_balanced.bin"),
        (carp_module, "data/train_carpenter_specialization.bin"),
        (sft_module, "data/train_balanced_sft.bin"),
    ]
)
def test_main_gate_raises_on_default_allow_frozen_write(mod, expected_frozen_bin):
    """(A) main() varsayılan allow_frozen_write=False ile çağrıldığında donmuş çıktı kapısı
    nedeniyle derhal RuntimeError fırlatmalıdır (hızlı ve yazmasız).
    """
    with pytest.raises(RuntimeError) as exc_info:
        mod.main()
    err_msg = str(exc_info.value)
    assert "Donmus yola yazma engellendi" in err_msg, (
        f"{mod.__name__}.main() beklenen hata mesajını vermedi: {err_msg}"
    )
    assert expected_frozen_bin in err_msg, (
        f"{mod.__name__}.main() hata mesajında {expected_frozen_bin} geçmiyor: {err_msg}"
    )


# ==============================================================================
# 4. T-0043: Yapısal Test (AST ile check_output_path ve tofile Sıralaması) (B)
# ==============================================================================

@pytest.mark.parametrize(
    "mod",
    [
        chat_module,
        carp_module,
        sft_module,
    ]
)
def test_ast_main_has_gate_before_tofile(mod):
    """(B) Yapısal test (AST): main() gövdesinde:
    1. En az bir check_output_path çağrısı bulunmalıdır.
    2. En az bir tofile çağrısı bulunmalıdır.
    3. Son check_output_path çağrısının satır numarası ilk tofile çağrısından küçük olmalıdır.
    """
    source = inspect.getsource(mod.main)
    tree = ast.parse(source)

    check_output_lines = []
    tofile_lines = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # ast.Call, func ast.Name id == "check_output_path"
            if isinstance(node.func, ast.Name) and node.func.id == "check_output_path":
                check_output_lines.append(node.lineno)
            # ast.Call, func ast.Attribute attr == "tofile"
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "tofile":
                tofile_lines.append(node.lineno)

    assert len(check_output_lines) >= 1, (
        f"{mod.__name__}.main() gövdesinde 'check_output_path' çağrısı bulunamadı!"
    )
    assert len(tofile_lines) >= 1, (
        f"{mod.__name__}.main() gövdesinde 'tofile' çağrısı bulunamadı!"
    )
    assert max(check_output_lines) < min(tofile_lines), (
        f"{mod.__name__}.main() içinde check_output_path çağrısı tofile çağrısından önce gelmiyor! "
        f"Son kapı: {max(check_output_lines)}, İlk tofile: {min(tofile_lines)}"
    )


# ==============================================================================
# 5. T-0043: Carpenter Muhasebe ve Normalize Encode Hatası Testi (C)
# ==============================================================================

class CarpenterFailNormalizeDummyTokenizer:
    """Yalnızca normalize SFT isteminde hata veren test tokenizer'ı."""
    def encode(self, text: str) -> list:
        if "Ahşap uzmanı olarak cevapla." in text:
            raise ValueError("Normalize SFT encode fail")
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_carpenter_accounting_normalize_encode_failure(tmp_path):
    """(C) Sahte tokenizer yalnızca normalize metinde hata fırlatır:
    Beklenen:
      sampled=2, json_errors=0, encode_errors=0, normalize_encode_errors=2,
      lines_ok=2, kept=2, short_skipped=0
      D4 Özdeşliği: sampled == json_errors + encode_errors + lines_ok (2 == 0 + 0 + 2)
      Invariant: kept <= 2 * lines_ok (2 <= 4)
    T-0042'de bu senaryo assert ateşliyordu (2 != 0 + 2 + 2); T-0043 ile düzeltildi.
    """
    lines = [
        json.dumps({"instruction": "Soru 1", "input": "Ağaç nasıl kesilir?", "output": "Cevap 1"}),
        json.dumps({"instruction": "Soru 2", "input": "Ceviz ağacı özellikleri nelerdir?", "output": "Cevap 2"}),
    ]
    sample_file = tmp_path / "carpenter_sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    tok = CarpenterFailNormalizeDummyTokenizer()
    records, stats = tokenize_carpenter_jsonl(str(sample_file), tok)

    assert stats["sampled"] == 2
    assert stats["json_errors"] == 0
    assert stats["encode_errors"] == 0
    assert stats["normalize_encode_errors"] == 2
    assert stats["lines_ok"] == 2
    assert stats["kept"] == 2
    assert stats["short_skipped"] == 0
    assert len(records) == 2

    # D4 Özdeşliği
    assert stats["sampled"] == stats["json_errors"] + stats["encode_errors"] + stats["lines_ok"]
    # Tavan invariantı
    assert stats["kept"] <= 2 * stats["lines_ok"]

