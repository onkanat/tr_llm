"""src/llm/frozen_guard.py

Kristal-Vektörel: Donmuş Yol Güvenlik Muhafızı (Frozen Guard)
==============================================================
Bu modül `.agent-bus/frozen.json` kütüğünü tek hakikat kaynağı (Single Source of Truth)
olarak okur ve donmuş yollara yapılacak yetkisiz yazma işlemlerini denetler.

Tasarım ilkeleri:
- Saf fonksiyonlar, tip ipuçları tam.
- FAIL-CLOSED: Dosya yoksa, okunamıyorsa veya JSON bozuksa/desen içermiyorsa RuntimeError fırlatılır.
  Asla sessizce boş liste veya False dönülmez.
- `fnmatch.fnmatch` hem `os.path.normpath(path)` hem de ham `path` üzerinde uygulanır.
"""

from __future__ import annotations

import os
import json
import fnmatch
from typing import List

DEFAULT_FROZEN_FILE = ".agent-bus/frozen.json"


def load_frozen_patterns(frozen_file: str = DEFAULT_FROZEN_FILE) -> List[str]:
    """Donmuş desenler listesini frozen_file dosyasından okur.

    Fail-closed mantığıyla çalışır: Dosya mevcut değilse, okunamıyorsa,
    JSON biçimi geçersizse veya 'patterns' listesi eksik/boş ise RuntimeError üretir.
    """
    if not os.path.exists(frozen_file):
        raise RuntimeError(f"[FAIL-CLOSED] Donmuş desen dosyası bulunamadı: {frozen_file}")

    try:
        with open(frozen_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"[FAIL-CLOSED] Donmuş desen dosyası okunamadı veya bozuk: {frozen_file} ({e})") from e

    if not isinstance(data, dict) or "patterns" not in data or not isinstance(data["patterns"], list):
        raise RuntimeError(f"[FAIL-CLOSED] Geçersiz desen formatı: {frozen_file} ('patterns' listesi bulunamadı)")

    patterns = data["patterns"]
    if not patterns:
        raise RuntimeError(f"[FAIL-CLOSED] Donmuş desen listesi boş olamaz: {frozen_file}")

    return patterns


def is_frozen_path(path: str, frozen_file: str = DEFAULT_FROZEN_FILE) -> bool:
    """Belirtilen yolun donmuş yol desenlerinden birine uyup uymadığını kontrol eder.

    Fail-closed: Dosya yüklenemezse RuntimeError fırlatılır.
    Hem ham yol hem de normalize edilmiş yol üzerinde desen eşleştirmesi yapılır.
    """
    patterns = load_frozen_patterns(frozen_file)
    norm_path = os.path.normpath(path)
    for p in patterns:
        if fnmatch.fnmatch(norm_path, p) or fnmatch.fnmatch(path, p):
            return True
    return False


def check_frozen_save_path(path: str, allow_frozen_write: bool = False, frozen_file: str = DEFAULT_FROZEN_FILE) -> None:
    """Belirtilen kaydetme yolunun donmuş olup olmadığını denetleyen kanonik yardımcı.

    Fail-closed: Donmuş yola yazma izni (allow_frozen_write=True) açıkça verilmemişse RuntimeError fırlatır.
    """
    if is_frozen_path(path, frozen_file=frozen_file) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {path} (allow_frozen_write=False)")

