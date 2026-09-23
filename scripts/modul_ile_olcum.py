#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KANONİK YETENEK KABI + MODÜL — yeteneği (ezber · ROUGE-L · kesişim) modülle ölç.

NEDEN SARMA, NEDEN YENİ DOSYA
  `scripts/evaluate_carpenter_anka.py` ceket eksenini (ezber · tutarsızlık · ROUGE-L ·
  içerik kesişimi) oracle katmanıyla birlikte tam ölçer, ama modül YÜKLEYEMEZ. Ona
  `--module` eklemek SATIR KAYDIRIR; o betiğin `:62-69` satırları T-0098'in
  `test_esik_referanslari` kapısında REFERANS olarak geçiyor ⇒ kapanmış bir kapı düşer.
  Bu yüzden kanonik betik HİÇ DEĞİŞTİRİLMEZ: `model_yukle` burada SARILIR ve modül, model
  yüklendikten SONRA takılır. Yetenek sayılarını üreten kod kanonik kodun KENDİSİDİR ⇒
  T-0095/T-0096 kayıtlarıyla doğrudan kıyaslanabilir.

  Sarma sırası bilerek seçildi: kanonik `model_yukle` modeli `.to(device)` + `.eval()`
  yapar, modül TAKMA ondan SONRA gelir — gerçek değerlendirme yolunun sırası budur
  (yükle → cihaza taşı → tak) ve `LoRAKatmani`'nin cihaz devralma düzeltmesini zorunlu
  ateşler (T-0098'de ölçülen üçüncü cihaz kusuru tam bu sırada yaşıyordu).

VAKUM KAPISI (zorunlu): modül takılmadan ÖNCE ve SONRA aynı sabit girdiyle logit farkı
ölçülür. Fark TAM 0 ise modül eğitilmemiş (delta=0, no-op) demektir ⇒ koşum VAKUMDUR:
"yetenek değişmedi" hükmü kurulamaz, DURULUR.

BASELINE TUZAĞI (ölçüldü, tasarım gereği kapatıldı): modül koşumunda `--model` de
`--baseline` da AYNI taban dosyasıdır — modül yan dosyadır. Sarma yalnız **ilk** uyan
çağrıya takılır ⇒ `--baseline` çağrısı TEMİZ kalır. Aksi hâlde "unutma" kıyaslaması
modülü tabana da takar ve ölçüm kendini ölçer (sahte sıfır).

Kullanım (kanonik bayraklar AYNEN geçer):
  venv/bin/python scripts/modul_ile_olcum.py --module modules/x.mod.pt \
      --model data/anka_a1r.pt --baseline data/anka_a1r.pt --ceket-ekseni \
      --output data/eval/... .json --modul-cikti data/eval/... .modul.json --device mps
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import scripts.evaluate_carpenter_anka as ECA  # kanonik kap — DEĞİŞTİRİLMEZ
from src.llm.modules import modul_katmanlari, modul_ozeti, modul_yukle, sha256_dosya

# Kanonik yükleyicinin ORİJİNALİ, sarmadan ÖNCE alınır. Sonra alınırsa sarma kendini
# çağırır (ölçüldü: RecursionError, 995 seviye) — kopya değil, aynı fonksiyon.
_ORIJINAL_MODEL_YUKLE = ECA.model_yukle

KENDI_BAYRAKLAR: Tuple[str, ...] = ("--module", "--modul-cikti")

# --- sarma durumu (yalnız bu betiğin içinde; kanonik betiğe sızmaz) ---
_HEDEF_ABS: Optional[str] = None
_MODUL_YOL: Optional[str] = None
_TAKILDI: List[Dict[str, Any]] = []


def durdur(mesaj: str) -> None:
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def _deger(argv: List[str], ad: str, varsayilan: Optional[str] = None) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == ad and i + 1 < len(argv):
            return argv[i + 1]
    return varsayilan


def sha256_yol_safe(yol: Optional[str]) -> Optional[str]:
    """Kanonik `sha256_yol` yerine güvenli sarmalayıcı: yol yoksa None (kanonik None döner)."""
    if not yol or not os.path.exists(yol):
        return None
    return sha256_dosya(yol)


def _ayikla(argv: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Kendi bayraklarımızı argv'den ÇIKAR — kanonik argparse bilmediği bayrakta ölür."""
    alinan: Dict[str, str] = {}
    kalan: List[str] = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in KENDI_BAYRAKLAR:
            if i + 1 >= len(argv):
                durdur(f"{a} değer bekliyor")
            alinan[a] = argv[i + 1]
            i += 2
            continue
        kalan.append(a)
        i += 1
    return alinan, kalan


def _sarilan_model_yukle(model_path: str, vocab: Any, device: torch.device):
    """Kanonik yükleyiciyi çağır, SONRA (yalnız hedef çağrıya) modülü tak."""
    model, kim = _ORIJINAL_MODEL_YUKLE(model_path, vocab, device)
    if _TAKILDI or _MODUL_YOL is None:
        return model, kim
    if os.path.abspath(str(model_path)) != _HEDEF_ABS:
        return model, kim

    # --- VAKUM KAPISI: takmadan önce/sonra AYNI sabit girdi ---
    torch.manual_seed(99)
    x = torch.randint(0, len(vocab.stoi), (1, 24), dtype=torch.long, device=device)
    with torch.no_grad():
        once, _ = model(x)
    meta = modul_yukle(model, _MODUL_YOL, str(model_path))  # digest + vocab kapıları içinde
    model.eval()
    with torch.no_grad():
        sonra, _ = model(x)
    fark = float((sonra - once).abs().max().item())
    # KANONİK YÜKLEM KULLANILIR (ölçüldü, 22 Eyl 2026): burada katmanlar SINIF ADI DİZESİYLE
    # sayılıyordu ("LoRAKatmani"). `LoRAEmbedding` eklenince sayaç onu saymadı (38 -> 37) ve
    # fail-closed kapı haklı olarak DURDU — sayı üretmek yerine. Kusur modülde değil, yüklemin
    # KOPYASINDA: yazan ile arayan aynı olmalı ([[indeks-yazan-ile-arayan-normalizasyonu-ayni-olmali]]).
    katman = len(modul_katmanlari(model))
    print(f"[sarma] modül TAKILDI: {_MODUL_YOL}\n"
          f"[sarma] {katman} katman · {meta.get('modul_parametre', -1):,} parametre · "
          f"eğitim adımı={meta.get('adim')} · vocab={meta.get('vocab_size')}\n"
          f"[sarma] vakum kontrolü: takma öncesi/sonrası logit farkı maks {fark:.6e}", flush=True)
    if fark == 0.0:
        durdur("VAKUM: modül takıldı ama çıktı BİREBİR aynı (delta tam 0). Eğitilmemiş ya da "
               "boş modül ⇒ 'yetenek değişmedi' hükmü KURULAMAZ.")
    if katman == 0 or meta.get("katman_sayisi") != katman:
        durdur(f"katman sayısı tutmuyor: takılan {katman}, meta {meta.get('katman_sayisi')}")
    _TAKILDI.append({"cagri_yolu": str(model_path), "logit_fark_maks": fark, "meta": meta})
    return model, kim


def main() -> int:
    alinan, kalan = _ayikla(list(sys.argv))
    modul_yol = alinan.get("--module")
    if not modul_yol:
        durdur("--module ZORUNLU (modülsüz ölçüm için kanonik betiği doğrudan koşun)")
    if not os.path.exists(modul_yol):
        durdur(f"--module dosyası YOK: {modul_yol}")

    model_yol = _deger(kalan, "--model")
    if not model_yol:
        durdur("--model ZORUNLU (kanonik kap da zorunlu tutar)")
    if "--ceket-ekseni" not in kalan:
        durdur("--ceket-ekseni ZORUNLU: bu kabın varlık nedeni YETENEK eksenidir; "
               "kapalıyken çalıştırmak ölçümsüz koşumdur")
    if "--output" not in kalan:
        durdur("--output ZORUNLU (kanonik kap da zorunlu tutar)")
    if not os.path.exists(model_yol):
        durdur(f"--model dosyası YOK: {model_yol}")

    global _HEDEF_ABS, _MODUL_YOL
    _HEDEF_ABS = os.path.abspath(model_yol)
    _MODUL_YOL = modul_yol

    om = modul_ozeti(modul_yol)
    print("=" * 78)
    print(" KANONİK YETENEK KABI + MODÜL (T-0095/T-0096 ile kıyaslanabilir şema)")
    print("=" * 78)
    print(f"modül  : {modul_yol} ({sha256_dosya(modul_yol)[:16]}…) adım={om.get('adim')} "
          f"r={om['spec'].get('r')} lr={om.get('lr')} veri={om.get('veri')}", flush=True)
    print(f"taban  : {model_yol} ({sha256_dosya(model_yol)[:16]}…)", flush=True)
    print(f"modülün kaydettiği taban: {om.get('base_yol')} {str(om.get('base_sha256'))[:16]}…",
          flush=True)

    ECA.model_yukle = _sarilan_model_yukle  # kanonik main bu global'i çağırır
    sys.argv = kalan
    rc = ECA.main()

    if rc == 0 and _TAKILDI:
        cikti = _deger(kalan, "--output")
        yan = alinan.get("--modul-cikti")
        if yan:
            kayit: Dict[str, Any] = {
                "modul": modul_yol, "modul_sha256": sha256_dosya(modul_yol),
                "taban": model_yol, "taban_sha256": sha256_dosya(model_yol),
                "modul_meta": om, "takma": _TAKILDI[0],
                "kanonik_kayit": cikti,
                "kanonik_kayit_sha256": sha256_yol_safe(cikti),
                "not": "Sayılar kanonik `evaluate_carpenter_anka.py` tarafından üretildi; "
                       "bu dosya yalnız modül kimliğini ve vakum kontrolünü kaydeder.",
            }
            os.makedirs(os.path.dirname(os.path.abspath(yan)), exist_ok=True)
            with open(yan, "w", encoding="utf-8") as f:
                json.dump(kayit, f, ensure_ascii=False, indent=2)
            print(f"[yan] modül kimliği yazıldı: {yan}")
    elif rc == 0 and not _TAKILDI:
        durdur("kanonik koşum bitti ama modül HİÇ takılmadı (sarma hedefi tutmadı) — "
               f"hedef {_HEDEF_ABS}, --model {model_yol}. Ölçüm TABAN ölçümüdür.")
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main() or 0)
