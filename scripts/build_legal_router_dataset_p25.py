#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.5 Faz-2B — legal router veri seti derleyicisi (v3).

data/pedagogy/legal_canonical_sft.jsonl'in 463 kaydından (462 benzersiz istem)
SEED-42 ile 50 istem seçer + 10 el derlemesi doğal hukuk sorgusu ekler
(probe sorguları HARİÇ) → data/pedagogy/legal_router_v3_dataset.jsonl (60 kayıt).

Kayıt kalıbı: kaynak dosya ailesiyle birebir {"instruction", "input", "output"}
(istem `instruction` alanında, `input` boş — train_router_v2_pure._metinler
fallback zinciri input→instruction→soru bunu okur; kanıt: legal_canonical_sft.jsonl).

Uydurmasızlık: doğal sorgular soru-cümlesidir, madde no/alıntı uydurmaz;
kanun adları korpusun kapsadığı gerçek kanunlardır (657/6331/6446/6502/5271/
6698/4708/6100/5378/2577). Probe 10 sorgusu (p25 probu) bu dosyada YOK.
"""
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAYNAK = "data/pedagogy/legal_canonical_sft.jsonl"
KAYNAK_SHA = "69670bc6ddbc033cff4a38a044ef0ccd5a82c278cfd8f564755ea70634734e9c"
CIKTI = "data/pedagogy/legal_router_v3_dataset.jsonl"
SEED = 42
N_ISTEM = 50

DOAL_SORGULAR = [
    "Devlet Memurları Kanunu'na göre memurlara verilen disiplin cezaları nelerdir?",
    "İş Sağlığı ve Güvenliği Kanunu'nda işverenin risk değerlendirmesi yükümlülüğü nedir?",
    "Elektrik Piyasası Kanunu'nda lisans başvurusu hangi koşullara tabidir?",
    "Tüketicinin Korunması Hakkında Kanun'da mesafeli satışlarda cayma hakkı süresi ne kadardır?",
    "Ceza Muhakemeleri Kanunu'nda soruşturma evresinde şüphelinin hakları nelerdir?",
    "Kişisel Verilerin Korunması Kanunu'na göre veri sorumlusunun aydınlatma yükümlülüğü nedir?",
    "Yapı Denetimi Hakkında Kanun'a göre yapı denetim hizmet sözleşmesi kimlerle yapılır?",
    "Hukuk Muhakemeleri Kanunu'nda kanıtların sunulması için öneri süresi ne kadardır?",
    "Engelliler Hakkında Kanun'a göre erişilebilirlik denetimi kim tarafından yapılır?",
    "İdari Yargılama Usulü Kanunu'na göre vergi mahkemesinde dava açma süresi nedir?",
]


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def main() -> int:
    tam = os.path.join(REPO_KOK, KAYNAK)
    olc = _sha256(tam)
    if olc != KAYNAK_SHA:
        print("KAYNAK-SHA SAPMA: %s (beklenen %s, ölçülen %s)" % (KAYNAK, KAYNAK_SHA[:12], olc[:12]),
              file=sys.stderr)
        return 2

    istemler: List[str] = []
    with open(tam, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            m = (rec.get("instruction") or "").strip()
            if m:
                istemler.append(m)
    benzersiz = sorted(set(istemler))
    rng = random.Random(SEED)
    rng.shuffle(benzersiz)
    secilen_istem = benzersiz[:N_ISTEM]

    kayitlar: List[Dict[str, Any]] = []
    for istem in secilen_istem:
        kayitlar.append({"instruction": istem, "input": "", "output": ""})
    for sorgu in DOAL_SORGULAR:
        kayitlar.append({"instruction": sorgu, "input": "", "output": ""})

    cikti_tam = os.path.join(REPO_KOK, CIKTI)
    with open(cikti_tam, "w", encoding="utf-8") as f:
        for rec in kayitlar:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tum = [r["instruction"] for r in kayitlar]
    ozet = {
        "kaynak": KAYNAK, "kaynak_sha256": olc,
        "cikti": CIKTI, "cikti_sha256": _sha256(cikti_tam),
        "benzersiz_istem_havuzu": len(set(istemler)),
        "secilen_istem": len(secilen_istem),
        "dogal_sorgu": len(DOAL_SORGULAR),
        "toplam_kayit": len(kayitlar),
        "benzersiz_kayit": len(set(tum)),
        "sablon": "instruction/input/output (legal_canonical_sft kalıbı; input boş)",
        "seed": SEED,
    }
    print(json.dumps(ozet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())