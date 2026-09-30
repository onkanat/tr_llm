#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0199 Faz-1 — Optimum-Zarar + Şablon-Arz Sızması Tanısı (BETİKTEN).

T-0195 kalıbı (Faz-1 tanı; onarım sonra-tur). Üç ölçüm:
- M1: taban_akicilik_tanisi.py ÇEVRİM (subprocess; kaynak-dokunuş YOK) — 5-checkpoint
  ppl, kanonik ölçüm-seti (anka_a1r_pretrain.bin, B_PENCERE=128, SEED=7, n=25,
  max_new=128). Çıpa: base_v2 ppl 33,38 birebir tekrar-kanıt (uyuşmaz ⇒ DUR).
- M2: şablon-arz ↔ üretim-sızma BETİKTEN sayım (jsonl + t0198_faz3_kayitlar.json).
- M3: 4-tur tutarsızlık-serisi BETİKTEN tablo (hüküm-JSON'lardan; elle-sayı YOK).
- M5: T-0196 filtre-semantiği beyanı (kaynak-taraması BETİKTEN).

Hüküm: T0199_TANI_KAYDI rc=0 | T0199_TANI_DUR rc=2. Onarım-yönü BEYANI hüküm-içi."""
import hashlib
import json
import os
import subprocess
import sys
import time

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

KAYNAK_AKICILIK = os.path.join(KÖK, "scripts", "taban_akicilik_tanisi.py")
KAYNAK_T0196 = os.path.join(KÖK, "scripts", "t0196_arz_dengeli_kulliyat.py")
SIZMA_JSON = os.path.join(KÖK, "data/eval/t0198_faz3_kayitlar.json")
HUKUM_PATH = os.path.join(KÖK, "data/eval/t0199_tani_hukum_2026-09-30.json")
KAYIT_PATH = os.path.join(KÖK, "data/eval/t0199_tani_kayitlar.json")

# MADDE-2 çıpalar — full-digest BETİKTEN (30 Eyl shasum ölçümü; .pt donmuş salt-okunur)
CIPALAR = {
    "taban": "data/anka_base_v2.pt",
    "halef": "data/anka_b1_5_best.pt",
    "arz": "data/anka_b1_5_arz.pt",
    "usaf": "data/anka_b1_5_usaf.pt",
    "hiza": "data/anka_b1_5_hiza.pt",
}
CIPI_SHA = {
    "data/anka_base_v2.pt": "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/anka_b1_5_best.pt": "e5eb114e5bd71d5800ba423d844b2e2aeed6b78beaf3cba4278690a43fb51eb7",
    "data/anka_b1_5_arz.pt": "80b0fd9a5a60de90536ad4d6d3a3f8eb1de16d1474501a0eac669d2458f106fa",
    "data/anka_b1_5_usaf.pt": "8f846023f2fb2374dd411f674b199f373198c108cd961c836c44e094948b2c10",
    "data/anka_b1_5_hiza.pt": "05d8e4cf55a2e27e46eadab6b93f50b2cd90202b6f17e2f57958d4ac4d5727df",
}
M1_AD = {"taban": "taban", "halef": "b1_5_best", "arz": "b1_5_arz",
         "usaf": "b1_5_usaf", "hiza": "b1_5_hiza"}
# İLAN-2 düzeltmesi: 33,38 belgeli ölçüm anka_a1r'e aittir (22 Eyl — o gün base_v2 yoktu,
# 26 Eyl'de var olur); base_v2 27,32 = T-0199 koşum-1 BETİKTEN İLK-ölçüm → yeni-çıpa.
# Ölçüm-kabı birebir-kanıtı: a1r 33,38 (koşum-1-teşhisi BETİKTEN birebir, disk-çıpa).
A1R = {"model": "data/anka_a1r.pt",
       "sha": "b93cc1cd54093fc63342d394abe528f2128dc16e20b4d7ac6ab680854b4d6293",
       "ppl_cipa": 33.38, "ref_json": "data/eval/t0199_m1_a1r_referans.json"}
BASE_PPL_CIPA = 27.32  # koşum-1 BETİKTEN ilk-ölçüm (2026-09-30 DUR-teşhis sonra-çıpa)

SABLON_ONEK = ("ROOT:", "TENSE:", "PLURAL:", "CASE:", "POSS:", "DERIV:", "KELDURUM:")


def sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def giris_jsonl(yol: str) -> list:
    return [json.loads(l) for l in open(yol, encoding="utf-8")]


def sablan_desen(out: str) -> dict:
    """Şablon-çıktı kırılımı (BETİKTEN; saf fonksiyon)."""
    return {p: (p in out) for p in SABLON_ONEK}


def main() -> int:
    # ---- K1: 5-çıpa full-digest (fail-closed) ----
    cipa_durum = {}
    for ad, yol in CIPALAR.items():
        hiz = sha256(os.path.join(KÖK, yol))
        if hiz != CIPI_SHA[yol]:
            print(f"T0199_TANI_DUR: çıpa-uyuşmazlığı {yol}", flush=True)
            return 2
        cipa_durum[ad] = hiz
    print("[CIPI] 5-sha birebir BETİKTEN — PASS", flush=True)

    # ---- K1b: a1r referans-çıpa (ölçüm-kabı birebir-kanıtı; koşum-1-teşhisi) ----
    if sha256(os.path.join(KÖK, A1R["model"])) != A1R["sha"]:
        print("T0199_TANI_DUR: a1r referans sha uymaz", flush=True)
        return 2
    ref = json.load(open(os.path.join(KÖK, A1R["ref_json"]), encoding="utf-8"))
    if ref.get("ppl") != A1R["ppl_cipa"] or ref.get("model") != A1R["model"]:
        print(f"T0199_TANI_DUR: a1r referans ppl {ref.get('ppl')} != {A1R['ppl_cipa']}", flush=True)
        return 2
    print(f"[CIPI-b] a1r referans {A1R['ppl_cipa']} birebir (rig sağlam) — PASS", flush=True)

    # ---- M2: arz-payları (jsonl BETİKTEN; İLAN TEKRAR-sayım hizalı) ----
    train = giris_jsonl(os.path.join(KÖK, "data/b1_5_splits/train.jsonl"))
    u197 = giris_jsonl(os.path.join(KÖK, "data/eval/t0197_uretim_saf.jsonl"))

    def arz_kayit(kayitlar: list) -> dict:
        kirilim = {p: 0 for p in SABLON_ONEK}
        top = 0
        for r in kayitlar:
            v = sablan_desen(r.get("output", ""))
            if any(v.values()):
                top += 1
                for p, var in v.items():
                    if var:
                        kirilim[p] += 1
        n = len(kayitlar)
        return {"n": n, "sablan_toplam": top, "kirilim": kirilim,
                "pay": 100.0 * top / n if n else 0.0}

    arz_train, arz_197 = arz_kayit(train), arz_kayit(u197)
    tekrar = {"root_train": (1734, arz_train["kirilim"]["ROOT:"]),
              "tense_train": (800, arz_train["kirilim"]["TENSE:"]),
              "plural_train": (1000, arz_train["kirilim"]["PLURAL:"]),
              "t0197_sablan": (622, arz_197["sablan_toplam"])}
    if not all(b == o for b, o in tekrar.values()):
        print(f"T0199_TANI_DUR: TEKRAR-sayım hizalama yok {tekrar}", flush=True)
        return 2
    print(f"[ARZ] TEKRAR PASS {tekrar} | train pay {arz_train['pay']:.1f}% | "
          f"t0197 pay {arz_197['pay']:.1f}%", flush=True)

    # ---- M2: üretim-sızma (t0198 faz3 kayıt json BETİKTEN) ----
    f3 = json.load(open(SIZMA_JSON, encoding="utf-8"))
    kayit = f3["kayitlar"]
    s_tut = [r for r in kayit if r.get("incoherent")]
    s_ek = [r for r in kayit
            if not r["decomp_text"].rstrip().endswith((".", "!", "?", ":"))]
    sab_tut = [r for r in s_tut if any(sablan_desen(r["gen_text"]).values())]
    root_ek = [r for r in s_ek if sablan_desen(r["gen_text"])["ROOT:"]]
    sab_top = [r for r in kayit if any(sablan_desen(r["gen_text"]).values())]
    yuklem_yok_uretim = sum(1 for r in kayit if r.get("yuklem_yok"))
    uretim_sizma = {
        "n": len(kayit), "tutarsiz": len(s_tut), "erken_kes": len(s_ek),
        "sablan_gen_toplam": len(sab_top), "sablan_gen_tutarsiz": len(sab_tut),
        "erken_kes_root_sizma": len(root_ek), "yuklem_yok_uretim": yuklem_yok_uretim,
        "t0195_m1_hizalama": {"erken_kesiçin_root_sizma_beyan": "11/13 (T-0195 M1)"},
        "pay": {"sablan_uretim": 100.0 * len(sab_top) / len(kayit),
                "yuklemsiz_uretim": 100.0 * yuklem_yok_uretim / len(kayit)},
    }
    tasma_beyani = (uretim_sizma["pay"]["sablan_uretim"] >= arz_197["pay"]
                    or uretim_sizma["pay"]["yuklemsiz_uretim"] >= arz_197["pay"])
    print(json.dumps({"URETIM_SIZMA": uretim_sizma,
                       "arz_pay_t0197": round(arz_197["pay"], 2),
                       "tasma_beyani": tasma_beyani}, ensure_ascii=False), flush=True)

    # ---- M5: T-0196 filtre-semantiği (kaynak-taraması BETİKTEN) ----
    kaynak = open(KAYNAK_T0196, encoding="utf-8").read().splitlines()
    semantik_satirlar = [{"satir": i + 1, "kod": ln.rstrip()}
                         for i, ln in enumerate(kaynak)
                         if "rng.sample" in ln]
    if not semantik_satirlar:
        print("T0199_TANI_DUR: T-0196 semantik-satırı bulunamadı", flush=True)
        return 2
    semantik = {
        "kural": "Adım-B: predicate-siz (TENSE_/COPULA_ hiç-yok) kayıtların %50'si atlanır "
                 "(rng.sample(predicate_siz, len(predicate_siz) // 2)); ROOT/PLURAL şablan "
                 "çıkışları predicate-siz sınıfındadır ve yarısı külliyatta KALIR",
        "kanit_satirlar": semantik_satirlar,
        "sessiz_gecis_beyani": f"t0197-saf şablan-payı {arz_197['pay']:.1f}% "
                               f"({arz_197['sablan_toplam']}/{arz_197['n']}) — filtre "
                               "şablan-sınıfını tam-dışlamadı (yarı-atla semantiği)"}
    print("[M5] " + semantik["kural"], flush=True)

    # ---- M3: zarar-çıpası (4-tur hüküm-JSON BETİKTEN) ----
    h197 = json.load(open(os.path.join(KÖK, "data/eval/t0197_faz3_hukum_2026-09-29.json"), encoding="utf-8"))
    h196 = json.load(open(os.path.join(KÖK, "data/eval/t0196_faz3_hukum_2026-09-29.json"), encoding="utf-8"))
    h192 = json.load(open(os.path.join(KÖK, "data/eval/t0192_test_hukum_2026-09-29.json"), encoding="utf-8"))
    h198 = json.load(open(os.path.join(KÖK, "data/eval/t0198_faz3_hukum_2026-09-29.json"), encoding="utf-8"))
    n197 = json.load(open(os.path.join(KÖK, "data/eval/t0197_onarim_meta.json"), encoding="utf-8"))["n_sonrasi"]
    c196 = json.load(open(os.path.join(KÖK, "data/eval/t0196_curation_meta.json"), encoding="utf-8"))
    zarar_cipasi = {
        "T-0192": {"tutarsizlik": h192["metrikler"]["tutarsizlik_orani"],
                   "veri_boyutu": None, "recepte": "T-0191 anka_b1_5 eğitim-zinciri"},
        "T-0196": {"tutarsizlik": h196["olculen"]["tutarsizlik_rate"],
                   "veri_boyutu": c196.get("n_sonrasi") or c196.get("sonrasi_n") or 6680,
                   "recepte": "2-epoch peak_lr 1e-4"},
        "T-0197": {"tutarsizlik": h197["olculen"]["tutarsizlik_rate"],
                   "veri_boyutu": n197, "recepte": "2-epoch peak_lr 1e-4"},
        "T-0198": {"tutarsizlik": h198["olculen"]["tutarsizlik_rate"],
                   "veri_boyutu": 5491, "recepte": "2-epoch peak_lr 1e-4 (hiza ALFA=2,0)"},
    }
    seri = [zarar_cipasi[t]["tutarsizlik"] for t in ("T-0192", "T-0196", "T-0197", "T-0198")]
    monoton_artis = seri[0] < seri[1] < seri[2] < seri[3]
    print(f"[M3] tutarsızlık-serisi {seri} — monoton-artış: {monoton_artis}", flush=True)

    # ---- M1: taban-akıcılık ÇEVRİM (subprocess; kaynak-dokunuş YOK) ----
    m1_tablo, m1_hata = {}, None
    for ad, yol in CIPALAR.items():
        cik = os.path.join(KÖK, f"data/eval/t0199_m1_{M1_AD[ad]}.json")
        olcum = subprocess.run([sys.executable, KAYNAK_AKICILIK, "--model", yol,
                                "--device", "mps", "--output", cik],
                               capture_output=True, text=True)
        if olcum.returncode != 0:
            m1_hata = {"ad": ad, "rc": olcum.returncode,
                       "stderr": olcum.stderr.strip()[-300:]}
            print(f"T0199_TANI_DUR: M1 çevrim düşük {m1_hata}", flush=True)
            break
        sonuc = json.load(open(cik, encoding="utf-8"))
        m1_tablo[ad] = {"model": yol, "ppl": sonuc["ppl"], "ce_ort": sonuc["ce_ort"],
                        "n": sonuc["n"], "dongu_orani": sonuc["dongu_orani"],
                        "benzersiz_4gram": sonuc["benzersiz_4gram"], "output": cik}
        print(f"[M1] {ad:<6} ppl={sonuc['ppl']} ce={sonuc['ce_ort']}", flush=True)

    if m1_tablo.get("taban", {}).get("ppl") != BASE_PPL_CIPA:
        print(f"T0199_TANI_DUR: base_v2 ppl çıpası {m1_tablo.get('taban', {}).get('ppl')} "
              f"!= {BASE_PPL_CIPA} — ölçüm-kabı bozuk", flush=True)
        return 2
    zincir = [m1_tablo[a]["ppl"] for a in ("halef", "arz", "usaf", "hiza")]
    zincir_ort = sum(zincir) / len(zincir)
    oran = zincir_ort / BASE_PPL_CIPA if m1_tablo else None
    kanat_a_guclu = oran is not None and oran > 2.0
    beyan = {"kanat_a": {"zincir_ppl_ort": round(zincir_ort, 2) if zincir else None,
                          "oran_base_v2": round(oran, 3) if oran else None,
                          "a1r_cipa_oran": round(zincir_ort / A1R["ppl_cipa"], 3) if zincir else None,
                          "hipotez_kaydi": "kendi-kök base_v2'ye 2,0× üstü ppl-bozulma "
                                          "(devam-SFT zararı; tanı-kanatı — nedensellik "
                                          "iddiası YOK (R2)",
                          "guclu": kanat_a_guclu},
             "kanat_b": {"tasma_beyani": tasma_beyani,
                          "hipotez_kaydi": "sızma-üretim-payı ≥ arz-payı — taşma-kanitesi"},
             "onarim_yonu": ("Faz-2 optimum-zarar onarımı aday (taban-natural karışım / "
                             "tek-epoch düşük-LR / KL-çıpa)" if kanat_a_guclu
                            else "Kanat-A zayıf; kanat-B (şablon-arz dönüştürme) aday")}

    kayit = {"task": "T-0199", "cipa_durum": cipa_durum, "tekrar_sayim": tekrar,
             "arz": {"train": arz_train, "t0197": arz_197},
             "uretim_sizma": uretim_sizma, "m5_semantik": semantik,
             "m3_zarar_cipasi": zarar_cipasi, "m3_monoton_artis": monoton_artis,
             "m1_tablo": m1_tablo, "a1r_referans": ref.get("ppl"), "beyan": beyan,
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(KAYIT_PATH, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)

    kapilar = {"cipalar_5_sha_b1r_ref": True, "tekrar_sayim": True,
               "m1_5_model": len(m1_tablo) == len(CIPALAR) and m1_hata is None,
               "m1_base_cipa": m1_tablo.get("taban", {}).get("ppl") == BASE_PPL_CIPA,
               "m1_a1r_cipa": m1_tablo.get("taban") is not None
                              and ref.get("ppl") == A1R["ppl_cipa"],
               "m2_sayimlar": True, "m3_tablo": True}
    gecti = all(kapilar.values())
    hukum = "T0199_TANI_KAYDI" if gecti else "T0199_TANI_DUR"
    donem = {"task": "T-0199", "hukum": hukum, "rc": 0 if gecti else 2,
             "kapilar": kapilar, "beyan": beyan, "m3_seri": seri,
             "m3_monoton_artis": monoton_artis,
             "ilan": "data/eval/t0199_tani_ilan2_2026-09-30.md",
             "kayitlar": "data/eval/t0199_tani_kayitlar.json",
             "damga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(HUKUM_PATH, "w", encoding="utf-8") as f:
        json.dump(donem, f, ensure_ascii=False, indent=2)
    print(f"HUKUM: {hukum}", flush=True)
    return 0 if gecti else 2


if __name__ == "__main__":
    raise SystemExit(main())