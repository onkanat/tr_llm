#!/usr/bin/env python
"""T-0150 FAZ-2 — P1 kalan-6 A2+B2+C onarım-kontrol koşumu.

Operatör kararı (27 Eyl 2026, T-0149 koşum-2 KEŞIF_TAMAM üzerinde; kayıt
commit 79a8584) ve İLAN-1 (data/eval/onarim_p1_faz2_ilan1_2026-09-27.md)
uyarınca: src/compiler/core.py onarım-kanıt koşumu. Kanonik kod IMPORT
edilir; FAZ-B/FAZ-A/FAZ-C ölçüm kodu dogrulama_p1_roundtrip'ten import
edilir (İLAN-3 betik-sha 94523084… DOKUNULMAZ — aynı örnekleme birebir).

HÜKÜM (İLAN-1 §1 — koşum öncesi sabit, operatör onayı 27 Eyl 2026):
  K1 FAZ-B ONARIM: ornek==9188 VE bit_esit==9188 VE düşen == {}
     (İLAN-3 kalan-6 üç-sınıfı boşalır; SEED-42 örnekleme birebir)
  K2 DEC-YÜZEY-KORUMA: kalan-6 çiftleri decompile_tags ×6 birebir
  K3 COMPILE-YOL-ÇAPA: 6 yüzey compile() token_vector[0]==lemma + geri==yüzey
  K4 ROOT-ORACLE: 25/25 (koşum-öncesi baseline; başarısız == {})
  K5 KSUR-1/2/3/4 SABİT: 3.627/6.352 + 44/50 · 20/20 · 79/70 · 200/200
  K6 FAZ-A: istisna==0 VE sessiz-boş==0
  K7 ŞEMA-KORUMA: compile() 6-anahtar + find_stems dönüş-şekli + tokenizer
     yolu (bayraksız _score_paths) bit-özdeş
  → P1_FAZ2_GECTİ (rc=0); aksi her dal → DUR (rc=2).
Çıktı: rapor + hüküm JSON betikten (elle sayı YOK). rc ∈ {0, 2}.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from src.compiler.lexicon import LexiconManager, turkish_lower  # noqa: E402
from src.compiler.morphotactics import build_default_graph, State  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.decompiler import MorphemeDecompiler  # noqa: E402

# İLAN-3 betiği import: FAZ-B/FAZ-A/FAZ-C ölçüm kodu birebir aynı
sys.path.insert(0, os.path.join(REPO_KOK, "scripts"))
import dogrulama_p1_roundtrip as p1  # noqa: E402

ORNEKLEM_HEDEFI = p1.ORNEKLEM_HEDEFI

# İLAN-3 kalan-6 örnek-beyanı (koşum-2 KANIT-birebir; commit 50143b8)
KALAN6 = [
    {"lemma": "zeyrek", "tags": ["zeyrek", "CASE_GEN"], "yuzey": "zeyrekin"},
    {"lemma": "meçhul", "tags": ["meçhul", "CASE_GEN"], "yuzey": "meçhlun"},
    {"lemma": "nakil", "tags": ["nakil", "CASE_GEN"], "yuzey": "naklin"},
    {"lemma": "güç", "tags": ["güç", "POSS_1SG"], "yuzey": "güçüm"},
    {"lemma": "hacir", "tags": ["hacir", "POSS_1PL"], "yuzey": "hacrimiz"},
    {"lemma": "bil", "tags": ["bil", "TENSE_AORIST_VOWEL"], "yuzey": "biler"},
]
ROOT_GOREVI = "Kelimedeki kök morfemini bul."
ROOT_ORACLE_HEDEF = 25


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _stderr(mesaj: str) -> None:
    print("[FAZ2] " + mesaj, file=sys.stderr, flush=True)


def _root_oracle(comp: CrystalCompiler) -> Dict[str, Any]:
    with open("tests/morphology_regression_100.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    root_items = [x for x in data if x["instruction"] == ROOT_GOREVI]
    basarisiz: Dict[str, Any] = {}
    for x in root_items:
        bek = x["output"].split("ROOT:")[1].strip()
        an = comp.compile(x["input"]).get("analyses") or []
        gel = an[0]["morphemes"][0]["id"] if an else None
        if gel != bek:
            basarisiz[x["input"]] = (bek, gel)
    return {"n": len(root_items), "basarili": len(root_items) - len(basarisiz),
            "basarisiz": basarisiz}


def _kalan6_kontrol(dec: MorphemeDecompiler, comp: CrystalCompiler
                    ) -> List[Dict[str, Any]]:
    kayitlar: List[Dict[str, Any]] = []
    for ornek in KALAN6:
        lemma, tags, yuzey = ornek["lemma"], ornek["tags"], ornek["yuzey"]
        dec_yuzey = dec.decompile_tags(tags)
        paket = comp.compile(yuzey)
        tv = paket["token_vector"]
        geri = dec.decompile_tags(tv) if tv else None
        kayitlar.append({"lemma": lemma, "dec_yuzey": dec_yuzey,
                         "dec_birebir": dec_yuzey == yuzey,
                         "tv": tv, "tv0_lemma": (tv[0] == lemma if tv else False),
                         "geri": geri, "bit_esit": geri == yuzey})
    return kayitlar


def _sema_kontrol(comp: CrystalCompiler) -> Dict[str, Any]:
    paket = comp.compile("göz")
    compile_anahtarlar = sorted(paket.keys())
    beklenen = ["analyses", "best_surface", "input", "language",
                "needs_disambiguation", "token_vector"]
    stems = comp.lexicon.find_stems("gözüm")
    stem_sekli = all(isinstance(s, tuple) and len(s) == 2
                     and isinstance(s[0], str) and isinstance(s[1], dict)
                     and {"lemma", "pos", "attributes"} <= set(s[1].keys())
                     for s in stems)
    # tokenizer-yolu: bayraksız init_path + _score_paths bit-özdeş davranış
    res: List[Dict[str, Any]] = []
    init = [{"type": "ROOT", "id": "göz", "surface": "göz", "pos": "NOUN",
             "attributes": "-"}]
    comp._find_paths_recursive("gözüm", "göz", State.NOUN_ROOT, init, res)
    skorlu = comp._score_paths(res)
    tok_tv = [m["id"] for m in skorlu[0]["morphemes"][1:]] if skorlu else None
    return {"compile_anahtarlar": compile_anahtarlar,
            "compile_sema_birebir": compile_anahtarlar == beklenen,
            "find_stems_sema_birebir": stem_sekli,
            "tokenizer_yol_token_vector": tok_tv,
            "tokenizer_yol_beklenen": ["POSS_1SG"]}


def hukum(faz_b: Dict[str, Any], faz_c: Dict[str, Any], faz_a: Dict[str, Any],
          root: Dict[str, Any], kalan6: List[Dict[str, Any]],
          sema: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    kapilar: List[Dict[str, Any]] = []
    kapilar.append({
        "ayrac": "K1 FAZ-B ONARIM: ornek==9188 VE bit_esit==9188 VE düşen=={}",
        "olcum": {"ornek": faz_b["ornek_sayisi"], "bit_esit": faz_b["bit_esit"],
                  "dusen": faz_b["dusen_siniflari"]},
        "gec": (faz_b["ornek_sayisi"] == 9188 and faz_b["bit_esit"] == 9188
                and not faz_b["dusen_siniflari"]),
    })
    k2 = all(k["dec_birebir"] for k in kalan6)
    kapilar.append({
        "ayrac": "K2 DEC-YÜZEY-KORUMA: kalan-6 decompile_tags ×6 birebir",
        "olcum": {k["lemma"]: k["dec_yuzey"] for k in kalan6},
        "gec": k2,
    })
    k3 = all(k["tv0_lemma"] and k["bit_esit"] for k in kalan6)
    kapilar.append({
        "ayrac": "K3 COMPILE-YOL-ÇAPA: 6 yüzey token_vector[0]==lemma + geri==yüzey",
        "olcum": {k["lemma"]: {"tv": k["tv"], "geri": k["geri"]} for k in kalan6},
        "gec": k3,
    })
    k4 = (root["n"] == ROOT_ORACLE_HEDEF and root["basarili"] == ROOT_ORACLE_HEDEF
          and not root["basarisiz"])
    kapilar.append({
        "ayrac": "K4 ROOT-ORACLE: 25/25 (koşum-öncesi baseline; başarısız == {})",
        "olcum": {"basarili": root["basarili"], "basarisiz": root["basarisiz"]},
        "gec": k4,
    })
    c1 = (faz_c["ksur1_adj_noun_yok"] == 3627
          and faz_c["ksur1_adj_toplam"] == 6352
          and faz_c["ksur1_probe"]["n"] == 50
          and faz_c["ksur1_probe"]["yol0"] == 44)
    c2 = (faz_c["ksur2_probe"]["n"] == 20
          and faz_c["ksur2_probe"]["kesmeli"] == 20
          and "'" in faz_c["ksur2_probe"]["placeholder_yuzey"])
    c3 = faz_c["ksur3"]["satir"] == 79 and faz_c["ksur3"]["benzersiz"] == 70
    c4 = (faz_c["ksur4_probe"]["sessiz_bos"] == 200
          and not faz_c["ksur4_probe"]["beklenmedik"])
    kapilar.append({
        "ayrac": "K5 KSUR SABİT: 3.627/6.352 + 44/50 · 20/20 · 79/70 · 200/200",
        "olcum": {"ksur1": [faz_c["ksur1_adj_noun_yok"], faz_c["ksur1_adj_toplam"],
                            faz_c["ksur1_probe"]["yol0"], faz_c["ksur1_probe"]["n"]],
                  "ksur2": [faz_c["ksur2_probe"]["kesmeli"], faz_c["ksur2_probe"]["n"]],
                  "ksur3": faz_c["ksur3"],
                  "ksur4": [faz_c["ksur4_probe"]["sessiz_bos"],
                            len(faz_c["ksur4_probe"]["beklenmedik"])]},
        "gec": c1 and c2 and c3 and c4,
    })
    sessiz_bos = sum(d.get("sessiz_bos", 0) for d in faz_a["pos_kirilim"].values())
    k6 = faz_a["istisna_toplam"] == 0 and sessiz_bos == 0
    kapilar.append({
        "ayrac": "K6 FAZ-A: istisna==0 VE sessiz-boş==0",
        "olcum": {"istisna": faz_a["istisna_toplam"], "sessiz_bos": sessiz_bos},
        "gec": k6,
    })
    k7 = (sema["compile_sema_birebir"] and sema["find_stems_sema_birebir"]
          and sema["tokenizer_yol_token_vector"] == sema["tokenizer_yol_beklenen"])
    kapilar.append({
        "ayrac": "K7 ŞEMA-KORUMA: compile 6-anahtar + find_stems tuple-şema"
                 " + tokenizer-yolu (bayraksız) bit-özdeş",
        "olcum": sema,
        "gec": k7,
    })
    gecti = all(k["gec"] for k in kapilar)
    return ("P1_FAZ2_GECTI" if gecti else "DUR"), kapilar


def main() -> int:
    ap = argparse.ArgumentParser(description="T-0150 FAZ-2 onarım-koşumu")
    ap.add_argument("--lexicon", default="data/lexicon/roots.tsv")
    ap.add_argument("--ilan", default="data/eval/onarim_p1_faz2_ilan1_2026-09-27.md")
    ap.add_argument("--rapor", default="data/eval/onarim_p1_faz2_rapor_2026-09-27.md")
    ap.add_argument("--hukum-json",
                    default="data/eval/onarim_p1_faz2_hukum_2026-09-27.json")
    args = ap.parse_args()

    ilan_sha = _sha256(args.ilan)
    _stderr("İLAN-1: %s sha256=%s" % (args.ilan, ilan_sha))

    lex = LexiconManager()
    lex.load_from_tsv(args.lexicon)
    graph = build_default_graph()
    comp = CrystalCompiler(lex, graph)
    dec = MorphemeDecompiler(comp, None)

    satirlar = p1._tsv_oku(args.lexicon)
    pos_kume: Dict[str, Any] = {}
    for satir in satirlar:
        pos_kume.setdefault(satir["lemma"], set()).add(satir.get("pos", "NOUN"))

    f_c = p1.faz_c(satirlar, dec, comp)
    _stderr("FAZ-C bitti: KSUR1=%d/%d probe=%d" %
            (f_c["ksur1_adj_noun_yok"], f_c["ksur1_adj_toplam"],
             f_c["ksur1_probe"]["yol0"]))

    havuz: Dict[str, List[Tuple[str, str]]] = {p: [] for p in
                                               ("NOUN", "VERB", "ADJ", "ADV",
                                                "DIGER")}
    for lemma, poslar in pos_kume.items():
        if lemma != lemma.lower() or "'" in lemma or " " in lemma:
            continue
        for pos in ("NOUN", "VERB", "ADJ", "ADV"):
            if pos in poslar:
                havuz[pos].append((lemma, pos))
        if poslar <= {"NUM", "PRON", "POSTP", "INTERJ", "CONJ"}:
            havuz["DIGER"].append((lemma, sorted(poslar)[0]))
    secilenler: List[Tuple[str, str]] = []
    rng_b = random.Random(p1.SEED)
    for pos, hedef in ORNEKLEM_HEDEFI.items():
        if not havuz[pos]:
            raise SystemExit("DUR: FAZ-B havuzu boş: " + pos)  # p1 birebir fail-closed
        secilenler.extend(rng_b.sample(havuz[pos], min(hedef, len(havuz[pos]))))

    f_b = p1.faz_b(secilenler, graph, dec, comp)
    _stderr("FAZ-B bitti: %d örnek, bit-özdeşlik %.6f" %
            (f_b["ornek_sayisi"], f_b["bit_ozdeslik_orani"]))

    benzersiz_sirali = sorted(pos_kume.items(), key=lambda x: x[0])
    f_a = p1.faz_a(comp, [(lemma, sorted(poslar)[0])
                          for lemma, poslar in benzersiz_sirali])
    _stderr("FAZ-A bitti: %d lemma" % f_a["lemma_sayisi"])

    root = _root_oracle(comp)
    _stderr("ROOT-oracle: %d/%d" % (root["basarili"], root["n"]))
    kalan6 = _kalan6_kontrol(dec, comp)
    sema = _sema_kontrol(comp)

    hukum_adi, kapilar = hukum(f_b, f_c, f_a, root, kalan6, sema)
    rc = 0 if hukum_adi == "P1_FAZ2_GECTI" else 2

    betik_sha = _sha256(os.path.abspath(__file__))
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.hukum_json, "w", encoding="utf-8") as f:
        json.dump({"hukum": hukum_adi, "rc": rc, "kapilar": kapilar,
                   "hukum_kaynagi": "BU BETİK — elle sayı YOK",
                   "ilan": args.ilan, "ilan_sha256": ilan_sha, "damga": damga},
                  f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(args.hukum_json)

    s: List[str] = []
    s.append("# T-0150 FAZ-2 ONARIM-KOŞUMU — P1 kalan-6 A2+B2+C (sonuç)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC, `time.gmtime`) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (koşum ÖNCESİ; sha256 `%s`)" % (args.ilan, ilan_sha))
    s.append("")
    s.append("## 1. Hüküm kapıları (betikten)")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for kap in kapilar:
        s.append("| %s | `%s` | %s |" % (kap["ayrac"],
                                         json.dumps(kap["olcum"], ensure_ascii=False),
                                         "GEÇTİ" if kap["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## 2. FAZ-B onarım-sonrası (birincil)")
    s.append("")
    s.append("- Hedef/örnek: %d / **%d** (çıkmaz-atılan %d)"
             % (f_b["hedef"], f_b["ornek_sayisi"], f_b["cikmaz_atilan"]))
    s.append("- **Yüzey-bit-özdeşlik: %d/%d = %.6f** (İLAN-1 hedef %%100)"
             % (f_b["bit_esit"], f_b["ornek_sayisi"], f_b["bit_ozdeslik_orani"]))
    s.append("- Düşen sınıfları: `%s`" % json.dumps(f_b["dusen_siniflari"],
                                                    ensure_ascii=False))
    s.append("- İkincil: tv_es %.4f · best_surface %.4f · ambig %.4f"
             % (f_b["tv_es_orani"], f_b["best_surface_esit_orani"], f_b["ambig_payi"]))
    s.append("- Süre: %s sn" % f_b["sure_sn"])
    if f_b["dusen_ornekler"]:
        s.append("- Düşen örnekler:")
        for k in f_b["dusen_ornekler"][:30]:
            s.append("  - `%s`" % json.dumps(
                {a: k.get(a) for a in ("lemma", "tags", "yuzey", "geri", "hata")},
                ensure_ascii=False))
    s.append("")
    s.append("## 3. FAZ-A — Tam lemma taraması")
    s.append("")
    s.append("- Benzersiz lemma: **%d**, istisna: **%d**, süre %s sn"
             % (f_a["lemma_sayisi"], f_a["istisna_toplam"], f_a["sure_sn"]))
    sessiz_bos_toplam = sum(d.get("sessiz_bos", 0)
                            for d in f_a["pos_kirilim"].values())
    s.append("- Sessiz-boş toplam: %d" % sessiz_bos_toplam)
    s.append("")
    s.append("## 4. Kalan-6 onarım-kanıtı (birebir)")
    s.append("")
    s.append("| lemma | dec_yuzey | tv0_lemma | geri | bit |")
    s.append("|---|---|---|---|---|")
    for k in kalan6:
        s.append("| %s | `%s` | %s | `%s` | %s |"
                 % (k["lemma"], k["dec_yuzey"], k["tv0_lemma"], k["geri"],
                    k["bit_esit"]))
    s.append("")
    s.append("## 5. Tam SHA-256 digest tablosu")
    s.append("")
    s.append("| Dosya | SHA-256 |")
    s.append("|---|---|")
    s.append("| `%s` | `%s` |" % (args.lexicon, _sha256(args.lexicon)))
    s.append("| `%s` | `%s` |" % (args.ilan, ilan_sha))
    s.append("| `scripts/dogrulama_p1_faz2_onarim.py` | `%s` |" % betik_sha)
    s.append("| `scripts/dogrulama_p1_roundtrip.py` (import) | `%s` |"
             % _sha256(os.path.join(REPO_KOK, "scripts", "dogrulama_p1_roundtrip.py")))
    s.append("| `src/compiler/core.py` (onarım) | `%s` |"
             % _sha256(os.path.join(REPO_KOK, "src", "compiler", "core.py")))
    s.append("| `%s` | `%s` |" % (args.hukum_json, hukum_sha))
    s.append("")
    with open(args.rapor, "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum_json=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())