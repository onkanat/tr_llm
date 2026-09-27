#!/usr/bin/env python
"""T-0149 FAZ-1 — P1 kalan-6 onarım-genişletme KEŞİF-KOŞUMU (tanı-betik).

Operatör onayı ("Onaylıyorum", 27 Eyl 2026) + İLAN-1
(data/eval/kesif_p1_ilan1_2026-09-27.md) uyarınca: kalan-6'nın İKİ
alt-sınıfının mekanizma-tanısı. Kanonik kod IMPORT edilir (yazım YOK;
src/compiler/** DONMUŞ-desen). Saf CPU, tek süreç, ağ YOK, seed yok
(deterministik — rastgelelik YOK).

HÜKÜM (İLAN-2 §1 — koşum öncesi sabit, operatör onayı 27 Eyl 2026):
  K1 SABİT-İMZA: 5 YOL_0 yüzeyi compile() → 0 analiz (İLAN-3
     kapanış-çıpası 6428b180… ile SABİT; beklenmedik yol = keşif-çıkışı
     çıpa-çürütme → DUR) [koşum-1 kanıt-birebir GEÇTİ — KORUNUR]
  K2 DEC-YÜZEY-BİREBİR: decompile_tags([lemma, affix_id]) == koşum-2
     yüzeyi ×5 birebir [koşum-1 kanıt-birebir GEÇTİ — KORUNUR]
  K3 ENVANTER-DOLU (İLAN-2 revizyonu — pos-ÇOKLU-SATIR): her 5 örnek
     için TSV-satır-çekimi > 0 VE HER TSV-satırı için ölçüm-alan-şeması
     dolu: transition-envanter (boş liste = ölçülmüş gecis-yok imzası)
     + template VAR ise resolve-ikili + split-aday mukayetesi,
     template YOK ise gecis_yok=true ölçülmüş-imza; hiçbir alan null.
     [koşum-1 K3 düşüşü: tsv[0]-pos varsayımı — İLAN-formülasyon-
     kusuru: çok-POS'lu satırlarda pos ÖLÇÜLMEMİŞ tahmin edilmişti]
  K4 BILBI-KIRILIMI: compile("biler") analiz ≥ 1 VE seçilen yol lemma-id
     "Bi" VE geri-decompile == "Bi'ler" (İLAN-3 kanıt-birebir) VE
     kırılım-alanları dolu [koşum-1 kanıt-birebir GEÇTİ — KORUNUR]
  K5 İSTİSNA: istisna == 0
  → KEŞIF_TAMAM (rc=0); aksi her dal → DUR (rc=2).
Çıktı: rapor + hüküm JSON betikten (elle sayı YOK). rc ∈ {0, 2}.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from src.compiler.lexicon import LexiconManager, turkish_lower  # noqa: E402
from src.compiler.morphotactics import build_default_graph, State  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.decompiler import MorphemeDecompiler  # noqa: E402
from src.compiler.phonology import PhonologyEngine  # noqa: E402

POS_TO_STATE = {
    "VERB": State.VERB_ROOT,
    "NOUN": State.NOUN_ROOT,
    "ADJ": State.ADJ_ROOT,
    "ADV": State.ADV_ROOT,
    "NUM": State.NOUN_ROOT,
    "PRON": State.NOUN_ROOT,
    "POSTP": State.NOUN_ROOT,
}

# İLAN-1 §2 — 5 YOL_0 örneği (İLAN-3 kapanış-çıpası tablosu BİREBİR;
# affix_id koşum-2 son-ek-kırılımından; beklenen decompile-yüzeyi
# koşum-2 rapor-yüzeyinden kanıt-birebir).
YOL0_ORNEKLER = [
    {"lemma": "zeyrek", "affix": "CASE_GEN", "yuzey": "zeyrekin"},
    {"lemma": "meçhul", "affix": "CASE_GEN", "yuzey": "meçhlun"},
    {"lemma": "nakil", "affix": "CASE_GEN", "yuzey": "naklin"},
    {"lemma": "güç", "affix": "POSS_1SG", "yuzey": "güçüm"},
    {"lemma": "hacir", "affix": "POSS_1PL", "yuzey": "hacrimiz"},
]
BILBI = {"surface": "biler", "lemmalar": ["bil", "Bi"],
         "bil_tags": ["bil", "TENSE_AORIST_VOWEL"],
         "Bi_tags": ["Bi", "PLURAL"], "beklenen_geri": "Bi'ler"}


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _stderr(mesaj: str) -> None:
    print("[KEŞİF] " + mesaj, file=sys.stderr, flush=True)


def _tsv_satirlari(lemma_ogleri: Tuple[str, ...]) -> List[Dict[str, str]]:
    """roots.tsv'de verilen lemma'ların TÜM satırları (TSV donmuş — salt-okuma)."""
    hits: List[Dict[str, str]] = []
    with open("data/lexicon/roots.tsv", "r", encoding="utf-8") as f:
        okuyucu = csv.DictReader(f, delimiter="\t")
        for satir in okuyucu:
            if satir.get("lemma", "") in lemma_ogleri:
                hits.append(dict(satir))
    return hits


def _stem_envanteri(comp: CrystalCompiler, kelime: str) -> List[Dict[str, Any]]:
    return [{"matched_prefix": p, "lemma": s["lemma"], "pos": s["pos"],
             "attributes": s.get("attributes", "-"),
             "is_case_alias": s.get("is_case_alias", False)}
            for p, s in comp.lexicon.find_stems(kelime)]


def _transition_envanteri(graph: Any, pos: str, affix_id: str) -> Dict[str, Any]:
    start = POS_TO_STATE.get(pos, State.NOUN_ROOT)
    return {"start_state": start,
            "affix_gecisleri": [{"affix_id": t.affix_id,
                                 "template": t.affix_template,
                                 "attributes": t.attributes,
                                 "to_state": t.to_state}
                                for t in graph.get_valid_transitions(start)
                                if t.affix_id == affix_id]}


def _yol0_tani(comp: CrystalCompiler, graph: Any, dec: MorphemeDecompiler,
               ornek: Dict[str, str]) -> Dict[str, Any]:
    lemma = ornek["lemma"]
    affix = ornek["affix"]
    yuzey = ornek["yuzey"]
    tsv = _tsv_satirlari((lemma,))

    # find_stems: yüzey (compile girişi) + lemma (decompiler girişi)
    stems_yuzey = _stem_envanteri(comp, yuzey)
    stems_lemma = _stem_envanteri(comp, lemma)

    # decompiler-yüzeyi (koşum-2 birebir beklenir) + compile (0 beklenir)
    try:
        dec_yuzey = dec.decompile_tags([lemma, affix])
        dec_istisna = None
    except Exception as exc:  # istisna da ölçülür — K5 fail-closed
        dec_yuzey = None
        dec_istisna = type(exc).__name__ + ": " + str(exc)
    try:
        paket = comp.compile(yuzey)
        analiz_sayi = len(paket["analyses"])
        compile_istisna = None
    except Exception as exc:
        paket = None
        analiz_sayi = None
        compile_istisna = type(exc).__name__ + ": " + str(exc)

    # İLAN-2: pos-ÇOKLU-SATIR — HER TSV-satırı için ölçüm (koşum-1'deki
    # tsv[0]-pos varsayımı İLAN-formülasyon-kusuruydu; pos ÖLÇÜLMÜŞ olur)
    satir_olcumleri: List[Dict[str, Any]] = []
    for t in tsv:
        pos_t = t.get("pos", "NOUN")
        attrs_t = t.get("attributes", "-")
        trans = _transition_envanteri(graph, pos_t, affix)
        template = (trans["affix_gecisleri"][0]["template"]
                    if trans["affix_gecisleri"] else None)
        if template:
            mut_t, aff_t = PhonologyEngine.resolve_affix(lemma, template,
                                                         attrs_t)
            mut_n, aff_n = PhonologyEngine.resolve_affix(lemma, template, "-")
            resolve_tsv = {"mutated_stem": mut_t, "affix_surface": aff_t,
                           "new_string": mut_t + aff_t,
                           "startswith_yuzey": (yuzey.startswith(mut_t + aff_t))}
            resolve_nok = {"mutated_stem": mut_n, "affix_surface": aff_n,
                           "new_string": mut_n + aff_n,
                           "startswith_yuzey": (yuzey.startswith(mut_n + aff_n))}
            # Yüzey-split aday mukayetesi: find_stems(yüzey)'deki her kök-
            # aday için kalan-ek + adayın KENDİ attrs'iyle resolve
            split_adaylari = []
            for aday in stems_yuzey:
                prefix = aday["matched_prefix"]
                kalan = yuzey[len(prefix):]
                mut, aff = PhonologyEngine.resolve_affix(prefix, template,
                                                         aday["attributes"])
                split_adaylari.append({
                    "aday_prefix": prefix, "kalan": kalan,
                    "mutated": mut, "affix_surface": aff,
                    "new_string": mut + aff,
                    "birebir_yuzey": (mut + aff == yuzey)})
        else:
            # gecis-yok ölçülmüş-imza (graph kapsamı envanteri)
            resolve_tsv = None
            resolve_nok = None
            split_adaylari = None
        satir_olcumleri.append({
            "pos": pos_t, "tsv_attrs": attrs_t, "tsv_is_case_alias": t.get("is_case_alias"),
            "transition_envanteri": trans, "template": template,
            "template_yok": template is None,
            "resolve_tsv_attrs": resolve_tsv, "resolve_nok_attrs": resolve_nok,
            "split_adaylari": split_adaylari})

    return {"lemma": lemma, "affix": affix, "yuzey": yuzey,
            "tsv_satir": tsv, "satir_olcumleri": satir_olcumleri,
            "stems_yuzey": stems_yuzey, "stems_lemma": stems_lemma,
            "dec_yuzey": dec_yuzey, "dec_istisna": dec_istisna,
            "compile_analiz_sayisi": analiz_sayi,
            "compile_istisna": compile_istisna}


def _bilbi_tani(comp: CrystalCompiler, dec: MorphemeDecompiler) -> Dict[str, Any]:
    tsv = _tsv_satirlari(("bil", "Bi"))
    stems = _stem_envanteri(comp, BILBI["surface"])
    try:
        paket = comp.compile(BILBI["surface"])
        analyses = [{"morpheme_ids": [m["id"] for m in a["morphemes"]],
                     "types": [m["type"] for m in a["morphemes"]],
                     "score": a.get("score"),
                     "surface_form": a["surface_form"]}
                    for a in paket["analyses"]]
        secilen_ids = paket["token_vector"]
        needs_disambiguation = paket["needs_disambiguation"]
        istisna = None
    except Exception as exc:
        analyses = None
        secilen_ids = None
        needs_disambiguation = None
        istisna = type(exc).__name__ + ": " + str(exc)
    try:
        geri_secilen = dec.decompile_tags(secilen_ids) if secilen_ids else None
    except Exception as exc:
        geri_secilen = None
    try:
        dec_bil = dec.decompile_tags(BILBI["bil_tags"])
        dec_bil_istisna = None
    except Exception as exc:
        dec_bil = None
        dec_bil_istisna = type(exc).__name__ + ": " + str(exc)
    try:
        dec_bi = dec.decompile_tags(BILBI["Bi_tags"])
        dec_bi_istisna = None
    except Exception as exc:
        dec_bi = None
        dec_bi_istisna = type(exc).__name__ + ": " + str(exc)
    return {"surface": BILBI["surface"], "tsv_satirlar": tsv,
            "stems_surface": stems,
            "compile_analiz_sayisi": len(analyses) if analyses is not None else None,
            "analyses": analyses, "secilen_token_vector": secilen_ids,
            "geri_decompile_secilen": geri_secilen,
            "needs_disambiguation": needs_disambiguation,
            "compile_istisna": istisna,
            "dec_bil_tags": dec_bil, "dec_bil_istisna": dec_bil_istisna,
            "dec_Bi_tags": dec_bi, "dec_Bi_istisna": dec_bi_istisna}


def hukum(yol0: List[Dict[str, Any]], bilbi: Dict[str, Any],
          istisna_toplam: int) -> Tuple[str, List[Dict[str, Any]]]:
    kapilar: List[Dict[str, Any]] = []

    k1 = all(r["compile_analiz_sayisi"] == 0 and r["compile_istisna"] is None
             for r in yol0)
    kapilar.append({
        "ayrac": "K1 SABİT-İMZA: 5 YOL_0 yüzeyi compile() → 0 analiz"
                 " (İLAN-3 çıpa 6428b180… SABİT; beklenmedik yol = çıpa-çürütme)",
        "olcum": {r["lemma"]: r["compile_analiz_sayisi"] for r in yol0},
        "gec": k1,
    })
    k2 = all(r["dec_yuzey"] == r["yuzey"] and r["dec_istisna"] is None
             for r in yol0)
    kapilar.append({
        "ayrac": "K2 DEC-YÜZEY-BİREBİR: decompile_tags([lemma, affix])"
                 " == koşum-2 yüzeyi ×5 birebir",
        "olcum": {r["lemma"]: r["dec_yuzey"] for r in yol0},
        "gec": k2,
    })
    # İLAN-2: pos-ÇOKLU-SATIR şema-dolu — her satır için transition-
    # envanter dolu (boş liste = gecis-yok ölçülmüş-imza) VE template
    # VAR ise resolve-ikili + split-adaylar dolu, YOK ise gecis_yok=true
    def _satir_dolu(so: Dict[str, Any]) -> bool:
        if not so.get("transition_envanteri") or so.get("transition_envanteri").get("start_state") is None:
            return False
        if so.get("template_yok"):
            return True
        return (so.get("resolve_tsv_attrs") is not None
                and so.get("resolve_nok_attrs") is not None
                and so.get("split_adaylari") is not None)
    k3 = all(r.get("tsv_satir") and r.get("satir_olcumleri")
             and all(_satir_dolu(so) for so in r["satir_olcumleri"])
             for r in yol0)
    kapilar.append({
        "ayrac": "K3 ENVANTER-DOLU (İLAN-2 pos-ÇOKLU-SATIR): 5 örneğin"
                 " HER TSV-satırı için ölçüm-alan-şeması dolu"
                 " (transition-envanter + template? resolve-ikili +"
                 " split-adaylar / gecis_yok ölçülmüş-imza)",
        "olcum": {r["lemma"]: {"tsv_satir_sayi": len(r["tsv_satir"]),
                               "satir_olcumleri_dolu": [_satir_dolu(so)
                                                        for so in r["satir_olcumleri"]]}
                  for r in yol0},
        "gec": k3,
    })
    k4 = (bilbi["compile_analiz_sayisi"] is not None
          and bilbi["compile_analiz_sayisi"] >= 1
          and bilbi["compile_istisna"] is None
          and isinstance(bilbi["secilen_token_vector"], list)
          and len(bilbi["secilen_token_vector"]) >= 1
          and bilbi["secilen_token_vector"][0] == "Bi"
          and bilbi["geri_decompile_secilen"] == BILBI["beklenen_geri"]
          and bilbi["stems_surface"] and bilbi["analyses"] is not None
          and bilbi["tsv_satirlar"] and bilbi["dec_Bi_tags"] is not None)
    kapilar.append({
        "ayrac": "K4 BILBI-KIRILIMI: compile('biler') ≥1 analiz + seçilen-yol"
                 " lemma 'Bi' + geri-decompile 'Bi'ler' (İLAN-3 kanıt-birebir)"
                 " + kırılım-alanları dolu",
        "olcum": {"analiz_sayisi": bilbi["compile_analiz_sayisi"],
                  "secilen_token_vector": bilbi["secilen_token_vector"],
                  "geri_decompile": bilbi["geri_decompile_secilen"],
                  "stems_surface_lemma_sirasi": [s["matched_prefix"]
                                                 for s in bilbi["stems_surface"]]},
        "gec": k4,
    })
    k5 = istisna_toplam == 0
    kapilar.append({
        "ayrac": "K5 İSTİSNA: tanı-koşumu istisna FIRLATMAZ (istisna == 0)",
        "olcum": istisna_toplam,
        "gec": k5,
    })
    gecti = all(k["gec"] for k in kapilar)
    return ("KEŞIF_TAMAM" if gecti else "DUR"), kapilar


def main() -> int:
    ap = argparse.ArgumentParser(description="T-0149 FAZ-1 keşif-koşumu")
    ap.add_argument("--lexicon", default="data/lexicon/roots.tsv")
    ap.add_argument("--ilan", default="data/eval/kesif_p1_ilan1_2026-09-27.md")
    ap.add_argument("--rapor", default="data/eval/kesif_p1_rapor_2026-09-27.md")
    ap.add_argument("--hukum-json", default="data/eval/kesif_p1_hukum_2026-09-27.json")
    args = ap.parse_args()

    ilan_sha = _sha256(args.ilan)
    _stderr("İLAN-1: %s sha256=%s" % (args.ilan, ilan_sha))

    lex = LexiconManager()
    lex.load_from_tsv(args.lexicon)
    graph = build_default_graph()
    comp = CrystalCompiler(lex, graph)
    dec = MorphemeDecompiler(comp, None)

    basla = time.time()
    istisna_toplam = 0
    yol0: List[Dict[str, Any]] = []
    for ornek in YOL0_ORNEKLER:
        try:
            kayit = _yol0_tani(comp, graph, dec, ornek)
        except Exception as exc:
            istisna_toplam += 1
            kayit = {"lemma": ornek["lemma"], "istisna":
                     type(exc).__name__ + ": " + str(exc)}
        yol0.append(kayit)
        _stderr("YOL_0 tanısı: %s — istisna=%d" % (ornek["lemma"], istisna_toplam))
    try:
        bilbi = _bilbi_tani(comp, dec)
    except Exception as exc:
        istisna_toplam += 1
        bilbi = {"surface": BILBI["surface"],
                 "istisna": type(exc).__name__ + ": " + str(exc)}

    hukum_adi, kapilar = hukum(yol0, bilbi, istisna_toplam)
    rc = 0 if hukum_adi == "KEŞIF_TAMAM" else 2

    betik_sha = _sha256(os.path.abspath(__file__))
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.hukum_json, "w", encoding="utf-8") as f:
        json.dump({"hukum": hukum_adi, "rc": rc, "kapilar": kapilar,
                   "hukum_kaynagi": "BU BETİK — elle sayı YOK",
                   "ilan": args.ilan, "ilan_sha256": ilan_sha, "damga": damga,
                   "sure_sn": round(time.time() - basla, 2)},
                  f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(args.hukum_json)

    # Rapor (betikten; ölçüm-JSON'lar birebir işlenir)
    s: List[str] = []
    s.append("# T-0149 FAZ-1 KEŞİF-RAPORU — P1 kalan-6 derin-mekanizma tanısı")
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
    s.append("## 2. YOL_0 tanı-kırılımı (birebir ölçüm)")
    s.append("")
    for r in yol0:
        s.append("### %s + %s → %s" % (r.get("lemma"), r.get("affix"), r.get("yuzey")))
        s.append("")
        s.append("```json")
        s.append(json.dumps(r, ensure_ascii=False, indent=1, sort_keys=True))
        s.append("```")
        s.append("")
    s.append("## 3. bil/Bi iki-farklı-lemma kırılımı (birebir ölçüm)")
    s.append("")
    s.append("```json")
    s.append(json.dumps(bilbi, ensure_ascii=False, indent=1, sort_keys=True))
    s.append("```")
    s.append("")
    s.append("## 4. Tam SHA-256 digest tablosu")
    s.append("")
    s.append("| Dosya | SHA-256 |")
    s.append("|---|---|")
    s.append("| `%s` | `%s` |" % (args.lexicon, _sha256(args.lexicon)))
    s.append("| `%s` | `%s` |" % (args.ilan, ilan_sha))
    s.append("| `scripts/kesif_p1_yol0_derin_mekanizma.py` | `%s` |" % betik_sha)
    s.append("| `%s` | `%s` |" % (args.hukum_json, hukum_sha))
    s.append("")
    with open(args.rapor, "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum_json=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())