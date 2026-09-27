#!/usr/bin/env python
"""T-0141 — MİMARİ DOĞRULAMA PAKET-1: Lexicon → Compile → Decompile gidiş-dönüş.

Operatör onaylı plan (.claude/plans/enchanted-wiggling-moon.md, 27 Eyl 2026) ve
koşum-öncesi İLAN (data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md) uyarınca
kanonik zincirin gidiş-dönüş doğrulaması. Kanonik kod IMPORT edilir (kopya YASAK;
src/compiler/** DONMUŞ). Saf CPU, seed 42, tek süreç, ağ YOK.

HÜKÜM (İLAN-3 §1 — koşum öncesi sabit, operatör onayı 27 Eyl 2026):
  FAZ-B birincil kapı: İLAN-3 BİREBİR-ÇIPA — bit_esit==9182 VE ornek==9188
  VE kalan-6 düşen örnek birebir-dislamalı (YOL_0 5 + BIT_UYUMSUZ 1;
  {lemma, tags, yuzey, geri} sıra-duyarlı birebir; betik deterministik —
  SEED 42 + TSV donmuş; koşum-2/3 bit-özdeş ölçüldü, çıpa koşum-2
  KANIT-ölçümünden formüle edilir, onarım davranışı bu koşumda DEĞİŞMEZ)
  VE FAZ-C 4 bilinen-ksur sınıfının İLANLI sayılarla birebir eşleşmesi
  VE FAZ-A'da istisna-sızlık (compile istisna FIRLATMAZ ilanlı davranış)
  → P1_GECTİ (rc=0); aksi her dal → DUR (rc=2).
  Bu koşum ONARIM KOŞUMU DEĞİLDİR — İLAN-3 KAPANIŞ-ÇIPASI: decompiler-
  yüzeyi A-1/A-2 onarımından bu yana DEĞİŞMEDİ; koşum kalan-6'yı beyanlı
  çıpa ile mühürler (kalan-6 iki alt-sınıf T-0147 raporunda AÇIKÇA
  dislanmıştır: iki-farklı-lemma 'bil/Bi' + VOWEL_DROP/kök-seçim
  derin-mekanizması — onarım-genişletme ayrı İLAN'lı turdur).

Çıktı: rapor + hüküm JSON betikten (elle sayı YOK). rc ∈ {0, 2}.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from src.compiler.lexicon import LexiconManager, turkish_lower  # noqa: E402
from src.compiler.morphotactics import build_default_graph, State  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.decompiler import MorphemeDecompiler  # noqa: E402

SEED = 42
YURUYUS_MAKS = 6  # ek zinciri tavanı; terminal/çıkmazda erken durur
ORNEKLEM_HEDEFI = {"NOUN": 5000, "VERB": 2500, "ADJ": 1500, "ADV": 800, "DIGER": 200}

# --- İLAN'lı sabitler (fail-closed: betikte tek kaynak, İLAN ile birebir) ---
KSUR1_ADJ_NOUN_YOK_ILANLI = 3627   # NOUN satırı olmayan ADJ lemma
KSUR1_ADJ_TOPLAM_ILANLI = 6352     # ADJ lemma toplamı
KSUR1_PROBE_N = 50                 # yalnız-ADJ + PLURAL probe havuzu
# İLAN-2 (T-0147 onarım-turu): 6 yalnız-ADJ lemma MEŞRU alternatif-kok
# çözülemesidir (koşum-1 §7: abuklar=VERB abukla-+AORIST; ademimerkeziyetçiler
# =+DERIV_CI; akışkanlar=ayrı lemma; …). A-1/A-2 onarımı bu probe'u
# ETKİLEMEZ (yalnız-ADJ+PLURAL; GERUND_KEN yok; ikiz-satır etkisi yok) →
# beklenen 0-yol sayısı 44/50 (koşum-1/3 birebir çıpası, seed-42 kararlı).
KSUR1_PROBE_BEKLENEN_YOL0 = 44
KSUR2_PROBE_N = 20                 # büyük-harfli lemma probe → %100 kesmeli
KSUR3_SATIR_ILANLI = 79            # 72 İ + 7 I (satır bazlı)
KSUR3_BENZERSIZ_ILANLI = 70        # 63 İ + 7 I (benzersiz lemma; İLAN §6)
KSUR4_PROBE_N = 200                # OOV probe → 200/200 sessiz-boş, istisna YOK
YURUYUS_DENEME = 3                 # lemma başına yürüyüş denemesi

# --- İLAN-3 birebir-çıpa (kapanış-çıpası; operatör onayı) ---
# Betik deterministik (SEED 42 + TSV donmuş) ve koşum-2/3 bit-özdeş
# ölçüldü (T-0147) → kalan-6 düşen-örnek-beyanı birebir çıpa olarak
# İLAN'lıdır. Çıpa koşum-2 KANIT-ölçümünden formüle edilir; bu koşum
# onarım koşumu DEĞİLDİR (decompiler-yüzeyi değişmedi) — çıpa meaningful.
FAZ_B_ORNEK_CIPA = 9188            # ornek_sayisi birebir
FAZ_B_BIT_ESIT_CIPA = 9182         # bit_esit birebir (215→6 onarım-sonrası)
FAZ_B_KALAN_CIPA = 6               # toplam düşen
FAZ_B_DISLANALI_SINIFLARI = ("YOL_0", "BIT_UYUMSUZ")
# kalan-6 örnek-beyanı (İLAN-3 tablosu): dusen_ornekler projeksiyonu
# {lemma, tags, yuzey, geri} ile sıra-duyarlı birebir karşılaştırılır.
# İki alt-sınıf: YOL_0 → VOWEL_DROP/kök-seçim derin-mekanizması
# (zeyrek/meçhul/nakil CASE_GEN; güç POSS_1SG; hacir POSS_1PL);
# BIT_UYUMSUZ → iki-farklı-lemma (bil VERB ↔ 'Bi' büyük-harfli —
# iki-ayrı-lemma TSV-sembolik kararı ayrı turdadir).
FAZ_B_DISLANALI_ORNEKLER: List[Dict[str, Any]] = [
    {"lemma": "zeyrek", "tags": ["zeyrek", "CASE_GEN"], "yuzey": "zeyrekin", "geri": None},
    {"lemma": "meçhul", "tags": ["meçhul", "CASE_GEN"], "yuzey": "meçhlun", "geri": None},
    {"lemma": "nakil", "tags": ["nakil", "CASE_GEN"], "yuzey": "naklin", "geri": None},
    {"lemma": "güç", "tags": ["güç", "POSS_1SG"], "yuzey": "güçüm", "geri": None},
    {"lemma": "hacir", "tags": ["hacir", "POSS_1PL"], "yuzey": "hacrimiz", "geri": None},
    {"lemma": "bil", "tags": ["bil", "TENSE_AORIST_VOWEL"], "yuzey": "biler",
     "geri": "Bi'ler"},
]


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _stderr(mesaj: str) -> None:
    print("[P1] " + mesaj, file=sys.stderr, flush=True)


def _tsv_oku(yol: str) -> List[Dict[str, str]]:
    satirlar: List[Dict[str, str]] = []
    with open(yol, "r", encoding="utf-8") as f:
        okuyucu = csv.DictReader(f, delimiter="\t")
        alanlar = okuyucu.fieldnames or []
        if "lemma" not in alanlar or "pos" not in alanlar:
            raise RuntimeError("DUR: TSV alanları beklenmedik: " + str(alanlar))
        for satir in okuyucu:
            satirlar.append(satir)
    return satirlar


def _pos_to_state(pos: str) -> str:
    """compile'ın (core.py) pos→kök-state sözleşmesinin aynısı."""
    if pos == "VERB":
        return State.VERB_ROOT
    if pos == "ADJ":
        return State.ADJ_ROOT
    if pos == "ADV":
        return State.ADV_ROOT
    return State.NOUN_ROOT  # NOUN + NUM/PRON/POSTP/INTERJ/CONJ/bilinmeyen


def _yuruyus(graph: Any, baslangic: str, rng: random.Random) -> Optional[List[str]]:
    """Kök state'ten başlayıp terminalde duran geçerli geçiş yürüyüşü (en az 1 ek)."""
    ekler: List[str] = []
    durum = baslangic
    while True:
        if ekler and graph.is_terminal(durum):
            return ekler
        gecisler = graph.get_valid_transitions(durum)
        if not gecisler:
            return None  # çıkmaz: örnek atılır (İLAN §2 — DUR değil)
        secilen = rng.choice(gecisler)
        ekler.append(secilen.affix_id)
        durum = secilen.to_state
        if len(ekler) >= YURUYUS_MAKS:
            return ekler if graph.is_terminal(durum) else None


def _gidi_donus(dec: MorphemeDecompiler, comp: CrystalCompiler,
                lemma: str, ekler: List[str]) -> Dict[str, Any]:
    tags = [lemma] + ekler
    kayit: Dict[str, Any] = {"lemma": lemma, "tags": tags}
    try:
        yuzey = dec.decompile_tags(tags)
    except Exception as exc:  # beklenmeyen = yeni sınıf (fail-closed DUR)
        kayit["sinif"] = "DECOMPILE_ISTISNA"
        kayit["hata"] = type(exc).__name__ + ": " + str(exc)
        return kayit
    kayit["yuzey"] = yuzey
    if not yuzey:
        kayit["sinif"] = "BOS_YUZEY"
        return kayit
    try:
        paket = comp.compile(yuzey)
    except Exception as exc:
        kayit["sinif"] = "COMPILE_ISTISNA"
        kayit["hata"] = type(exc).__name__ + ": " + str(exc)
        return kayit
    if not paket["analyses"]:
        kayit["sinif"] = "YOL_0"  # bilinen-ksur dışı 0-yol → yeni sınıf → DUR
        return kayit
    try:
        geri = dec.decompile_tags(paket["token_vector"])
    except Exception as exc:
        kayit["sinif"] = "DECOMPILE_ISTISNA"
        kayit["hata"] = type(exc).__name__ + ": " + str(exc)
        return kayit
    kayit["geri"] = geri
    kayit["bit_esit"] = (geri == yuzey)
    kayit["best_surface_esit"] = (paket["best_surface"] == yuzey)
    kayit["tv_es"] = (paket["token_vector"] == tags)
    kayit["ambig"] = bool(paket["needs_disambiguation"])
    kayit["yol_sayisi"] = len(paket["analyses"])
    if not kayit["bit_esit"]:
        kayit["sinif"] = "BIT_UYUMSUZ"
    return kayit


def faz_b(secilenler: List[Tuple[str, str]], graph: Any,
          dec: MorphemeDecompiler, comp: CrystalCompiler) -> Dict[str, Any]:
    rng = random.Random(SEED)
    ornekler: List[Dict[str, Any]] = []
    cikmaz = 0
    basla = time.time()
    for i, (lemma, pos) in enumerate(secilenler):
        ekler = None
        for _ in range(YURUYUS_DENEME):
            aday = _yuruyus(graph, _pos_to_state(pos), rng)
            if aday:
                ekler = aday
                break
        if not ekler:
            cikmaz += 1
            continue
        kayit = _gidi_donus(dec, comp, lemma, ekler)
        kayit["pos"] = pos
        ornekler.append(kayit)
        if (i + 1) % 1000 == 0:
            _stderr("FAZ-B %d/%d — %.1f sn" % (i + 1, len(secilenler), time.time() - basla))
    dusen_siniflari: Dict[str, int] = {}
    dusenler: List[Dict[str, Any]] = []
    basarili = 0
    tv_es = 0
    best_es = 0
    ambig = 0
    dusen_son_ek: Dict[str, int] = {}
    bit_uyumsuz_apostrof = 0
    for k in ornekler:
        if k.get("bit_esit"):
            basarili += 1
            if k.get("tv_es"):
                tv_es += 1
            if k.get("best_surface_esit"):
                best_es += 1
            if k.get("ambig"):
                ambig += 1
        else:
            sinif = k.get("sinif", "BILINMEYEN")
            dusen_siniflari[sinif] = dusen_siniflari.get(sinif, 0) + 1
            tags = k.get("tags") or []
            son_ek = tags[-1] if len(tags) > 1 else "YALNIZ_KOK"
            dusen_son_ek[son_ek] = dusen_son_ek.get(son_ek, 0) + 1
            if k.get("geri") and "'" in k["geri"]:
                bit_uyumsuz_apostrof += 1
            if len(dusenler) < 200:
                dusenler.append(k)
    ornek_sayisi = len(ornekler)
    return {
        "hedef": len(secilenler),
        "ornek_sayisi": ornek_sayisi,
        "cikmaz_atilan": cikmaz,
        "bit_esit": basarili,
        "bit_ozdeslik_orani": (basarili / ornek_sayisi) if ornek_sayisi else 0.0,
        "dusen_siniflari": dusen_siniflari,
        "dusen_son_ek_kirilimi": dusen_son_ek,
        "bit_uyumsuz_apostrof": bit_uyumsuz_apostrof,
        "dusen_ornekler": dusenler,
        "tv_es_orani": (tv_es / basarili) if basarili else 0.0,
        "best_surface_esit_orani": (best_es / basarili) if basarili else 0.0,
        "ambig_payi": (ambig / basarili) if basarili else 0.0,
        "sure_sn": round(time.time() - basla, 2),
    }


def faz_a(comp: CrystalCompiler,
          benzersiz_lemma_pos: List[Tuple[str, str]]) -> Dict[str, Any]:
    pos_kirilim: Dict[str, Dict[str, int]] = {}
    bos_ornekler: List[Dict[str, Any]] = []
    istisna_toplam = 0
    basla = time.time()
    for i, (lemma, pos) in enumerate(benzersiz_lemma_pos):
        try:
            paket = comp.compile(lemma)
        except Exception as exc:  # İLAN'lı davranış: istisna FIRLATMAZ → istisna = DUR
            istisna_toplam += 1
            pos_kirilim.setdefault(pos, {}).setdefault("istisna", 0)
            pos_kirilim[pos]["istisna"] += 1
            if len(bos_ornekler) < 200:
                bos_ornekler.append({"lemma": lemma, "pos": pos,
                                     "hata": type(exc).__name__ + ": " + str(exc)})
            continue
        pos_kirilim.setdefault(pos, {}).setdefault("toplam", 0)
        pos_kirilim[pos]["toplam"] += 1
        if not paket["analyses"]:
            pos_kirilim[pos].setdefault("sessiz_bos", 0)
            pos_kirilim[pos]["sessiz_bos"] += 1
            if len(bos_ornekler) < 200:
                bos_ornekler.append({"lemma": lemma, "pos": pos})
        if (i + 1) % 5000 == 0:
            _stderr("FAZ-A %d/%d — %.1f sn" % (i + 1, len(benzersiz_lemma_pos),
                                               time.time() - basla))
    return {"lemma_sayisi": len(benzersiz_lemma_pos), "pos_kirilim": pos_kirilim,
            "istisna_toplam": istisna_toplam, "bos_ornekler": bos_ornekler,
            "sure_sn": round(time.time() - basla, 2)}


def faz_c(satirlar: List[Dict[str, str]], dec: MorphemeDecompiler,
          comp: CrystalCompiler) -> Dict[str, Any]:
    pos_kume: Dict[str, Set[str]] = {}
    satir_sayi = 0
    ksur3_satir = 0
    for satir in satirlar:
        lemma = satir["lemma"]
        pos_kume.setdefault(lemma, set()).add(satir.get("pos", "NOUN"))
        satir_sayi += 1
        if lemma.lower() != turkish_lower(lemma):
            ksur3_satir += 1
    benzersiz = set(pos_kume)
    ksur3_benzersiz = sum(1 for l in benzersiz if l.lower() != turkish_lower(l))
    adj_lemma = [l for l in benzersiz if "ADJ" in pos_kume[l]]
    adj_noun_yok = [l for l in adj_lemma if "NOUN" not in pos_kume[l]]
    yalniz_adj = [l for l in benzersiz if pos_kume[l] == {"ADJ"}]
    yalniz_adj_kucuk = [l for l in yalniz_adj
                        if l == l.lower() and "'" not in l and " " not in l]

    # KSUR-1 probe: küçük-harf yalnız-ADJ + PLURAL → beklenen N/N 0-yol
    ornek1 = sorted(yalniz_adj_kucuk)[:KSUR1_PROBE_N]
    probe1_yol0 = 0
    probe1_yol_bulunan: List[Dict[str, Any]] = []
    for lemma in ornek1:
        yuzey = dec.decompile_tags([lemma, "PLURAL"])
        paket = comp.compile(yuzey)
        if not paket["analyses"]:
            probe1_yol0 += 1
        elif len(probe1_yol_bulunan) < 10:
            probe1_yol_bulunan.append({"lemma": lemma, "yuzey": yuzey,
                                       "tv": paket["token_vector"]})

    # KSUR-2 probe: büyük-harfli lemma → yüzeyde apostrof (davranış envanteri)
    isupper_havuz = sorted(l for l in benzersiz
                           if l != l.lower() and "'" not in l and " " not in l)
    ornek2 = isupper_havuz[:KSUR2_PROBE_N]
    probe2_kesmeli = 0
    for lemma in ornek2:
        if "'" in dec.decompile_tags([lemma, "CASE_DAT"]):
            probe2_kesmeli += 1
    placeholder_yuzey = dec.decompile_tags(["[Özel İsim]", "CASE_DAT"])

    # KSUR-4 probe: OOV → sessiz-boş (istisna YOK); trie'de kök yokluğu teyitli
    heceler = ["bar", "tek", "mor", "sul", "ker", "zan", "tip", "gus", "lüv", "çem"]
    probe4_sessiz = 0
    probe4_beklenmedik: List[str] = []
    idx = 0
    while probe4_sessiz < KSUR4_PROBE_N and idx < KSUR4_PROBE_N * 20:
        kelime = (heceler[idx % 10] + heceler[(idx * 3 + 5) % 10]
                  + heceler[(idx * 7 + 2) % 10] + str(idx % 10))
        idx += 1
        if comp.lexicon.find_stems(kelime):
            continue  # trie'de kök bulundu → OOV probe değil, atla
        try:
            paket = comp.compile(kelime)
            if paket["analyses"]:
                probe4_beklenmedik.append(kelime)  # OOV'de yol = beklenmedik sınıf
            else:
                probe4_sessiz += 1
        except Exception as exc:
            probe4_beklenmedik.append(kelime + "→" + type(exc).__name__ + ": " + str(exc))
    return {
        "satir_sayisi": satir_sayi,
        "benzersiz_lemma": len(benzersiz),
        "ksur1_adj_toplam": len(adj_lemma),
        "ksur1_adj_noun_yok": len(adj_noun_yok),
        "ksur1_yalniz_adj": len(yalniz_adj),
        "ksur1_yalniz_adj_kucuk_harf": len(yalniz_adj_kucuk),
        "ksur1_probe": {"n": len(ornek1), "yol0": probe1_yol0,
                        "yol_bulunan": probe1_yol_bulunan},
        "ksur2_probe": {"n": len(ornek2), "kesmeli": probe2_kesmeli,
                        "placeholder_yuzey": placeholder_yuzey,
                        "isupper_havuz": len(isupper_havuz)},
        "ksur3": {"satir": ksur3_satir, "benzersiz": ksur3_benzersiz},
        "ksur4_probe": {"sessiz_bos": probe4_sessiz,
                        "beklenmedik": probe4_beklenmedik[:10]},
    }


def hukum(faz_b_sonuc: Dict[str, Any], faz_c_sonuc: Dict[str, Any],
          faz_a_sonuc: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    kapilar: List[Dict[str, Any]] = []

    # İLAN-3 birebir-çıpa (kapanış-çıpası): %100 hedefi koşum-2'de
    # DUR-kanıtlandı (kalan-6 üçüncü-sınıf; T-0147 operatör kararı) —
    # İLAN-3'te çıpa bit_esit==9182 + ornek==9188 + kalan-6 birebir-dislama.
    dislanali_disi = {s: n for s, n in faz_b_sonuc["dusen_siniflari"].items()
                      if s not in FAZ_B_DISLANALI_SINIFLARI}
    kalan_6 = [{"lemma": k.get("lemma"), "tags": k.get("tags"),
                "yuzey": k.get("yuzey"), "geri": k.get("geri")}
               for k in faz_b_sonuc["dusen_ornekler"]]
    kapilar.append({
        "ayrac": "FAZ-B: İLAN-3 birebir-çıpa — bit_esit==9182 / ornek==9188"
                 " + kalan-6 (YOL_0 5 + BIT_UYUMSUZ 1) birebir-dislamalı",
        "olcum": {"ornek": faz_b_sonuc["ornek_sayisi"],
                  "bit_esit": faz_b_sonuc["bit_esit"],
                  "dusen": faz_b_sonuc["dusen_siniflari"],
                  "dislanali_disi": dislanali_disi,
                  "kalan_6_uyum": kalan_6 == FAZ_B_DISLANALI_ORNEKLER},
        "gec": faz_b_sonuc["ornek_sayisi"] == FAZ_B_ORNEK_CIPA
        and faz_b_sonuc["bit_esit"] == FAZ_B_BIT_ESIT_CIPA
        and sum(faz_b_sonuc["dusen_siniflari"].values()) == FAZ_B_KALAN_CIPA
        and not dislanali_disi
        and kalan_6 == FAZ_B_DISLANALI_ORNEKLER,
    })
    kapilar.append({
        "ayrac": "FAZ-A: compile istisna FIRLATMAZ (istisna == 0; ilanlı davranış)",
        "olcum": faz_a_sonuc["istisna_toplam"],
        "gec": faz_a_sonuc["istisna_toplam"] == 0,
    })
    c1 = (faz_c_sonuc["ksur1_adj_noun_yok"] == KSUR1_ADJ_NOUN_YOK_ILANLI
          and faz_c_sonuc["ksur1_adj_toplam"] == KSUR1_ADJ_TOPLAM_ILANLI
          and faz_c_sonuc["ksur1_probe"]["n"] == KSUR1_PROBE_N
          and faz_c_sonuc["ksur1_probe"]["yol0"] == KSUR1_PROBE_BEKLENEN_YOL0)
    kapilar.append({
        "ayrac": "KSUR-1: NOUN-satırı-yok ADJ == 3.627 / 6.352 ADJ + probe 44/50 0-yol"
                 " (İLAN-2: 6 yol-bulunan meşru-alternatif-kok dislamalı)",
        "olcum": {"adj_noun_yok": faz_c_sonuc["ksur1_adj_noun_yok"],
                  "adj_toplam": faz_c_sonuc["ksur1_adj_toplam"],
                  "probe_yol0": faz_c_sonuc["ksur1_probe"]["yol0"],
                  "probe_n": faz_c_sonuc["ksur1_probe"]["n"]},
        "gec": c1,
    })
    c2 = (faz_c_sonuc["ksur2_probe"]["n"] > 0
          and faz_c_sonuc["ksur2_probe"]["kesmeli"] == faz_c_sonuc["ksur2_probe"]["n"]
          and "'" in faz_c_sonuc["ksur2_probe"]["placeholder_yuzey"])
    kapilar.append({
        "ayrac": "KSUR-2: kesme-apostrof davranışı (probe %100 kesmeli + placeholder)",
        "olcum": {"kesmeli": faz_c_sonuc["ksur2_probe"]["kesmeli"],
                  "n": faz_c_sonuc["ksur2_probe"]["n"],
                  "placeholder": faz_c_sonuc["ksur2_probe"]["placeholder_yuzey"]},
        "gec": c2,
    })
    c3 = (faz_c_sonuc["ksur3"]["satir"] == KSUR3_SATIR_ILANLI
          and faz_c_sonuc["ksur3"]["benzersiz"] == KSUR3_BENZERSIZ_ILANLI)
    kapilar.append({
        "ayrac": "KSUR-3: İ/I düz-lower — satır == 79 (72 İ+7 I) VE benzersiz == 70 (63+7)",
        "olcum": faz_c_sonuc["ksur3"],
        "gec": c3,
    })
    c4 = (faz_c_sonuc["ksur4_probe"]["sessiz_bos"] == KSUR4_PROBE_N
          and not faz_c_sonuc["ksur4_probe"]["beklenmedik"])
    kapilar.append({
        "ayrac": "KSUR-4: OOV → 200/200 sessiz-boş (istisna/yol YOK)",
        "olcum": {"sessiz_bos": faz_c_sonuc["ksur4_probe"]["sessiz_bos"],
                  "beklenmedik": faz_c_sonuc["ksur4_probe"]["beklenmedik"]},
        "gec": c4,
    })
    gecti = all(k["gec"] for k in kapilar)
    return ("P1_GECTI" if gecti else "DUR"), kapilar


def _ozet(kayit: Dict[str, Any], anahtarlar: List[str]) -> str:
    return json.dumps({a: kayit.get(a) for a in anahtarlar}, ensure_ascii=False)


def _rapor_yaz(rapor_yol: str, ilan_yol: str, ilan_sha: str, lex_yol: str,
               lex_sha: str, betik_sha: str, hukum_json_yol: str, hukum_sha: str,
               hukum_adi: str, kapilar: List[Dict[str, Any]], f_a: Dict[str, Any],
               f_b: Dict[str, Any], f_c: Dict[str, Any], damga: str) -> None:
    s: List[str] = []
    s.append("# MİMARİ DOĞRULAMA PAKET-1 SONUÇ — Gidiş-Dönüş Doğrulaması (T-0141)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC, `time.gmtime`) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (koşum ÖNCESİ; sha256 `%s`)" % (ilan_yol, ilan_sha))
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
    s.append("## 2. FAZ-B — Gidiş-dönüş örneklemi (birincil)")
    s.append("")
    s.append("- Hedef/örnek: %d / **%d** (çıkmaz-atılan %d)"
             % (f_b["hedef"], f_b["ornek_sayisi"], f_b["cikmaz_atilan"]))
    s.append("- **Yüzey-bit-özdeşlik: %d/%d = %.6f** (İLAN-3 birebir-çıpa:"
             " bit_esit==%d, kalan-6 dislamalı)"
             % (f_b["bit_esit"], f_b["ornek_sayisi"], f_b["bit_ozdeslik_orani"],
                FAZ_B_BIT_ESIT_CIPA))
    s.append("- Düşen sınıfları: `%s`" % json.dumps(f_b["dusen_siniflari"],
                                                    ensure_ascii=False))
    s.append("- Düşen örneklerin son-ek kırılımı (kök-neden kaydı): `%s`"
             % json.dumps(f_b.get("dusen_son_ek_kirilimi", {}), ensure_ascii=False))
    s.append("- BIT_UYUMSUZ örneklerinde geri-yüzeyde apostrof: **%d** "
             "(özel-ad ikiz-satır deseni; kök neden §7)" % f_b.get("bit_uyumsuz_apostrof", 0))
    s.append("- İkincil kayıt (hükümde DEĞİL): `token_vector==tags` %.4f · "
             "`best_surface` eşliği %.4f · ambigüite payı %.4f"
             % (f_b["tv_es_orani"], f_b["best_surface_esit_orani"], f_b["ambig_payi"]))
    s.append("- Süre: %s sn (saf CPU)" % f_b["sure_sn"])
    if f_b["dusen_ornekler"]:
        s.append("- Düşen ilk örnekler:")
        s.append("")
        s.append("| lemma · tags · yüzey · geri | sınıf |")
        s.append("|---|---|")
        for k in f_b["dusen_ornekler"][:30]:
            s.append("| `%s` | `%s` |" % (_ozet(k, ["lemma", "tags", "yuzey", "geri",
                                                    "hata"]),
                                          k.get("sinif", "?")))
    s.append("")
    s.append("## 3. FAZ-A — Tam lemma taraması (envanter)")
    s.append("")
    s.append("- Benzersiz lemma: **%d**, istisna: **%d**, süre %s sn"
             % (f_a["lemma_sayisi"], f_a["istisna_toplam"], f_a["sure_sn"]))
    s.append("")
    s.append("| POS | toplam | sessiz-boş (analyses==[]) | istisna |")
    s.append("|---|---|---|---|")
    for pos in sorted(f_a["pos_kirilim"]):
        d = f_a["pos_kirilim"][pos]
        s.append("| %s | %d | %d | %d |" % (pos, d.get("toplam", 0),
                                            d.get("sessiz_bos", 0),
                                            d.get("istisna", 0)))
    s.append("")
    s.append("## 4. FAZ-C — Bilinen-ksur envanteri (İLAN'lı sayılar)")
    s.append("")
    s.append("| Sınıf | Ölçülen | İLAN'lı |")
    s.append("|---|---|---|")
    s.append("| KSUR-1 NOUN-satırı-yok ADJ | **%d** / ADJ toplam %d | 3.627 / 6.352 |"
             % (f_c["ksur1_adj_noun_yok"], f_c["ksur1_adj_toplam"]))
    s.append("| KSUR-1 yalnız-ADJ (rapor) | %d (küçük-harf havuz %d) | 3.282 / 3.249 |"
             % (f_c["ksur1_yalniz_adj"], f_c["ksur1_yalniz_adj_kucuk_harf"]))
    s.append("| KSUR-1 probe 0-yol | %d/%d | 44/50 (İLAN-2: 6 meşru-alternatif-kok) |"
             % (f_c["ksur1_probe"]["yol0"], f_c["ksur1_probe"]["n"]))
    s.append("| KSUR-2 kesmeli probe | %d/%d (havuz %d) | %%100 |"
             % (f_c["ksur2_probe"]["kesmeli"], f_c["ksur2_probe"]["n"],
                f_c["ksur2_probe"]["isupper_havuz"]))
    s.append("| KSUR-3 İ/I satır | **%d** | 79 (72 İ+7 I) |" % f_c["ksur3"]["satir"])
    s.append("| KSUR-3 İ/I benzersiz | **%d** | 70 (63 İ+7 I) |"
             % f_c["ksur3"]["benzersiz"])
    s.append("| KSUR-4 OOV sessiz-boş | %d (beklenmedik %d) | 200/200 |"
             % (f_c["ksur4_probe"]["sessiz_bos"],
                len(f_c["ksur4_probe"]["beklenmedik"])))
    if f_c["ksur4_probe"]["beklenmedik"]:
        s.append("")
        s.append("- KSUR-4 beklenmedik örnekleri: `%s`"
                 % json.dumps(f_c["ksur4_probe"]["beklenmedik"], ensure_ascii=False))
    s.append("")
    s.append("## 5. Kapsam-dışı beyanı (İLAN §2)")
    s.append("")
    s.append("FAZ-B örneklem havuzu dışı lemma'lar (büyük-harfli/apostroflu/çok-sözcüklü) "
             "bit-özdeşlik ölçütüne SOKULMAMIŞTIR; bu sınıflar KSUR-2/3 probe'larında ve "
             "FAZ-A sessiz-boş kırılımında envanterlenir. KSUR-1 probe havuzu: küçük-harf "
             "yalnız-ADJ (İLAN §6 revizyonu).")
    s.append("")
    s.append("## 6. Tam SHA-256 digest tablosu")
    s.append("")
    s.append("| Dosya | SHA-256 |")
    s.append("|---|---|")
    s.append("| `%s` | `%s` |" % (lex_yol, lex_sha))
    s.append("| `%s` | `%s` |" % (ilan_yol, ilan_sha))
    s.append("| `scripts/dogrulama_p1_roundtrip.py` | `%s` |" % betik_sha)
    s.append("| `%s` | `%s` |" % (hukum_json_yol, hukum_sha))
    s.append("")
    with open(rapor_yol, "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="T-0141 Paket-1 gidiş-dönüş doğrulaması")
    ap.add_argument("--lexicon", default="data/lexicon/roots.tsv")
    ap.add_argument("--ilan", default="data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md")
    ap.add_argument("--rapor", default="data/eval/mimari_dogrulama_p1_sonuc_2026-09-27.md")
    ap.add_argument("--hukum-json",
                    default="data/eval/mimari_dogrulama_p1_hukum_2026-09-27.json")
    args = ap.parse_args()

    ilan_sha = _sha256(args.ilan)
    _stderr("İLAN: %s sha256=%s" % (args.ilan, ilan_sha))

    lex = LexiconManager()
    lex.load_from_tsv(args.lexicon)
    graph = build_default_graph()
    comp = CrystalCompiler(lex, graph)
    dec = MorphemeDecompiler(comp, None)

    satirlar = _tsv_oku(args.lexicon)
    pos_kume: Dict[str, Set[str]] = {}
    for satir in satirlar:
        pos_kume.setdefault(satir["lemma"], set()).add(satir.get("pos", "NOUN"))

    # FAZ-C önce: envanter FAZ-B havuz tanımıyla hizalı
    f_c = faz_c(satirlar, dec, comp)
    _stderr("FAZ-C bitti: KSUR1=%d/%d KSUR3(satır)=%d KSUR3(benzersiz)=%d"
            % (f_c["ksur1_adj_noun_yok"], f_c["ksur1_adj_toplam"],
               f_c["ksur3"]["satir"], f_c["ksur3"]["benzersiz"]))

    # FAZ-B havuzu: çekimlenebilir sınıf — küçük-harf, boşluksuz, apostrofsuz lemma
    havuz: Dict[str, List[Tuple[str, str]]] = {p: [] for p in ("NOUN", "VERB", "ADJ",
                                                               "ADV", "DIGER")}
    for lemma, poslar in pos_kume.items():
        if lemma != lemma.lower() or "'" in lemma or " " in lemma:
            continue  # İLAN §2 kapsam-dışı; adet FAZ-C kalemlerinden raporlanır
        for pos in ("NOUN", "VERB", "ADJ", "ADV"):
            if pos in poslar:
                havuz[pos].append((lemma, pos))
        if poslar <= {"NUM", "PRON", "POSTP", "INTERJ", "CONJ"}:
            havuz["DIGER"].append((lemma, sorted(poslar)[0]))
    secilenler: List[Tuple[str, str]] = []
    rng_b = random.Random(SEED)
    for pos, hedef in ORNEKLEM_HEDEFI.items():
        havuz_pos = havuz[pos]
        if not havuz_pos:
            raise SystemExit("DUR: FAZ-B havuzu boş: " + pos)  # fail-closed
        secilenler.extend(rng_b.sample(havuz_pos, min(hedef, len(havuz_pos))))
    f_b = faz_b(secilenler, graph, dec, comp)
    _stderr("FAZ-B bitti: %d örnek, bit-özdeşlik %.6f, süre %s sn"
            % (f_b["ornek_sayisi"], f_b["bit_ozdeslik_orani"], f_b["sure_sn"]))

    # FAZ-A: tüm benzersiz lemma'lar (lemma, alfabetik-ilk POS)
    benzersiz_sirali = sorted(pos_kume.items(), key=lambda x: x[0])
    f_a = faz_a(comp, [(lemma, sorted(poslar)[0]) for lemma, poslar in benzersiz_sirali])
    _stderr("FAZ-A bitti: %d lemma, süre %s sn" % (f_a["lemma_sayisi"], f_a["sure_sn"]))

    hukum_adi, kapilar = hukum(f_b, f_c, f_a)
    rc = 0 if hukum_adi == "P1_GECTI" else 2

    betik_sha = _sha256(os.path.abspath(__file__))
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(args.hukum_json, "w", encoding="utf-8") as f:
        json.dump({"hukum": hukum_adi, "rc": rc, "kapilar": kapilar,
                   "hukum_kaynagi": "BU BETİK — elle sayı YOK",
                   "ilan": args.ilan, "ilan_sha256": ilan_sha, "damga": damga},
                  f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(args.hukum_json)
    _rapor_yaz(args.rapor, args.ilan, ilan_sha, args.lexicon, _sha256(args.lexicon),
               betik_sha, args.hukum_json, hukum_sha, hukum_adi, kapilar,
               f_a, f_b, f_c, damga)
    _stderr("HÜKÜM: %s rc=%d hukum_json=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())