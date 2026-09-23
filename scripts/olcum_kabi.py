#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0098 Faz 1 — Olcum kabinin KENDINI dogrulamasi.

Bu arac bir olcum uretmez; olcum ureten araclarin **kabini** denetler. T-0097 kapanisinda
kayda gecen 17 kusurun olcum-kabi sinifini (K6, K7, K10, K11, K14, K15, K16) tek bir
yurutulebilir yuzeye cevirir. Kural yazili oldugu icin degil, **dusecek bir kapiya bagli
olduugu** icin yururluktedir (T-0097/K15'in dersi).

Ilan edilen kapilar
-------------------
G1 `say_adim_satirlari`  — IKI bagimsiz sayac (capali + alan-ayristirma) UYUSMAZSA hata;
                           naif sayaclar (tek-bosluk ve alt-dizgi) KANIT olarak reddedilir.
                           Kapatir: K15 (kor sayac) + K6 (sise sayac).
G2 `tuple_alan_denetimi` — cok alanli yapida indeks->ad haritasini BIR KEZ basar.
                           Kapatir: K16 (sutun kaymasi).
G3 `yon_adli`            — yon, eksenden BAGIMSIZ ADIYLA verilir; semantik
                           `scratch/t0097_rapor.py:134-163`'ten OKUNARAK kopyalanmistir
                           (o dosya DUZENLENMEZ). Kapatir: K7 (isaret ters okuma).
G4 `sayi_kaynagi`        — her sayi kaynagini tasir: turetilmis / kayitli / olculmus.
                           Kapatir: K14 (olculmeden yazilan seri).
G5 `yokluk_beyani`       — eksik anahtar `OLCULEMEDI`'dir, ASLA "degisti" degil; eksikler
                           ADIYLA listelenir (kismi anahtar deligi dahil). Kapatir: K10.
G6 `referans_dogrula`    — kaynak satir numarasi CANLI dosyadan dogrulanir. Kapatir: K11.

Cikis: rc=0 BUTUN kapilar gecti · rc=2 fail-closed (mesaj stderr'e).
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

KOK: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EPS: float = 1e-9

YONLER: Tuple[str, ...] = ("artmaz", "dusmez", "duser", "yuksek")
EKSEN_IYI: Dict[str, str] = {"rouge": "yuksek = IYI", "A": "yuksek = KOTU"}
SAYI_TIPLERI: Tuple[str, ...] = ("turetilmis", "kayitli", "olculmus")
YOKLUK_HUKMU: str = "OLCULEMEDI"

# Adim satirinin ILAN EDILEN alan etiketleri (sirayla). Sayac, bu dort etiketin
# DORDUNU de gordugu satiri adim satiri sayar — tek etikete bakmak yetmez.
ALAN_ETIKETLERI: Tuple[str, ...] = ("Adım", "Kayıp", "Adım Süresi", "Toplam Süre")


class SayacUyusmazligi(ValueError):
    """Iki bagimsiz sayac ayni dosya icin ayni sayiyi vermedi."""


class AlanKaymasi(ValueError):
    """Cok alanli yapinin alan sayisi ad listesiyle uyusmuyor."""


class ReferansUyusmazligi(ValueError):
    """Ilan edilen kaynak satir numarasi canli dosyada o sabiti tasimiyor."""


# ---------------------------------------------------------------------------
# G1 — K15 + K6: adim satiri sayaci (iki bagimsiz sayac + iki naif kanit sayaci)
# ---------------------------------------------------------------------------

_CAPALI_DESENI: str = r"(?m)^[ \t]*Adım[ \t]+(\d+)[ \t]*/"
_NAIF_TEK_BOSLUK_DESENI: str = r"Ad[ıi]m (\d+)/"
_NAIF_ALT_DIZGI: str = "Adım "


def _alan_satirlari(metin: str) -> List[Tuple[int, int]]:
    """Adim satirlarini ALAN AYRISTIRMASIYLA bulur: `|` ile bolunmis dort alanin
    DORDU de ilan edilen etiketle baslamali. Capali desenden BAGIMSIZ sayactir."""
    bulunan: List[Tuple[int, int]] = []
    for i, satir in enumerate(metin.splitlines()):
        alanlar = [a.strip() for a in satir.split("|")]
        if len(alanlar) < len(ALAN_ETIKETLERI):
            continue
        if not all(alanlar[j].startswith(ALAN_ETIKETLERI[j]) for j in range(len(ALAN_ETIKETLERI))):
            continue
        m = re.match(r"^Adım[ \t]+(\d+)[ \t]*/", alanlar[0])
        if m is None:
            continue
        bulunan.append((i, int(m.group(1))))
    return bulunan


def _sayac_ozeti(adimlar: Sequence[int], etiket: str) -> Dict[str, Any]:
    if not adimlar:
        return {"ad": etiket, "sayi": 0, "ilk": None, "son": None, "gecerli": False,
                "red_nedeni": "hic adim satiri bulunamadi"}
    return {"ad": etiket, "sayi": len(adimlar), "ilk": adimlar[0], "son": adimlar[-1],
            "gecerli": True, "red_nedeni": None}


def _naif_ozeti(sayi: int, ilk: Optional[int], son: Optional[int], dogru_sayi: int,
                dogru_ilk: int, dogru_son: int, etiket: str) -> Dict[str, Any]:
    nedenler: List[str] = []
    if sayi != dogru_sayi:
        nedenler.append(f"sayi {sayi} != dogru {dogru_sayi}")
    if ilk != dogru_ilk:
        nedenler.append(f"ilk adim {ilk} != dogru {dogru_ilk} (saga yasli sayi sessizce dusuyor)")
    if son != dogru_son:
        nedenler.append(f"son adim {son} != dogru {dogru_son}")
    return {"ad": etiket, "sayi": sayi, "ilk": ilk, "son": son,
            "gecerli": not nedenler, "red_nedeni": None if not nedenler else "; ".join(nedenler)}


def say_adim_satirlari(metin: str) -> Dict[str, Any]:
    """Adim satirlarini IKI BAGIMSIZ sayacla sayar; uyusmazlarsa RAISE eder.

    Naif sayaclar da hesaplanir ama DOGRU sayilmaz: hangi yonde yanildiklari
    `red_nedeni` alaninda ADIYLA gosterilir. Boylece hem kor sayac (K15: tek bosluk
    varsayimi saga yasli sayiyi dusurur) hem sise sayac (K6: `Adım ` alt dizgisi
    `Adım Süresi:` icinde de gecer) ayni cagride gorunur olur.
    """
    if not isinstance(metin, str):
        raise TypeError("metin str olmali")

    capali = [(m.start(), int(m.group(1))) for m in re.finditer(_CAPALI_DESENI, metin)]
    alan = _alan_satirlari(metin)

    if not capali and not alan:
        raise SayacUyusmazligi("hic adim satiri bulunamadi: girdi adim gunlugu degil")

    capali_adimlar = [a for _, a in capali]
    alan_adimlar = [a for _, a in alan]

    if len(capali_adimlar) != len(alan_adimlar):
        raise SayacUyusmazligi(
            f"capali={len(capali_adimlar)} != alan={len(alan_adimlar)}: "
            f"iki bagimsiz sayac uyusmuyor ⇒ sayac KUSURLU, veri degil"
        )
    if (capali_adimlar[0], capali_adimlar[-1]) != (alan_adimlar[0], alan_adimlar[-1]):
        raise SayacUyusmazligi(
            f"ilk/son adim uyusmuyor: capali={capali_adimlar[0]}..{capali_adimlar[-1]} "
            f"alan={alan_adimlar[0]}..{alan_adimlar[-1]}"
        )

    dogru_sayi = len(capali_adimlar)
    dogru_ilk, dogru_son = capali_adimlar[0], capali_adimlar[-1]

    naif_tek = [int(x) for x in re.findall(_NAIF_TEK_BOSLUK_DESENI, metin)]
    naif_alt_sayi = metin.count(_NAIF_ALT_DIZGI)

    return {
        "sayi": dogru_sayi,
        "ilk": dogru_ilk,
        "son": dogru_son,
        "satir_sayisi": dogru_sayi,
        "sayaclar": {
            "capali": _sayac_ozeti(capali_adimlar, "capali ^[ \\t]*Adım[ \\t]+\\d+/"),
            "alan": _sayac_ozeti(alan_adimlar, "alan-ayristirma (4 etiket)"),
        },
        "naif": {
            "tek_bosluk": _naif_ozeti(len(naif_tek), naif_tek[0] if naif_tek else None,
                                      naif_tek[-1] if naif_tek else None, dogru_sayi, dogru_ilk,
                                      dogru_son, "naif 'Ad[ıi]m (\\d+)/' (K15)"),
            # Alt-dizgi sayaci SATIR saymaz, GECIS sayar: ilk/son ADIM degeri YOKTUR.
            # Bu yuzden ilk/son None birakilir ve red_nedeni bunu acikca soyler.
            "alt_dizgi": _naif_ozeti(naif_alt_sayi, None, None, dogru_sayi, dogru_ilk,
                                     dogru_son, "naif metin.count('Adım ') (K6)"),
        },
    }


# ---------------------------------------------------------------------------
# G2 — K16: cok alanli yapida indeks -> ad haritasi
# ---------------------------------------------------------------------------

def tuple_alan_denetimi(ornek: Sequence[Any], alanlar: Sequence[str]) -> Dict[str, Any]:
    """Indeks->ad haritasini BIR KEZ uretir ve basar. Kayma, sessizce okunmaz."""
    if len(ornek) != len(alanlar):
        raise AlanKaymasi(
            f"ornek {len(ornek)} alan tasiyor ama {len(alanlar)} ad verildi: "
            f"harita kurulamaz ⇒ erisim INDEKSLE yapilamaz"
        )
    harita = {i: ad for i, ad in enumerate(alanlar)}
    baslik = " · ".join(f"[{i}]={ad}" for i, ad in harita.items())
    cozum = {ad: ornek[i] for i, ad in harita.items()}
    return {"harita": harita, "baslik": baslik, "ornek": cozum, "alan_sayisi": len(alanlar)}


# ---------------------------------------------------------------------------
# G3 — K7: yon, eksenden BAGIMSIZ ADIYLA
# ---------------------------------------------------------------------------

def yon_adli(eksen: str, yon: str) -> Dict[str, Any]:
    """Yonu ADIYLA dogrular. Iki eksenin "iyi"si ZIT isaretlidir; tek bir
    'yeni kotu' kuralini iki eksene birden uygulamak ISARETI TERSINE cevirir."""
    if eksen not in EKSEN_IYI:
        raise ValueError(f"bilinmeyen eksen: {eksen}")
    if yon not in YONLER:
        raise ValueError(f"bilinmeyen yon: {yon}")
    return {"eksen": eksen, "yon": yon, "eksen_notu": EKSEN_IYI[eksen],
            "not": f"{eksen} ({EKSEN_IYI[eksen]}): kural '{yon}'"}


def karsilastir(ad: str, eksen: str, yeni: Optional[float], eski: Optional[float],
                yon: str, esik: float) -> Dict[str, Any]:
    """Tek karsilastirmayi KENDI sayilariyla kurar (sabit sablon YASAK).

    Yon semantigi `scratch/t0097_rapor.py:134-163`'ten OKUNARAK kopyalanmistir; o dosya
    DUZENLENMEZ (donmus kayit). Kopyanin ayni 6 kosulla sinanmasi `tests/`tedir.
    """
    if yeni is None or eski is None:
        return {"ad": ad, "eksen": eksen, "olculemedi": True,
                "metin": f"{ad} [{eksen}]: {YOKLUK_HUKMU} (yeni={yeni}, eski={eski})"}
    fark = yeni - eski
    if yon == "artmaz":
        atesledi = fark <= EPS
    elif yon == "dusmez":
        atesledi = fark >= -EPS
    elif yon == "duser":
        atesledi = fark < -EPS
    elif yon == "yuksek":
        atesledi = fark > EPS
    else:
        raise ValueError(f"bilinmeyen yon: {yon}")
    if eksen not in EKSEN_IYI:
        raise ValueError(f"bilinmeyen eksen: {eksen}")
    return {"ad": ad, "eksen": eksen, "yeni": yeni, "eski": eski, "fark": fark,
            "yon": yon, "atesledi": bool(atesledi), "esik": esik,
            "metin": (f"{ad} [{eksen}] ({EKSEN_IYI[eksen]}): yeni {yeni:.4f} - eski {eski:.4f} "
                      f"= {fark:+.4f} · kural '{yon}' ⇒ "
                      f"{'ATESLEDI' if atesledi else 'ATESLEMEDI'}")}


# ---------------------------------------------------------------------------
# G4 — K14: her sayi kaynagini tasir
# ---------------------------------------------------------------------------

def sayi_kaynagi(deger: Any, tip: str) -> Dict[str, Any]:
    """Sayiya kaynagini ilistirir. `turetilmis` = betikle hesaplandi ·
    `kayitli` = bir artefaktta yazili · `olculmus` = bu kosumda olculdu."""
    if tip not in SAYI_TIPLERI:
        raise ValueError(f"bilinmeyen sayi tipi: {tip} (gecerli: {SAYI_TIPLERI})")
    return {"deger": deger, "tip": tip, "etiket": f"{deger} [{tip}]"}


# ---------------------------------------------------------------------------
# G5 — K10: yokluk "degisti" DEGILDIR
# ---------------------------------------------------------------------------

def yokluk_beyani(sozluk: Optional[Dict[str, Any]],
                  beklenen_anahtarlar: Sequence[str]) -> Dict[str, Any]:
    """Eksik anahtar `OLCULEMEDI` hukmu alir; ASLA "degisti"/"ayni" denemez.

    `None` govde de yokluk sayilir: kapanis araci kosum bitmeden `korunan_sonra`yi
    goremezse bu, 9/9 SAHTE DUS uretmisti (T-0097/K10). Kismi anahtar deligi de ayni
    siniftadir: govde VAR ama icinde anahtar YOKSA o anahtar OLCULEMEDI'dir.
    """
    if sozluk is None:
        return {"hukum": YOKLUK_HUKMU, "eksik": list(beklenen_anahtarlar),
                "mevcut": {}, "gerekce": "govde YOK (None)", "degisti_denebilir": False,
                "metin": f"{YOKLUK_HUKMU}: govde yok, {len(beklenen_anahtarlar)} anahtar okunamadi"}
    eksik = [a for a in beklenen_anahtarlar if a not in sozluk or sozluk[a] is None]
    mevcut = {a: sozluk[a] for a in beklenen_anahtarlar if a not in eksik}
    if eksik:
        return {"hukum": YOKLUK_HUKMU, "eksik": eksik, "mevcut": mevcut,
                "gerekce": f"eksik anahtar(lar): {', '.join(eksik)}", "degisti_denebilir": False,
                "metin": (f"{YOKLUK_HUKMU}: {len(eksik)} anahtar okunamadi "
                          f"({', '.join(eksik)}) ⇒ 'degisti' hukmu VERILEMEZ")}
    return {"hukum": "OLCULDU", "eksik": [], "mevcut": mevcut, "gerekce": "tam sozluk",
            "degisti_denebilir": True,
            "metin": f"OLCULDU: {len(mevcut)} anahtar okundu"}


# ---------------------------------------------------------------------------
# G6 — K11: kaynak satir numarasi CANLI dosyadan dogrulanir
# ---------------------------------------------------------------------------

def _deger_geciyor(satir: str, deger: Any) -> bool:
    if isinstance(deger, float):
        if re.search(rf"(?<![\d.]){re.escape(str(deger))}(?![\d])", satir):
            return True
        if float(deger).is_integer():
            return re.search(rf"(?<![\d.]){int(deger)}(?![\d.])", satir) is not None
        return False
    return str(deger) in satir


def referans_dogrula(dosya: str, satir: int, sabit_ad: str, deger: Any) -> Dict[str, Any]:
    """Ilan edilen kaynak satir numarasini CANLI dosyada dogrular.

    Ezberden yazilan referans, sayilar dogru olsa bile YANLISTIR (T-0097/K11: 6 satirin
    3'u yanlisti). Bu kapi sabitin ADINI ve DEGERINI o satirin UZERINDE arar.
    """
    if not os.path.isfile(dosya):
        raise ReferansUyusmazligi(f"dosya yok: {dosya}")
    with open(dosya, encoding="utf-8") as f:
        satirlar = f.readlines()
    if satir < 1 or satir > len(satirlar):
        raise ReferansUyusmazligi(f"{dosya}: satir {satir} aralik disi (1..{len(satirlar)})")
    metin = satirlar[satir - 1]
    if sabit_ad not in metin:
        raise ReferansUyusmazligi(
            f"{dosya}:{satir} '{sabit_ad}' tasimiyor ⇒ referans YANLIS · satir: {metin.rstrip()!r}"
        )
    if not _deger_geciyor(metin, deger):
        raise ReferansUyusmazligi(
            f"{dosya}:{satir} '{sabit_ad}' var ama deger {deger!r} o satirda yok · "
            f"satir: {metin.rstrip()!r}"
        )
    return {"dosya": dosya, "satir": satir, "sabit_ad": sabit_ad, "deger": deger,
            "metin": metin.rstrip()}


# ---------------------------------------------------------------------------
# Kendini sinama (POZITIF KONTROL DAHIL — T-0097/K12: beklenen deger ilandan okunur)
# ---------------------------------------------------------------------------

ESIK_DOSYASI: str = os.path.join(KOK, "scripts", "evaluate_carpenter_anka.py")

# Ilandan birebir okunan esik referanslari: (satir, sabit ad, deger).
# P2/0c-0d (23 Eyl 2026): ESIK_KESISIM kalibre edildi 80,0 -> 10,92 (insan tavani %13,0
# x k=0,84) ve satir numaralari evaluate_carpenter_anka.py'deki yorum bloguyla kaydi:
# EZBER 68, TUTARSIZ 69, ROUGE 70, KESISIM 71, A_ARTIS 74, B_DUSUS 75.
# Ilan: data/eval/anka_p2_esik_kalibrasyon_ilani_2026-09-23.md
ESIK_REFERANSLARI: Tuple[Tuple[int, str, float], ...] = (
    (68, "ESIK_EZBER", 10.0),
    (69, "ESIK_TUTARSIZ", 5.0),
    (70, "ESIK_ROUGE", 0.35),
    (71, "ESIK_KESISIM", 10.92),
    (74, "ESIK_A_ARTIS", 10.0),
    (75, "ESIK_B_DUSUS", 5.0),
)

# T-0097/K11'de EZBERDEN yazilip YANLIS cikan referanslar — kapinin pozitif kontrolu:
# bunlar referans_dogrula'dan GECMEMELIDIR. P2/0d'de ayni kanarya sinifi gucellendi:
# kalibrasyon-sinifi ezber hatalari (eski esik, ham tavan, esik karisikligi).
K11_YANLIS_REFERANSLAR: Tuple[Tuple[int, str, float], ...] = (
    (71, "ESIK_KESISIM", 80.0),      # Ç4'ün ulaşılamaz eski eşiği geri yazılırsa DÜŞMELİ
    (70, "ESIK_ROUGE", 0.4164),      # ham insan tavanı eşik yazılırsa (tavan ≠ tavan×k)
    (74, "ESIK_A_ARTIS", 5.0),       # B_DUSUS eşiğiyle karıştırılırsa DÜŞMELİ
)


def _kontrol(kapilar: List[str], hatalar: List[str], ad: str, gecti: bool, not_: str) -> None:
    """FAIL satiri TEK BASINA kalamaz: dusen kontrol ayni cagride `hatalar`a yazilir,
    yoksa kapi 'FAIL' basip rc=0 donebilir (fail-open)."""
    kapilar.append(f"{'PASS' if gecti else 'FAIL'} {ad} — {not_}")
    if not gecti:
        hatalar.append(f"{ad} — {not_}")


def main() -> int:
    kapilar: List[str] = []
    hatalar: List[str] = []

    # --- G1: gercek adim gunlugu varsa kalibre et (yoksa OLCULEMEDI, uydurma yok) ---
    gercek_log = os.path.join(KOK, "scratch", "t0097_kos", "seg_2.log")
    if os.path.isfile(gercek_log):
        with open(gercek_log, encoding="utf-8", errors="replace") as f:
            metin = f.read()
        ozet = say_adim_satirlari(metin)
        naif_tb = ozet["naif"]["tek_bosluk"]
        naif_ad = ozet["naif"]["alt_dizgi"]
        _kontrol(kapilar, hatalar, "G1 sayac (gercek log)",
                 ozet["sayi"] > 0 and naif_tb["gecerli"] is False and naif_ad["gecerli"] is False,
                 f"dogru={ozet['sayi']} (adim {ozet['ilk']}..{ozet['son']}) · "
                 f"naif tek-bosluk={naif_tb['sayi']} · naif alt-dizgi={naif_ad['sayi']} "
                 f"⇒ ikisi de REDDEDILDI")
    else:
        _kontrol(kapilar, hatalar, "G1 sayac (gercek log)", True,
                 f"{YOKLUK_HUKMU}: {gercek_log} yok ⇒ kalibrasyon yapilmadi (fikstur tests/te)")

    # --- G1 kanaryasi: saga yasli fikstur naif sayaci DUSURMELI ---
    fikstur = ("Adım    1/4 | Kayıp (Loss): 1.1000 | Adım Süresi: 0.50s | Toplam Süre: 0.5s\n"
               "Adım 1500/4 | Kayıp (Loss): 0.9000 | Adım Süresi: 0.40s | Toplam Süre: 0.4s\n")
    f = say_adim_satirlari(fikstur)
    _kontrol(kapilar, hatalar, "G1 kanarya (saga yasli)",
             f["sayi"] == 2 and f["naif"]["tek_bosluk"]["gecerli"] is False,
             f"dogru={f['sayi']} · naif tek-bosluk={f['naif']['tek_bosluk']['sayi']} "
             f"ilk={f['naif']['tek_bosluk']['ilk']} ⇒ {f['naif']['tek_bosluk']['red_nedeni']}")

    # --- G1 kanaryasi: iki sayac uyusmazsa RAISE ---
    try:
        say_adim_satirlari("Adım 1/4 | Kayıp: 1.0 | Adım Süresi: 0.5s\n")
        hatalar.append("G1: tek alanli satir hata VERMEDI (iki sayac uyusmazligi yutuldu)")
    except SayacUyusmazligi:
        _kontrol(kapilar, hatalar, "G1 kanarya (uyusmazlik)", True, "tek alanli girdi SayacUyusmazligi verdi")

    # --- G2: tuple alan denetimi + kanarya ---
    t = tuple_alan_denetimi((1, 1.1342, 0.95), ("adim", "kayip", "sure"))
    _kontrol(kapilar, hatalar, "G2 tuple haritasi", t["ornek"]["sure"] == 0.95 and t["ornek"]["kayip"] == 1.1342,
             t["baslik"])
    try:
        tuple_alan_denetimi((1, 1.1342, 0.95), ("adim", "kayip"))
        hatalar.append("G2: eksik ad listesi hata VERMEDI")
    except AlanKaymasi:
        _kontrol(kapilar, hatalar, "G2 kanarya (kayma)", True, "3 alan / 2 ad ⇒ AlanKaymasi verdi")

    # --- G3: dort yon + kanarya ---
    for eksen, yon in (("rouge", "artmaz"), ("A", "dusmez"), ("rouge", "duser"), ("A", "yuksek")):
        try:
            y = yon_adli(eksen, yon)
            _kontrol(kapilar, hatalar, f"G3 yon {eksen}/{yon}", True, y["not"])
        except ValueError as e:
            hatalar.append(f"G3 {eksen}/{yon}: {e}")
    k = karsilastir("Ç3", "rouge", 0.1983, 0.3035, "duser", 35.0)
    _kontrol(kapilar, hatalar, "G3 isaret (rouge dusuyor)", k["atesledi"] is True, k["metin"])
    k = karsilastir("Ç2", "A", 62.2644, 56.3044, "yuksek", 0.0)
    _kontrol(kapilar, hatalar, "G3 isaret (A yukseliyor)", k["atesledi"] is True, k["metin"])
    try:
        yon_adli("rouge", "yeni_kotu")
        hatalar.append("G3: bilinmeyen yon hata VERMEDI")
    except ValueError:
        _kontrol(kapilar, hatalar, "G3 kanarya (bilinmeyen yon)", True, "ValueError verdi")

    # --- G4: sayi kaynagi + kanarya ---
    s = sayi_kaynagi(3.6509, "olculmus")
    _kontrol(kapilar, hatalar, "G4 sayi kaynagi", s["etiket"].endswith("[olculmus]"), s["etiket"])
    try:
        sayi_kaynagi(1.0, "tahmin")
        hatalar.append("G4: bilinmeyen tip hata VERMEDI")
    except ValueError:
        _kontrol(kapilar, hatalar, "G4 kanarya (bilinmeyen tip)", True, "ValueError verdi")

    # --- G5: yokluk = OLCULEMEDI, asla "degisti" ---
    y = yokluk_beyani({"a": 1, "b": 2}, ("a", "b", "c", "d"))
    _kontrol(kapilar, hatalar, "G5 kismi anahtar",
             y["hukum"] == YOKLUK_HUKMU and y["eksik"] == ["c", "d"]
             and y["degisti_denebilir"] is False,
             y["metin"])
    y = yokluk_beyani(None, ("a",))
    _kontrol(kapilar, hatalar, "G5 govde yok",
             y["hukum"] == YOKLUK_HUKMU and y["eksik"] == ["a"] and y["degisti_denebilir"] is False,
             y["metin"])
    y = yokluk_beyani({"a": 1}, ("a",))
    _kontrol(kapilar, hatalar, "G5 tam sozluk",
             y["hukum"] == "OLCULDU" and y["degisti_denebilir"] is True, y["metin"])

    # --- G6: esik referanslari + K11 kanaryasi ---
    if os.path.isfile(ESIK_DOSYASI):
        for satir, ad, deger in ESIK_REFERANSLARI:
            try:
                referans_dogrula(ESIK_DOSYASI, satir, ad, deger)
                _kontrol(kapilar, hatalar, f"G6 referans :{satir} {ad}", True, f"= {deger}")
            except ReferansUyusmazligi as e:
                hatalar.append(f"G6 :{satir} {ad} — {e}")
        for satir, ad, deger in K11_YANLIS_REFERANSLAR:
            try:
                referans_dogrula(ESIK_DOSYASI, satir, ad, deger)
                hatalar.append(f"G6 kanarya :{satir} {ad} GECTI ama K11'de YANLIS cikmisti")
            except ReferansUyusmazligi:
                _kontrol(kapilar, hatalar, f"G6 kanarya :{satir} {ad}", True,
                         "K11'in yanlis referansi REDDEDILDI")
    else:
        hatalar.append(f"G6: esik dosyasi yok: {ESIK_DOSYASI}")

    for satir in kapilar:
        print(satir)
    print(f"--- {len([k for k in kapilar if k.startswith('PASS')])} PASS · "
          f"{len(hatalar)} HATA ---")
    if hatalar:
        for h in hatalar:
            print(f"HATA: {h}", file=sys.stderr)
        print(f"KAPI DUSTU: {len(hatalar)} hata", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
