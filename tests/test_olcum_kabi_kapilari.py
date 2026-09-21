#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0098 Faz 3 — Olcum kabi ve kosum isletmesi kapilarinin YURUTUCU katmani.

Bu dosya kuralı YAZMAZ; kuralı **dusurur**. T-0097/K15'in dersi: yazili kural kendini
dayatmaz, kapiya baglanmasi gerekir. Her test IKI YONLU sinanir (T-0075): kapi bozuk
fiksturde **atesler**, iyi fiksturde **gecer**. Yalniz atesleme sinamak, kapinin her seye
"dur" diyen bir sabit olup olmadigini ayirt edemez.

K12'nin dersi bu dosyaya GOMULUDUR: yon beklentileri sezgiden DEGIL, donmus on-kayit
belgesinden (`data/eval/anka_r18_ceket_ilani_2026-09-21.md` §5) AYRISTIRILARAK okunur.
T-0097'de beklenen deger elle yazildigi icin DOGRU kod suclu ilan edilmisti.

K9: bu dosyanin kosumu `tests/__pycache__/*.pyc` uretir ⇒ T-0098 `writes[]`inde beyan
edilmistir (T-0097'de edilmemisti).
"""

import importlib.util
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pytest

KOK: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

ILAN_YOL: str = os.path.join(KOK, "data/eval/anka_r18_ceket_ilani_2026-09-21.md")
ESIK_DOSYASI: str = os.path.join(KOK, "scripts", "evaluate_carpenter_anka.py")
RAPOR_KAYNAK: str = os.path.join(KOK, "scratch", "t0097_rapor.py")
GERCEK_SURUCU: str = os.path.join(KOK, "scratch", "t0097_marangoz.py")
GERCEK_LOG: str = os.path.join(KOK, "scratch", "t0097_kos", "seg_2.log")


def _yukle(ad: str, yol: str) -> Any:
    """Kapi betigini YOLUNDAN yukler (scripts/ paket degil; conftest de yok).

    K9 disiplini: import edilen modul `scratch/` gibi **beyan edilmemis** bir dizinde
    oturuyorsa, Python oraya `.pyc` yazabilir ⇒ `sys.dont_write_bytecode` ile kapatilir.
    (Olculdu: `scratch/__pycache__/t0097_rapor.cpython-314.pyc` bu kosumda YAZILMADI,
    cunku onbellek tazeydi; ama YAZMA YETENEGI bir kusurdur — kapinin kendisi K9
    uretemez.)
    """
    spec = importlib.util.spec_from_file_location(ad, yol)
    assert spec is not None and spec.loader is not None, f"yuklenemedi: {yol}"
    mod = importlib.util.module_from_spec(spec)
    onceki = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = onceki
    return mod


OLCUM = _yukle("olcum_kabi", os.path.join(KOK, "scripts", "olcum_kabi.py"))
KOSUM = _yukle("kosum_kapisi", os.path.join(KOK, "scripts", "kosum_kapisi.py"))


# ---------------------------------------------------------------------------
# Ilan §5'ten okunan beklentiler (K12: sezgiden degil, belgeden)
# ---------------------------------------------------------------------------

_TR_HARF: Dict[str, str] = {"ü": "u", "ı": "i", "ş": "s", "ç": "c", "ö": "o", "ğ": "g",
                            "Ü": "U", "İ": "I", "Ş": "S", "Ç": "C", "Ö": "O", "Ğ": "G"}


def _sadelestir(metin: str) -> str:
    for a, b in _TR_HARF.items():
        metin = metin.replace(a, b)
    return metin.replace("*", " ").lower()


def _ilan_curutuculeri() -> List[Tuple[str, str]]:
    """§5 tablosunun satirlarini (id, curutucu_metni) olarak doner."""
    with open(ILAN_YOL, encoding="utf-8") as f:
        metin = f.read()
    blok = re.search(r"## §5(.*?)## §6", metin, re.S)
    assert blok is not None, "ilan §5 blogu bulunamadi"
    satirlar: List[Tuple[str, str]] = []
    for satir in blok.group(1).splitlines():
        if not satir.startswith("| **Ç"):
            continue
        hucreler = satir.split("|")
        assert len(hucreler) >= 5, f"§5 satiri bozuk: {satir!r}"
        satirlar.append((hucreler[1].strip(), hucreler[3].strip()))
    return satirlar


def _ilan_kurallari() -> Dict[Tuple[str, str], str]:
    """§5'ten (eksen, yon) -> hangi madde oldugunu AYRISTIRIR.

    Kural adlari ve eksenleri belgeden OKUNUR; bu dosyada beklenen deger YAZILMAZ.
    """
    kural: Dict[Tuple[str, str], str] = {}
    for cid, curutucu in _ilan_curutuculeri():
        sade = _sadelestir(curutucu)
        for eksen_ham, eksen in (("rouge", "rouge"), ("a", "A")):
            m = re.search(rf"\b{eksen_ham}\b", sade)
            if m is None:
                continue
            for yon in OLCUM.YONLER:
                if re.search(rf"\b{yon}\b", sade[m.end():]):
                    kural[(eksen, yon)] = cid
                    break
    return kural


def test_yon_eksenden_okunur() -> None:
    """K7 — yon, eksenden bagimsiz ADIYLA verilir; sozluk ILANDAN dogrulanir.

    Kapinin kural sozlugu ile donmus ilanin curutucu metni IKI YONLU hizalanir:
    (a) kodda ilanin ATTEST ETMEDIGI kural olmamali (uydurma kural yasak),
    (b) ilanin ilan ettigi her kural kodda BULUNMALI (sessiz dusme yasak).
    """
    ilan = _ilan_kurallari()
    assert ilan, "ilan §5'ten hic kural ayristirilamadi"

    # (a) ilanin attest ettigi kural adlari
    ilan_yonler: Set[str] = {yon for _, yon in ilan}
    ilan_eksenler: Set[str] = {eksen for eksen, _ in ilan}
    # (b) kodun kural sozlugu
    kod_yonler: Set[str] = set(OLCUM.YONLER)
    kod_eksenler: Set[str] = set(OLCUM.EKSEN_IYI)

    assert kod_yonler == ilan_yonler, (
        f"kural sozlugu ilanla hizasiz: kodda fazla={sorted(kod_yonler - ilan_yonler)} · "
        f"ilanda olup kodda olmayan={sorted(ilan_yonler - kod_yonler)}"
    )
    assert kod_eksenler == ilan_eksenler, (
        f"eksen sozlugu ilanla hizasiz: kod={sorted(kod_eksenler)} ilan={sorted(ilan_eksenler)}"
    )

    # Eksen->kural eslesmesi BIREBIR ayni olmali (K7: iki eksenin 'iyi'si zit isaretli)
    assert ilan[("rouge", "artmaz")] == "**Ç1**" and ilan[("A", "dusmez")] == "**Ç1**", (
        "Ç1 iki eksene AYNI ANDA iki farkli kural bagliyor olmali"
    )
    assert ilan[("A", "yuksek")] == "**Ç2**" and ilan[("rouge", "duser")] == "**Ç3**", ilan

    # --- K7 KANARYASI: tek kural iki eksene uygulanirsa ISARET TERSINE DONER ---
    # T-0097 24.000 segmenti (kayitli): rouge DUSTU, A YUKSELDI.
    rouge_yeni, rouge_eski = 0.1983, 0.3035
    a_yeni, a_eski = 62.2644, 56.3044
    dogru_rouge = OLCUM.karsilastir("Ç3", "rouge", rouge_yeni, rouge_eski, "duser", 0.0)
    yanlis_rouge = OLCUM.karsilastir("Ç3-yanlis", "rouge", rouge_yeni, rouge_eski, "yuksek", 0.0)
    assert dogru_rouge["atesledi"] is True, dogru_rouge["metin"]
    assert yanlis_rouge["atesledi"] is False, (
        "K7 kanaryasi dusmedi: A ekseninin kurali rouge'a uygulaninca hala atesliyor"
    )
    dogru_a = OLCUM.karsilastir("Ç2", "A", a_yeni, a_eski, "yuksek", 0.0)
    # AYNI kural kelimesi ('duser') iki eksende ZIT hukum verir: rouge'da ATESLER,
    # A'da ATESLEMEZ. Tek bir 'yeni kotu' kurali bu yuzden bir ekseni TERS okur (K7).
    a_yanlis = OLCUM.karsilastir("Ç2-yanlis", "A", a_yeni, a_eski, "duser", 0.0)
    assert dogru_a["atesledi"] is True and a_yanlis["atesledi"] is False, (
        f"A ekseni kanaryasi: {dogru_a['metin']} · {a_yanlis['metin']}"
    )
    assert dogru_rouge["atesledi"] is True, (
        "ayni 'duser' kurali rouge'da ateslemeliydi ⇒ kanarya ayirt edici degil"
    )

    # Dort kuralin DOGRULUK TABLOSU (esik genisligi SIFIR: EPS kadar)
    tablo = {(0.5, "artmaz"): False, (-0.5, "artmaz"): True, (0.0, "artmaz"): True,
             (0.5, "dusmez"): True, (-0.5, "dusmez"): False, (0.0, "dusmez"): True,
             (0.5, "duser"): False, (-0.5, "duser"): True, (0.0, "duser"): False,
             (0.5, "yuksek"): True, (-0.5, "yuksek"): False, (0.0, "yuksek"): False}
    for (fark, yon), beklenen in tablo.items():
        k = OLCUM.karsilastir("t", "A", 1.0 + fark, 1.0, yon, 0.0)
        assert k["atesledi"] is beklenen, f"{yon} fark={fark}: {k['metin']}"

    # Bilinmeyen kural SESSIZ gecmez
    with pytest.raises(ValueError):
        OLCUM.karsilastir("t", "rouge", 1.0, 0.0, "yeni_kotu", 0.0)


def test_kopya_donmus_kaynagiyla_ayni() -> None:
    """`yon_adli` mantigi `scratch/t0097_rapor.py:134-163`'ten kopyalandi; kopya AYNI olmali.

    Kaynak dosya DONMUSTUR ve DUZENLENMEZ; karsilastirma yalnizca OKUR. Kaynak yoksa
    (gitignored `scratch/` temizlenmisse) bu bir GECME degil, `skip`tir: sessiz yesil yok.
    """
    if not os.path.isfile(RAPOR_KAYNAK):
        pytest.skip(f"OLCULEMEDI: kaynak yok: {RAPOR_KAYNAK}")
    rapor = _yukle("t0097_rapor", RAPOR_KAYNAK)
    n = 0
    for yon in OLCUM.YONLER:
        for eksen in OLCUM.EKSEN_IYI:
            for yeni, eski in ((1.5, 1.0), (1.0, 1.5), (1.0, 1.0),
                               (1.0 + 1e-12, 1.0), (1.0 - 1e-12, 1.0)):
                a = rapor._karsilastir("t", eksen, yeni, eski, yon, 0.0)
                b = OLCUM.karsilastir("t", eksen, yeni, eski, yon, 0.0)
                assert a["atesledi"] == b["atesledi"], (
                    f"kopya kaymasi: {eksen}/{yon} fark={yeni - eski:.1e} "
                    f"kaynak={a['atesledi']} kopya={b['atesledi']}"
                )
                n += 1
    assert n == 40, f"beklenen 40 karsilastirma, yapilan {n}"


def test_sayac_iki_yonlu() -> None:
    """K15 — sayac KOR kalmamali (saga yasli sayi sessizce dusmez) ve FAZLA saymamali.

    Iki yon de ayni cagride olculur: dogru sayi, kor sayac ve sise sayac yan yana.
    """
    saga_yasli = ("Adım    1/4 | Kayıp (Loss): 1.1000 | Adım Süresi: 0.50s | Toplam Süre: 0.5s\n"
                  "Adım 1500/4 | Kayıp (Loss): 0.9000 | Adım Süresi: 0.40s | Toplam Süre: 0.4s\n")
    o = OLCUM.say_adim_satirlari(saga_yasli)

    # Iki BAGIMSIZ sayac ayni fikirdе olmali (capali + alan-ayristirma)
    assert o["sayi"] == 2 and o["ilk"] == 1 and o["son"] == 1500
    assert o["sayaclar"]["capali"]["sayi"] == o["sayaclar"]["alan"]["sayi"] == 2

    # KOR sayac (K15): tek bosluk varsayimi ilk satiri DUSURUR ⇒ ilk adim 1 degil 1500
    kor = o["naif"]["tek_bosluk"]
    assert kor["gecerli"] is False, "K15 kanaryasi dusmedi: kor sayac gecerli sayildi"
    assert kor["sayi"] == 1 and kor["ilk"] == 1500, kor
    assert "ilk adim" in (kor["red_nedeni"] or ""), kor["red_nedeni"]

    # SISE sayac (K6): 'Adım Süresi:' icinde 'Adım ' gecer
    sise = o["naif"]["alt_dizgi"]
    assert sise["gecerli"] is False, "K6 kanaryasi dusmedi: sise sayac gecerli sayildi"

    # --- NEGATIF KONTROL: duz hizali logda naif sayac DOGRU olmali ---
    # (kapi her seye "yanlis" demiyorsa, yukaridaki ret bir sey KANITLAMAZ)
    duz = ("Adım 1/4 | Kayıp (Loss): 1.1 | Adım Süresi: 0.5s | Toplam Süre: 0.5s\n"
           "Adım 2/4 | Kayıp (Loss): 1.0 | Adım Süresi: 0.5s | Toplam Süre: 1.0s\n")
    d = OLCUM.say_adim_satirlari(duz)
    assert d["naif"]["tek_bosluk"]["gecerli"] is True, (
        f"negatif kontrol dustu: duz hizali logda naif sayac da dogru olmali — "
        f"{d['naif']['tek_bosluk']}"
    )

    # Iki sayac uyusmazsa SESSIZ kalmaz
    with pytest.raises(OLCUM.SayacUyusmazligi):
        OLCUM.say_adim_satirlari("Adım 1/4 | Kayıp: 1.0 | Adım Süresi: 0.5s\n")
    with pytest.raises(OLCUM.SayacUyusmazligi):
        OLCUM.say_adim_satirlari("bu bir adim gunlugu degil\n")


def test_sayac_fazla_saymaz() -> None:
    """K6 — alt dizgi sayimi satir basina IKI eslesme verir; kapi bunu ADIYLA reddeder."""
    n = 7
    satirlar = [f"Adım {i:4d}/{n} | Kayıp (Loss): 1.0000 | Adım Süresi: 0.50s | "
                f"Toplam Süre: 0.5s" for i in range(1, n + 1)]
    o = OLCUM.say_adim_satirlari("\n".join(satirlar) + "\n")
    assert o["sayi"] == n, o
    assert o["naif"]["alt_dizgi"]["sayi"] == 2 * n, (
        f"K6 kanaryasi: alt dizgi sayimi {2 * n} olmali, {o['naif']['alt_dizgi']['sayi']} bulundu"
    )
    assert o["naif"]["alt_dizgi"]["gecerli"] is False, o["naif"]["alt_dizgi"]

    # Gercek log varsa: T-0097/K15'in sayilari oradan da dogrulanir
    if os.path.isfile(GERCEK_LOG):
        with open(GERCEK_LOG, encoding="utf-8", errors="replace") as f:
            g = OLCUM.say_adim_satirlari(f.read())
        assert g["sayi"] == 201 and (g["ilk"], g["son"]) == (1, 2000), g
        assert g["naif"]["tek_bosluk"]["sayi"] == 101, g["naif"]["tek_bosluk"]
        assert g["naif"]["alt_dizgi"]["sayi"] == 403, g["naif"]["alt_dizgi"]


def test_tuple_alan_kaymasi() -> None:
    """K16 — cok alanli yapi kurulup LOSS sutunu 'sure' diye okunmustu; harita bunu kapatir."""
    t = OLCUM.tuple_alan_denetimi((1, 1.1342, 0.95), ("adim", "kayip", "sure"))
    assert t["ornek"] == {"adim": 1, "kayip": 1.1342, "sure": 0.95}, t
    assert "[1]=kayip" in t["baslik"] and "[2]=sure" in t["baslik"], t["baslik"]
    with pytest.raises(OLCUM.AlanKaymasi):
        OLCUM.tuple_alan_denetimi((1, 1.1342, 0.95), ("adim", "kayip"))


def test_sayi_kaynagi_zorunlu() -> None:
    """K14 — olculmeden yazilan seri: her sayi kaynagini TASIMALI."""
    s = OLCUM.sayi_kaynagi(3.6509, "olculmus")
    assert s["tip"] == "olculmus" and s["etiket"] == "3.6509 [olculmus]", s
    assert set(OLCUM.SAYI_TIPLERI) == {"turetilmis", "kayitli", "olculmus"}
    with pytest.raises(ValueError):
        OLCUM.sayi_kaynagi(4.2, "tahmin")


def test_yokluk_degisti_degildir() -> None:
    """K10 — 'yokluk' ASLA 'degisti' degildir; kismi anahtar deligi de kapali."""
    y = OLCUM.yokluk_beyani({"a": 1, "b": 2}, ("a", "b", "c", "d"))
    assert y["hukum"] == OLCUM.YOKLUK_HUKMU
    assert y["eksik"] == ["c", "d"] and y["degisti_denebilir"] is False, y
    assert set(y["mevcut"]) == {"a", "b"}, y

    # K10 kanaryasi: T-0097'de eksik govde 9/9 SAHTE DUS uretmisti
    govde_yok = OLCUM.yokluk_beyani(None, ("korunan_sonra", "x"))
    assert govde_yok["hukum"] == OLCUM.YOKLUK_HUKMU and govde_yok["degisti_denebilir"] is False

    # None DEGER de yokluk sayilir (anahtar var ama okunamadi)
    none_deger = OLCUM.yokluk_beyani({"a": None}, ("a",))
    assert none_deger["hukum"] == OLCUM.YOKLUK_HUKMU and none_deger["eksik"] == ["a"]

    # Negatif kontrol: tam sozluk OLCULDU'dur (kapi her seye OLCULEMEDI demiyor)
    tam = OLCUM.yokluk_beyani({"a": 1, "b": 2}, ("a", "b"))
    assert tam["hukum"] == "OLCULDU" and tam["degisti_denebilir"] is True


def test_esik_referanslari() -> None:
    """K11 — kaynak satir numarasi EZBERDEN yazilamaz; CANLI dosyadan dogrulanir."""
    assert os.path.isfile(ESIK_DOSYASI), ESIK_DOSYASI
    with open(ESIK_DOSYASI, encoding="utf-8") as f:
        satirlar = f.readlines()

    # (a) Ilan edilen referanslar canli dosyada TUTUYOR mu
    for satir, ad, deger in OLCUM.ESIK_REFERANSLARI:
        OLCUM.referans_dogrula(ESIK_DOSYASI, satir, ad, deger)

    # (b) Ilan edilen satir numaralari dosyanin KENDISINDEN turetilen numarayla AYNI mi
    for satir, ad, _deger in OLCUM.ESIK_REFERANSLARI:
        bulunan = [i + 1 for i, s in enumerate(satirlar) if re.match(rf"^{ad}\s*=", s)]
        assert bulunan == [satir], (
            f"{ad}: ilan edilen satir {satir}, dosyada {bulunan} ⇒ referans BAYAT (T-0097/K11)"
        )

    # (c) K11 KANARYASI: T-0097'de ezberden yazilip YANLIS cikan referanslar DUSMELI
    assert OLCUM.K11_YANLIS_REFERANSLAR, "kanarya listesi bos olamaz"
    for satir, ad, deger in OLCUM.K11_YANLIS_REFERANSLAR:
        with pytest.raises(OLCUM.ReferansUyusmazligi):
            OLCUM.referans_dogrula(ESIK_DOSYASI, satir, ad, deger)

    # (d) Esik SOZLUGU ilani ile esik DOSYASI celismemeli (V7 sinifi: bayat formul)
    gercek = {ad: deger for _s, ad, deger in OLCUM.ESIK_REFERANSLARI}
    assert gercek["ESIK_ROUGE"] == 0.35 and gercek["ESIK_KESISIM"] == 80.0, gercek
    with pytest.raises(OLCUM.ReferansUyusmazligi):
        OLCUM.referans_dogrula(ESIK_DOSYASI, 62, "ESIK_EZBER", 99.0)


def test_timeout_kapisi(tmp_path: Any) -> None:
    """K4 — her `subprocess` cagrisi `timeout=` tasimali; kapinin KENDISI de dahil."""
    kotu = tmp_path / "surucu_kotu.py"
    kotu.write_text("import subprocess\n"
                    "subprocess.run(['echo','a'], capture_output=True)\n"
                    "subprocess.call(['echo','b'])\n"
                    "subprocess.run(['echo','c'], timeout=5)\n", encoding="utf-8")
    d = KOSUM.timeout_denetimi(str(kotu))
    assert {k["satir"] for k in d["eksik"]} == {2, 3}, d
    assert len(d["timeoutlu"]) == 1 and d["gecildi"] is False

    # Negatif kontrol: hepsi timeout'lu ⇒ GECER
    iyi = tmp_path / "surucu_iyi.py"
    iyi.write_text("import subprocess\n"
                   "subprocess.run(['echo'], timeout=5)\n", encoding="utf-8")
    assert KOSUM.timeout_denetimi(str(iyi))["gecildi"] is True

    # STRING/YORUM kanaryasi: kod icinde YAZILI ornek sayaci SISIRMEMELI (K6 kardesi)
    metin = tmp_path / "surucu_metin.py"
    metin.write_text('ORNEK = "subprocess.run([\'x\'], timeout=9)"\n'
                     "# subprocess.run(['y'])\n", encoding="utf-8")
    d = KOSUM.timeout_denetimi(str(metin))
    assert d["toplam"] == 0 and d["gecildi"] is True, d

    # Kapi KENDI kaynagini denetler (kendi kuralina uyar)
    assert KOSUM.timeout_denetimi(os.path.join(KOK, "scripts", "kosum_kapisi.py"))["gecildi"]

    # Gercek surucu: kapinin kusura KOR olmadiginin kaniti ⇒ kayitli :179'dan FAZLASI
    if os.path.isfile(GERCEK_SURUCU):
        g = KOSUM.timeout_denetimi(GERCEK_SURUCU)
        assert {k["satir"] for k in g["eksik"]} == {179, 228}, (
            f"K4 kapisi kayitli olandan fazlasini gormeli: {g['eksik']}"
        )
        assert KOSUM.timeout_kapisi(GERCEK_SURUCU) == 2


def test_tek_egitici_kapisi(tmp_path: Any) -> None:
    """K5 — iki egitici ayni veriyi tutuyorsa ikincisi BASLATILMAZ (iki dal da sinanir)."""
    veri = tmp_path / "sahte.bin"
    veri.write_bytes(b"x" * 32)

    with open(veri, "rb"):
        tek = KOSUM.tek_egitici_onkontrol(str(veri))
        assert tek["tutucu"] == 1 and tek["gecildi"] is True, tek

        tutucu = subprocess.Popen(
            [sys.executable, "-c", f"import time; f=open({str(veri)!r},'rb'); time.sleep(15)"],
        )
        try:
            son = time.time() + 10
            pidler: List[int] = []
            while time.time() < son:
                pidler = KOSUM.tutan_pidler(str(veri))
                if len(pidler) >= 2:
                    break
                time.sleep(0.5)
            assert len(pidler) >= 2, f"tutucu surec gorunmedi: {pidler}"
            with pytest.raises(KOSUM.CokluEgitici):
                KOSUM.tek_egitici_onkontrol(str(veri))
        finally:
            tutucu.terminate()
            tutucu.wait(timeout=30)

    # Fail-closed: lsof okunamiyorsa GECMEZ
    with pytest.raises(KOSUM.OrtamHatasi):
        KOSUM.tutan_pidler(str(veri), lsof="/nonexistent/lsof")
    with pytest.raises(KOSUM.OrtamHatasi):
        KOSUM.tutan_pidler(str(tmp_path / "yok.bin"))


def test_kosum_canli_kapisi(tmp_path: Any) -> None:
    """K17 — kosum CANLIYKEN kapanis/kabul araci calismaz (rc=2)."""
    veri = tmp_path / "sahte.bin"
    veri.write_bytes(b"x" * 32)
    sonuc = tmp_path / "sonuc.json"
    sonuc.write_text("{}", encoding="utf-8")

    # GECEN dal: tutucu yok
    d = KOSUM.kosum_canli_mi(str(veri), str(sonuc))
    assert d["canli"] is False and KOSUM.kosum_bekcisi(str(veri), str(sonuc), "t") == 0, d

    # DURAN dal: tutucu VAR (sonuc.json taze olmasa bile KATI olcut durur)
    with open(veri, "rb"):
        d = KOSUM.kosum_canli_mi(str(veri), str(sonuc), simdi=time.time() + 10_000)
        assert d["canli"] is True, d
        assert d["sonuc_taze"] is False and d["olcut"].startswith("tutucu >= 1"), d
        assert KOSUM.kosum_bekcisi(str(veri), str(sonuc), "t") == 2

    # sonuc.json YOKken de DURAR (ilan edilen AND olcutu fail-open olurdu — bilerek sapildi)
    with open(veri, "rb"):
        d = KOSUM.kosum_canli_mi(str(veri), str(tmp_path / "yok.json"))
        assert d["canli"] is True and d["sonuc_taze"] is False, d
