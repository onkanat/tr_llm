#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0098 Faz 2 — Kosum isletmesi kapilari.

T-0097'nin kosum-isletmesi sinifi kusurlarini (K4, K5, K17) yurutulebilir kapilara cevirir.
Bu araclar egitimi BASLATMAZ; egitimin **nasil isletildigini** denetler.

Ilan edilen kapilar
-------------------
H1 `tek_egitici_onkontrol` — veri dosyasini tutan surec sayisi > 1 ise DUR. `ps`/`pgrep`
                             sandbox'ta yasak, **`lsof` calisir** (T-0097/K5). Kapatir: K5.
H2 `kosum_canli_mi`        — veri dosyasini tutan surec VARSA kosum CANLIDIR ⇒ kapanis/kabul
                             araclari rc=2 ile DURAR. Kapatir: K17.
H3 `timeout_denetimi`      — surucudeki HER `subprocess.<islev>(` cagrisi `timeout=` tasimali.
                             Kapatir: K4 (kayitli tek site :179 DEGIL, gercek iki site).

Fail-closed: okunamayan sinyalde DURULUR, gecilmez. `lsof` yoksa bu bir HATA'dir, "tutucu yok"
degildir. Bu, `timeout_denetimi(__file__)` ile **kendini de** denetler: kapi kendi kuralina uyar.

ILAN EDILEN KAPSAM SINIRI (gizli delik degil, acik sinir)
---------------------------------------------------------
`timeout=` yalnizca onu KABUL EDEN islevlerde aranir: `run`, `call`, `check_call`,
`check_output`. **`Popen` DISARIDA tutulur** — `Popen.__init__`'in `timeout` parametresi
YOKTUR (`inspect.signature` ile olculdu; gecirmek `TypeError` verir), dolayisiyla ondan
`timeout=` istemek **yanlis pozitif uretir**. `Popen` icin ayri ve daha zayif bir olcut
uygulanir: degiskene baglanan `Popen` icin `.wait(timeout=` / `.communicate(timeout=`
aranir; baglanmayan `Popen` **`sinirlama`** listesinde ADIYLA raporlanir (gizlenmez,
ama `rc`yi dusurmez — cunku orada `timeout`'un varligi bu kaptan olculemez).

Kaynak taramasi STRING ve YORUM icini GORMEZ (`_kod_metni` maskeler): aksi halde kod
icinde yazili bir ornek metin sayaci sisirir — bu, K6'nin (alt-dizgi sayaci) tam kardesidir
ve bu kapinin ilk surumunde **olculdu**: kendi fiksturumu 2 sahte pozitif olarak saydi.

Cikis: rc=0 BUTUN kapilar gecti · rc=2 fail-closed (mesaj stderr'e).
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

KOK: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: `sonuc.json` bu sureden daha taze ise kosumun izi CANLI sayilir (destekleyici kanit).
TAZE_SN: int = 900

#: `timeout=` parametresini KABUL EDEN islevler. Yalnizca bunlarda `timeout=` aranir.
SAYILAN_ISLEVLER: Tuple[str, ...] = ("run", "call", "check_call", "check_output")

#: `timeout=` KABUL ETMEYEN islev: `Popen.__init__` (olculdu). Ayri olcutle denetlenir.
POPEN_ISLEVI: str = "Popen"

_LSOF_ZAMAN_ASIMI_SN: int = 30


class OrtamHatasi(RuntimeError):
    """Gerekli ortam araci yok/okunamiyor ⇒ DUR (fail-closed)."""


class CokluEgitici(RuntimeError):
    """Veri dosyasini birden fazla surec tutuyor: ayni GPU'da iki egitici."""


# ---------------------------------------------------------------------------
# H1 — K5: tek egitici on-kontrolu
# ---------------------------------------------------------------------------

def tutan_pidler(yol: str, lsof: str = "lsof") -> List[int]:
    """`lsof -t <yol>` ile dosyayi tutan surecleri doner. Okunamazsa OrtamHatasi."""
    if not os.path.exists(yol):
        raise OrtamHatasi(f"yol yok: {yol}")
    try:
        cp = subprocess.run([lsof, "-t", yol], capture_output=True, text=True,
                            timeout=_LSOF_ZAMAN_ASIMI_SN)
    except FileNotFoundError as e:
        raise OrtamHatasi(f"'{lsof}' bulunamadi: tutucu sayisi OKUNAMAZ ⇒ DUR (fail-closed)") from e
    except subprocess.TimeoutExpired as e:
        raise OrtamHatasi(f"'{lsof}' {_LSOF_ZAMAN_ASIMI_SN} sn'de donmedi ⇒ DUR") from e
    if cp.returncode > 1:
        raise OrtamHatasi(f"lsof rc={cp.returncode}: {cp.stderr.strip()[:200]}")
    return sorted({int(x) for x in cp.stdout.split() if x.strip().isdigit()})


def tek_egitici_onkontrol(veri_yolu: str, lsof: str = "lsof") -> Dict[str, Any]:
    """>1 tutucu ⇒ CokluEgitici. T-0097/K5: 'onceki kosum oldu' VARSAYILDI, ustune ikincisi
    baslatildi; iki egitici MPS'te kilitlendi (adim 0,39 → 18,78 sn)."""
    pidler = tutan_pidler(veri_yolu, lsof=lsof)
    sonuc = {"veri_yolu": veri_yolu, "pidler": pidler, "tutucu": len(pidler),
             "gecildi": len(pidler) <= 1}
    if len(pidler) > 1:
        raise CokluEgitici(
            f"{veri_yolu}: {len(pidler)} surec tutuyor (PID {pidler}) ⇒ "
            f"ikinci egitici BASLATILMAZ (T-0097/K5: MPS'te kilitlenir)"
        )
    return sonuc


# ---------------------------------------------------------------------------
# H2 — K17: kosum canli mi?
# ---------------------------------------------------------------------------

def kosum_canli_mi(veri_yolu: str, sonuc_json: str, simdi: Optional[float] = None,
                   taze_sn: int = TAZE_SN, lsof: str = "lsof") -> Dict[str, Any]:
    """Canli kosum imzasi. `canli` = veri dosyasini tutan surec VAR (KATI olcut).

    DIKKAT — ilan edilen metinden BILEREK SAPMA: plan 'tutucu VE sonuc.json taze' diyordu;
    bu bir AND'dir ve `sonuc.json` silindiginde kapiyi **fail-open** yapardi. T-0075'in
    dersi: okunamayan sinyalde durmayan koruma kapi degildir. Bu yuzden DURMA karari
    KATI olcute baglandi (tutucu >= 1); `sonuc_taze` yalnizca DESTEKLEYICI kanit olarak
    raporlanir. Gerekce rapora ve `.agent-bus/notes/T-0098.md`'ye yazildi.
    """
    pidler = tutan_pidler(veri_yolu, lsof=lsof)
    canli_katı = len(pidler) >= 1
    if os.path.isfile(sonuc_json):
        yas = (simdi if simdi is not None else time.time()) - os.path.getmtime(sonuc_json)
        taze = yas <= taze_sn
    else:
        yas, taze = None, False
    return {"veri_yolu": veri_yolu, "sonuc_json": sonuc_json, "pidler": pidler,
            "canli": canli_katı, "olcut": "tutucu >= 1 (KATI)",
            "sonuc_taze": taze, "sonuc_yas_sn": None if yas is None else round(yas, 1),
            "karar": ("DUR: kosum CANLI" if canli_katı else "gecilir: tutucu yok")}


def kosum_bekcisi(veri_yolu: str, sonuc_json: str, arac_adi: str, **kw: Any) -> int:
    """Kapanis/kabul araclari icin ortak kapi: kosum canliysa rc=2 ile DURUR."""
    d = kosum_canli_mi(veri_yolu, sonuc_json, **kw)
    if d["canli"]:
        print(f"[{arac_adi}] KOSUM CANLI (PID {d['pidler']}) ⇒ arac CALISTIRILMAZ (T-0097/K17)",
              file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# H3 — K4: her subprocess cagrisinda timeout=
# ---------------------------------------------------------------------------

def _cagri_metni(metin: str, bas: int) -> Tuple[str, int]:
    """`bas` = islev adinin baslangici. Dengeli parantez taramasi: tirnak icindeki
    parantezler SAYILMAZ. Cagrinin metnini ve `(`'nin indeksini doner."""
    i = metin.find("(", bas)
    if i < 0:
        return "", -1
    derinlik, j, tirnak = 0, i, None
    while j < len(metin):
        c = metin[j]
        if tirnak is not None:
            if c == "\\":
                j += 2
                continue
            if c == tirnak:
                tirnak = None
        elif c in "\"'":
            tirnak = c
        elif c == "(":
            derinlik += 1
        elif c == ")":
            derinlik -= 1
            if derinlik == 0:
                return metin[i:j + 1], i
        j += 1
    return metin[i:], i


def _kod_metni(metin: str) -> str:
    """String literallerini ve yorumlari BOSLUKLA maskeler; uzunluk ve satir sayisi KORUNUR
    ⇒ satir numaralari kaymaz. Sayac, kod icinde YAZILI bir ornegi kod sanmaz."""
    cikti = list(metin)
    i, n = 0, len(metin)
    while i < n:
        c = metin[i]
        if c == "#":
            while i < n and metin[i] != "\n":
                cikti[i] = " "
                i += 1
        elif c in "\"'":
            tirnak = c * 3 if metin.startswith(c * 3, i) else c
            i += len(tirnak)
            while i < n:
                if metin[i] == "\\":
                    cikti[i] = " "
                    if i + 1 < n:
                        cikti[i + 1] = " "
                    i += 2
                    continue
                if metin.startswith(tirnak, i):
                    i += len(tirnak)
                    break
                if metin[i] != "\n":  # yeni satirlar korunur (satir no kaymasi olmasin)
                    cikti[i] = " "
                i += 1
        else:
            i += 1
    return "".join(cikti)


def _satir_no(metin: str, indeks: int) -> int:
    return metin.count("\n", 0, indeks) + 1


def timeout_denetimi(surucu_dosya: str) -> Dict[str, Any]:
    """Surucudeki `subprocess.<islev>(` cagrilarini denetler (bkz. modul basligindaki
    ILAN EDILEN KAPSAM SINIRI).

    K4 yalnizca `scratch/t0097_marangoz.py:179` icin kayitliydi; gercekte **iki** site var
    (`:179` egit() ve `:228` degerlendir()). Kapi, kayitli olandan FAZLASINI gorur —
    kapinin kusura kor olmadiginin kaniti budur.
    """
    if not os.path.isfile(surucu_dosya):
        raise OrtamHatasi(f"surucu yok: {surucu_dosya}")
    with open(surucu_dosya, encoding="utf-8") as f:
        kod = _kod_metni(f.read())

    desen = re.compile(r"subprocess\.(" + "|".join(SAYILAN_ISLEVLER) + r")\s*\(")
    eksik: List[Dict[str, Any]] = []
    tamam: List[Dict[str, Any]] = []
    for m in desen.finditer(kod):
        cagri, _ = _cagri_metni(kod, m.start())
        kayit = {"satir": _satir_no(kod, m.start()), "islev": f"subprocess.{m.group(1)}",
                 "cagri": " ".join(cagri.split())[:120]}
        (tamam if re.search(r"\btimeout\s*=", cagri) else eksik).append(kayit)

    # Popen: kurucusunda timeout YOK ⇒ bekci .wait(timeout=)/.communicate(timeout=)
    popen_desen = re.compile(r"(?:(\w+)\s*=\s*)?subprocess\.Popen\s*\(")
    popen_bekcili: List[Dict[str, Any]] = []
    sinirlama: List[Dict[str, Any]] = []
    for m in popen_desen.finditer(kod):
        satir = _satir_no(kod, m.start())
        degisken = m.group(1)
        bekci = None
        if degisken:
            for yontem in ("wait", "communicate"):
                if re.search(rf"\b{re.escape(degisken)}\.{yontem}\s*\([^)]*\btimeout\s*=", kod):
                    bekci = f"{degisken}.{yontem}(timeout=...)"
                    break
        kayit = {"satir": satir, "bekci": bekci,
                 "cagri": " ".join(_cagri_metni(kod, m.start())[0].split())[:120]}
        (popen_bekcili if bekci else sinirlama).append(kayit)

    return {"dosya": surucu_dosya, "toplam": len(eksik) + len(tamam),
            "eksik": eksik, "timeoutlu": tamam,
            "popen_bekcili": popen_bekcili, "sinirlama": sinirlama,
            "gecildi": not eksik}


def timeout_kapisi(surucu_dosya: str, arac_adi: str = "timeout_kapisi") -> int:
    """Eksik `timeout=` varsa rc=2, site ADIYLA (satir numarasiyla) bildirilir."""
    d = timeout_denetimi(surucu_dosya)
    for k in d["sinirlama"]:
        print(f"[{arac_adi}] SINIRLAMA (olculemedi): {os.path.basename(surucu_dosya)}:"
              f"{k['satir']} Popen — bekci .wait(timeout=) bulunamadi", file=sys.stderr)
    if d["eksik"]:
        for k in d["eksik"]:
            print(f"[{arac_adi}] TIMEOUT YOK: {os.path.basename(surucu_dosya)}:{k['satir']} "
                  f"{k['islev']} — asili kalma kesme kuralindan KACAR", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# Kendini sinama — kapinin KENDI kuralina uymasi dahil
# ---------------------------------------------------------------------------

GERCEK_SURUCU: str = os.path.join(KOK, "scratch", "t0097_marangoz.py")


def _kontrol(kapilar: List[str], hatalar: List[str], ad: str, gecti: bool, not_: str) -> None:
    """FAIL satiri tek basina kalamaz: dusen kontrol `hatalar`a da yazilir (fail-open'a karsi)."""
    kapilar.append(f"{'PASS' if gecti else 'FAIL'} {ad} — {not_}")
    if not gecti:
        hatalar.append(f"{ad} — {not_}")


def main() -> int:
    kapilar: List[str] = []
    hatalar: List[str] = []

    # --- H3 kanaryasi: timeout'suz fikstur ATESLEMELI ---
    with tempfile.TemporaryDirectory() as td:
        fikstur = os.path.join(td, "surucu_kotu.py")
        with open(fikstur, "w", encoding="utf-8") as f:
            f.write("import subprocess\n"
                    "subprocess.run(['echo', 'a'], capture_output=True)\n"
                    "subprocess.run(['echo', 'b'], timeout=5)\n")
        d = timeout_denetimi(fikstur)
        _kontrol(kapilar, hatalar, "H3 kanarya (timeout'suz)",
                 len(d["eksik"]) == 1 and d["eksik"][0]["satir"] == 2
                 and len(d["timeoutlu"]) == 1,
                 f"eksik={[(k['satir'], k['islev']) for k in d['eksik']]} · "
                 f"timeoutlu={len(d['timeoutlu'])}")

        # --- H3 kanaryasi: kod ICINDE YAZILI ornek sayaci SISIRMEMELI (K6 kardesi) ---
        fikstur_metin = os.path.join(td, "surucu_metin.py")
        with open(fikstur_metin, "w", encoding="utf-8") as f:
            f.write('ORNEK = "subprocess.run([' + "'x'" + '], timeout=9)"\n'
                    "# subprocess.run(['y'])\n")
        d = timeout_denetimi(fikstur_metin)
        _kontrol(kapilar, hatalar, "H3 kanarya (string/yorum)",
                 d["toplam"] == 0 and d["gecildi"],
                 f"string+yorum icindeki 2 ornek KOD SAYILMADI (toplam={d['toplam']})")

        # --- H3: kapi KENDI kaynagini denetler ---
        d = timeout_denetimi(os.path.abspath(__file__))
        _kontrol(kapilar, hatalar, "H3 kendini denetim", d["gecildi"],
                 f"{d['toplam']} cagrinin {len(d['eksik'])}'i timeout'suz · "
                 f"Popen bekcili={len(d['popen_bekcili'])} sinirlama={len(d['sinirlama'])}")

        # --- H3: gercek surucude KAYITLI olandan FAZLASINI gormeli (2 site) ---
        if os.path.isfile(GERCEK_SURUCU):
            d = timeout_denetimi(GERCEK_SURUCU)
            _kontrol(kapilar, hatalar, "H3 gercek surucu",
                     len(d["eksik"]) == 2 and {k["satir"] for k in d["eksik"]} == {179, 228},
                     f"eksik siteler={sorted(k['satir'] for k in d['eksik'])} "
                     f"(K4 yalnizca :179'u kaydetmisti)")
        else:
            _kontrol(kapilar, hatalar, "H3 gercek surucu", True,
                     f"OLCULEMEDI: {GERCEK_SURUCU} yok (fikstur tests/te)")

        # --- H1: tek tutucu ⇒ GECER · iki tutucu ⇒ DURUR (iki dal da sinanir) ---
        veri = os.path.join(td, "sahte_veri.bin")
        sonuc = os.path.join(td, "sonuc.json")
        with open(veri, "wb") as f:
            f.write(b"x" * 64)
        with open(veri, "rb") as tutamac:
            tek = tek_egitici_onkontrol(veri)
            _kontrol(kapilar, hatalar, "H1 tek tutucu (gecen dal)",
                     tek["tutucu"] == 1 and tek["gecildi"], f"PID {tek['pidler']}")
            ikinci = subprocess.Popen(
                [sys.executable, "-c",
                 f"import time; f=open({veri!r},'rb'); time.sleep(20)"],
            )
            try:
                time.sleep(1.5)
                try:
                    tek_egitici_onkontrol(veri)
                    hatalar.append("H1: iki tutucuda DURMADI (fail-open)")
                except CokluEgitici as e:
                    _kontrol(kapilar, hatalar, "H1 iki tutucu (duran dal)", True, str(e)[:110])

                # --- H2 (DURAN dal): kontrol, tutucu HALA CANLIYIKE yapilir ---
                with open(sonuc, "w", encoding="utf-8") as f:
                    f.write("{}")
                canli = kosum_canli_mi(veri, sonuc)
                _kontrol(kapilar, hatalar, "H2 canli kosum (duran dal)",
                         canli["canli"] is True and kosum_bekcisi(veri, sonuc, "test") == 2,
                         f"olcut={canli['olcut']} · PID {canli['pidler']} · {canli['karar']}")
            finally:
                ikinci.terminate()
                ikinci.wait(timeout=30)

        # --- H2 (GECEN dal): tutucu birakildiktan SONRA ---
        with open(sonuc, "w", encoding="utf-8") as f:
            f.write("{}")
        bos = os.path.join(td, "tutucusuz.bin")
        with open(bos, "wb") as f:
            f.write(b"y")
        durgun = kosum_canli_mi(bos, sonuc)
        _kontrol(kapilar, hatalar, "H2 durgun (gecen dal)",
                 durgun["canli"] is False and kosum_bekcisi(bos, sonuc, "test") == 0,
                 f"{durgun['karar']} · sonuc_taze={durgun['sonuc_taze']}")

        # --- fail-closed: okunamayan sinyalde GECMEZ ---
        try:
            tutan_pidler(bos, lsof="/nonexistent/lsof")
            hatalar.append("H1: lsof yokken GECTI (fail-open)")
        except OrtamHatasi as e:
            _kontrol(kapilar, hatalar, "H1 kanarya (lsof yok)", True, str(e)[:110])

    for satir in kapilar:
        print(satir)
    n_pass = len([k for k in kapilar if k.startswith("PASS")])
    print(f"--- {n_pass} PASS · {len(hatalar)} HATA ---")
    if hatalar:
        for h in hatalar:
            print(f"HATA: {h}", file=sys.stderr)
        print(f"KAPI DUSTU: {len(hatalar)} hata", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
