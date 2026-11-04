#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P2.5 FAZ-1 — HUKUK KORPUSU GETİRİCİ (mevzuat.gov.tr)
====================================================
Kamu portalı mevzuat.gov.tr'den ölçülü, saygılı (aralıklı) getirme ile temel
kanun metinlerini indirir, madde numaralı satır yapısını koruyarak
`data/legal_korpus/` altına düz metin dosyaları yazar.

Hukuki çerçeve: TBMM/mevzuat metinleri (kanunlar, tüzükler) FSEK m.19(b)
uyarınca telif dışıdır; kamu portalından ölçülü getirme yapılır.

Kural: ham HTML önbelleği `scratch/mvz_raw/` (gitignore) altında tutulur;
çıktılar YENİ dosyalardır. İzlenen dosyalara dokunulmaz.

Kullanım:
    venv/bin/python scripts/fetch_mevzuat_corpus.py [--sleep 3.5] [--only 4857]
"""

import argparse
import html as htmllib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "data", "legal_korpus")
RAW_DIR = os.path.join(REPO, "scratch", "mvz_raw")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# MevzuatTertip: 2069 ve üzeri kanun numaraları 5. tertiptir.
# Seçim ölçütü: kullanıcının FMS-200/elektrik müh. bağlamı (iş, yapı, elektrik,
# veri) + temel özel hukuk/kamu hukuku kodları. TDK kanunu DEĞİL.
LAWS = [
    (657,   "Devlet Memurları Kanunu", 5),
    (2577,  "İdari Yargılama Usulü Kanunu", 5),
    (2872,  "Çevre Kanunu", 5),
    (3194,  "İmar Kanunu", 5),
    (4708,  "Yapı Denetimi Hakkında Kanun", 5),
    (4721,  "Türk Medeni Kanunu", 5),
    (4857,  "İş Kanunu", 5),
    (5237,  "Türk Ceza Kanunu", 5),
    (5271,  "Ceza Muhakemeleri Kanunu", 5),
    (5378,  "Engelliler Hakkında Kanun", 5),
    (5510,  "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu", 5),
    (6098,  "Türk Borçlar Kanunu", 5),
    (6100,  "Hukuk Muhakemeleri Kanunu", 5),
    (6331,  "İş Sağlığı ve Güvenliği Kanunu", 5),
    (6446,  "Elektrik Piyasası Kanunu", 5),
    (6502,  "Tüketicinin Korunması Hakkında Kanun", 5),
    (6698,  "Kişisel Verilerin Korunması Kanunu", 5),
]

URL_TMPL = (
    "https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe"
    "?MevzuatNo={no}&MevzuatTur=1&MevzuatTertip={tertip}"
)

MADDE_RE = re.compile(
    r"((?:GEÇİCİ\s+|EK\s+)?MADDE\s*(\d+)\s*[-–—])", re.I
)


def slugify(name: str) -> str:
    tr = "çÇğĞıİöÖşŞüÜ"
    repl = "cCgGiIoOsSuU"
    s = "".join(repl[tr.index(c)] if c in tr else c for c in name)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def fetch_law(no: int, tertip: int, sleep_s: float, retries: int = 3) -> bytes:
    """Tek kanunun HTML'ini indirir (saygılı aralık + yeniden deneme)."""
    url = URL_TMPL.format(no=no, tertip=tertip)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 20_000:
                raise ValueError(f"aşırı küçük yanıt ({len(data)} bayt)")
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            sys.stderr.write(
                f"[fetch] {no} deneme {attempt}/{retries} başarısız: {e}\n"
            )
            time.sleep(sleep_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"Kanun {no} getirilemedi: {last_err}")


def html_to_text(raw: bytes) -> str:
    """Mevzuat HTML'i -> madde numaralı satır yapısını koruyan düz metin."""
    # Gövde Windows-1254 beyan etse de sunucu fiilen UTF-8 yazar.
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("windows-1254", errors="replace")

    # DİKKAT: txt.lower() 'İ' (U+0130) karakterini İKİ karaktere büyüttüğünden
    # indeks kayar; büyük-küçük harfsiz arama daima re.IGNORECASE ile yapılır.
    m_body = re.search(r"<body[^>]*>", txt, flags=re.I)
    if m_body:
        txt = txt[m_body.end():]

    # Stil/beton blokları at.
    txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<script.*?</script>", " ", txt, flags=re.S | re.I)

    # Paragraf/satır/hücre sonları \x00 işaretiyle ayrılır; ham \r\n asla
    # bölümleyici değildir (span içi kırılmalar kelimeleri bölmesin).
    txt = re.sub(r"(?i)</p\s*>", "\x00", txt)
    txt = re.sub(r"(?i)<br[^>]*>", "\x00", txt)
    txt = re.sub(r"(?i)</tr\s*>", "\x00", txt)
    txt = re.sub(r"(?i)</td\s*>", " ", txt)
    # Satır içi etiketleri ve geri kalan tüm etiketleri sil.
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = htmllib.unescape(txt).replace("\xa0", " ")
    # Tüm boşlukları (raw \r\n dahil) tek boşluğa indir; "Madde\r\n1 -" kendiliğinden birleşir.
    txt = re.sub(r"\s+", " ", txt)

    lines = []
    for para in txt.split("\x00"):
        para = para.strip()
        if not para:
            continue
        # Madde başlığı satırın ortasındaysa: öncesini koru, maddeyi yeni satıra al.
        while True:
            m = MADDE_RE.search(para)
            if not m:
                break
            pre = para[: m.start()].strip()
            rest = para[m.start():].strip()
            if pre:
                lines.append(pre)
            # rest'in başında madde başlığı var; içinde başka madde olabilir.
            m2 = MADDE_RE.search(rest, len(m.group(1)))
            if m2:
                nxt = rest[m2.start():].strip()
                head_body = rest[: m2.start()].strip()
                # İlk maddeyi normalize ederek satır yap.
                lines.append(normalize_madde_head(head_body))
                para = nxt
            else:
                lines.append(normalize_madde_head(rest))
                para = ""
                break
        if para:
            lines.append(para)
    return "\n".join(lines)


def normalize_madde_head(chunk: str) -> str:
    """'MADDE 12 – metin' biçimini 'Madde 12 - metin' kurallı biçimine indirger."""
    m = MADDE_RE.match(chunk)
    if not m:
        return chunk
    head = re.sub(r"\s+", " ", m.group(1)).strip()      # 'MADDE 12 –'
    head = head.replace("–", "-").replace("—", "-")
    # Tüm-büyük biçemleri kurallı küçük-başlığa indir (MADDE→Madde, GEÇİCİ→Geçici, EK→Ek).
    head = re.sub(r"(?i)^(GEÇİCİ)(\s+MADDE)", r"Geçici\2", head)
    head = re.sub(r"(?i)^(EK)(\s+MADDE)", r"Ek\2", head)
    head = re.sub(r"(?i)MADDE", "Madde", head)
    return head + " " + chunk[m.end():].strip()


def parse_metadata(text: str, expected_no: int) -> dict:
    """Gövde ilk satırlarından kanun adı/no/tarih/RG üst verisini çıkarır."""
    meta = {}
    for key, pat in [
        ("kanun_no", r"Kanun Numarası\s*:\s*(\d+)"),
        ("kabul_tarihi", r"Kabul Tarihi\s*:\s*([\d/.]+)"),
        ("rg", r"Resm[iî]?\s*Gazete\s*:\s*Tarih[iî]?\s*:\s*([\d/.]+)\s*Sayı\s*:\s*(\d+)"),
    ]:
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(1) if key != "rg" else f"{m.group(1)} - Sayı {m.group(2)}"
    # Kanun adı: başlık birkaç paragraga bölünebilir; 'KANUN' geçene kadar birleştir.
    cand = ""
    for ln in text.split("\n")[:4]:
        cand = (cand + " " + ln).strip()
        if re.search(r"KANUN", ln, re.I):
            break
        if "Numarası" in ln:
            cand = ""
            break
    cand = re.sub(r"\s*\[\d+\]", "", cand).strip()
    if (
        len(cand) < 120
        and re.match(r"^[A-ZÇĞİÖŞÜ0-9\s.,()''`–-]+$", cand)
        and re.search(r"KANUN", cand, re.I)
        and "Numarası" not in cand
    ):
        meta["kanun_adi"] = cand
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=3.5, help="istekler arası saniye")
    ap.add_argument("--only", type=str, default="", help="yalnız bu numaralı kanun (virgülle)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
    access_date = time.strftime("%Y-%m-%d")
    manifest = []
    failures = []

    todo = LAWS if not only else [l for l in LAWS if l[0] in only]
    for idx, (no, name, tertip) in enumerate(todo):
        out_path = os.path.join(OUT_DIR, f"{no:04d}_{slugify(name)}.txt")
        raw_path = os.path.join(RAW_DIR, f"{no}.html")
        if os.path.exists(out_path):
            print(f"[{idx+1}/{len(todo)}] {no} zaten mevcut: {os.path.basename(out_path)}")
            continue
        try:
            if os.path.exists(raw_path):
                raw = open(raw_path, "rb").read()
                print(f"[{idx+1}/{len(todo)}] {no} önbellekten")
            else:
                raw = fetch_law(no, tertip, args.sleep)
                with open(raw_path, "wb") as f:
                    f.write(raw)
                if idx < len(todo) - 1:
                    time.sleep(args.sleep)
        except Exception as e:  # noqa: BLE001
            failures.append({"no": no, "hata": str(e)})
            print(f"[{idx+1}/{len(todo)}] {no} BAŞARISIZ: {e}")
            continue

        text = html_to_text(raw)
        meta = parse_metadata(text, no)
        got_no = int(meta.get("kanun_no", -1))
        if got_no != no:
            failures.append({"no": no, "hata": f"üst veri kanun no uyuşmaz: {got_no}"})
            print(f"[{idx+1}/{len(todo)}] {no} ÜST VERİ UYUŞMAZLIĞI (no={got_no}), atlandı.")
            continue

        url = URL_TMPL.format(no=no, tertip=tertip)
        header = (
            f"# KANUN ADI: {meta.get('kanun_adi', name)}\n"
            f"# KANUN NO: {no}\n"
            f"# KABUL TARİHİ: {meta.get('kabul_tarihi', 'BULUNAMADI')}\n"
            f"# RESMÎ GAZETE: {meta.get('rg', 'BULUNAMADI')}\n"
            f"# KAYNAK URL: {url}\n"
            f"# ERİŞİM TARİHİ: {access_date}\n"
            f"# TELİF: Kamu mevzuatı — kanun/tüzük metinleri FSEK m.19(b) uyarınca telif dışı\n"
        )
        # Üst veri bloğu gövdede tekrar etmesin: ilk 'BÖLÜM/KISIM/Madde' satırından başlat.
        m_body = re.search(
            r"\n(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|KISIM|BÖLÜM|Madde\s*\d+\s*[-–—])",
            text, flags=re.I,
        )
        body = text[m_body.start() + 1:] if m_body else text

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + "\n" + body.strip() + "\n")

        n_chars = os.path.getsize(out_path)
        n_madde = len(
            re.findall(r"^(?:Geçici\s+|Ek\s+)?Madde\s*\d+\s*[-–—]", body, flags=re.M | re.I)
        )
        manifest.append({
            "no": no, "ad": name, "kanun_adi_sayfa": meta.get("kanun_adi", ""),
            "kabul": meta.get("kabul_tarihi", ""), "rg": meta.get("rg", ""),
            "url": url, "dosya": os.path.basename(out_path),
            "karakter": n_chars, "madde_sayisi": n_madde,
        })
        print(f"[{idx+1}/{len(todo)}] {no} OK: {name} ({n_chars:,} bayt, {n_madde} madde satırı)")

    if failures:
        print("\nBAŞARISIZLAR:", json.dumps(failures, ensure_ascii=False, indent=1))
    if manifest:
        mpath = os.path.join(OUT_DIR, "korpus_manifest.json")
        old = []
        if os.path.exists(mpath):
            try:
                old = json.load(open(mpath, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                old = []
        merged = {m["no"]: m for m in old + manifest}
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(sorted(merged.values(), key=lambda x: x["no"]),
                      f, ensure_ascii=False, indent=1)
        print(f"\nManifest: {mpath} ({len(merged)} kanun)")


if __name__ == "__main__":
    main()