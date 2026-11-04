#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.5 KALEM-3 — HUKUK SFT VERİ SETİ ÜRETİCİSİ v3 (KART-KİMLİĞİ DOZU)
===================================================================
Girdi: data/pedagogy/legal_canonical_sft_v2.jsonl (4.813 kayıt, DOKUNULMAZ)
Çıktı: data/pedagogy/legal_canonical_sft_v3.jsonl (+ .meta.json) — YENİ dosya.

ZEMİN (p25c hükmü, 2026-11-04): K1 (alan-CE) v2 verisiyle İLK KEZ geçti
(Δ −0,523 nat) ama K2 kanonik-20 0/20 kaldı: model %84'lük b1∪b3
'Mevzuat' çerçevesine çöküyor, kart-tip ayrımı üretmiyor ve kanun/madde
numarası halüsinasyonu yapıyor ('Kanun No. 477' — veride yok). Ortak-önek
0,0→0,9: yalnız '[Mevzuat]' ilk jetonu edinilmiş; kart İSKELETİ (ilk-4/ilk-16
jeton) kurulmamış.

K2 RİG'İNİN ŞEKLİ (scripts/train_legal_ceket.py::kanonik_dogruluk, ölçüldü):
  - tohum-42 ile TÜM kayıt listesinden 20 örnek; her kayıtta argmax üretim
    (≤48 jeton) altın gövde jeton-id'leriyle karşılaştırılır.
  - kart4    = gen[:4]  == gold[:4]   → kart-TİP iskeleti (ölçülmüş ilk-4'ler:
    'mevzuat <ENT> <CAP> h' [Hükmü] · 'mevzuat yanıt POSS_3SG' [Yanıtı] ·
    'kanun tanım POSS_3SG' [Tanımı] · 'kavram ayrım POSS_3SG' [Ayrımı]).
  - etiket16 = gen[:16] == gold[:16] → kart-tip + kanun ADI + '( kanun No'
    öneki (kısa adlı Yanıtı kartlarında TAM kanun numarası dahil).
  - K2 barı: etiket16 ≥ 5/20.

MODELİN EDİNEMEDİĞİ ŞEKİL (v3'ün hedefi):
  (i)  kart-tip iskeletinin ilk-4/ilk-16 jeton bütünlüğü (sabit iskelet dozu),
  (ii) talimat→kart-tip eşlemesi (tip-dengeli doz; v2 yeni-blok %93 b-çerçeveydi),
  (iii) kanun adı↔numara bağının açık denetimi ('(Kanun No. N)' talimatta+çıktıda),
  (iv) KONU→MADDE-NO ve MADDE-NO→METİN eşlemeleri (v2'de ikisi de yok —
       halüsinasyonun ölçülen kaynağı).

ÜÇ DESEN (tüm alıntılar korpus madde satırından; uydurma YOK; FAZ-1 kapıları
+ v2 dedupe/birebir kalıbı aynen):
  (a) MADDE-BULMA  : "'konu' hangi maddede düzenlenmiştir?" → Mevzuat Yanıtı
      kartı: kanun+madde NO + birebir alıntı (talimatta numara YOK — cevapta VAR).
      konu = resmi madde başlığı (%60) ya da madde metninden birebir öbek (%40,
      'madde' geçen öbekler elenir; öbeğin metinde geçişi kapıyla doğrulanır).
  (b) TERS-KİMLİK  : "{k}. maddesi ne düzenler?" → Mevzuat Hükmü kartı:
      birebir alıntı + kısa öz (başlıklı: yapısal başlık özü; başlıksız:
      alıntının ilk cümlesi — birebir). Havuza başlıksız/Geçici-Ek maddeler
      de karışır (b3 sınıfından ~%25).
  (c) KART-İSKELET DOZU (5 alt-tip, tip-DENGELİ):
      c1 Mevzuat Hükmü  — talimatta '(Kanun No. N)' + etiket; çıktı b-çerçevesi
                          TAM (f1.rec_b çıktısı birebir).
      c2 Mevzuat Yanıtı — kaynak-madde sorusu; çıktı rec_b_hedefli çerçevesi TAM.
      c3 Kanun Tanımı   — f1.rec_a çıktısı TAM, YENİ talimat şablonu.
      c4 Kavram Ayrımı  — YENİ bent çiftleri (i, i+2) (v2'de (i, i+1) idi);
                          f1.rec_c çıktı biçimi birebir.
      c5 Mevzuat Dizisi — YENİ pencereler (ofset-1; v2 ofset-0 idi);
                          f2.rec_d_dizi çıktısı birebir.

MONTAJ SÖZLEŞMESİ (görev talimatı):
  v3 = tohum-42 karışmış [v2[:-50] + yeni kayıtlar] + v2[-50:] (SON-50 val
  kuyruğu BAYT-BİREBİR — 2A rig'inin val avlusu değişmez; eski satırlar
  yeniden serileştirilmez). Yazım sonrası: satır çokluğu, son-50 özdeşliği,
  şema ve alıntı-birebirliği programatik doğrulanır; sha256 programatik.

Kullanım:
    venv/bin/python scripts/prepare_legal_dataset_v3.py [--dry-run]
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

import scripts.prepare_legal_dataset as f1      # FAZ-1 kapıları/kurucuları
import scripts.prepare_legal_dataset_v2 as f2   # v2 (rec_d_dizi kalıbı)

V2_JSONL = os.path.join(KÖK, "data", "pedagogy", "legal_canonical_sft_v2.jsonl")
OUT_JSONL = os.path.join(KÖK, "data", "pedagogy", "legal_canonical_sft_v3.jsonl")
OUT_META = OUT_JSONL + ".meta.json"
TOHUM = 42

# Hedef sayılar (bant: 300-600 toplam)
A_HEDEF, B_HEDEF = 150, 150          # (a) madde-bulma, (b) ters-kimlik
CB_HEDEF, CY_HEDEF = 60, 50          # (c1) Hükmü-kart, (c2) Yanıtı-kart
CA_HEDEF, CC_HEDEF, CD_HEDEF = 50, 50, 30   # (c3) Tanım, (c4) Ayrım, (c5) Dizi
VAL_N = 50

# --- Şablonlar (v2 şablon setlerinden FARKLI; UNK ölçülü — 'hükmü' UNK, kullanılmaz) ---
A_SABLON = [  # (a) madde-bulma: konu→numara; talimatta madde numarası YOK
    "{kanun}'nda '{konu}' hangi maddede düzenlenmiştir?",
    "Mevzuat sorusu: {kanun} (Kanun No. {no}) kapsamında '{konu}' hangi maddede yer alır?",
    "{kanun} metninde '{konu}' kaçıncı maddede düzenlenmiştir?",
    "Mevzuat araması: {kanun}'da '{konu}' ifadesi hangi maddeye dayanır?",
    "Hukuk danışmanı olarak: {kanun}'nda '{konu}' düzenlemesini hangi madde taşır?",
]
B_SABLON = [  # (b) ters-kimlik: numara→metin
    "{kanun}'nun {kisi} maddesi ne düzenler?",
    "Mevzuat sorusu: {kanun} (Kanun No. {no}) {kisi} maddesinin metni nedir?",
    "{kanun} {kisi} madde metnini aynen ver.",
    "Ters kimlik sorusu: {kanun}'nun {kisi} maddesi ne der?",
    "Hukuk danışmanı olarak: {kanun}'nun {kisi} maddesi hangi konuyu düzenler, metniyle ver?",
]
CB_SABLON = [  # (c1) kart-iskelet: Mevzuat Hükmü — ad↔numara bağı talimatta
    "Mevzuat uzmanı: {kanun} (Kanun No. {no}) {label} metnini aynen aktar.",
    "Kanun metni sorusu: {kanun} (Kanun No. {no}) {label} ne der?",
    "Mevzuat kartı: {kanun} (Kanun No. {no}) '{baslik}' düzenlemesini aynen ver.",
    "{kanun} (Kanun No. {no}) {label} metni nedir?",
]
CY_SABLON = [  # (c2) kart-iskelet: Mevzuat Yanıtı
    "Mevzuat yanıtı: {kanun}'nun {label} ne der, kaynak madde hangisidir?",
    "Kaynak sorusu: {kanun} {label} metnini alıntıyla ver.",
    "{kanun} {label} ne der, kaynağı hangi maddedir?",
]
CA_SABLON = [  # (c3) kart-iskelet: Kanun Tanımı
    "Tanım sorusu: {kanun}'da '{kavram}' kavramının kanun tanımı nedir?",
    "Hukuk terimi: '{kavram}' terimi {kanun} içinde nasıl tanımlanmıştır?",
    "Kanun tanımı sorusu: '{kavram}' terimi {kanun}'da ne anlama gelir?",
]
CC_SABLON = [  # (c4) kart-iskelet: Kavram Ayrımı
    "Kavram ayrımı sorusu: '{kx}' ve '{ky}' kavramlarını {kanun} tanımları üzerinden karşılaştır.",
    "Hukuk kavram karşılaştırması: '{kx}' ile '{ky}' arasındaki ayrımı kanun tanımlarıyla göster.",
]


# ---------------------------------------------------------------------------
# Korpus yardımcıları
# ---------------------------------------------------------------------------
def b1_eligible(mads):
    """v2 b1 filtre seti BİREBİR (başlıklı, non-mülga madde)."""
    return [m for m in mads
            if not m.gecici and m.baslik and not m.is_mulga()
            and len(m.text) >= f1.MIN_MADDE_CHARS
            and m.text.count("yerine işlenmiştir") < 2
            and not ("yerine işlenmiştir" in m.text and len(m.text) < 400)
            and not m.baslik.lower().startswith(("birinci", "ikinci"))
            and m.baslik.lower() not in
            ("genel hükümler", "kapsam", "amaç", "yürürlük", "kabul")]


def kisi_etiket(m):
    """'Madde 63' → '63.'; 'Geçici Madde 3' → 'geçici 3.' (ters-kimlik talimat biçimi)."""
    mm = re.match(r"^(Geçici|Ek)?\s*Madde\s+(\d+)$", m.label.rstrip(" -"))
    if not mm:
        return None
    onek = (mm.group(1).lower() + " ") if mm.group(1) else ""
    return f"{onek}{mm.group(2)}."


def konu_ifadesi(m):
    """Madde gövdesinden birebir konu öbeği (20-90 kr, ≥3 kelime).

    Uydurma kapısı: öbek yalnız ham gövde satırından kesilir ve m.text içinde
    birebir geçiği doğrulanır. Süzgeçler: 'madde' geçen öbekler (numara
    sızıntısı), dipnot/kurulum satırları ('(' ile başlayan, Değişik/Mülga/
    İptal içeren), liste-kurulum öbekleri (';', ':' ile biten) ve genel
    girişler ('Bu Kanun…') elenir."""
    for ln in m.lines[:6]:
        s = ln.strip()
        if not s or f1.ANNOT_LINE.match(s):
            continue
        s = f1.INLINE_ANNOT.sub(" ", s)
        s = re.sub(r"^\(\d+\)\s*", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 20 or s.startswith("("):
            continue
        if s.lower().startswith(("bu kanun", "bu madde", "aşağıda")):
            continue
        kelime = s.split()
        for adet in range(min(len(kelime), 7), 2, -1):
            öbek = " ".join(kelime[:adet])
            if not (20 <= len(öbek) <= 90):
                continue
            düşük = öbek.lower()
            if any(k in düşük for k in ("madde", "değişik", "mülga", "iptal")):
                continue
            if öbek[-1] in ";:,":
                continue
            if düşük in m.text.lower():
                return öbek
    return None


def bagimsiz_temiz(m):
    """Alıntı-birebir doğrulayıcısı — clean_quote'dan BAĞIMSIZ yol:
    gövde satırlarına belgelenen temizlik (dipnot satırı düşür, satır-içi
    dipnot sil, fıkra numarası düşür, boşluğu normalleştir) uygular ve
    tek metne birleştirir. Kurucunun taşıdığı her alıntı bu metnin içinde
    (ya da öneki) olmalıdır."""
    parts = []
    for ln in m.lines:
        ln = ln.strip()
        if not ln or f1.ANNOT_LINE.match(ln):
            continue
        ln = f1.INLINE_ANNOT.sub(" ", ln)
        ln = re.sub(r"^\(\d+\)\s*", "", ln)
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            parts.append(ln)
    return " ".join(parts)


def kaynak_varyantlar(m):
    """Birebir doğrulama için kaynak metin varyantları:
      V1 — temiz: clean_quote'un belgelenen satır hattı BAĞIMSIZ yeniden
           kurulumu (dipnot satırı düşür → satır-içi dipnot sil → fıkra
           numarası düşür → boşluk normalleştir → boş satır düşür).
           clean_quote tabanlı alıntılar bunun öneki/alt-dizisidir;
      V2 — ham: dipnot satırı düşür + boşluk normalleştir (satır-içi
           düzeltme '(Değişik:…)' ve fıkra numaraları DURUR — FAZ-1 rec_a
           ham-satır alıntılarının birebirlik zemini);
      V3/V4 — kolon-normalize kopyalar: 'Kavram : tanım' → 'Kavram: tanım'
           (FAZ-1/v2 rec_c 'Kavram: tanım' önek kurulumunun biçim
           toleransı; tanım GÖVDESİ birebir kalır)."""
    v1, v2 = [], []
    for ln in m.lines:
        s = ln.strip()
        if not s or f1.ANNOT_LINE.match(s):
            continue
        ham = re.sub(r"\s+", " ", s)
        v2.append(ham)
        t = f1.INLINE_ANNOT.sub(" ", ham)
        t = re.sub(r"^\(\d+\)\s*", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            v1.append(t)
    metinler = [" ".join(v1), " ".join(v2)]
    kolon = [re.sub(r"\s*:\s*", ": ", t) for t in metinler]
    return metinler + kolon


def alinti_birebir_mi(quote, m):
    """«» içeriği kaynak maddeden izlenebilir mi (birebirlik kapısı).

    Kabul: alıntı (kesme '…' düşülmüş hali) kaynak varyantlarından birinin
    öneki ya da alt-dizisi. c3/c4 bent-kartlarının 'Kavram: tanım' öneki
    kurucuda kanonik biçimde yeniden kurulur (v2 kalıbı) — gövde birebir
    doğrulanır, önek biçimi (kolon boşluğu) tolere edilir."""
    q = quote.strip()
    while q.endswith("…"):
        q = q[:-1].rstrip()
    q = q.strip()
    if not q:
        return False
    for temiz in kaynak_varyantlar(m):
        if temiz.startswith(q) or (q in temiz):
            return True
    return False


def kisa_oz_cumle(quote, cap=140):
    """Alıntının ilk cümlesi — birebir, ≤cap kr (kesilirse '…')."""
    idx = quote.find(". ")
    c = quote[: idx + 1] if idx > 0 else quote
    if len(c) > cap:
        c = c[:cap].rsplit(" ", 1)[0] + "…"
    return c


def rr_sec(havuzlar, hedef):
    """Kanunlar arası yuvarlak-robin seçim (havuz başına ön-karıştırılmış)."""
    sonuc = []
    kanunlar = sorted(havuzlar)
    while len(sonuc) < hedef:
        hareket = False
        for no in kanunlar:
            if len(sonuc) >= hedef:
                break
            lst = havuzlar[no]
            if lst:
                sonuc.append((no, lst.pop(0)))
                hareket = True
        if not hareket:
            break
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(" P2.5 KALEM-3 — LEGAL CANONICAL SFT v3 ÜRETİMİ (kart-kimliği dozu)")
    print("=" * 72)

    rng = random.Random(TOHUM)
    laws = f1.load_corpus()
    print(f"Korpus: {len(laws)} kanun, "
          f"{sum(len(v) for v in laws.values()):,} madde satırı")

    # ---- Havuzlar (tohum-42 ön-karıştırma; b1 dörde bölünür, b havuzuna ~%25
    # ---- başlıksız/Geçici-Ek karışır) --------------------------------------
    havuz = {}
    for no, mads in laws.items():
        e1 = b1_eligible(mads)
        rng.shuffle(e1)
        b1_keyler = {(no, m.label) for m in e1}
        e3 = [m for m in mads
              if not m.is_mulga() and len(m.text) >= f1.MIN_MADDE_CHARS
              and (no, m.label) not in b1_keyler]
        rng.shuffle(e3)
        p = {"a": [], "b": [], "cb": [], "cy": []}
        adlar = ["a", "b", "cb", "cy"]
        for i, m in enumerate(e1):
            p[adlar[i % 4]].append(m)
        ek = max(0, len(p["b"]) // 3)
        p["b"] = p["b"] + e3[:ek]
        rng.shuffle(p["b"])
        havuz[no] = p
    print("Havuzlar hazır (b1 4'e bölündü; b'ye başlıksız karıştı).")

    yeni = []        # (kayıt, provenans) — provenans: tip, kanun, [(madde, alıntı)]
    dusen = []
    per_type = Counter()

    def ekle(r, tip, no, kaynaklar):
        yeni.append((r, {"tip": tip, "kanun": no, "kaynaklar": kaynaklar}))
        per_type[tip] += 1

    # ---- (a) MADDE-BULMA: konu→madde numarası (Mevzuat Yanıtı kartı) --------
    a_sec = rr_sec({no: havuz[no]["a"] for no in havuz}, A_HEDEF)
    for i, (no, m) in enumerate(a_sec):
        ad = f1.LAW_NAMES[no]
        konu = konu_ifadesi(m) if rng.random() < 0.4 else None
        if konu is None:
            konu = m.baslik
        ins = A_SABLON[i % len(A_SABLON)].format(kanun=ad, no=no, konu=konu)
        if re.search(r"madde\s*\d", ins, re.I):   # numara sızıntısı kapısı
            dusen.append(("a", f"talimatta numara sızdı: {ins[:60]}"))
            continue
        r = f1.rec_b_hedefli(no, ad, m, ins)
        quote = re.search(r"«(.*?)»", r["output"], re.S).group(1)
        ekle(r, "a_madde_bulma", no, [(m, quote)])
    print(f"(a) madde-bulma: {per_type['a_madde_bulma']}")

    # ---- (b) TERS-KİMLİK: madde numarası→metin (Mevzuat Hükmü kartı + öz) ---
    b_sec = rr_sec({no: havuz[no]["b"] for no in havuz}, B_HEDEF)
    for i, (no, m) in enumerate(b_sec):
        ad = f1.LAW_NAMES[no]
        kisi = kisi_etiket(m)
        if kisi is None:
            dusen.append(("b", f"etiket çözülmedi: {m.label}"))
            continue
        label = m.label.rstrip(" -")
        quote = m.clean_quote(f1.QUOTE_CAP)
        if m.baslik:
            govde = (f"Mevzuat Hükmü — {ad} (Kanun No. {no}), {label} "
                     f"({m.baslik}): «{quote}»")
            oz = f"Kısa öz: hüküm '{m.baslik}' konusundadır."
        else:
            govde = f"Mevzuat Hükmü — {ad} (Kanun No. {no}), {label}: «{quote}»"
            oz = f"Kısa öz: {kisa_oz_cumle(quote)}"
        ins = B_SABLON[i % len(B_SABLON)].format(kanun=ad, no=no, kisi=kisi)
        if kisi.split()[0].rstrip(".") not in ins:
            dusen.append(("b", f"numara talimatta yok: {ins[:60]}"))
            continue
        ekle({"instruction": ins, "input": "", "output": govde + " " + oz},
             "b_ters_kimlik", no, [(m, quote)])
    print(f"(b) ters-kimlik: {per_type['b_ters_kimlik']}")

    # ---- (c1) KART-İSKELET: Mevzuat Hükmü (çıktı = f1.rec_b, TAM) ------------
    cb_sec = rr_sec({no: havuz[no]["cb"] for no in havuz}, CB_HEDEF)
    for i, (no, m) in enumerate(cb_sec):
        ad = f1.LAW_NAMES[no]
        r = f1.rec_b(no, ad, m, i)
        r["instruction"] = CB_SABLON[i % len(CB_SABLON)].format(
            kanun=ad, no=no, label=m.label.rstrip(" -"), baslik=m.baslik)
        quote = re.search(r"«(.*?)»", r["output"], re.S).group(1)
        ekle(r, "c1_kart_hukum", no, [(m, quote)])
    print(f"(c1) kart Hükmü: {per_type['c1_kart_hukum']}")

    # ---- (c2) KART-İSKELET: Mevzuat Yanıtı (çıktı = rec_b_hedefli, TAM) ------
    cy_sec = rr_sec({no: havuz[no]["cy"] for no in havuz}, CY_HEDEF)
    for i, (no, m) in enumerate(cy_sec):
        ad = f1.LAW_NAMES[no]
        ins = CY_SABLON[i % len(CY_SABLON)].format(
            kanun=ad, label=m.label.rstrip(" -"))
        r = f1.rec_b_hedefli(no, ad, m, ins)
        quote = re.search(r"«(.*?)»", r["output"], re.S).group(1)
        ekle(r, "c2_kart_yanit", no, [(m, quote)])
    print(f"(c2) kart Yanıtı: {per_type['c2_kart_yanit']}")

    # ---- (c3) KART-İSKELET: Kanun Tanımı (çıkıtı = f1.rec_a, TAM) -----------
    defs = f1.bent_definitions(laws)
    kept_defs = []
    for (no, m, kavram, tanim, raw_lines) in defs:
        if len(kavram) < 3 or len(tanim) < 30:
            continue
        if len(kavram) > 48 or len(kavram.split()) > 5:
            continue
        if m.is_mulga():
            continue
        if "yerine işlenmiştir" in tanim and len(tanim) < 400:
            continue
        if raw_lines[0] not in m.text:
            continue
        kept_defs.append((no, m, kavram, tanim, raw_lines))
    seen_kv = set()
    kanun_defs = defaultdict(list)
    for (no, m, kavram, tanim, raw_lines) in kept_defs:
        key = (no, kavram.lower())
        if key in seen_kv:
            continue
        seen_kv.add(key)
        kanun_defs[no].append((m, kavram, tanim, raw_lines))
    ca_sec = rr_sec({no: list(dl) for no, dl in kanun_defs.items()}, CA_HEDEF)
    for i, (no, d) in enumerate(ca_sec):
        m, kavram, tanim, raw_lines = d
        r = f1.rec_a(no, f1.LAW_NAMES[no], m, kavram, tanim, i,
                    " ".join(raw_lines))
        r["instruction"] = CA_SABLON[i % len(CA_SABLON)].format(
            kanun=f1.LAW_NAMES[no], kavram=kavram)
        quote = re.search(r"«(.*?)»", r["output"], re.S).group(1)
        ekle(r, "c3_kart_tanim", no, [(m, quote)])
    print(f"(c3) kart Tanımı: {per_type['c3_kart_tanim']}")

    # ---- (c4) KART-İSKELET: Kavram Ayrımı (YENİ çiftler: i, i+2) ------------
    cift_havuz = {}
    for no, dl in kanun_defs.items():
        ad = f1.LAW_NAMES[no]
        lst = []
        for i in range(len(dl) - 2):
            (mx, kx, tx, _), (my, ky, ty, _) = dl[i], dl[i + 2]
            dx = f"{kx}: {tx}"[: f1.DEF_CAP] + ("…" if len(f"{kx}: {tx}") > f1.DEF_CAP else "")
            dy = f"{ky}: {ty}"[: f1.DEF_CAP] + ("…" if len(f"{ky}: {ty}") > f1.DEF_CAP else "")
            ayrim = (f"Her iki kavram da {ad} içinde ayrı bentlerde "
                     f"tanımlanmıştır; aralarındaki ayrım yukarıdaki kanun "
                     f"tanımlarının kendi içeriğinden okunur.")
            lst.append((mx, my, kx, ky, dx, dy, ayrim))
        cift_havuz[no] = lst
    cc_sec = rr_sec(cift_havuz, CC_HEDEF)
    for i, (no, cift) in enumerate(cc_sec):
        mx, my, kx, ky, dx, dy, ayrim = cift
        ad = f1.LAW_NAMES[no]
        r = f1.rec_c(ad, no, mx, kx, dx, my, ky, dy, ayrim)
        r["instruction"] = CC_SABLON[i % len(CC_SABLON)].format(
            kanun=ad, kx=kx, ky=ky)
        q1 = re.findall(r"«(.*?)»", r["output"], re.S)
        ekle(r, "c4_kart_ayrim", no, [(mx, q1[0]), (my, q1[1])])
    print(f"(c4) kart Ayrımı: {per_type['c4_kart_ayrim']}")

    # ---- (c5) KART-İSKELET: Mevzuat Dizisi (YENİ pencereler: ofset-1) -------
    d_havuz = {}
    for no, mads in laws.items():
        eligible = b1_eligible(mads)
        lst, i = [], 1
        while i + 3 <= len(eligible) and len(lst) < 2:
            grup = eligible[i: i + 3]
            if any(grup[j + 1].no != grup[j].no + 1 for j in range(len(grup) - 1)):
                i += 1
                continue
            quotes, ok = [], True
            for m in grup:
                q = m.clean_quote(320)
                if len(q) < 40:
                    ok = False
                    break
                quotes.append(q)
            if not ok:
                i += 1
                continue
            lst.append((grup, quotes))
            i += 3
        d_havuz[no] = lst
    cd_sec = rr_sec(d_havuz, CD_HEDEF)
    for i, (no, paket) in enumerate(cd_sec):
        grup, quotes = paket
        r = f2.rec_d_dizi(f1.LAW_NAMES[no], no, grup, quotes)
        ekle(r, "c5_kart_dizi", no, list(zip(grup, quotes)))
    print(f"(c5) kart Dizisi: {per_type['c5_kart_dizi']}")

    hedef_toplam = sum((A_HEDEF, B_HEDEF, CB_HEDEF, CY_HEDEF, CA_HEDEF,
                        CC_HEDEF, CD_HEDEF))
    print(f"Üretim öncesi toplam: {len(yeni)} (hedef {hedef_toplam})")

    # ---- v2 küfünü yükle + dedupe + şema kapısı (v2 kalıbı aynen) -----------
    v2_satirlar = [l for l in open(V2_JSONL, encoding="utf-8") if l.strip()]
    v2_recs = [json.loads(l) for l in v2_satirlar]
    v2_hashler = {hashlib.sha256(
        (r["instruction"] + "||" + r["output"]).encode()).hexdigest()
        for r in v2_recs}
    print(f"v2: {len(v2_recs)} kayıt yüklendi (dokunulmaz)")

    valid, gecen_provenans = [], []
    dup_sayi = 0
    for (r, prov) in yeni:
        if not f1.validate_record(r, dusen, len(valid)):
            dusen.append((prov["tip"], "şema dışı kayıt"))
            continue
        h = hashlib.sha256(
            (r["instruction"] + "||" + r["output"]).encode()).hexdigest()
        if h in v2_hashler:
            dup_sayi += 1
            continue
        v2_hashler.add(h)     # yeni-yeni dedupe
        valid.append(r)
        gecen_provenans.append(prov)
    print(f"Şema+dedupe sonrası: {len(valid)} (v2-örtüşme düşüş: {dup_sayi})")

    # ---- ALINTI-BİREBİR KAPISI (bağımsız doğrulayıcı, her alıntı) -----------
    alinti_ok = 0
    alinti_toplam = 0
    alinti_hatalar = []
    for (r, prov) in zip(valid, gecen_provenans):
        for (m, q) in prov["kaynaklar"]:
            alinti_toplam += 1
            if alinti_birebir_mi(q, m):
                alinti_ok += 1
            else:
                alinti_hatalar.append((prov["tip"], m.label, q))
                dusen.append((prov["tip"],
                              f"alıntı izlenemedi: {q[:60]!r} @ {m.label}"))
    print(f"Alıntı-birebir: {alinti_ok}/{alinti_toplam}")
    for (tip, lab, q) in alinti_hatalar[:12]:
        print(f"  [BİREBİR-DÜŞÜŞ] {tip} @ {lab}: {q[:90]!r}")
    if alinti_ok != alinti_toplam:
        raise SystemExit("P25D_DUR: birebirliği düşen alıntı var — yazılmamalı")

    # ---- Token/UNK ölçümü (v2 rig'i aynen: mode=True, kayıt-dışı) -----------
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary

    print("Token ölçümü (anka 33.114, literal_entity_mode=True)…")
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data", "lexicon", "roots.tsv"))
    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data", "rebuild",
                            "vocab_anka_r1_33114.json"))
    tok = KristalTokenizer(CrystalCompiler(lexicon, build_default_graph()),
                            vocab, literal_entity_mode=True, verbose=False)
    unk_id = vocab.stoi["<UNK>"]

    yeni_tok = yeni_unk = 0
    kayit_unk = []
    unk_kayit_sayisi = 0
    talimat_unk_kayit = 0
    for r in valid:
        ids = tok.encode(json.dumps(r, ensure_ascii=False))
        u = sum(1 for j in ids if j == unk_id)
        yeni_tok += len(ids)
        yeni_unk += u
        kayit_unk.append(u)
        if u:
            unk_kayit_sayisi += 1
        ins_ids = tok.encode(r["instruction"])
        if sum(1 for j in ins_ids if j == unk_id):
            talimat_unk_kayit += 1
    print(f"YENİ blok: {yeni_tok:,} token, UNK {yeni_unk:,} "
          f"(%{100 * yeni_unk / max(1, yeni_tok):.2f}) | UNK'lu kayıt: "
          f"{unk_kayit_sayisi}/{len(valid)} | maks/kayıt {max(kayit_unk)} | "
          f"talimatında UNK olan kayıt: {talimat_unk_kayit}")

    # ---- MONTAJ: v3 = shuffle(v2[:-50] + yeni, tohum-42) + v2[-50:] ---------
    v2_train_satir = v2_satirlar[: -VAL_N]
    v2_val_satir = v2_satirlar[-VAL_N:]
    yeni_satir = [json.dumps(r, ensure_ascii=False) + "\n" for r in valid]
    orta = [l if l.endswith("\n") else l + "\n" for l in v2_train_satir]
    orta = orta + yeni_satir
    random.Random(TOHUM).shuffle(orta)
    kuyruk = [l if l.endswith("\n") else l + "\n" for l in v2_val_satir]
    print(f"Montaj: orta {len(orta)} (v2-train {len(v2_train_satir)} + yeni "
          f"{len(yeni_satir)}, tohum-{TOHUM}) + val kuyruğu {len(kuyruk)}")

    if args.dry_run:
        print("[dry-run] yazım yapılmadı.")
        gorulen_tip = set()
        for (r, prov) in zip(valid, gecen_provenans):
            if prov["tip"] in gorulen_tip:
                continue
            gorulen_tip.add(prov["tip"])
            print(f"ÖRNEK [{prov['tip']}] INS:", r["instruction"][:95])
            print(f"      OUT:", r["output"][:120].replace("\n", " "))
            if len(gorulen_tip) >= 8:
                break
        print("Düşenler:", dusen[:10])
        return

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for l in orta + kuyruk:
            f.write(l)

    # ---- YAZIM SONRASI DOĞRULAMALAR -----------------------------------------
    v3_satirlar = [l for l in open(OUT_JSONL, encoding="utf-8") if l.strip()]
    son50_v3 = [l.rstrip("\n") for l in v3_satirlar[-VAL_N:]]
    son50_v2 = [l.rstrip("\n") for l in v2_satirlar[-VAL_N:]]
    ozdes = son50_v3 == son50_v2
    print(f"VAL-KANIT: v3 son-50 == v2 son-50 → {ozdes} | v3 toplam {len(v3_satirlar)}")
    if not ozdes:
        raise SystemExit("P25D_DUR: son-50 özdeşliği bozuldu")

    beklenen_orta = Counter([l.rstrip("\n") for l in v2_train_satir]) + \
        Counter([l.rstrip("\n") for l in yeni_satir])
    gercek_orta = Counter([l.rstrip("\n") for l in v3_satirlar[: -VAL_N]])
    print(f"ORTA-KANIT: çokluk özdeş → {beklenen_orta == gercek_orta}")

    sema_ok = 0
    for l in v3_satirlar:
        r = json.loads(l)
        if set(r.keys()) == {"instruction", "input", "output"} and \
                all(isinstance(v, str) for v in r.values()):
            sema_ok += 1
    print(f"ŞEMA-KANIT: {sema_ok}/{len(v3_satirlar)} kayıt {{instruction,input,output}}")

    # tam dosya token/UNK (v2 meta ölçümüyle aynı rig)
    tot_tok = unk_tok = 0
    for l in v3_satirlar:
        ids = tok.encode(l)
        tot_tok += len(ids)
        unk_tok += sum(1 for j in ids if j == unk_id)
    print(f"TAM v3: {tot_tok:,} token, UNK {unk_tok:,} "
          f"(%{100 * unk_tok / max(1, tot_tok):.3f})")

    # kanonik-20 önizleme (K2 örneklemi yeni nüfusta — kalem-5 beklentisi)
    kanon_idx = sorted(random.Random(42).sample(range(len(v3_satirlar)), 20))
    kanon_tipler = Counter()
    for i in kanon_idx:
        r = json.loads(v3_satirlar[i])
        kart = r["output"].split("—")[0].split(":")[0].strip()
        kanon_tipler[kart] += 1
    print(f"KANONİK-20 önizleme (tohum-42, nüfus {len(v3_satirlar)}): "
          f"{dict(kanon_tipler)}")

    def sha(yol):
        return hashlib.sha256(open(yol, "rb").read()).hexdigest()

    meta = {
        "uretim_tarihi": time.strftime("%Y-%m-%d %H:%M"),
        "gorev": "P2.5 KALEM-3 — legal SFT v3 (kart-kimliği dozu; K2 hedefli)",
        "kaynak_korpus": "data/legal_korpus/ (17 kanun, mevzuat.gov.tr)",
        "taban_v2": {
            "dosya": "data/pedagogy/legal_canonical_sft_v2.jsonl",
            "sha256": sha(V2_JSONL),
            "kayit": len(v2_recs),
            "dokunulmaz": True,
        },
        "vocab": "vocab_anka_r1_33114.json",
        "literal_entity_mode": True,
        "kayit_sayisi": len(v3_satirlar),
        "yeni_kayit": len(valid),
        "v2_kayit": len(v2_recs),
        "val_sozlesmesi": ("v2 son-50 (2A val avlusu) dosya sonunda BAYT-BİREBİR; "
                            "yeni kayıtlar tohum-42 ile v2-train bloğuna KARIŞTIRILDI"),
        "desenler": {
            "a_madde_bulma": ("konu→madde numarası (Mevzuat Yanıtı kartı); konu "
                              "başlık (%60) veya gövdeden birebir öbek (%40)"),
            "b_ters_kimlik": ("madde numarası→metin (Mevzuat Hükmü kartı + kısa öz); "
                              "havuz ~%25 başlıksız/Geçici-Ek"),
            "c_kart_iskelet": ("c1 Hükmü/c2 Yanıtı/c3 Tanımı/c4 Ayrımı/c5 Dizisi — "
                              "çıktı kalıbı FAZ-1/v2 kurucularından birebir; "
                              "talimatlar YENİ şablonlar; tip-dengeli doz"),
        },
        "tip_dagilimi_yeni": dict(per_type),
        "dedupe_v2_ortusme": dup_sayi,
        "alinti_birebir": {"gecen": alinti_ok, "toplam": alinti_toplam},
        "token_olcumu": {
            "yeni_blok": {
                "toplam_token": yeni_tok,
                "unk_token": yeni_unk,
                "unk_yuzde": round(100 * yeni_unk / max(1, yeni_tok), 3),
                "unklu_kayit": unk_kayit_sayisi,
                "talimatinda_unk_kayit": talimat_unk_kayit,
                "kayit": len(valid),
                "maks_kayit_unk": max(kayit_unk),
            },
            "tam_v3": {
                "toplam_token": tot_tok,
                "unk_token": unk_tok,
                "unk_yuzde": round(100 * unk_tok / max(1, tot_tok), 3),
                "ortalama_kayit_token": round(tot_tok / max(1, len(v3_satirlar)), 1),
            },
        },
        "kanonik20_onizleme": {
            "tohum": 42,
            "nufus": len(v3_satirlar),
            "orneklem_idx": kanon_idx,
            "kart_dagilimi": dict(kanon_tipler),
        },
        "dogrulamalar": {
            "son50_ozdes": ozdes,
            "orta_cokluk_ozdes": beklenen_orta == gercek_orta,
            "sema_ok": sema_ok == len(v3_satirlar),
            "satir_sayisi": len(v3_satirlar),
        },
        "dusenler": dusen,
        "montaj_toz": TOHUM,
        "sha256_v3": sha(OUT_JSONL),
        "sha256_uretici": sha(os.path.abspath(__file__)),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"Yazıldı: {OUT_JSONL} ({len(valid)} yeni + {len(v2_recs)} v2 = "
          f"{len(v3_satirlar)} kayıt) | sha256 {meta['sha256_v3']}")
    print(f"Meta: {OUT_META}")
    print("=" * 72)


if __name__ == "__main__":
    main()