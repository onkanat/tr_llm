#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P2.5 FAZ-2C — HUKUK SFT VERİ SETİ ÜRETİCİSİ v2 (ÖLÇEKLENDİRİLMİŞ)
=================================================================
FAZ-1 üreticisi (`scripts/prepare_legal_dataset.py`, 463 kayıt) ile AYNI
kalite kapıları; ölçek farkı yalnız YAPAY KAPLARIN KALDIRILMASI ve
YENİ korpus-temelli desenlerdedir. FAZ-1 modülü IMPORT EDİLİR — filtreler,
şema doğrulaması, alıntı kurucuları BİREBİR yeniden kullanılır.

ÖLÇEK DEĞİŞİKLİKLERİ (FAZ-1'e göre):
  (a) kavram tanımı     : 28/kanun tanım dengesi KALDIRILDI → tüm geçerli
                          bent tanımları (kapılar aynen: uzunluk, bent
                          biçimi, mülga/mezar filtresi, birebir-satır kapısı).
  (b1) başlıklı madde   : PER_LAW_CAP KALDIRILDI → tüm eligible başlıklı
                          maddeler (FAZ-1 filtre seti aynen).
  (b3) YENİ desen       : b1 kapsamı DIŞINDAKİ tüm madde satırları (başlıksız
                          ve Geçici/Ek maddeler dahil) için hedefli-soru
                          kartı — "17 kanunun TÜM madde satırları" güvencesi:
                          her non-mülga madde (≥60 kr) b1∪b3'te en az bir
                          kartla temsil edilir.
  (c) kavram ayrımı     : FAZ-1'in 11 el-kürasyonlu çifti eski 463'te korunur;
                          v2 AYNI kanun içindeki bent-tanım çiftlerinden
                          kayan-pencere çiftleri üretir. 'Ayrım:' cümlesi
                          YAPISAL-DOĞRU şablondur (iki kavramın ayrı
                          bentlerde tanımlı olduğu korpus yapısından
                          doğrulanabilir) — hukuk iddiası içermez, tırnak
                          içi alıntılar korpus metnidir.
  (d) madde dizisi      : prosedür deseninin otomatik karşılığı — ardışık
                          numaralı başlıklı maddelerin sıralı alıntısı.
                          'Prosedür' iddiası YOK ('Mevzuat Dizisi'); her
                          adım gerçek madde alıntısıdır.

UYDURMA KAPILARI (FAZ-1 aynen): her alıntı korpus madde satırından;
çözümlenemeyen düşürülür; mülga/boş elenir; şema doğrulama + dedupe
(yeni-yeni VE yeni-eski463) + token/UNK ölçümü meta kütüğüne.

VAL TUTARLILIK SÖZLEŞMESİ (görev talimatı): eski 463 kaydın SON-50'si
2A rig'inin val avlusu — v2 dosyasında BİREBİR korunur. Yeni kayıtlar
BAŞA eklenir; eski 463 satırı (bayt-düzeyinde, yeniden serileştirilmeden)
kuyrukta durur. Yazım sonrası son-50 özdeşliği programatik doğrulanır.

Kullanım:
    venv/bin/python scripts/prepare_legal_dataset_v2.py [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)

import scripts.prepare_legal_dataset as f1  # FAZ-1 üreticisi (kapılar aynen)

ESKI_JSONL = os.path.join(KÖK, "data", "pedagogy", "legal_canonical_sft.jsonl")
OUT_JSONL = os.path.join(KÖK, "data", "pedagogy", "legal_canonical_sft_v2.jsonl")
OUT_META = OUT_JSONL + ".meta.json"

B3_SABLONLARI = [
    "Mevzuat sorusu: {kanun} (Kanun No. {no}) {label} ne der?",
    "Hukuk danışmanı olarak: {kanun} (Kanun No. {no}) {label} hükmünü aynen aktarır mısın?",
    "Mevzuat uzmanı: {kanun} (Kanun No. {no}) {label} metnini ver.",
    "{kanun} (Kanun No. {no}) {label} kapsamında hangi hüküm yürür?",
    "Kanun hükmü sorusu: {kanun} (Kanun No. {no}) {label} düzenlemesi ne öngörür?",
]

D_PER_LAW_CAP = 12   # madde-dizisi pencere/kanun dengesi
D_PENCERE = 3        # ardışık madde sayısı (numara sürekliliği şartıyla)


def rec_b3(name, no, m, idx):
    """(b3) b1 dışı madde kartı — alıntı FAZ-1 clean_quote (birebir)."""
    sablon = B3_SABLONLARI[idx % len(B3_SABLONLARI)]
    quote = m.clean_quote(f1.QUOTE_CAP)
    label = m.label.rstrip(" -")
    instruction = sablon.format(kanun=name, no=no, label=label)
    output = f"Mevzuat Hükmü — {name} (Kanun No. {no}), {label}: «{quote}»"
    return {"instruction": instruction, "input": "", "output": output}


def rec_d_dizi(name, no, maddeler, quotes):
    """(d-auto) ardışık madde dizisi — sıralı alıntı, usul iddiası yok."""
    l0 = maddeler[0].label.rstrip(" -")
    l1 = maddeler[-1].label.rstrip(" -")
    ins = (f"Mevzuat dizisi sorusu: {name} (Kanun No. {no}) {l0}–{l1} "
           f"arasındaki ardışık maddelerin hükümlerini madde alıntılarıyla "
           f"sırasıyla ver.")
    lines = [f"Mevzuat Dizisi — {name} (Kanun No. {no}): ardışık {len(maddeler)} madde."]
    for i, (m, q) in enumerate(zip(maddeler, quotes), 1):
        lines.append(f"{i}) {m.label.rstrip(' -')} ({m.baslik}): «{q}»")
    return {"instruction": ins, "input": "", "output": "\n".join(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(" P2.5 FAZ-2C — LEGAL CANONICAL SFT v2 ÜRETİMİ (FAZ-1 kapıları, ölçekli)")
    print("=" * 72)

    laws = f1.load_corpus()
    print(f"Korpus: {len(laws)} kanun, {sum(len(v) for v in laws.values()):,} madde satırı")

    drops = []
    records = []      # YENİ kayıtlar (eski 463 sonda birebir eklenir)
    per_type = Counter()

    # ---- (a) Kavram tanımları — FAZ-1 kapıları, 28/kanun dengesi YOK --------
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
    seen_kavram = set()
    for (no, m, kavram, tanim, raw_lines) in kept_defs:
        key = (no, kavram.lower())
        if key in seen_kavram:
            continue
        seen_kavram.add(key)
        records.append(f1.rec_a(no, f1.LAW_NAMES[no], m, kavram, tanim,
                                len(records), " ".join(raw_lines)))
        per_type["a_kavram_tanimi"] += 1
    print(f"(a) kayıt: {per_type['a_kavram_tanimi']} (kapı-geçer bent tanımı: {len(kept_defs)})")

    # ---- (b1) Başlıklı madde — FAZ-1 filtre seti, PER_LAW_CAP YOK -----------
    b1_keys = set()
    for no, mads in laws.items():
        eligible = [m for m in mads
                    if not m.gecici and m.baslik and not m.is_mulga()
                    and len(m.text) >= f1.MIN_MADDE_CHARS
                    and m.text.count("yerine işlenmiştir") < 2
                    and not ("yerine işlenmiştir" in m.text and len(m.text) < 400)
                    and not m.baslik.lower().startswith(("birinci", "ikinci"))
                    and m.baslik.lower() not in
                    ("genel hükümler", "kapsam", "amaç", "yürürlük", "kabul")]
        for m in eligible:
            b1_keys.add((no, m.label))
            records.append(f1.rec_b(no, f1.LAW_NAMES[no], m, len(records)))
            per_type["b_madde_sorusu"] += 1
    print(f"(b1) kayıt (başlıklı madde, kap YOK): {per_type['b_madde_sorusu']}")

    # ---- (b3) YENİ — b1 dışı TÜM madde satırları (başlıksız + Geçici/Ek) ----
    n_gecici = 0
    for no, mads in laws.items():
        for m in mads:
            if m.is_mulga() or len(m.text) < f1.MIN_MADDE_CHARS:
                continue
            if (no, m.label) in b1_keys:
                continue
            records.append(rec_b3(f1.LAW_NAMES[no], no, m, len(records)))
            per_type["b3_tum_madde"] += 1
            if m.gecici:
                n_gecici += 1
    print(f"(b3) kayıt (b1 dışı madde): {per_type['b3_tum_madde']} (geçici: {n_gecici})")

    # ---- (c) Kavram ayrımları — otomatik aynı-kanun bent çiftleri -----------
    # FAZ-1'in 11 el-kürasyonlu çifti eski 463 içinde korunur; burada YENİ
    # çiftler kayan-pencere (i, i+1) ile — ayrım cümlesi yapısal şablon.
    c_cift = 0
    per_law_defs = {}
    for (no, m, kavram, tanim, raw_lines) in kept_defs:
        per_law_defs.setdefault(no, []).append((m, kavram, tanim))
    for no, dl in per_law_defs.items():
        name = f1.LAW_NAMES[no]
        for i in range(len(dl) - 1):
            (mx, kx, tx), (my, ky, ty) = dl[i], dl[i + 1]
            dx = f"{kx}: {tx}"[: f1.DEF_CAP] + ("…" if len(f"{kx}: {tx}") > f1.DEF_CAP else "")
            dy = f"{ky}: {ty}"[: f1.DEF_CAP] + ("…" if len(f"{ky}: {ty}") > f1.DEF_CAP else "")
            ayrim = (f"Her iki kavram da {name} içinde ayrı bentlerde "
                     f"tanımlanmıştır; aralarındaki ayrım yukarıdaki kanun "
                     f"tanımlarının kendi içeriğinden okunur.")
            records.append(f1.rec_c(name, no, mx, kx, dx, my, ky, dy, ayrim))
            per_type["c_kavram_ayrimi"] += 1
            c_cift += 1
    print(f"(c) kayıt (otomatik bent çifti): {c_cift}")

    # ---- (d) Madde dizisi — ardışık numaralı başlıklı maddeler --------------
    # Prosedür deseninin otomatik karşılığı: usul iddiası YOK, salt sıralı
    # madde alıntısı. Numara sürekliliği şart; adım alıntısı <40 kr düşürülür.
    d_adet = 0
    for no, mads in laws.items():
        eligible = [m for m in mads
                    if not m.gecici and m.baslik and not m.is_mulga()
                    and len(m.text) >= f1.MIN_MADDE_CHARS
                    and m.text.count("yerine işlenmiştir") < 2
                    and not ("yerine işlenmiştir" in m.text and len(m.text) < 400)
                    and not m.baslik.lower().startswith(("birinci", "ikinci"))
                    and m.baslik.lower() not in
                    ("genel hükümler", "kapsam", "amaç", "yürürlük", "kabul")]
        kanun_d, i = 0, 0
        while i + D_PENCERE <= len(eligible) and kanun_d < D_PER_LAW_CAP:
            grup = eligible[i: i + D_PENCERE]
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
            records.append(rec_d_dizi(f1.LAW_NAMES[no], no, grup, quotes))
            per_type["d_madde_dizisi"] += 1
            kanun_d += 1
            d_adet += 1
            i += D_PENCERE  # çakışmayan pencereler
    print(f"(d) kayıt (madde dizisi): {d_adet}")

    # ---- Şema doğrulama (FAZ-1 aynen) + dedupe (yeni-yeni, yeni-eski463) ----
    eski_satirlar = [l for l in open(ESKI_JSONL, encoding="utf-8") if l.strip()]
    eski_recs = [json.loads(l) for l in eski_satirlar]
    eski_hashler = {hashlib.sha256(
        (r["instruction"] + "||" + r["output"]).encode()).hexdigest() for r in eski_recs}
    print(f"Eski 463: {len(eski_recs)} kayıt (kuyruk bloğu, birebir korunur)")

    valid, seen = [], set()
    for i, r in enumerate(records):
        if not f1.validate_record(r, drops, i):
            drops.append((i, "şema dışı kayıt"))
            continue
        h = hashlib.sha256((r["instruction"] + "||" + r["output"]).encode()).hexdigest()
        if h in seen or h in eski_hashler:
            per_type["dup"] = per_type.get("dup", 0) + 1
            continue
        seen.add(h)
        valid.append(r)
    print(f"Yeni kayıt (şema+dedupe sonrası): {len(valid)} "
          f"(dedupe düşüş: {per_type.get('dup', 0)})")

    # ---- Token/UNK ölçümü (FAZ-1 rig'i aynen: mode=True) --------------------
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.compiler.core import CrystalCompiler
    from src.llm.tokenizer import KristalTokenizer, Vocabulary

    print("Token ölçümü (anka 33.114, literal_entity_mode=True)…")
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(KÖK, "data", "lexicon", "roots.tsv"))
    vocab = Vocabulary()
    vocab.load(os.path.join(KÖK, "data", "rebuild", "vocab_anka_r1_33114.json"))
    tok = KristalTokenizer(
        CrystalCompiler(lexicon, build_default_graph()), vocab,
        literal_entity_mode=True, verbose=False,
    )
    unk_id = vocab.stoi["<UNK>"]
    tum_recs = valid + eski_recs
    tot_tok = unk_tok = 0
    len_hist = []
    for r in tum_recs:
        ids = tok.encode(json.dumps(r, ensure_ascii=False))
        tot_tok += len(ids)
        unk_tok += sum(1 for j in ids if j == unk_id)
        len_hist.append(len(ids))
    print(f"Toplam token: {tot_tok:,}  UNK: {unk_tok:,} "
          f"(%{100 * unk_tok / max(1, tot_tok):.2f})  "
          f"ort. kayıt: {tot_tok / max(1, len(tum_recs)):.0f} token")

    meta = {
        "uretim_tarihi": time.strftime("%Y-%m-%d %H:%M"),
        "gorev": "P2.5 FAZ-2C — legal SFT v2 (463 -> ~5K; FAZ-1 kapıları aynen)",
        "kaynak_korpus": "data/legal_korpus/ (17 kanun, mevzuat.gov.tr)",
        "vocab": "vocab_anka_r1_33114.json",
        "literal_entity_mode": True,
        "kayit_sayisi": len(valid) + len(eski_recs),
        "yeni_kayit": len(valid),
        "eski_463_kuyruk": len(eski_recs),
        "olcek_degisiklikleri": {
            "a_kavram_tanimi": "28/kanun dengesi KALDIRILDI (kapılar aynen)",
            "b_madde_sorusu": "PER_LAW_CAP KALDIRILDI (FAZ-1 filtre seti aynen)",
            "b3_tum_madde": "YENİ: b1 dışı tüm madde satırları (başlıksız+Geçici/Ek)",
            "c_kavram_ayrimi": "YENİ otomatik aynı-kanun bent çiftleri (yapısal ayrım cümlesi)",
            "d_madde_dizisi": "YENİ ardışık-madde dizileri (usul iddiası yok; "
                              f"pencere {D_PENCERE}, kanun tavanı {D_PER_LAW_CAP})",
        },
        "val_sozlesmesi": "eski 463'ün son-50'si dosya sonunda BİREBİR (2A val avlusu); "
                          "yeni kayıtlar başta; eski satırlar yeniden serileştirilmez",
        "tip_dagilimi_yeni": {k: v for k, v in per_type.items() if k != "dup"},
        "tip_dagilimi_eski463": json.load(
            open(ESKI_JSONL + ".meta.json", encoding="utf-8"))["tip_dagilimi"],
        "token_olcumu": {
            "toplam_token": tot_tok,
            "unk_token": unk_tok,
            "unk_yuzde": round(100 * unk_tok / max(1, tot_tok), 3),
            "ortalama_kayit_token": round(tot_tok / max(1, len(tum_recs)), 1),
            "maks_kayit_token": max(len_hist) if len_hist else 0,
        },
        "dusenler": drops,
    }

    if args.dry_run:
        print("[dry-run] yazım yapılmadı.")
        for r in valid[:3]:
            print("ÖRNEK-YENİ:", json.dumps(r, ensure_ascii=False)[:200])
        return

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for l in eski_satirlar:          # eski 463 satırı BAYT-BİREBİR kuyruk
            f.write(l if l.endswith("\n") else l + "\n")
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    # ---- val tutarlılığı programatik kanıtı ---------------------------------
    v2_satirlar = [l for l in open(OUT_JSONL, encoding="utf-8") if l.strip()]
    son50_v2 = [l.rstrip("\n") for l in v2_satirlar[-50:]]
    son50_eski = [l.rstrip("\n") for l in eski_satirlar[-50:]]
    ozdes = son50_v2 == son50_eski
    print(f"VAL-KANIT: v2 son-50 == eski son-50 → {ozdes} | v2 toplam {len(v2_satirlar)}")
    if not ozdes:
        raise SystemExit("P25C_DUR: son-50 özdeşliği bozuldu — dosya yazılmamalıydı")

    print(f"Yazıldı: {OUT_JSONL} ({len(valid)} yeni + {len(eski_recs)} eski = "
          f"{len(valid) + len(eski_recs)} kayıt)")
    print(f"Meta: {OUT_META}")
    print("=" * 72)


if __name__ == "__main__":
    main()