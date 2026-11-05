#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 — RAG-ALINTI DONMUŞ KAPILAR (GÖREV 4; İLAN'dan ÖNCE koşulur)
================================================================
Üç kapı, EŞİKLERİ BU BAŞLIKTA DONMUŞ (koşumdan önce dondurulur; tarif yetersiz
kalırsa dürüstçe KALDI hükmü verilir):

K1 — KOŞULLANMALI ÜRETİMDE ALINTI (VG-1 rig'i; Qdrant YOK — donmuş gerçek
     retrieval belgeleri data/eval/vg1_uretim_sonrasi.json'dan okunur):
     donmuş İHBAR + 4857-süre sorguları, belgeyle koşullu üretim
     (build_rag_prompt_tokens + üretim-hattı generate_tokens semantiği:
     tekrar cezası 1.4/pencere 10, yapısal-jeton bastırması, argmax,
     max_new=100; talimat = üretim varsayılanı 'Belgeye göre cevapla.').
     GEÇER (sorgu başına, anka_rag_alinti modeli): paylaşılan 4-gram(çıktı,
     belge) ≥ 3 VE doğru kaynak: belge başlığındaki madde numarası çıktıda
     (rakam) VE çıktıda madde/kanun/4857 referansı. Öncesi (VG-1 donmuş
     kanıt): 0 4-gram, en-uzun-ortak-dizi 1 sözcük.

K2 — ALINTISIZ DÜRÜSTLÜK: aynı sorgular belge YOKKEN (doc_text=None →
     çıplak sorgu). GEÇER (sorgu başına): morfem ≥ 5, UNK ≤ %3, dejenerasyon
     yok (tekil-morfem oranı ≥ 0.3). 'Belge yok' tarzı dürüst ifadeler
     RAPORLANIR (kapıya girmeyen gözlem).

K3 — GENEL DAVRANIŞ NOBETİ: 3 donmuş sonda sorgusu (zımpara/fidan/kontrol),
     trllm_sonda rig'i birebir (argmax + 0.8 frekans cezası, pencere 48,
     max_new=45, T-0157 maske). GEÇER (sorgu başına): |morf_model −
     morf_taban| ≤ 0.30 × morf_taban VE model UNK% ≤ 3.

Çıktı: data/eval/p1_rag_alinti_kapilar_{TARIH}.json (birebir üretimler dâhil).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import torch

KÖK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KÖK)
os.chdir(KÖK)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import resize_state_dict
from src.rag.rag_pipeline import build_rag_prompt_tokens
from src.rag.epistemic_agent import YAPISAL_BASTIRMA_JETONLARI, TEMIZLIK_JETONLARI
from src.compiler.decompiler import MorphemeDecompiler
from scripts.train_step_demo import KristalLM

TARIH = time.strftime("%Y-%m-%d")

# ---- DONMUŞ SORGULAR / EŞİKLER (değiştirilmez) --------------------------------
VG1_KANIT = "data/eval/vg1_uretim_sonrasi.json"   # donmuş gerçek retrieval belgeleri
URETIM_TALIMAT = "Belgeye göre cevapla."          # gateway ask() varsayılanı (VG-1 rig)
EGITIM_TALIMAT = "Belgeye dayanarak yanıtla."     # SFT veri kümesinin talimatı (tanı)
MAX_NEW = 100                                      # VG-1 rig koşumu
K1_4GRAM_ESIK = 3
K2_MORF_MIN, K2_UNK_YZDE, K2_TEKIL_ORAN = 5, 3.0, 0.30
K3_MORF_BAND, K3_UNK_YZDE = 0.30, 3.0
SONDA_SORGULAR = [
    ("MARANGOZ", "Marangozluk sorusunu yanıtla.", "Zımpara nedir ve nasıl kullanılır?"),
    ("BAHCIVAN", "Bahçıvanlık sorusunu yanıtla.", "Fidan nasıl dikilir?"),
    ("KONTROL", "Soruyu kısaca yanıtla.", "Okula gitmek için hangi ekler kullanılır?"),
]
# K1 teşhis kontrolü: dağıtım-içi (~25 morf) belge — 4857 M17 İLK CÜMLESİ.
# M17'nin TAM gövdesi (~90 morf) veri setine ALINMADI (kısa-değil kuralı) →
# kontrol belgesi eğitim setinde YOK; kopya-devresi/uzun-belge-OOD ayrımı için.
K1_KONTROL_BELGE = ("İş Kanunu (Sayı: 4857) Madde 17 - Belirsiz süreli iş sözleşmelerinin "
                    "feshinden önce durumun diğer tarafa bildirilmesi gerekir.")

# vg1_alinti_taramasi.py kalıbı (birebir):


def kelimeler(s):
    return re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9']+", s.lower())


def en_uzun_ortak_dizi(a, b):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0, ""
    onceki = [0] * (lb + 1)
    en_iyi, son_idx = 0, 0
    for i in range(1, la + 1):
        simdi = [0] * (lb + 1)
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                simdi[j] = onceki[j - 1] + 1
                if simdi[j] > en_iyi:
                    en_iyi, son_idx = simdi[j], i
        onceki = simdi
    return en_iyi, (" ".join(a[son_idx - en_iyi: son_idx]) if en_iyi else "")


def ngram_say(a, b, n=4):
    ga = {" ".join(a[i:i + n]) for i in range(len(a) - n + 1)}
    gb = {" ".join(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(ga & gb)


class Rig:
    def __init__(self, device):
        self.device = device
        lexicon = LexiconManager(); lexicon.load_from_tsv("data/lexicon/roots.tsv")
        self.compiler = CrystalCompiler(lexicon, build_default_graph())
        self.vocab = Vocabulary(); self.vocab.load("data/rebuild/vocab_anka_r1_33114.json")
        self.tok = KristalTokenizer(self.compiler, self.vocab)
        self.dec = MorphemeDecompiler(self.compiler, self.vocab)
        self.eos_id = self.vocab.stoi.get("<EOS>", 3)
        self.out_end_id = self.vocab.stoi.get("</OUTPUT>", -1)
        self.bastirma = [self.vocab.stoi[t] for t in YAPISAL_BASTIRMA_JETONLARI
                         if t in self.vocab.stoi]
        self.temizlik = {self.eos_id, self.out_end_id} | {
            self.vocab.stoi[t] for t in TEMIZLIK_JETONLARI if t in self.vocab.stoi}

    def model_yukle(self, yol):
        m = KristalLM(vocab_size=len(self.vocab.stoi), n_embd=768, vocab=self.vocab,
                     block_size=4096, n_layer=6, n_head=6)
        # Güvenli bağlam: dosya bu repoda kendi eğitim koşumlarımızın çıktısıdır
        # (provenans sha'ları koşum hükümlerinde donmuş); tensor-only state dict.
        try:
            sd = torch.load(yol, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(yol, map_location="cpu")
        for k in [k for k in sd.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
            del sd[k]
        sd = resize_state_dict(m, sd)
        m.load_state_dict(sd, strict=False); m.to(self.device); m.eval()
        return m

    # epistemic_agent.generate_tokens semantiği BİREBİR (üretim hattı paritesi)
    def uretim(self, model, prompt_ids, max_new=MAX_NEW, rep=1.4, pencere=10):
        gen = [t for t in prompt_ids if 0 <= t < len(self.vocab.stoi)]
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([gen], dtype=torch.long, device=self.device)
                logits, _ = model(x)
                last = logits[0, -1, :].clone()
                cikan = gen[len(prompt_ids):]
                if rep > 1.0 and cikan:
                    for tid in set(cikan[-pencere:]):
                        if last[tid] > 0:
                            last[tid] /= rep
                        else:
                            last[tid] *= rep
                for sid in self.bastirma:
                    last[sid] = float("-inf")
                nxt = int(last.argmax().item())
                gen.append(nxt)
                if nxt in (self.eos_id, self.out_end_id):
                    break
        yanit_ids = [t for t in gen[len(prompt_ids):] if t not in self.temizlik]
        morf = " ".join(self.vocab.decode(t) for t in yanit_ids)
        metin = ""
        if morf:
            try:
                metin = self.dec.decompile_sentence(morf, capitalize=True)
            except Exception:
                metin = morf
        unk = sum(1 for t in yanit_ids if self.vocab.decode(t) == "<UNK>")
        return {"ids_say": len(yanit_ids), "morfem_say": len(morf.split()),
                "unk": unk, "unk_yzde": round(100.0 * unk / max(1, len(morf.split())), 2),
                "morfem": morf, "metin": metin}

    # trllm_sonda rig'i BİREBİR (argmax + 0.8 frekans cezası, pencere 48, max_new=45)
    def sonda(self, model, instruction, inp, max_new=45):
        case = {"instruction": instruction, "input": inp, "output": ""}
        ids = self.tok.encode(json.dumps(case, ensure_ascii=False))
        out_id = self.vocab.stoi.get("<OUTPUT>", -1)
        if out_id not in ids:
            return {"morfem_say": 0, "unk": 0, "morfem": "", "metin": "(<OUTPUT> yok)"}
        i = ids.index(out_id) + 1
        ctx = ids[:i]
        mask = [self.vocab.stoi.get(t, -1) for t in
                ("<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>")]
        mask = [t for t in mask if t >= 0]
        with torch.no_grad():
            for _ in range(max_new):
                x = torch.tensor([ctx[-4096:]], dtype=torch.long, device=self.device)
                logits, _ = model(x)
                last = logits[0, -1, :].clone()
                for tid in mask:
                    last[tid] = float("-inf")
                for tid, c in Counter(ctx[-48:]).items():
                    if tid not in mask:
                        last[tid] /= (1.0 + 0.8 * c)
                nxt = int(last.argmax().item())
                if nxt == self.eos_id:
                    break
                ctx.append(nxt)
        resp = ctx[i:]
        unk = sum(1 for t in resp if self.vocab.decode(t) == "<UNK>")
        morf = " ".join(self.vocab.decode(t) for t in resp)
        try:
            metin = self.dec.decompile_sentence(morf, capitalize=True)
        except Exception:
            metin = morf
        return {"morfem_say": len(morf.split()), "unk": unk, "morfem": morf, "metin": metin}


def kaynak_denetim(metin, beklenen_madde):
    """K1 kaynak kapısı: doğru madde numarası + kanun/madde referansı."""
    rakamlar = re.findall(r"\d+", metin)
    numara_var = beklenen_madde in rakamlar
    ref = ("madde" in metin.lower() or "4857" in metin
           or "iş kanunu" in metin.lower())
    return {"beklenen_madde": beklenen_madde, "numara_var": numara_var,
            "referans_var": ref, "tam_4857": "4857" in metin,
            "kaynak_ok": bool(numara_var and ref)}


def main():
    p = argparse.ArgumentParser(description="P1 RAG-alıntı donmuş kapıları (K1/K2/K3)")
    p.add_argument("--taban", default="data/anka_taban_pure.pt")
    p.add_argument("--model", default="data/anka_rag_alinti.pt")
    p.add_argument("--out", default=f"data/eval/p1_rag_alinti_kapilar_{TARIH}.json")
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[KAPILAR] cihaz={device} | taban={args.taban} | model={args.model}", flush=True)
    rig = Rig(device)

    vg1 = json.load(open(VG1_KANIT, encoding="utf-8"))
    sorgular = list(vg1["sorgular"].keys())
    belgeler = {}
    for s in sorgular:
        doc = vg1["sorgular"][s].get("rag_document", "") or ""
        m = re.search(r"Madde (\d+)", doc)
        assert doc and m, f"KAPILAR_DUR: donmuş belge/madde numarası yok: {s}"
        belgeler[s] = (doc, m.group(1))
        print(f"[KAPILAR] donmuş belge: {s[:40]!r} → Madde {m.group(1)} "
              f"({len(kelimeler(doc))} sözcük, VG-1 skor {vg1['sorgular'][s]['rag_score']})", flush=True)

    taban = rig.model_yukle(args.taban)
    model = rig.model_yukle(args.model)

    rapor = {
        "tarih": TARIH,
        "esikler": {
            "K1": {"paylasilan_4gram": K1_4GRAM_ESIK, "kaynak": "madde no (rakam) VE madde/4857/İş Kanunu referansı",
                   "once_kanit": "vg1_fark_tablosu.json: 0 4-gram, LCS 1 sözcük"},
            "K2": {"morf_min": K2_MORF_MIN, "unk_yzde": K2_UNK_YZDE, "tekil_oran": K2_TEKIL_ORAN},
            "K3": {"morf_band": K3_MORF_BAND, "unk_yzde": K3_UNK_YZDE,
                   "sorgular": [s[2] for s in SONDA_SORGULAR]},
        },
        "modeller": {"taban": args.taban, "model": args.model},
        "K1": {}, "K2": {}, "K3": {},
        "hukum": {},
    }

    # ---------------- K1: belgeyle koşullu üretim (önce taban, sonra model)
    print("\n=== K1: KOŞULLANMALI ÜRETİMDE ALINTI ===", flush=True)
    k1_hukum = True
    for s in sorgular:
        doc, madde_no = belgeler[s]
        satir = {"beklenen_madde": madde_no}
        for ad, m in (("taban", taban), ("alinti_modeli", model)):
            t0 = time.time()
            pr = build_rag_prompt_tokens(rig.tok, rig.vocab, s, doc_text=doc,
                                        instruction=URETIM_TALIMAT)
            u = rig.uretim(m, pr)
            yk, dk = kelimeler(u["metin"]), kelimeler(doc)
            lcs, ornek = en_uzun_ortak_dizi(yk, dk)
            satir[ad] = {"sure_s": round(time.time() - t0, 1),
                         "metin": u["metin"], "morfem": u["morfem"],
                         "morfem_say": u["morfem_say"], "unk_yzde": u["unk_yzde"],
                         "paylasilan_4gram": ngram_say(yk, dk, 4),
                         "en_uzun_ortak_sozcuk": lcs, "lcs_ornek": ornek,
                         **kaynak_denetim(u["metin"], madde_no)}
            print(f"[K1:{ad}] {s[:40]!r} → 4-gram={satir[ad]['paylasilan_4gram']} "
                  f"LCS={lcs} kaynak_ok={satir[ad]['kaynak_ok']}", flush=True)
            print(f"    yanıt: {u['metin'][:220]!r}", flush=True)
        # tanımlayıcı ek koşum: EĞİTİM talimatıyla (raporlama amaçlı; kapı üretim talimatıyla)
        pr = build_rag_prompt_tokens(rig.tok, rig.vocab, s, doc_text=doc,
                                     instruction=EGITIM_TALIMAT)
        u = rig.uretim(model, pr)
        yk, dk = kelimeler(u["metin"]), kelimeler(doc)
        satir["alinti_modeli_egitim_talimatli"] = {
            "metin": u["metin"], "paylasilan_4gram": ngram_say(yk, dk, 4),
            **kaynak_denetim(u["metin"], madde_no)}
        g = satir["alinti_modeli"]
        satir["hukum"] = {
            "kosul_4gram": g["paylasilan_4gram"] >= K1_4GRAM_ESIK,
            "kosul_kaynak": g["kaynak_ok"],
            "gecti": bool(g["paylasilan_4gram"] >= K1_4GRAM_ESIK and g["kaynak_ok"]),
        }
        k1_hukum = k1_hukum and satir["hukum"]["gecti"]
        rapor["K1"][s] = satir

    # ---- K1-teşhis: dağıtım-içi KISA belge kontrolü (kopya-devresi var mı?)
    print("\n=== K1-TEŞHİS: KISA BELGE KONTROLÜ (dağıtım-içi) ===", flush=True)
    k1_kontrol = {}
    for ad, m in (("taban", taban), ("alinti_modeli", model)):
        pr = build_rag_prompt_tokens(rig.tok, rig.vocab, sorgular[0],
                                    doc_text=K1_KONTROL_BELGE, instruction=URETIM_TALIMAT)
        u = rig.uretim(m, pr)
        yk, dk = kelimeler(u["metin"]), kelimeler(K1_KONTROL_BELGE)
        k1_kontrol[ad] = {"metin": u["metin"], "morfem": u["morfem"],
                          "paylasilan_4gram": ngram_say(yk, dk, 4),
                          **kaynak_denetim(u["metin"], "17")}
        print(f"[K1-kontrol:{ad}] 4-gram={k1_kontrol[ad]['paylasilan_4gram']} "
              f"kaynak_ok={k1_kontrol[ad]['kaynak_ok']}", flush=True)
        print(f"    yanıt: {u['metin'][:220]!r}", flush=True)
    rapor["K1_kisa_belge_kontrolu"] = k1_kontrol

    # ---------------- K2: belge YOKKEN dürüstlük
    print("\n=== K2: ALINTISIZ AKIŞ (BELGE YOK) ===", flush=True)
    k2_hukum = True
    for s in sorgular:
        satir = {}
        for ad, m in (("taban", taban), ("alinti_modeli", model)):
            t0 = time.time()
            pr = build_rag_prompt_tokens(rig.tok, rig.vocab, s, doc_text=None,
                                        instruction=URETIM_TALIMAT)
            u = rig.uretim(m, pr)
            morfler = u["morfem"].split()
            tekil_oran = (len(set(morfler)) / len(morfler)) if morfler else 0.0
            durust_ifade = any(k in u["metin"].lower() for k in
                               ("belge", "bilgi", "bilmiyorum", "açık değil", "yeterli"))
            satir[ad] = {"sure_s": round(time.time() - t0, 1), "metin": u["metin"],
                         "morfem": u["morfem"], "morfem_say": u["morfem_say"],
                         "unk_yzde": u["unk_yzde"], "tekil_oran": round(tekil_oran, 3),
                         "durust_ifade": durust_ifade}
        g = satir["alinti_modeli"]
        satir["hukum"] = {
            "kosul_morf": g["morfem_say"] >= K2_MORF_MIN,
            "kosul_unk": g["unk_yzde"] <= K2_UNK_YZDE,
            "kosul_tekil": g["tekil_oran"] >= K2_TEKIL_ORAN,
            "gecti": bool(g["morfem_say"] >= K2_MORF_MIN and g["unk_yzde"] <= K2_UNK_YZDE
                          and g["tekil_oran"] >= K2_TEKIL_ORAN),
        }
        k2_hukum = k2_hukum and satir["hukum"]["gecti"]
        rapor["K2"][s] = satir
        print(f"[K2:alinti_modeli] {s[:40]!r} → morf={g['morfem_say']} unk%={g['unk_yzde']} "
              f"tekil={g['tekil_oran']} dürüst={g['durust_ifade']} → {satir['hukum']['gecti']}", flush=True)
        print(f"    yanıt: {g['metin'][:220]!r}", flush=True)

    # ---------------- K3: genel davranış nöbeti (sonda rig)
    print("\n=== K3: GENEL DAVRANIŞ NOBETİ (sonda rig) ===", flush=True)
    k3_hukum = True
    for ad_, instr, inp in SONDA_SORGULAR:
        satir = {}
        for ad, m in (("taban", taban), ("alinti_modeli", model)):
            u = rig.sonda(m, instr, inp)
            satir[ad] = u
            print(f"[K3:{ad}] {ad_} → morf={u['morfem_say']} unk={u['unk']} "
                  f"unk%={round(100.0*u['unk']/max(1,u['morfem_say']),2)}", flush=True)
            print(f"    yüzey: {u['metin'][:200]!r}", flush=True)
        mt, mm = satir["taban"]["morfem_say"], satir["alinti_modeli"]["morfem_say"]
        band = abs(mm - mt) <= K3_MORF_BAND * mt
        unk_ok = round(100.0 * satir["alinti_modeli"]["unk"] / max(1, mm), 2) <= K3_UNK_YZDE
        satir["hukum"] = {"taban_morf": mt, "model_morf": mm,
                          "sapma_yzde": round(100.0 * abs(mm - mt) / max(1, mt), 1),
                          "kosul_band": bool(band), "kosul_unk": bool(unk_ok),
                          "gecti": bool(band and unk_ok)}
        k3_hukum = k3_hukum and satir["hukum"]["gecti"]
        rapor["K3"][inp] = satir

    rapor["hukum"] = {
        "K1_alinti": "GECTI" if k1_hukum else "KALDI",
        "K2_durustluk": "GECTI" if k2_hukum else "KALDI",
        "K3_nobet": "GECTI" if k3_hukum else "KALDI",
        "genel": "GECTI" if (k1_hukum and k2_hukum and k3_hukum) else "KALDI",
    }
    del taban, model
    if device.type == "mps":
        torch.mps.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print(f"\n[KAPILAR] hüküm: {rapor['hukum']} → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
