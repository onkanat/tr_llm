#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-0094 / Faz 2 — ANKA'YA BAĞLI MARANGOZ ÜRETİM VE UNUTMA KABI
=============================================================

NEDEN YENİ BETİK: `scripts/evaluate_carpenter_generation_100.py` Anka'ya BAĞLANAMIYOR
(ölçüldü): sözlüğü `data/vocab.json` (31.357), lexicon `roots.tsv`, varsayılan model
yolu SİLİNMİŞ (`kristal_b1_5_best.pt`/`kristal_model.pt`), zarfı ELLE kurar (:128-132),
`--output` verilmezse DONMUŞ dizine yazar, çıktı şeması model kimliği TAŞIMAZ ve
`</OUTPUT>` id'sini sessiz varsayılanla (`get(..., 9)`) alır. O betik DEĞİŞTİRİLMEZ.

Bu kap dört ekseni birlikte ölçer (T-0059: marangoz metriği TEK BAŞINA hüküm veremez):
  K2  ceket ekseni  : ezber · tutarsızlık · ROUGE-L · içerik kesişimi
  K3  unutma ekseni : A (Wikipedia CE) + B (noktalama top-1), tabanla EŞLEŞTİRİLMİŞ
  K4  oracle katmanı: kimlik · distraktör · sabit-tahmin tabanı (ölçüt AYIRT EDİYOR mu)
  K5  kimlik         : model/sözlük/lexicon digest'leri + literal_entity_mode çıktıda

Fail-closed: `--model` ve `--output` ZORUNLU; sözlük boyutu != checkpoint kafası ⇒ rc=2;
özel jetonlar sözlükte YOKSA ⇒ rc=2 (sessiz varsayılan YASAK).

Kanonik yardımcılar (kopya YASAK): `render_prompt` + `resize_state_dict`
(`src/llm/prompt_contract.py`), `wilson_ci` + `compute_completion_logp`
(`scripts/evaluate_mcq_conditioning.py`), `rouge_l_score`
(`scripts/evaluate_b1_5_rigorous.py`), `KristalLM` (`scripts/train_step_demo.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.compiler.core import CrystalCompiler
from src.compiler.decompiler import MorphemeDecompiler
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.llm.prompt_contract import render_prompt, resize_state_dict
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.evaluate_b1_5_rigorous import rouge_l_score
from scripts.evaluate_mcq_conditioning import wilson_ci
from scripts.train_step_demo import KristalLM

# --- ilan edilen varsayilanlar (hepsi Anka'ya bagli) ---
VOCAB_VARSAYILAN = "data/rebuild/vocab_anka_r1_33114.json"
LEXICON_VARSAYILAN = "data/lexicon/roots_anka_r1.tsv"
HELDOUT_VARSAYILAN = "data/eval/anka_r17_heldout_2026-09-20.jsonl"
KAYNAK_VARSAYILAN = "data/pedagogy/carpenter_specialization_dataset.jsonl"
WIKI_BIN = "data/anka_a1r_pretrain.bin"

# K2 esikleri (T-0059 / Faz 0 §4'ten; ilan edildi, olcumden sonra DEGISMEZ)
# P2/Aşama 0c kalibrasyonu: ESIK_KESISIM ölçülmüş İNSAN TAVANINA (kesisim_sondasi +
# scratch/anka_p2_tavan_sondasi.json) kalibre edildi: eşik = insan_tavani × k.
# k = 0,84 (ROUGE'un mevcut 0,35/0,4164 oranıyla aynı formül). 80,0 ULAŞILAMAZDI (Ç4).
# LCS-F1 ölçütü de ölçüldü ve GERİ DÖNDÜRÜLDÜ (P2/0b: insan tavanı 0,0446 [JETON yolu]
# + 23/100 kayıtta YUZEY↔JETON kararı ayrışıyor ⇒ gösterim-uyuşmazlığı sınıfı;
# LCS-F1 yalnız TANISAL raporlanır, eşik ona bağlanmaz).
ESIK_EZBER = 10.0            # < %10
ESIK_TUTARSIZ = 5.0          # < %5  (insan tavanı %3,0 × 1,67 — beyanlı, değişmedi)
ESIK_ROUGE = 0.35            # >= 0.35 (insan tavanı 0,4164 × 0,84 — değişmedi)
ESIK_KESISIM = 10.92         # >= %10,92 (insan tavanı %13,0 × 0,84 — P2/0c)

# K3 esikleri: A artis <= +%10 · B dusus <= 5,0 puan (ILAN §4)
ESIK_A_ARTIS = 10.0
ESIK_B_DUSUS = 5.0
NK_BASI, NK_SONU = 32137, 32145      # olculdu: . , ? ! - : ; ( )
NK_CEKIRDEK = (32137, 32138, 32142)  # T-0059'un kumesi (onek invaryanti ile aynen)

BLOCK_SIZE = 4096            # checkpoint mask tamponlariyla ayni olmali
# --- Olcum gucu: esikler olcum gurultusunun ALTINDA kalmali (olculdu, 2026-09-20) ---
# B ekseni: T-0059'un n=250'si bu tabanda (p≈%41) Wilson YARI-GENISLIGI ±6,05 puandi
# ⇒ esik 5,0 puani COZEMIYOR (T-0059'un kendi n=250'si p=%84,4'te ±4,94 ile SINIRDA
# gecmis). n=2.500'de ±1,93 ⇒ esigin ~2,6× altinda. Kap bunu KENDISI dogrular.
# A ekseni: 64 pencere ±0,2205 (esik payi 0,3462 ⇒ yalniz 1,57×) → 256 pencere ±0,1103
# ⇒ 3,1× marj.
N_PENCERE = 256              # A/B ekseni icin deterministik pencere sayisi
B_PENCERE = 128
B_MIN_KONUM = 2500           # T-0059: 250 — bu tabanda esigi cozemiyor (yukari bak)
B_TARAMA_TAVANI = 12000      # tarama butcesi (deterministik sira)
SEED_B = 7                   # T-0059 kanonik seed'i
PREDICATE_TAGS = ("TENSE_", "COPULA_")


def damga() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_yol(p: str) -> Optional[str]:
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for parca in iter(lambda: f.read(1 << 22), b""):
            h.update(parca)
    return h.hexdigest()


def durdur(mesaj: str) -> None:
    sys.stderr.write(f"\nDURDURULDU: {mesaj}\n")
    raise SystemExit(2)


def kelimeler(s: str) -> List[str]:
    return re.findall(r"[\w']+", (s or "").lower())


def lcs_uzunluk(a: List[str], b: List[str]) -> int:
    """İki dizinin en-uzun-ortak-alt-dizisi uzunluğu (iki satırlı DP; deterministik)."""
    if not a or not b:
        return 0
    onceki = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        simdiki = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                simdiki[j] = onceki[j - 1] + 1
            else:
                simdiki[j] = max(onceki[j], simdiki[j - 1])
        onceki = simdiki
    return onceki[len(b)]


def lcs_f1(a: List[str], b: List[str]) -> float:
    """LCS-F1 — sıra duyarlı, kısmi kredi verir. F1 = 2·L / (|a| + |b|).

    P2/0b KARARI: hüküm ölçütü DEĞİLDİR. Ölçüldü (scratch/anka_p2_tavan_sondasi.json):
    insan uzman cevabı yalnız 0,0446 (JETON yolu) / 0,0984 (YUZEY) alır ve 23/100
    kayıtta YUZEY↔JETON kararı AYRIŞIR ⇒ gösterim-uyuşmazlığı sınıfı. Yalnız TANISAL
    alan olarak raporlanır; eşik ESIK_KESISIM (küme ölçütü, kalibre) bağlıdır.
    """
    l = lcs_uzunluk(a, b)
    toplam = len(a) + len(b)
    if toplam == 0:
        return 0.0
    return 2.0 * l / toplam


def trie_kok_sayisi(lex: LexiconManager) -> Tuple[int, int]:
    """(is_word dugum sayisi, toplam girdi) — `load_from_tsv` GERCEKTEN doldu mu?

    T-0078: `load_from_tsv` atlanirsa her girdi SESSIZCE bos doner. Var olmayan bir
    alani (`lex.roots`) okumak da ayni siniftir: AttributeError yerine once
    `self.root` uzerinden SAYARAK dogrulariz (fail-closed).
    """
    dugum = girdi = 0
    yigin = [lex.root]
    while yigin:
        n = yigin.pop()
        if n.is_word:
            dugum += 1
            girdi += len(n.entries)
        yigin.extend(n.children.values())
    return dugum, girdi


def birlesik_cevap(kayitlar: List[dict]) -> List[str]:
    """Sabit-tahmin tabani icin EN SIK egitim cevabini sec (deterministik)."""
    from collections import Counter
    c = Counter(" ".join(r.get("output", "").split()) for r in kayitlar)
    return kelimeler(c.most_common(1)[0][0]) if c else []


def egitim_satirlari(kaynak_yol: str, heldout_yol: str) -> List[dict]:
    """Egitim kumesi = kaynak EKSI held-out (tam satir cikarma; carve ile ayni)."""
    with open(heldout_yol, encoding="utf-8") as f:
        held = {l.strip() for l in f if l.strip()}
    out = []
    with open(kaynak_yol, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and l not in held:
                out.append(json.loads(l))
    return out


def dortgram_kumesi(kayitlar: List[dict]) -> set:
    s = set()
    for r in kayitlar:
        w = kelimeler(r.get("output", ""))
        for i in range(len(w) - 3):
            s.add(tuple(w[i:i + 4]))
    return s


# ---------------------------------------------------------------- K3: unutma
def a_ekseni(model: KristalLM, vocab: Vocabulary, device) -> Dict[str, Any]:
    """A ekseni: Wikipedia diliminde MASKESIZ CE (ilan §4.1). Deterministik pencereler."""
    if not os.path.exists(WIKI_BIN):
        durdur(f"A ekseni dilimi YOK: {WIKI_BIN}")
    mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    n_win = mm.size // B_PENCERE
    if n_win < N_PENCERE:
        durdur(f"A ekseni: yalniz {n_win} pencere var, {N_PENCERE} gerekli")
    rng = random.Random(SEED_B)
    baslar = sorted(rng.sample(range(n_win), N_PENCERE))
    kayiplar: List[float] = []
    with torch.no_grad():
        for b in baslar:
            seq = np.asarray(mm[b * B_PENCERE:(b + 1) * B_PENCERE], dtype=np.int64)
            x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(seq[1:], dtype=torch.long, device=device).unsqueeze(0)
            sign_mask = model.embedding.compute_sign_mask(x).to(device)
            _, loss = model(x, targets=y, sign_mask=sign_mask)
            kayiplar.append(float(loss.item()))
    return {"CE_ort": float(np.mean(kayiplar)), "CE_std": float(np.std(kayiplar)),
            "pencere": N_PENCERE, "blok": B_PENCERE, "seed": SEED_B,
            "dilim": WIKI_BIN, "maskesiz": True}


def b_ekseni(model: KristalLM, vocab: Vocabulary, device) -> Dict[str, Any]:
    """B ekseni: hedefi NOKTALAMA olan konumlarda top-1 (ilan §4; seed 7)."""
    mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    n_win = mm.size // B_PENCERE
    rng = random.Random(SEED_B)
    # yeterli konum toplanana dek pencere tara (deterministik sira)
    baslar = rng.sample(range(n_win), min(n_win, B_TARAMA_TAVANI))
    dogru = toplam = 0
    nk_kumesi = set(range(NK_BASI, NK_SONU + 1))
    cekirdek_dogru = cekirdek_toplam = 0
    taranan = 0
    with torch.no_grad():
        for b in baslar:
            if toplam >= B_MIN_KONUM:
                break
            taranan += 1
            seq = np.asarray(mm[b * B_PENCERE:(b + 1) * B_PENCERE], dtype=np.int64)
            x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            sign_mask = model.embedding.compute_sign_mask(x).to(device)
            logits, _ = model(x, sign_mask=sign_mask)
            tahmin = torch.argmax(logits[0], dim=-1).cpu().numpy()
            for i, hedef in enumerate(seq[1:]):
                if int(hedef) in nk_kumesi:
                    toplam += 1
                    if int(tahmin[i]) == int(hedef):
                        dogru += 1
                    if int(hedef) in NK_CEKIRDEK:
                        cekirdek_toplam += 1
                        if int(tahmin[i]) == int(hedef):
                            cekirdek_dogru += 1
    if toplam < B_MIN_KONUM:
        durdur(f"B ekseni: yalniz {toplam} noktalama konumu (< {B_MIN_KONUM}) — "
               f"tarama tavani {B_TARAMA_TAVANI} pencerede doldu")
    lo, hi = wilson_ci(dogru, toplam)
    yari = (hi - lo) / 2.0
    # FAIL-CLOSED: cozemeyecegi bir esigi sinayan kap, kap DEGILDIR.
    if yari >= ESIK_B_DUSUS:
        durdur(f"B ekseni COZUNURLUK YETERSIZ: Wilson yari-genisligi ±{yari:.2f} puan "
               f">= esik {ESIK_B_DUSUS} puan ⇒ kap esigi AYIRT EDEMEZ (n={toplam})")
    return {"top1": round(100.0 * dogru / toplam, 4), "konum": toplam,
            "dogru": dogru, "wilson": [lo, hi], "wilson_yari": round(yari, 4),
            "tarama_pencere": taranan, "seed": SEED_B,
            "blok": [NK_BASI, NK_SONU],
            "cekirdek_top1": round(100.0 * cekirdek_dogru / max(1, cekirdek_toplam), 4)
            if cekirdek_toplam else None,
            "cekirdek_konum": cekirdek_toplam, "kesme": NK_CEKIRDEK}


# ---------------------------------------------------------------- uretim
def uret(model: KristalLM, tokenizer: KristalTokenizer, vocab: Vocabulary,
         inst: str, inp: str, device, max_new: int,
         eos_id: int, out_end_id: int, out_id: int) -> Tuple[List[int], List[str]]:
    """Greedy uretim — kanonik zarf `render_prompt` ile.

    FAIL-CLOSED (T-0094/K7): `tokenizer.encode` sona **`<EOS>` EKLER**. Egitimde
    `<EOS>` her zaman KAYIT SINIRIDIR ve `<OUTPUT>`'tan SONRA hic gelmez; egitim
    dizisi `<OUTPUT> cevap </OUTPUT> <EOS>` seklindedir. Dolayisiyla `encode`
    ciktisini oldugu gibi vermek modele "<OUTPUT> <EOS>" gosterir — dagilim disi
    bir son konum — ve uretim coker (olculdu: `<PROPER_NOUN>` tekrari).

    Cozum: sonda `<EOS>` varsa KIRP ve son jetonun `<OUTPUT>` oldugunu dogrula.
    Ikisi de tutmazsa burada DURURUZ (sessiz kalip bozuk uretim yapmayiz).
    """
    prompt_str = render_prompt(inst, inp)
    prompt_ids = [int(t) for t in tokenizer.encode(prompt_str)]
    if prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]
    else:
        raise RuntimeError(
            f"encode ciktisi <EOS> ile bitmiyor (son={prompt_ids[-1] if prompt_ids else None}) "
            "— zarf sozlesmesi degismis olabilir; T-0094/K7"
        )
    if not prompt_ids or prompt_ids[-1] != out_id:
        raise RuntimeError(
            f"istem <OUTPUT> ile bitmiyor (son={prompt_ids[-1] if prompt_ids else None}) "
            "— render_prompt sozlesmesi ihlali; T-0094/K7"
        )
    curr = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen: List[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            sign_mask = model.embedding.compute_sign_mask(curr).to(device)
            logits, _ = model(curr, sign_mask=sign_mask)
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            if nxt in (eos_id, out_end_id):
                break
            gen.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return prompt_ids, [vocab.decode(t) for t in gen]


# ---------------------------------------------------------------- model yukle
def model_yukle(model_path: str, vocab: Vocabulary, device) -> Tuple[KristalLM, Dict[str, Any]]:
    if not os.path.exists(model_path):
        durdur(f"model dosyasi YOK: {model_path}")
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    n_vocab = len(vocab.stoi)
    # FAIL-CLOSED: kafa boyutu sozlukle AYNI olmali. Kirpma/doldurma SESSIZ bozulma olur
    # (T-0046: uyumsuzluk gorunur basilir, ama gorunur olmasi DOGRU oldugu anlamina gelmez).
    kafa = sd.get("lm_head.weight")
    if kafa is None:
        durdur("checkpoint'te 'lm_head.weight' YOK — kimlik dogrulanamaz")
    if int(kafa.shape[0]) != n_vocab:
        durdur(f"KAFA/SOZLUK UYUSMAZLIGI: checkpoint kafasi {int(kafa.shape[0]):,} != "
               f"sozluk {n_vocab:,} — boyle bir kosum checkpoint'i KIRPAR (T-0044 sinifi)")
    model = KristalLM(vocab_size=n_vocab, n_embd=768, vocab=vocab,
                      block_size=BLOCK_SIZE, n_layer=6, n_head=6)
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to(device)
    model.eval()
    return model, {"kafa": int(kafa.shape[0]), "sozluk": n_vocab}


def main() -> int:
    ap = argparse.ArgumentParser(description="Anka'ya bagli marangoz uretim kapı (T-0094/Faz 2)")
    ap.add_argument("--model", required=True, help="ZORUNLU: degerlendirilecek checkpoint")
    ap.add_argument("--output", required=True, help="ZORUNLU: sonuc JSON yolu")
    ap.add_argument("--baseline", default=None,
                    help="K3 esli karsilastirma icin TABAN checkpoint (or. anka_a1r.pt)")
    ap.add_argument("--vocab", default=VOCAB_VARSAYILAN)
    ap.add_argument("--lexicon", default=LEXICON_VARSAYILAN)
    ap.add_argument("--heldout", default=HELDOUT_VARSAYILAN)
    ap.add_argument("--train-source", default=KAYNAK_VARSAYILAN)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"],
                    help="Beklenen cihaz; MPS yoksa DUR (sessiz CPU dususu yasak)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--ceket-ekseni", action="store_true",
                    help="ceket (uretim) eksenini olc — taban/giydirilmemis kosumda KAPALI tut")
    args = ap.parse_args()

    print("=" * 78)
    print(" T-0094 / Faz 2 — ANKA'YA BAĞLI MARANGOZ ÜRETİM KABI")
    print("=" * 78, flush=True)

    # ---- G1: sozluk + lexicon (fail-closed; T-0078: load_from_tsv atlanirsa sessiz bos) ----
    vocab = Vocabulary()
    vocab.load(args.vocab, freeze=True)
    n_vocab = len(vocab.stoi)
    lex = LexiconManager()
    lex.load_from_tsv(args.lexicon)
    n_kok, n_girdi = trie_kok_sayisi(lex)
    if n_kok == 0 or n_girdi == 0:
        durdur(f"lexicon BOS dondu: {args.lexicon} ({n_kok} dugum, {n_girdi} girdi) — "
               f"load_from_tsv atlandi mi? (T-0078)")
    compiler = CrystalCompiler(lex, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=False)
    decompiler = MorphemeDecompiler(compiler, vocab)
    if len(vocab.stoi) != n_vocab:
        durdur("tokenizer sozlugu BUYUTTU")
    print(f"[G1] sozluk {n_vocab:,} ({args.vocab}) · lexicon {n_kok:,} dugum / "
          f"{n_girdi:,} girdi ({args.lexicon})", flush=True)

    # ---- G2: ozel jetonlar — sessiz varsayilan YASAK ----
    gerekli = ["<OUTPUT>", "</OUTPUT>", "<EOS>", "<BOS>", "<INSTRUCTION>", "<INPUT>"]
    eksik = [t for t in gerekli if t not in vocab.stoi]
    if eksik:
        durdur(f"ozel jetonlar sozlukte YOK: {eksik} (sessiz varsayilana DUSULMEZ)")
    out_id = vocab.stoi["<OUTPUT>"]
    out_end_id = vocab.stoi["</OUTPUT>"]
    eos_id = vocab.stoi["<EOS>"]
    if out_id == out_end_id:
        durdur("<OUTPUT> ve </OUTPUT> AYNI id — sozluk bozuk")
    print(f"[G2] <OUTPUT>={out_id} </OUTPUT>={out_end_id} <EOS>={eos_id} "
          f"<BOS>={vocab.stoi['<BOS>']}", flush=True)

    # ---- G4: ZARF SOZLESMESI — iki dali da olculur (T-0075/T-0093) ----
    # T-0094/K7: `encode` sona <EOS> ekler; egitimde <EOS> KAYIT SINIRIDIR ve
    # <OUTPUT>'tan SONRA hic gelmez. Sozlesme sessizce bozulursa uretim coker ve
    # olcum "ceket ogrenilmedi" diye YANLIS hukum verir. Bu yuzden once dogrula.
    _ornek = render_prompt("Ahşap uzmanı olarak cevapla.", "Marangozlukta eğe ne işe yarar?")
    _ham = [int(t) for t in tokenizer.encode(_ornek)]
    _kirp = _ham[:-1] if _ham and _ham[-1] == eos_id else _ham
    if not _ham or _ham[-1] != eos_id:
        durdur("G4: encode ciktisi <EOS> ile bitmiyor — zarf sozlesmesi DEGISMIS")
    if not _kirp or _kirp[-1] != out_id:
        durdur(f"G4: <EOS> kirpildiktan sonra son jeton <OUTPUT> degil ({_kirp[-1]})")
    print(f"[G4] zarf sozlesmesi ✓ encode→<EOS> KIRPILIR (ham son={_ham[-1]}, "
          f"kirpilmis son={_kirp[-1]}=<OUTPUT>) · istem {len(_kirp)} jeton", flush=True)

    # ---- G5: CIHAZ — sessiz CPU dususu YASAK (T-0094/B5) ----
    # Eski satir `"mps" if available else "cpu"` idi: sandbox'ta MPS GORUNMEZ
    # ([[sandbox-hides-mps-device]]) ⇒ kosum sessizce CPU'ya duser, 10x yavaslar ve
    # sayilar MPS kosumlariyla KIYASLANAMAZ hale gelir. Hicbir uyari verilmezdi.
    # Fail-closed: istenen cihaz yoksa DUR. Cihaz sonuca YAZILIR.
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            durdur("G5: --device mps istendi ama MPS GORUNMUYOR "
                   f"(built={torch.backends.mps.is_built()}). Sandbox icinde misiniz? "
                   "Olcumu sandbox DISINDA kosun; CPU kiyaslamasi icin --device cpu ACIKCA verin.")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
        print("[G5] UYARI: --device cpu ACIKCA istendi. Bu sayilar MPS kosumlariyla "
              "KIYASLANAMAZ (uretim RNG'si cihaza bagli).", flush=True)
    else:
        durdur(f"G5: bilinmeyen --device {args.device!r} (mps|cpu)")
    print(f"[cihaz] {device} (acikca istenen: {args.device})", flush=True)

    # ---- G3: model + kafa/sozluk uyumu ----
    model, mkim = model_yukle(args.model, vocab, device)
    msd = sha256_yol(args.model)
    print(f"[G3] model {args.model} · kafa {mkim['kafa']:,} = sozluk {mkim['sozluk']:,} · "
          f"sha256 {str(msd)[:16]}…", flush=True)

    # ---- K3: unutma eksenleri (ceket ekseninden ONCE, tabanla esli) ----
    print("\n[K3] unutma eksenleri (Wikipedia dilimi, Anka'nin kendi dagilimi)…", flush=True)
    A_m = a_ekseni(model, vocab, device)
    B_m = b_ekseni(model, vocab, device)
    print(f"  A CE {A_m['CE_ort']:.4f} ± {A_m['CE_std']:.4f}  ·  "
          f"B top-1 %{B_m['top1']:.2f} ({B_m['konum']} konum)", flush=True)
    A_b = B_b = None
    if args.baseline:
        print(f"[K3] taban: {args.baseline} …", flush=True)
        bmodel, bkim = model_yukle(args.baseline, vocab, device)
        if bkim["kafa"] != mkim["kafa"]:
            durdur("taban checkpoint kafasi FARKLI — esli karsilastirma GECERSIZ")
        A_b = a_ekseni(bmodel, vocab, device)
        B_b = b_ekseni(bmodel, vocab, device)
        print(f"  taban A CE {A_b['CE_ort']:.4f} ± {A_b['CE_std']:.4f}  ·  "
              f"taban B top-1 %{B_b['top1']:.2f} ({B_b['konum']} konum)", flush=True)

    # ---- K4: oracle katmanlari (olcut AYIRT EDIYOR mu) ----
    with open(args.heldout, encoding="utf-8") as f:
        held = [json.loads(l) for l in f if l.strip()]
    if not held:
        durdur(f"held-out BOS: {args.heldout}")
    rng = random.Random(args.seed)
    ornek = rng.sample(held, min(args.n, len(held)))

    ref_kel = [kelimeler(r["output"]) for r in ornek]
    kimlik = float(np.mean([rouge_l_score(w, w) for w in ref_kel]))
    distraktor = float(np.mean([rouge_l_score(ref_kel[i], ref_kel[(i + 1) % len(ref_kel)])
                                for i in range(len(ref_kel))]))
    egitim = egitim_satirlari(args.train_source, args.heldout)
    sabit = birlesik_cevap(egitim)
    sabit_rouge = float(np.mean([rouge_l_score(sabit, w) for w in ref_kel])) if sabit else None
    print(f"\n[K4] oracle: kimlik {kimlik:.4f} (tavan 1,0 olmali) · "
          f"distraktor {distraktor:.4f} · sabit-tahmin {sabit_rouge:.4f}", flush=True)
    if abs(kimlik - 1.0) > 1e-9:
        durdur(f"KIMLIK ORACLE'i 1,0 vermedi ({kimlik}) — olcut BOZUK")
    if distraktor >= ESIK_ROUGE:
        durdur(f"DISTRAKTOR {distraktor:.4f} >= esik {ESIK_ROUGE} — olcut AYIRT ETMIYOR")

    # ---- K2: ceket ekseni ----
    metrik: Dict[str, Any] = {"olculdu": False}
    if args.ceket_ekseni:
        train_4g = dortgram_kumesi(egitim)
        print(f"\n[K2] ceket ekseni: {len(ornek)} ornek · egitim 4-gram {len(train_4g):,} · "
              f"uretim {len(ornek)}×{args.max_new}…", flush=True)
        ezber = tutarsiz = kesisim = 0
        rouges: List[float] = []
        kes_f1ler: List[float] = []
        ornekler: List[dict] = []
        for i, r in enumerate(ornek):
            _, gen_tok = uret(model, tokenizer, vocab, r.get("instruction", ""),
                              r.get("input", ""), device, args.max_new, eos_id, out_end_id,
                              out_id)
            gm = " ".join(gen_tok)
            try:
                yuzey = decompiler.decompile_sentence(gm)
            except Exception:
                yuzey = gm
            gw = kelimeler(gm)
            cg = [tuple(gw[k:k + 4]) for k in range(len(gw) - 3)]
            mem = bool(cg) and (sum(1 for g in cg if g in train_4g) / len(cg)) >= 0.90
            if mem:
                ezber += 1
            son = gen_tok[-5:]
            yuklem = any(t.startswith(PREDICATE_TAGS) for t in son)
            dongu = any(gen_tok[k:k + 2] == gen_tok[k + 2:k + 4] == gen_tok[k + 4:k + 6]
                        for k in range(max(0, len(gen_tok) - 5)))
            inc = (not yuklem) or dongu
            if inc:
                tutarsiz += 1
            rl = rouge_l_score(gw, ref_kel[i])
            rouges.append(rl)
            giris_kel = kelimeler(r.get("input", ""))
            kes_f1ler.append(lcs_f1(giris_kel, gw))
            # LEGACY süreklilik raporu: ham küme kesişimi (≥2 ortak kelime)
            if len(set(giris_kel) & set(gw)) >= 2:
                kesisim += 1
            if i < 5:
                ornekler.append({"idx": i + 1, "input": r.get("input", "")[:150],
                                 "uretim_yuzey": yuzey[:200], "referans": r["output"][:200],
                                 "rouge_l": rl, "kesisim_f1": round(kes_f1ler[-1], 4),
                                 "ezber": mem, "tutarsiz": inc})
            if (i + 1) % 25 == 0:
                print(f"  … {i + 1}/{len(ornek)}", flush=True)
        n = len(ornek)
        ezber_r = 100.0 * ezber / n
        tut_r = 100.0 * tutarsiz / n
        kes_r = 100.0 * kesisim / n
        rouge_ort = float(np.mean(rouges))
        el, eh = wilson_ci(ezber, n)
        tl, th = wilson_ci(tutarsiz, n)
        kl, kh = wilson_ci(kesisim, n)
        metrik = {
            "olculdu": True, "n": n,
            "kesisim_f1_ort": round(float(np.mean(kes_f1ler)), 4),
            "kesisim_f1_medyan": round(float(np.median(kes_f1ler)), 4),
            "kesisim_f1_std": round(float(np.std(kes_f1ler)), 4),
            "ezber_orani": round(ezber_r, 4), "ezber_wilson": [el, eh],
            "tutarsizlik_orani": round(tut_r, 4), "tutarsizlik_wilson": [tl, th],
            "rouge_l_ort": round(rouge_ort, 4),
            "rouge_l_medyan": round(float(np.median(rouges)), 4),
            "rouge_l_std": round(float(np.std(rouges)), 4),
            "kesisim_orani": round(kes_r, 4), "kesisim_wilson": [kl, kh],
            "ornekler": ornekler,
        }
        print(f"  ezber %{ezber_r:.2f} · tutarsizlik %{tut_r:.2f} · "
              f"ROUGE-L {rouge_ort:.4f} · kesisim %{kes_r:.2f} "
              f"(tanı: LCS-F1 {metrik['kesisim_f1_ort']:.4f})", flush=True)
    else:
        print("\n[K2] ceket ekseni KAPALI (--ceket-ekseni verilmedi) — "
              "giydirilmemis kosumda beklenen", flush=True)

    # ---- esik hukumleri ----
    hukum: Dict[str, Any] = {}
    if metrik["olculdu"]:
        hukum = {
            "ezber_gec": metrik["ezber_orani"] < ESIK_EZBER,
            "tutarsizlik_gec": metrik["tutarsizlik_orani"] < ESIK_TUTARSIZ,
            "rouge_gec": metrik["rouge_l_ort"] >= ESIK_ROUGE,
            "kesisim_gec": metrik["kesisim_orani"] >= ESIK_KESISIM,
        }
        hukum["ceket_ekseni_gec"] = all(hukum.values())
    if A_b is not None:
        a_artis = 100.0 * (A_m["CE_ort"] / A_b["CE_ort"] - 1.0)
        b_dusus = B_b["top1"] - B_m["top1"]
        # FAIL-CLOSED: A ekseninin %95 yari-genisligi, esik payinin ALTINDA olmali.
        a_yari = 1.96 * A_m["CE_std"] / np.sqrt(A_m["pencere"])
        a_pay = (ESIK_A_ARTIS / 100.0) * A_b["CE_ort"]
        if a_yari >= a_pay:
            durdur(f"A ekseni COZUNURLUK YETERSIZ: %95 yari-genislik ±{a_yari:.4f} >= "
                   f"esik payi {a_pay:.4f} ⇒ kap esigi AYIRT EDEMEZ "
                   f"(pencere={A_m['pencere']})")
        hukum.update({
            "A_artis_yuzde": round(a_artis, 4), "A_gec": a_artis <= ESIK_A_ARTIS,
            "A_yari_genislik": round(float(a_yari), 4), "A_esik_payi": round(float(a_pay), 4),
            "B_dusus_puan": round(b_dusus, 4), "B_gec": b_dusus <= ESIK_B_DUSUS,
            "B_yari_genislik": B_m["wilson_yari"], "B_esik": ESIK_B_DUSUS,
        })
        hukum["unutma_gec"] = hukum["A_gec"] and hukum["B_gec"]
        print(f"\n[hüküm] A artis %{a_artis:+.2f} (esik ≤ +%{ESIK_A_ARTIS}) · "
              f"A yari-genislik ±{a_yari:.4f} < pay {a_pay:.4f} ✓", flush=True)
        print(f"[hüküm] B dusus {b_dusus:+.2f} puan (esik ≤ {ESIK_B_DUSUS}) · "
              f"B yari-genislik ±{B_m['wilson_yari']:.2f} < esik ✓", flush=True)

    # ---- cikti semasi: MODEL KIMLIGI TASIR (eski kap tasimiyordu) ----
    sonuc: Dict[str, Any] = {
        "gorev": "T-0094", "asama": "Faz 2 — Anka'ya bagli uretim kapı",
        "damga_utc": damga(),
        "model_path": args.model, "model_sha256": msd,
        "baseline_path": args.baseline, "baseline_sha256": sha256_yol(args.baseline) if args.baseline else None,
        "vocab_path": args.vocab, "vocab_sha256": sha256_yol(args.vocab),
        "vocab_size": n_vocab, "kafa_boyutu": mkim["kafa"],
        "lexicon_path": args.lexicon, "lexicon_sha256": sha256_yol(args.lexicon),
        "literal_entity_mode": False,
        "heldout_path": args.heldout, "heldout_sha256": sha256_yol(args.heldout),
        "train_source": args.train_source, "train_source_sha256": sha256_yol(args.train_source),
        "train_satir": len(egitim), "heldout_satir": len(held),
        "seed": args.seed, "n": len(ornek), "max_new": args.max_new,
        "device": str(device), "device_istenen": args.device,
        "device_notu": "Sayilar YALNIZ ayni cihazdaki kosumlarla kiyaslanabilir "
                       "(uretim RNG'si cihaza bagli; CPU ile MPS esit degil).",
        "ceket_ekseni": metrik,
        "A_ekseni": A_m, "B_ekseni": B_m,
        "A_ekseni_taban": A_b, "B_ekseni_taban": B_b,
        "oracle": {"kimlik_rouge": round(kimlik, 6), "distraktor_rouge": round(distraktor, 6),
                   "sabit_tahmin_rouge": round(sabit_rouge, 6) if sabit_rouge is not None else None,
                   "sabit_tahmin_kelime": len(sabit),
                   "tavan": 1.0, "esik_rouge": ESIK_ROUGE},
        "esikler": {"ezber": ESIK_EZBER, "tutarsizlik": ESIK_TUTARSIZ,
                    "rouge": ESIK_ROUGE, "kesisim": ESIK_KESISIM,
                    "A_artis": ESIK_A_ARTIS, "B_dusus": ESIK_B_DUSUS},
        "hukum": hukum,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"\n[out] {args.output} sha256={str(sha256_yol(args.output))[:16]}…")
    print(f"[hukum] {hukum}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
