#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K4 KOPYA KAPISI ölçüm aleti (donmuş eşik; v6 hazırlığı).

Ne ölçer: gerçek RAG prompt'u (build_rag_prompt_tokens: json {instruction,input,output}
+ <BELGE> doc </BELGE> sorgu, <OUTPUT>'a kadar) verilir; model greedy 30 jeton üretir.
  - `kopya_4gram`: üretim ile BELGE arasındaki 4-gram örtüşme sayısı (>0 = kopya var)
  - `span_tam`: altın span'ın jeton dizisi üretimin İÇİNDE bitişik geçiyor mu
  - `span_yarim`: span'ın ≥%50'si (en uzun ortak bitişik alt dizi) üretimde mi
Hüküm oranları: kopya_orani = kopya_4gram>0 olan örnek / toplam; span_tam_orani; span_yarim_orani.

Kullanım:
  venv/bin/python scripts/k4_kopya_olcum.py --model data/anka_ogrenci_v5.pt \
      --tutulan data/eval/k4_kopya_tutulan.jsonl --n 100 --out scratch/k4_v5.json

Qdrant YOK, yazım YOK; yalnız yerel ağırlıklar.
"""
import argparse
import json
import statistics
import sys

import torch

sys.path.append(".")
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from scripts.t0200_mutasyon_kanit import yukle_checkpoint
from src.rag.rag_pipeline import build_rag_prompt_tokens

VOCAB = "data/rebuild/vocab_anka_r1_33114.json"
LEXICON = "data/lexicon/roots.tsv"


def dortgram(a, b):
    B = {tuple(b[i:i + 4]) for i in range(len(b) - 3)}
    return sum(1 for i in range(len(a) - 3) if tuple(a[i:i + 4]) in B)


def en_uzun_ortak(a, b):
    """a'nın b içinde bitişik geçen en uzun alt dizisinin uzunluğu."""
    en = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            en = max(en, k)
    return en


def uret(model, vocab, prompt, n=30, rep_pen=1.4, rep_win=10):
    gen = list(prompt)
    out = []
    eos = vocab.stoi.get("<EOS>", -1)
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor([gen], dtype=torch.long, device="mps")
            logits, _ = model(x)
            lg = logits[0, -1, :]
            if out:
                for tid in set(out[-rep_win:]):
                    lg[tid] = lg[tid] / rep_pen if lg[tid] > 0 else lg[tid] * rep_pen
            nxt = int(torch.argmax(lg).item())
            if nxt == eos:
                break
            out.append(nxt); gen.append(nxt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tutulan", default="data/eval/k4_kopya_tutulan.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--uretim", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab = Vocabulary(); vocab.load(VOCAB)
    lex = LexiconManager(); lex.load_from_tsv(LEXICON)
    tok = KristalTokenizer(CrystalCompiler(lex, build_default_graph()), vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      n_layer=6, n_head=6, dropout=0.0, block_size=1024)
    yukle_checkpoint(args.model, model)
    model.eval().to("mps")

    ornekler = [json.loads(l) for l in open(args.tutulan, encoding="utf-8")][:args.n]
    satirlar = []
    for i, o in enumerate(ornekler, 1):
        doc_ids = tok.encode(o["doc"])[1:-1]
        span_ids = tok.encode(o["span"])[1:-1]
        prompt = build_rag_prompt_tokens(tok, vocab, o["sorgu"], doc_text=o["doc"],
                                         instruction=o["talimat"])
        out = uret(model, vocab, prompt, n=args.uretim)
        d4 = dortgram(out, doc_ids)
        lcs = en_uzun_ortak(span_ids, out)
        satirlar.append({
            "i": i, "prompt_jeton": len(prompt), "cikti_jeton": len(out),
            "kopya_4gram": d4,
            "span_tam": lcs >= len(span_ids),
            "span_oran": round(lcs / max(1, len(span_ids)), 3),
            "cikti": " ".join(vocab.decode(t) for t in out)[:160],
        })
        if i % 20 == 0:
            print(f"  {i}/{len(ornekler)} …", flush=True)

    n = len(satirlar)
    ozet = {
        "model": args.model, "n": n,
        "kopya_orani": round(sum(1 for s in satirlar if s["kopya_4gram"] > 0) / n, 4),
        "span_tam_orani": round(sum(1 for s in satirlar if s["span_tam"]) / n, 4),
        "span_yarim_orani": round(sum(1 for s in satirlar if s["span_oran"] >= 0.5) / n, 4),
        "kopya_4gram_ort": round(statistics.mean(s["kopya_4gram"] for s in satirlar), 3),
        "prompt_jeton_ort": round(statistics.mean(s["prompt_jeton"] for s in satirlar), 1),
        "cikti_jeton_ort": round(statistics.mean(s["cikti_jeton"] for s in satirlar), 1),
    }
    json.dump({"ozet": ozet, "satirlar": satirlar}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(ozet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
