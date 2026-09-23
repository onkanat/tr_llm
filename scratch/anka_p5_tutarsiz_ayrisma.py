#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 bileşen ayrışması: B-kolu tutarsızlık %39 = yüklemler-eksik mi, döngü mü?
T-0099 · 23 Eyl 2026 · sonuç: yüklemler-eksik 39/100, döngü 0/100, temiz 61/100.
Aynı orneklem (arena_base temiz, n=100, seed 42), sonda betiğinin birebir yolu."""
import importlib.util, json, os, random, sys
KOK = "/Users/hakankilicaslan/Git/tr_llm"
sys.path.insert(0, KOK)
os.chdir(KOK)
spec = importlib.util.spec_from_file_location("p5", "scratch/anka_p5_tavan_sondasi.py")
p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p5)
carve = p5.carve_denetimi()
temiz = carve["arena_base_temiz"]
print(f"temiz: {len(temiz)}")

from src.compiler.core import CrystalCompiler
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.evaluate_carpenter_anka import kelimeler
vocab = Vocabulary(); vocab.load("data/rebuild/vocab_anka_r1_33114.json", freeze=True)
lex = LexiconManager(); lex.load_from_tsv("data/lexicon/roots_anka_r1.tsv")
tok = KristalTokenizer(CrystalCompiler(lex, build_default_graph()), vocab, literal_entity_mode=False)

rng = random.Random(42)
ornek = rng.sample(temiz, min(100, len(temiz)))
PRED = ("TENSE_", "COPULA_")
yuklem_eksik = dongu = temiz_ = 0
ornekler = []
for r in ornek:
    ids = [int(t) for t in tok.encode(r["output"])]
    gen_tok = [vocab.decode(t) for t in ids]
    y = any(t.startswith(PRED) for t in gen_tok[-5:])
    lo = any(gen_tok[x:x+2] == gen_tok[x+2:x+4] == gen_tok[x+4:x+6]
             for x in range(max(0, len(gen_tok) - 5)))
    if not y: yuklem_eksik += 1
    if lo: dongu += 1
    if not y and not lo: temiz_ += 1
    if (not y or lo) and len(ornekler) < 6:
        ornekler.append({"jeton_son5": gen_tok[-5:], "yuklem": y, "dongu": lo,
                         "uz": len(gen_tok)})
print(f"yuklem eksik: {yuklem_eksik}/100 · dongu: {dongu}/100 · temiz: {temiz_}/100")
for o in ornekler:
    print(f"  yuklem={o['yuklem']} dongu={o['dongu']} uz={o['uz']:4d} son5={o['jeton_son5']}")
