#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: HİBRİT VARLIK TOKENIZER KELİME DAĞARCIĞI VE MODEL GENİŞLETME
=============================================================================
Bu betik:
1. `data/vocab.json` sözlüğünü yükler.
2. 36 yeni hibrit varlık belirtecini (<ENT>, </ENT>, <CAP>, <ALL_CAPS> ve 32 harfi) ekler.
3. Genişletilmiş sözlüğü `data/vocab_entity.json` olarak kaydeder.
4. `data/kristal_b1_5_best.pt` modelinin embedding ve lm_head tensörlerini cerrahiyle (expand_model_vocabulary)
   32.816'dan 32.852'ye genişletir (orijinal ağırlıkları %100 koruyarak).
5. Genişletilmiş modeli `data/kristal_b1_5_entity_ready.pt` olarak kaydeder.
"""

import os
import sys
import torch
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary, KristalTokenizer
from scripts.train_step_demo import KristalLM
from src.gateway.retrain_pipeline import expand_model_vocabulary

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASE_VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
EXP_VOCAB_PATH = os.path.join(DATA_DIR, "vocab_entity.json")
BASE_CKPT_PATH = os.path.join(DATA_DIR, "kristal_b1_5_best.pt")
EXP_CKPT_PATH = os.path.join(DATA_DIR, "kristal_b1_5_entity_ready.pt")


def main():
    print("=== [ADIM 2] HİBRİT VARLIK MODEL VE SÖZLÜK GENİŞLETME ===")
    
    # 1. Sözlük genişletme
    vocab = Vocabulary()
    vocab.load(BASE_VOCAB_PATH, freeze=False)
    old_vocab_len = len(vocab.stoi)
    print(f"Orijinal Vocab Boyutu: {old_vocab_len:,} token")

    new_tokens = list(KristalTokenizer.ENTITY_MARKERS) + list(KristalTokenizer.ENTITY_CHARS)
    
    # 2. Model yükleme
    print(f"B1.5 Checkpoint Yükleniyor: {BASE_CKPT_PATH}...")
    model = KristalLM(
        vocab_size=old_vocab_len,
        n_embd=768,
        n_head=6,
        n_layer=6,
        block_size=4096,
        vocab=vocab
    )
    sd = torch.load(BASE_CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)

    # 3. Model Ağırlık Cerrahisi (Zero-Forgetting Surgery)
    model, added_ids = expand_model_vocabulary(model, vocab, new_tokens, device="cpu")
    new_vocab_len = len(vocab.stoi)
    print(f"Genişletilmiş Vocab Boyutu: {new_vocab_len:,} token (+{new_vocab_len - old_vocab_len} yeni token)")

    # 4. Kaydetme
    vocab.save(EXP_VOCAB_PATH)
    print(f"Genişletilmiş Sözlük Kaydedildi: '{EXP_VOCAB_PATH}'")

    torch.save(model.state_dict(), EXP_CKPT_PATH)
    print(f"Genişletilmiş Checkpoint Kaydedildi: '{EXP_CKPT_PATH}'")
    print("=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
