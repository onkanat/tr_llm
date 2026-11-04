#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P2.5 FAZ-1 — HUKUK KORPUSU TOKENİZABİLİRLİK ÖLÇÜMÜ
===================================================
`data/legal_korpus/` altındaki kanun metinlerinden ölçülmüş bir örneklem
(≥100K karakter) alır, donmuş anka sözlüğü (33.114) + kristal tokenizer ile
kodlar ve şu ölçümleri üretir:

- UNK oranı (id=0) — hukuk dili sözlükte ne kadar karşılık buluyor?
- kelime başına ortalama morfem/token sayısı, karakter başına token
- UNK'ya düşen kelimeler (en sık 50) + 'sözlükte mi?' sorgusu
- belirli hukuk kelimeleri için sözlük derlenebilirlik sorgusu

Çıktı: data/legal_korpus/tokenizabilite_raporu.json (+ konsol özeti).
Sözlük DOKUNULMAZ — genişletme kalemi yalnız raporda işaretlenir.
"""

import glob
import json
import os
import re
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

VOCAB_PATH = os.path.join(REPO, "data", "rebuild", "vocab_anka_r1_33114.json")
CORPUS_DIR = os.path.join(REPO, "data", "legal_korpus")
OUT_PATH = os.path.join(CORPUS_DIR, "tokenizabilite_raporu.json")

WORD_RE = re.compile(r"[\w'’]+", re.UNICODE)

# Rapor için merak edilen hukuk kelimeleri (görev metninden + hukuk dili).
PROBE_WORDS = [
    "fıkra", "hükmün", "taraflarca", "kanunen", "mevzuat", "yargı", "mahkeme",
    "hâkim", "tazminat", "zamanaşımı", "sözleşme", "taraflarınca", "işveren",
    "işçi", "kanunu", "yürürlükten", "kaldırılan", "tüzüğün", "resmî",
    "soruşturma", "kovuşturma", "icra", "iflas", "rehin", "kabul", "red",
    "istinaf", "temyiz", "kanunoylarıyla", "vekalet", "vekâlet", "taraftara",
    "konusunda", "aykırı", "aykırılık", "tâbi", "mücbir", "katıksız",
]


def build_tokenizer():
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tok = KristalTokenizer(compiler, vocab, literal_entity_mode=True, verbose=False)
    return tok, vocab


def sample_corpus(min_chars: int = 150_000, seed: int = 42) -> tuple:
    """Kanus başına orantılı kota ile baş/orta/son dilimlerinden örneklem kurar."""
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    texts = {}
    for f in files:
        raw = open(f, encoding="utf-8").read()
        # Üstbilgi satırlarını at.
        body = "\n".join(
            l for l in raw.split("\n") if not l.startswith("#")
        )
        texts[os.path.basename(f)] = body
    total = sum(len(t) for t in texts.values())

    sample_parts = []
    import random
    rng = random.Random(seed)
    for name, text in texts.items():
        quota = max(4_000, int(min_chars * len(text) / total))
        # Üç eşit aralıklı pencere: baş, orta, son.
        n_windows = 3
        w = quota // n_windows
        if w >= 250 and len(text) > w * 2:
            for i in range(n_windows):
                start = int(i * (len(text) - w) / max(1, n_windows - 1)) if n_windows > 1 else 0
                # Pencereyi satır sınırına hizala.
                seg = text[start:start + w + 400]
                nl = seg.find("\n")
                if 0 <= nl < 400:
                    seg = seg[nl + 1:]
                sample_parts.append((name, seg[:w]))
        else:
            sample_parts.append((name, text[:quota]))
    sample = "\n".join(s for _, s in sample_parts)
    return sample, sample_parts, texts


def main():
    print("=" * 72)
    print(" P2.5 FAZ-1 — HUKUK KORPUSU TOKENİZABİLİRLİK ÖLÇÜMÜ")
    print("=" * 72)
    tok, vocab = build_tokenizer()
    unk_id = vocab.stoi["<UNK>"]
    pn_id = vocab.stoi.get("<PROPER_NOUN>")
    digit_ids = {vocab.stoi[d] for d in "0123456789" if d in vocab.stoi}

    sample, parts, texts = sample_corpus(min_chars=150_000)
    print(f"Örneklem: {len(sample):,} karakter "
          f"({len(parts)} dilim, {len(texts)} kanun; hedef ≥100K)")

    t0 = time.time()
    ids = tok.encode(sample)
    dt = time.time() - t0
    n_tok = len(ids)
    unk = sum(1 for i in ids if i == unk_id)
    pn = sum(1 for i in ids if pn_id is not None and i == pn_id)
    digits = sum(1 for i in ids if i in digit_ids)
    words = WORD_RE.findall(sample)
    n_words = len(words)
    uniq_words = Counter(words)

    print(f"Kodlama: {n_tok:,} token / {len(sample):,} karakter "
          f"({n_tok/len(sample):.3f} tok/karakter, {dt:.1f} sn, "
          f"{n_tok/max(dt,1e-9):,.0f} tok/sn)")
    print(f"UNK: {unk:,} (%{100*unk/n_tok:.2f})  "
          f"<PROPER_NOUN>: {pn:,}  rakam: {digits:,} (%{100*digits/n_tok:.1f})")
    print(f"Kelime: {n_words:,}  → ortalama {n_tok/n_words:.2f} token/kelime, "
          f"{len(sample)/n_tok:.2f} karakter/token")

    # Kelime düzeyinde UNK sayımı (UNK'ya düşen kelimeler hangileri?)
    unk_word_counts = Counter()
    unk_word_instances = 0
    checked = set()
    for w, c in uniq_words.items():
        if w in checked:
            continue
        checked.add(w)
        if not re.search(r"[0-9]", w) is None:
            continue  # rakam blokları UNK değildir, hanelere açılır
        try:
            wids = tok.encode(w)
        except Exception:  # noqa: BLE001
            continue
        core = wids[1:-1]  # BOS/EOS çıkar
        if unk_id in core:
            unk_word_counts[w] = c
            unk_word_instances += c
    print(f"UNK'ya düşen farklı kelime: {len(unk_word_counts):,} "
          f"({unk_word_instances:,} kez, tüm kelimelerin "
          f"%{100*unk_word_instances/n_words:.2f}'i)")

    top_unk = unk_word_counts.most_common(50)

    # Hukuk kelimeleri sözlük sorgusu
    probe = {}
    for w in PROBE_WORDS:
        in_stoi = w in vocab.stoi
        try:
            r = tok.compiler.compile(w)
            compiles = bool(r.get("token_vector"))
            vec = r.get("token_vector") or []
        except Exception:  # noqa: BLE001
            compiles, vec = False, []
        probe[w] = {
            "sozlukte_var": in_stoi,
            "derlenebiliyor": compiles,
            "token_vector": vec[:6],
            "unk_olur_mu": (not in_stoi and not compiles),
        }
    unk_probes = [w for w, v in probe.items() if v["unk_olur_mu"]]

    # Kanun bazlı UNK dağılımı (dilim bazında hızlı yaklaşım)
    per_law = {}
    for name, seg in parts:
        seg_ids = tok.encode(seg)
        l_unk = sum(1 for i in seg_ids if i == unk_id)
        per_law[name] = {
            "karakter": len(seg),
            "token": len(seg_ids),
            "unk": l_unk,
            "unk_yuzde": round(100 * l_unk / max(1, len(seg_ids)), 2),
        }

    report = {
        "olcum_tarihi": time.strftime("%Y-%m-%d %H:%M"),
        "vocab": os.path.basename(VOCAB_PATH),
        "vocab_boyut": len(vocab.stoi),
        "literal_entity_mode": True,
        "orneklem": {
            "karakter": len(sample),
            "dilim_sayisi": len(parts),
            "kanun_sayisi": len(texts),
            "hedef_alt_sinir": 100_000,
        },
        "olcumler": {
            "toplam_token": n_tok,
            "toplam_karakter": len(sample),
            "unk_token": unk,
            "unk_yuzde_token": round(100 * unk / n_tok, 3),
            "proper_noun_token": pn,
            "rakam_token": digits,
            "rakam_yuzde": round(100 * digits / n_tok, 2),
            "kelime_sayisi": n_words,
            "token_basina_karakter": round(len(sample) / n_tok, 3),
            "kelime_basina_token": round(n_tok / n_words, 3),
            "unk_kelime_cesidi": len(unk_word_counts),
            "unk_kelime_ornegi": unk_word_instances,
            "unk_kelime_yuzde": round(100 * unk_word_instances / n_words, 3),
        },
        "sozlukte_mi_sorgusu": probe,
        "sozlukte_olmayan_hukuk_kelimeleri": unk_probes,
        "en_sik_unk_kelimeler": [
            {"kelime": w, "sayi": c} for w, c in top_unk
        ],
        "kanun_bazli_unk": per_law,
        "sozluk_gelistirme_kalemi": (
            "AÇILDI: UNK oranı %10 üzerinde" if unk / n_tok > 0.10
            else "Gerek yok: UNK oranı %10 altında (sözlük donmuş 33.114)"
        ),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nRapor: {OUT_PATH}")
    print(f"En sık UNK kelimeler: {[w for w, _ in top_unk[:15]]}")
    print(f"Sözlükte bulunamayan hukuk kelimeleri: {unk_probes}")
    print("=" * 72)


if __name__ == "__main__":
    main()