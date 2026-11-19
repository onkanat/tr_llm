#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1 — DONMUŞ SORGU RETRIEVAL ÖLÇÜMÜ (ekim öncesi / sonrası)
=============================================================
İki donmuş hukuk sorgusunu, verilen koleksiyonlarda hibrit (dense+sparse RRF)
geri çağırma ile ölçer; skor/kök-eşleşme/metin kanıtını JSON'a yazar.

- Sorgu vektörleri ÜRETİM hattıyla AYNI yoldan kurulur: KristalTokenizer
  (anka vocab 33.114) + generate_kristal_vector(768) + generate_sparse_vector.
- Koleksiyon YOK ise ölçülür, YARATILMAZ (yazma yetkisi yok — prob salt-okunur).
- Kanal (zincir) davranışı ayrıca gerçek EpistemicCuriosityAgent.search_memory
  ile ölçülür (yama sonrası aux koleksiyonlar dahil); mock model/router yalnız
  kurulum içindir, search_memory onlara DOKUNMAZ.

Kullanım:
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/vg1_retrieval_olcum.py \
        --label oncesi --collections anka_bellek,simulasyon_bellek \
        --out data/eval/vg1_retrieval_oncesi.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from src.rag.vector_memory import VectorMemory, resolve_qdrant_url
from src.rag.embedding import generate_kristal_vector, generate_sparse_vector
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

# DONMUŞ SORGULAR (VG-1.2 — değiştirilmez)
DONMUS_SORGULAR = [
    "İş sözleşmesi feshinde ihbar süresi nedir?",
    "4857'de haftalık normal çalışma süresi hangi maddede düzenlenmiştir?",
]


def tokenizer_kur(vocab_path: str, lexicon_path: str) -> KristalTokenizer:
    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    return KristalTokenizer(compiler, vocab)


def koleksiyon_olc(bellek: VectorMemory, sorgu_satirlari: list, top_k: int) -> dict:
    """Tek koleksiyonda donmuş sorguları hibrit geri çağırma ile ölçer."""
    out = {}
    for sorgu, tok_ids, tags in sorgu_satirlari:
        dense = generate_kristal_vector(tok_ids, tags)
        sparse = generate_sparse_vector(tok_ids, tags)
        t0 = time.time()
        sonuclar = bellek.hybrid_recall(dense, sparse, top_k=top_k, query_tags=tags)
        sure_ms = (time.time() - t0) * 1000.0
        out[sorgu] = {
            "sure_ms": round(sure_ms, 1),
            "sonuclar": [
                {
                    "skor": round(float(r.get("score", 0.0)), 4),
                    "kok_eslesme": bool(r.get("has_root_match", False)),
                    "eslesen_kokler": r.get("matching_roots", []),
                    "metin": r.get("text", "")[:400],
                }
                for r in sonuclar
            ],
        }
    return out


def zincir_olc(memories: list, sorgu_satirlari: list) -> dict:
    """Gerçek EpistemicCuriosityAgent.search_memory ile ZİNCİR davranışı.

    memories sırası = zincir sırası: [birincil, yedek, aux1, aux2, ...]
    (Yama öncesi aux yoksa yalnız [birincil, yedek] verilir — bugünkü kod.)
    """
    import torch
    from src.rag.epistemic_agent import EpistemicCuriosityAgent
    from src.rag.merak import CuriosityEngine
    from src.llm.router import TriModalRouter

    class _KuklaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 64)

        def forward(self, x, return_hidden_states=False):
            logits = torch.zeros(x.shape[0], x.shape[1], 8)
            return (logits, None) if not return_hidden_states else (logits, None, torch.zeros(x.shape[0], x.shape[1], 64))

    ana, yedek = memories[0], (memories[1] if len(memories) > 1 else None)
    aux = memories[2:] if len(memories) > 2 else []
    ajan = EpistemicCuriosityAgent(
        model=_KuklaModel(),
        tokenizer=None,  # search_memory yalnız bellekleri kullanır; tokenizer gerekmez
        memory=ana,
        general_memory=yedek,
        curiosity_engine=CuriosityEngine(hidden_dim=64, curiosity_dim=64, tau=2.5),
        router=TriModalRouter(prompt_dim=64, merak_dim=64, rag_dim=64, router_dim=32, num_experts=4, top_k=2),
        tau=2.5,
        device="cpu",
    )
    ajan.auxiliary_memories = aux  # yama-öncesi koşumda boş/atanamazsa zincir bugünkü gibi
    out = {}
    for sorgu, tok_ids, tags in sorgu_satirlari:
        doc, kaynak = ajan.search_memory(tok_ids, tags)
        out[sorgu] = {
            "zincir_kaynak": kaynak,
            "skor": round(float(doc.get("score", 0.0)), 4) if doc else None,
            "kok_eslesme": bool(doc.get("has_root_match", False)) if doc else None,
            "metin": (doc.get("text", "")[:400] if doc else None),
        }
    return out


def main() -> int:
    ayarla = argparse.ArgumentParser(description="VG-1 donmuş sorgu retrieval ölçümü (salt-okuma)")
    ayarla.add_argument("--label", type=str, required=True, help="Evre etiketi: oncesi / sonrasi")
    ayarla.add_argument("--collections", type=str, default="anka_bellek,simulasyon_bellek",
                        help="Ölçülecek koleksiyonlar (virgülle; sıra = zincir sırası)")
    ayarla.add_argument("--out", type=str, required=True, help="Çıktı JSON yolu")
    ayarla.add_argument("--url", type=str, default=None)
    ayarla.add_argument("--vocab", type=str, default="data/rebuild/vocab_anka_r1_33114.json")
    ayarla.add_argument("--lexicon", type=str, default="data/lexicon/roots.tsv")
    ayarla.add_argument("--top-k", type=int, default=3)
    ayarla.add_argument("--timeout", type=float, default=30.0)
    args = ayarla.parse_args()

    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    print(f"[OLCUM] hedef Qdrant: {hedef[0]}:{hedef[1]} | evre: {args.label}")

    istemci = QdrantClient(host=hedef[0], port=hedef[1], timeout=args.timeout)
    koleksiyonlar = [k.strip() for k in args.collections.split(",") if k.strip()]
    mevcut = {k for k in koleksiyonlar if istemci.collection_exists(k)}
    yoklar = [k for k in koleksiyonlar if k not in mevcut]
    for k in koleksiyonlar:
        n = istemci.count(collection_name=k).count if k in mevcut else None
        print(f"[OLCUM] {k}: {'VAR' if k in mevcut else 'YOK'} (nokta: {n})")

    tokenizer = tokenizer_kur(args.vocab, args.lexicon)
    sorgu_satirlari = []
    for s in DONMUS_SORGULAR:
        ids = tokenizer.encode(s)
        tags = tokenizer.decode(ids)
        sorgu_satirlari.append((s, ids, tags))
        print(f"[OLCUM] sorgu etiketleri: {s!r} -> {tags[:120]}")

    rapor = {
        "evre": args.label,
        "qdrant": f"{hedef[0]}:{hedef[1]}",
        "zaman_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "esik": 0.33,
        "koleksiyonlar": {k: (istemci.count(collection_name=k).count if k in mevcut else None)
                          for k in koleksiyonlar},
        "koleksiyon_yok": yoklar,
        "per_koleksiyon": {},
    }

    bellekler = []
    for k in koleksiyonlar:
        if k not in mevcut:
            continue
        bellek = VectorMemory(collection_name=k, vector_size=768,
                              host=hedef[0], port=hedef[1], timeout=args.timeout)
        bellekler.append(bellek)
        rapor["per_koleksiyon"][k] = koleksiyon_olc(bellek, sorgu_satirlari, args.top_k)

    # Zincir: gerçek search_memory (aux yalnız yama-sonrası koşumda dolu gelir)
    try:
        rapor["zincir"] = zincir_olc(bellekler, sorgu_satirlari)
    except Exception as e:  # noqa: BLE001 — zincir ölçümü opsiyonel, hata raporlanır
        rapor["zincir_hata"] = f"{type(e).__name__}: {e}"

    for b in bellekler:
        b.close()
    istemci.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print(f"[OLCUM] yazıldı: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
