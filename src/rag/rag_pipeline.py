#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src.rag.rag_pipeline
Kanonik RAG (Arama - Koşullama - Üretim) Hattı Dikişi.

Vektörel bellekten (Qdrant) morfolojik kısıtlı hibrit geri çağırma,
eşik kapısı (RAG_MATCH_THRESHOLD = 0.40) kontrolü, kanonik prompt zarfı
inşası ve nedensel dil modeli ile autoregressive greedy decoding.
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

# Proje kök dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.llm.prompt_contract import build_rag_input, resize_state_dict
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector

# Kanonik koşullama eşiği (tek adlandırılmış sabit)
RAG_MATCH_THRESHOLD: float = 0.40


def build_query_vectors(query: str, tokenizer: Any) -> Tuple[List[int], str, Any, Any]:
    """Sorgu metnini tokenize eder ve yoğun/seyrek vektörlerini üretir."""
    query_token_ids = tokenizer.encode(query)
    query_tags = tokenizer.decode(query_token_ids)
    dense_vec = generate_kristal_vector(query_token_ids, query_tags)
    sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
    return query_token_ids, query_tags, dense_vec, sparse_vec


def retrieve_context(
    memory: Any,
    dense: Any,
    sparse: Any,
    query_tags: str,
    top_k: int = 1,
) -> Optional[Dict[str, Any]]:
    """Vektörel bellekten en uyumlu belgeleri sorgular; bulunamazsa None döner."""
    if memory is None:
        return None
    try:
        results = memory.hybrid_recall(dense, sparse, top_k=top_k, query_tags=query_tags)
    except Exception:
        return None
    if not results:
        return None
    return results[0]


def is_context_usable(match_score: float, threshold: float = RAG_MATCH_THRESHOLD) -> bool:
    """Eşik kapısı kontrolü: match_score >= threshold ise True döner."""
    return float(match_score) >= float(threshold)


def build_rag_prompt_tokens(
    tokenizer: Any,
    vocab: Any,
    query: str,
    doc_text: Optional[str] = None,
    instruction: str = "Belgeye göre cevapla.",
) -> List[int]:
    """
    Kanonik RAG prompt zarfı kurucu ve <OUTPUT> dilimleyici.
    doc_text sağlanmış ve boş değilse <BELGE> koşullaması yapılır.
    doc_text None/boş ise çıplak sorgu kullanılır.
    """
    if doc_text is not None and doc_text.strip():
        augmented_input = build_rag_input(doc_text, query)
    else:
        augmented_input = query

    prompt_dict = {
        "instruction": instruction,
        "input": augmented_input,
        "output": ""
    }
    prompt_json = json.dumps(prompt_dict, ensure_ascii=False)
    prompt_token_ids = tokenizer.encode(prompt_json)

    output_start_id = -1
    if vocab is not None and hasattr(vocab, "stoi"):
        output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    elif hasattr(tokenizer, "vocab") and hasattr(tokenizer.vocab, "stoi"):
        output_start_id = tokenizer.vocab.stoi.get("<OUTPUT>", -1)

    if output_start_id != -1 and output_start_id in prompt_token_ids:
        output_idx = prompt_token_ids.index(output_start_id)
        return prompt_token_ids[:output_idx + 1]
    return prompt_token_ids


def greedy_decode(
    model: Any,
    token_ids: List[int],
    eos_id: int,
    device: Any,
    max_new_tokens: int = 60,
    rep_window: int = 12,
    rep_penalty: float = 1.5,
    max_len: int = 128,
) -> List[int]:
    """
    Deterministik greedy argmax decoding.
    Örnekleme (sampling) içermez. Tekrarlama cezası sliding window üzerinde uygulanır.
    """
    current_tokens = list(token_ids)
    generated_tokens: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(current_tokens) >= max_len:
                break
            x_input = torch.tensor([current_tokens], dtype=torch.long, device=device)
            logits, _ = model(x_input)
            next_token_logits = logits[0, -1, :].clone()

            if generated_tokens and rep_penalty != 1.0:
                window_tokens = generated_tokens[-rep_window:]
                for token_id in set(window_tokens):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= rep_penalty
                    else:
                        next_token_logits[token_id] *= rep_penalty

            next_token_id = int(torch.argmax(next_token_logits).item())

            if next_token_id == eos_id:
                break

            generated_tokens.append(next_token_id)
            current_tokens.append(next_token_id)

            if len(current_tokens) >= max_len:
                break

    return generated_tokens


class RagPipeline:
    """Kanonik RAG boru hattı sınıfı."""

    def __init__(
        self,
        tokenizer: Any,
        vocab: Any,
        memory: Any,
        model: Any,
        device: Optional[Any] = None,
        threshold: float = RAG_MATCH_THRESHOLD,
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.memory = memory
        self.model = model
        self.device = device or (torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        self.threshold = threshold

    def answer(
        self,
        query: str,
        max_new_tokens: int = 60,
        instruction: str = "Belgeye göre cevapla.",
    ) -> Dict[str, Any]:
        """Sorguyu alır, retrieval yapar, eşik kapısını denetler, prompt kurar ve yanıt üretir."""
        query_token_ids, query_tags, dense_vec, sparse_vec = build_query_vectors(query, self.tokenizer)

        retrieved_doc = retrieve_context(self.memory, dense_vec, sparse_vec, query_tags, top_k=1)
        match_score = 0.0
        if retrieved_doc and "score" in retrieved_doc:
            match_score = float(retrieved_doc["score"])

        conditioned = False
        doc_text = None
        if retrieved_doc and is_context_usable(match_score, self.threshold):
            doc_text = retrieved_doc.get("text", "")
            conditioned = True

        prompt_tokens = build_rag_prompt_tokens(
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            query=query,
            doc_text=doc_text if conditioned else None,
            instruction=instruction,
        )

        eos_id = -1
        if self.vocab and hasattr(self.vocab, "stoi"):
            eos_id = self.vocab.stoi.get("<EOS>", -1)
        elif hasattr(self.tokenizer, "vocab") and hasattr(self.tokenizer.vocab, "stoi"):
            eos_id = self.tokenizer.vocab.stoi.get("<EOS>", -1)

        generated_tokens = greedy_decode(
            model=self.model,
            token_ids=prompt_tokens,
            eos_id=eos_id,
            device=self.device,
            max_new_tokens=max_new_tokens,
        )

        output_tags = self.tokenizer.decode(generated_tokens) if generated_tokens else ""

        return {
            "output_tags": output_tags,
            "retrieved_doc": retrieved_doc,
            "match_score": match_score,
            "conditioned": conditioned,
            "generated_tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
        }


def build_from_disk(
    lexicon_path: str = "data/lexicon/roots.tsv",
    vocab_path: Optional[str] = None,
    model_path: Optional[str] = None,
    collection_name: str = "simulasyon_bellek",
    vector_size: int = 768,
    host: str = "localhost",
    port: int = 6333,
    device: Optional[torch.device] = None,
    threshold: float = RAG_MATCH_THRESHOLD,
) -> RagPipeline:
    """Bileşenleri diskten yükler ve RagPipeline örneği döndürür.

    FAIL-CLOSED (T-0087): `vocab_path` ve `model_path` VARSAYILANI KALDIRILDI.
    Gerekce (olculdu): eski varsayilanlar (a) SILINMIS bir checkpoint adini tasiyordu,
    (b) BAYAT bir sozluge (data/vocab.json, 31.357) isaret ediyordu ve checkpoint'i
    SESSIZCE kirpiyordu. Ikisi de ACIKCA verilmelidir.
    """
    if not model_path:
        raise RuntimeError(
            "DURDURULDU: model_path verilmedi. Varsayilan KALDIRILDI (T-0087): eski "
            "varsayilan silinmis bir checkpoint zincirinin adini tasiyordu. "
            "Or. model_path='data/anka_a1r.pt'")
    if not vocab_path:
        raise RuntimeError(
            "DURDURULDU: vocab_path verilmedi. Varsayilan KALDIRILDI (T-0087): eski "
            "varsayilan BAYAT bir sozluktu (data/vocab.json, 31.357) ve checkpoint'i "
            "SESSIZCE kirpiyordu. Sozluk, checkpoint satir sayisiyla AYNI olmalidir; "
            "or. vocab_path='data/rebuild/vocab_anka_r1_33114.json'")

    # 1. Load compiler modules
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    # 2. Load vocabulary & tokenizer
    vocab = Vocabulary()
    vocab.load(vocab_path)
    tokenizer = KristalTokenizer(compiler, vocab)

    # 3. Connect to Qdrant VectorMemory
    is_fallback = False
    try:
        memory = VectorMemory(collection_name=collection_name, vector_size=vector_size, host=host, port=port)
    except Exception as e:
        print(f"Uyarı: Qdrant sunucusuna bağlanılamadı. Geçici bellek (In-Memory) modunda çalışılıyor. Detay: {e}")
        memory = VectorMemory(collection_name=collection_name, vector_size=vector_size)
        is_fallback = True

    if is_fallback:
        print("  -> Geçici belleğe örnek belgeler yükleniyor...")
        sample_texts = [
            "Okul müdürüyken okulun ek inşaatında hamallarla birlikte çalışmış.",
            "Su düzeyi.",
            "Kitap okumak insanı geliştirir."
        ]
        batch_dense = []
        batch_sparse = []
        batch_meta = []
        for text in sample_texts:
            t_ids = tokenizer.encode(text)
            t_tags = tokenizer.decode(t_ids)
            batch_dense.append(generate_kristal_vector(t_ids, t_tags))
            batch_sparse.append(generate_sparse_vector(t_ids, t_tags))
            batch_meta.append({
                "domain": "gts_sozluk",
                "crystal_tags": t_tags,
                "token_ids": t_ids
            })
        memory.add_documents_batch(sample_texts, batch_dense, batch_sparse, batch_meta)

    # 4. Lazy import of KristalLM (modül seviyesinde eğitim betiği import edilmez)
    from scripts.train_step_demo import KristalLM

    print("[2] Eğitilmiş dil modeli yükleniyor...")
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hata: Model dosyası '{model_path}' bulunamadı! Lütfen önce eğitimi tamamlayın.")

    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]

    # SOZLUK <-> CHECKPOINT TUTARLILIK KAPISI (T-0087). Gerekce OLCULDU: uyusmazlikta
    # resize_state_dict satirlari KIRPIYOR ve kosum DURMUYOR (bkz. K4 raporu).
    _emb = state_dict.get("embedding.embedding.weight")
    if _emb is None:
        raise RuntimeError(
            f"DURDURULDU: checkpoint'te 'embedding.embedding.weight' yok ({model_path}); "
            "sozluk uyumu DOGRULANAMAZ.")
    if int(_emb.shape[0]) != len(vocab.stoi):
        raise RuntimeError(
            f"DURDURULDU: sozluk/checkpoint UYUSMAZLIGI (T-0087). "
            f"sozluk={vocab_path} ({len(vocab.stoi)} giris) != checkpoint={model_path} "
            f"({int(_emb.shape[0])} satir), fark={int(_emb.shape[0]) - len(vocab.stoi)}. "
            "Bu cift yurutulurse resize_state_dict satirlari KIRPAR ve kosum sessizce "
            "devam eder. Eslesen sozlugu ACIKCA verin.")

    state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"  -> Model {device} cihazına başarıyla taşındı.")

    return RagPipeline(
        tokenizer=tokenizer,
        vocab=vocab,
        memory=memory,
        model=model,
        device=device,
        threshold=threshold,
    )


def main():
    """CLI giriş noktası (USER_GUIDE.md:280 ekran çıktısı sözleşmesini korur).

    T-0087: `--model` ve `--vocab` ACIKCA verilmelidir — `build_from_disk` varsayilanlari
    kaldirildi (silinmis checkpoint adi + bayat sozluk). Ilk bayraksiz arguman sorgudur.
    """
    print("=" * 60)
    print(" VEKTÖREL GEZGİN: UÇTAN UCA RAG (ARAMA-ÜRETİM) HATTI SİMÜLASYONU")
    print("=" * 60)

    model_path: Optional[str] = None
    vocab_path: Optional[str] = None
    query: str = "okul"
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] in ("--model", "--model-path") and i + 1 < len(argv):
            model_path = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--vocab" and i + 1 < len(argv):
            vocab_path = argv[i + 1]
            i += 2
            continue
        if not argv[i].startswith("--"):
            query = argv[i]
        i += 1

    print("\n[1] Qdrant vektörel belleğe bağlanılıyor...")
    pipeline = build_from_disk(model_path=model_path, vocab_path=vocab_path)

    print(f"\n[Girdi Sorgu]: '{query}'")

    query_token_ids, query_tags, dense_vec, sparse_vec = build_query_vectors(query, pipeline.tokenizer)
    print(f"  -> Sorgu Morfemleri: {query_tags}")

    print("\n[3] Vektörel bellekten en uyumlu döküman geri çağrılıyor (Retrieval)...")
    res = pipeline.answer(query)
    doc = res["retrieved_doc"]

    if not doc:
        print("  -> Uyarı: Hiçbir döküman bulunamadı!")
        return

    doc_text = doc["text"]
    doc_tags = doc.get("metadata", {}).get("crystal_tags", "")
    print(f"  -> Eşleşen Belge: '{doc_text}'")
    print(f"  -> Belge Morfemleri: {doc_tags}")
    print(f"  -> Eşleşme Skoru (RRF + Penalty): {res['match_score']:.4f}")

    eval_tokens = res["prompt_tokens"]
    print(f"\n[4] RAG Prompt Oluşturuldu (Uzunluk: {len(eval_tokens)}):")
    print(f"  -> Model Girdisi: {pipeline.tokenizer.decode(eval_tokens)}")

    print("\n[5] Model Yanıt Üretiyor (Generation)...")
    print(f"\n[6] Üretilen Yanıt Morfemleri:")
    print(f"  -> {res['output_tags']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
