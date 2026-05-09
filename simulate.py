import os
import hashlib
import random
from qdrant_client.http import models
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory

def generate_kristal_vector(token_ids, crystal_tags_str, size=768):
    """
    Kristal-Vektörel Mimarisi için yoğun (dense), konumsal (positional) 
    ve ağırlıklı (weighted) deterministik vektör üretimi.
    """
    if not token_ids:
        return [0.0] * size
        
    final_vec = [0.0] * size
    tags = crystal_tags_str.split() if crystal_tags_str else []
    
    for pos_idx, tid in enumerate(token_ids):
        weight = 1.0
        if pos_idx < len(tags):
            tag = tags[pos_idx]
            if tag in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]:
                weight = 0.1
            elif tag.isupper() or "_" in tag:
                weight = 0.5
            else:
                weight = 2.0

        hash_input = f"{tid}_pos{pos_idx}"
        seed_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        rng = random.Random(seed_val)
        
        for i in range(size):
            val = (rng.random() * 2.0) - 1.0
            final_vec[i] += val * weight
            
    norm = sum(v*v for v in final_vec) ** 0.5
    if norm > 1e-9:
        final_vec = [v/norm for v in final_vec]
        
    return final_vec

def generate_sparse_vector(token_ids):
    """
    BM25 hibrit arama için morfem frekanslarından seyrek (sparse) vektör üretir.
    """
    from collections import Counter
    counts = Counter(token_ids)
    return models.SparseVector(indices=list(counts.keys()), values=[float(v) for v in counts.values()])

def run_simulation():
    # 1. HAZIRLIK
    lexicon_path = 'data/lexicon/roots.tsv'
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    tokenizer = KristalTokenizer(compiler, vocab)
    
    # Vektör Bellek (768d)
    memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)

    # 2. SORGULAR
    queries = [
        "Durgun su ölümü hatırlatır",
        "Akmayan su kımıldanmayan yer",
        "Pıhtılaşan su ağırlaşan su",
        "Kitabı okudum ve temizledim",
        "Gidecekti ama gelmedi",
        "Demirkır, güney tepelerinin duldalarına çektiği atları gece yarısına doğru yeniden ovaya indirdi."
    ]

    # Find next report number (rapor01.txt, rapor02.txt, ...)
    report_idx = 1
    while os.path.exists(f"rapor{report_idx:02d}.txt"):
        report_idx += 1
        if report_idx > 99: break
    report_path = f"rapor{report_idx:02d}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write("="*60 + "\n")
        f_rep.write(" KRİSTAL-VEKTÖREL MİMARİSİ: VEKTÖREL GEZGİN RAPORU (HİBRİT ARAMA)\n")
        f_rep.write("="*60 + "\n\n")

        print("="*50)
        print(" KRİSTAL-VEKTÖREL MİMARİSİ: VEKTÖREL GEZGİN SİMÜLASYONU")
        print("="*50)

        for q_idx, query in enumerate(queries):
            f_rep.write(f"Sorgu #{q_idx+1}: '{query}'\n")
            f_rep.write("-" * 30 + "\n")
            
            print(f"\n[Sorgu #{q_idx+1}]: '{query}'")
            
            # A. KRİSTALİZASYON
            token_ids = tokenizer.encode(query)
            crystal_tags = tokenizer.decode(token_ids)
            
            f_rep.write(f"[1. Kristalizasyon]:\n")
            f_rep.write(f"  Morfem Etiketleri: {crystal_tags}\n")
            f_rep.write(f"  Token IDs: {token_ids}\n")

            # B. VEKTÖREL PROJEKSİYON (DENSE & SPARSE)
            query_dense_vector = generate_kristal_vector(token_ids, crystal_tags)
            query_sparse_vector = generate_sparse_vector(token_ids)
            
            f_rep.write(f"[2. Vektörel Projeksiyon]:\n")
            f_rep.write(f"  Dense Fingerprint (İlk 5): {query_dense_vector[:5]}\n")
            f_rep.write(f"  Sparse Indices: {query_sparse_vector.indices}\n")

            # C. HİBRİT GERİ ÇAĞIRMA (RRF)
            results = memory.hybrid_recall(query_dense_vector, query_sparse_vector, top_k=3)

            f_rep.write(f"[3. Hibrit Geri Çağırma (Dense + BM25)]:\n")
            if not results:
                f_rep.write("  !! Eşleşme bulunamadı.\n")
            else:
                for i, res in enumerate(results):
                    f_rep.write(f"  Eşleşme #{i+1} (Skor: {res['score']:.4f}):\n")
                    f_rep.write(f"    Metin: {res['text']}\n")
                    meta = res['metadata']
                    f_rep.write(f"    Morfemler: {meta.get('crystal_tags', 'Bilinmiyor')}\n")
                    f_rep.write(f"    Yazar: {meta.get('yazar', 'Anonim')}\n")
            
            f_rep.write("\n")
            print(f"  -> İşlendi, rapor dosyasına yazıldı.")

        f_rep.write("="*60 + "\n")
        f_rep.write(" RAPOR SONU\n")
        f_rep.write("="*60 + "\n")

    print("\n" + "="*50)
    print(f" SİMÜLASYON TAMAMLANDI. Çıktılar '{report_path}' dosyasına kaydedildi.")
    print("="*50)

if __name__ == "__main__":
    run_simulation()
