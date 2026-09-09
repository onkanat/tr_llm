import os
import sys
from qdrant_client.http import models
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector

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
    if memory.is_in_memory or memory.client.count(memory.collection_name).count == 0:
        seed_docs = [
            "Bana göre ölümün en büyük vasfı durgunluk, hareketsizliktir.",
            "Akmayan su kirlenir, yerinde sayan insan geriler.",
            "Pıhtılaşan kan ve ağırlaşan hava nefes almayı güçleştirdi.",
            "Kitap okumak zihni temizler ve yeni ufuklar açar.",
            "O gün gelecekti fakat işleri uzadığı için gelemedi.",
            "Demirkır atları ovaya indirdi ve rüzgar dindi."
        ]
        for s_doc in seed_docs:
            d_ids = tokenizer.encode(s_doc)
            d_tags = tokenizer.decode(d_ids)
            d_dense = generate_kristal_vector(d_ids, d_tags)
            d_sparse = generate_sparse_vector(d_ids, d_tags)
            memory.add_document(
                text=s_doc,
                dense_vector=d_dense,
                sparse_vector=d_sparse,
                metadata={"crystal_tags": d_tags}
            )

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

            # B. YENİDEN YAPILANDIRMA (DECOMPILATION)
            # Reconstruct the tokenized morphemes back to a surface string word-by-word
            from tests.test_decompiler import MorphemeDecompiler
            decompiler = MorphemeDecompiler(compiler, vocab)
            
            decompiled_sentence = decompiler.decompile_sentence(crystal_tags)
            
            f_rep.write(f"[1.5 Decompilation (Yeniden Yapılandırma)]:\n")
            f_rep.write(f"  Yapılandırılan Cümle: {decompiled_sentence}\n")

            # C. VEKTÖREL PROJEKSİYON (DENSE & SPARSE)
            query_dense_vector = generate_kristal_vector(token_ids, crystal_tags)
            query_sparse_vector = generate_sparse_vector(token_ids, crystal_tags)
            
            f_rep.write(f"[2. Vektörel Projeksiyon]:\n")
            f_rep.write(f"  Dense Fingerprint (İlk 5): {query_dense_vector[:5]}\n")
            f_rep.write(f"  Sparse Indices: {query_sparse_vector.indices}\n")

            # D. HİBRİT GERİ ÇAĞIRMA (RRF)
            results = memory.hybrid_recall(query_dense_vector, query_sparse_vector, top_k=3, query_tags=crystal_tags)


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
