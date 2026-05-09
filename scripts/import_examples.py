import json
import os
import sys
import hashlib
import random

# Proje kök dizinini path'e ekleyelim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vector_memory import VectorMemory
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

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
        # --- 2. Morfem Ağırlıklandırma ---
        weight = 1.0
        if pos_idx < len(tags):
            tag = tags[pos_idx]
            if tag in ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]:
                weight = 0.1 # Düşük ağırlık (Sistem tokenları)
            elif tag.isupper() or "_" in tag:
                weight = 0.5 # Orta ağırlık (Gramer ekleri, örn: TENSE_PAST)
            else:
                weight = 2.0 # Yüksek ağırlık (Kökler, örn: durgun, su)

        # --- 1. Konumsal Kodlama (Positional Encoding) ---
        # Seed değerine pos_idx ekleyerek kelimenin sırasını vektöre kodluyoruz.
        hash_input = f"{tid}_pos{pos_idx}"
        seed_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        rng = random.Random(seed_val)
        
        # Bu morfem için 768 boyutlu yoğun bir vektör oluştur ve ağırlıkla çarp
        for i in range(size):
            # -1.0 ile 1.0 arası deterministik rastgele değer
            val = (rng.random() * 2.0) - 1.0
            final_vec[i] += val * weight
            
    # Son cümleyi L2 ile normalize et
    norm = sum(v*v for v in final_vec) ** 0.5
    if norm > 1e-9:
        final_vec = [v/norm for v in final_vec]
        
    return final_vec

from qdrant_client.http import models

def generate_sparse_vector(token_ids):
    """
    BM25 hibrit arama için morfem frekanslarından seyrek (sparse) vektör üretir.
    """
    from collections import Counter
    counts = Counter(token_ids)
    return models.SparseVector(indices=list(counts.keys()), values=[float(v) for v in counts.values()])

def import_gts_examples(max_examples=100, reset=True):
    jsonl_path = 'data/poems/gts.json'
    lexicon_path = 'data/lexicon/roots.tsv'
    
    if not os.path.exists(jsonl_path):
        print(f"Hata: {jsonl_path} bulunamadı!")
        return
    
    if not os.path.exists(lexicon_path):
        print(f"Hata: {lexicon_path} bulunamadı!")
        return

    # 1. KRİSTAL DERLEYİCİ HAZIRLIĞI
    print(f"[1] Kristal Derleyici ve Tokenizer hazırlanıyor (Hedef: {max_examples} örnek)...")
    lexicon = LexiconManager()
    lexicon.load_from_tsv(lexicon_path)
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. RAPOR HAZIRLIĞI
    report_idx = 1
    while os.path.exists(f"import_rapor{report_idx:02d}.txt"):
        report_idx += 1
    report_path = f"import_rapor{report_idx:02d}.txt"
    f_rep = open(report_path, "w", encoding="utf-8")
    f_rep.write(f"KRİSTAL-VEKTÖREL İMPORT RAPORU - {report_path} (Kapasite: {max_examples})\n")
    f_rep.write("="*50 + "\n\n")

    # 3. VEKTÖR BELLEK HAZIRLIĞI
    print("[2] Vektör Bellek (Qdrant) hazırlanıyor...")
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
        if reset:
            memory.recreate_collection(768)
    except Exception as e:
        print(f"Uyarı: Qdrant sunucusuna bağlanılamadı, in-memory modunda devam ediliyor. Hata: {e}")
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768)
    
    processed_examples = 0
    batch_size = 50 
    batch_texts = []
    batch_dense_vectors = []
    batch_sparse_vectors = []
    batch_metadatas = []
    
    print(f"[3] GTS sözlüğünden ilk {max_examples} örnek işleniyor, OOV'ler {report_path} dosyasına kaydediliyor...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if processed_examples >= max_examples: break
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except:
                continue
                
            madde = obj.get("madde", "").strip()
            anlamlar = obj.get("anlamlarListe", [])
            
            if not anlamlar: continue
            
            for anlam_obj in anlamlar:
                if processed_examples >= max_examples: break
                ornekler = anlam_obj.get("orneklerListe", [])
                if not ornekler: continue
                
                for ornek_obj in ornekler:
                    if processed_examples >= max_examples: break
                    ornek_metin = ornek_obj.get("ornek", "").strip()
                    if not ornek_metin: continue
                    
                    # Yazar bilgisi
                    yazar_adi = "Anonim/Halk"
                    yazarlar = ornek_obj.get("yazar", [])
                    if yazarlar:
                        yazar_adi = yazarlar[0].get("tam_adi", yazar_adi)
                    
                    # --- KRİSTAL-VEKTÖREL ANALİZ ---
                    # OOV yakalamak için kelime bazlı kontrol (Pedagojik Bypass Uyumlu)
                    words = ornek_metin.split()
                    for w in words:
                        clean_word = w.strip(".,!?\"…—«»/()-;:")
                        if not clean_word: continue
                        
                        # Bypass kontrolleri
                        if clean_word.replace(".", "").replace(",", "").isdigit() or clean_word.isnumeric():
                            continue
                            
                        compile_word = clean_word.replace("'", "")
                        if not compiler.compile(compile_word)['analyses']:
                            if clean_word[0].isupper():
                                continue # PROPER_NOUN bypass
                            else:
                                f_rep.write(f"Warning OOV: '{clean_word}' (Cümle: {ornek_metin[:50]}...)\n")

                    token_ids = tokenizer.encode(ornek_metin)
                    crystal_tags = tokenizer.decode(token_ids)
                    dense_vector = generate_kristal_vector(token_ids, crystal_tags)
                    sparse_vector = generate_sparse_vector(token_ids)
                    
                    # Batch listelerine ekle
                    batch_texts.append(ornek_metin)
                    batch_dense_vectors.append(dense_vector)
                    batch_sparse_vectors.append(sparse_vector)
                    batch_metadatas.append({
                        "domain": "gts_sozluk",
                        "kok": madde,
                        "yazar": yazar_adi,
                        "crystal_tags": crystal_tags,
                        "token_ids": token_ids,
                        "method": "kristal_hybrid_v1"
                    })
                    
                    processed_examples += 1
                    
                    if len(batch_texts) >= batch_size:
                        memory.add_documents_batch(batch_texts, batch_dense_vectors, batch_sparse_vectors, batch_metadatas)
                        batch_texts, batch_dense_vectors, batch_sparse_vectors, batch_metadatas = [], [], [], []
                        print(f"  -> {processed_examples} adet örnek işlendi...")

    # (Rest of the loop logic remains the same)
    if batch_texts:
        memory.add_documents_batch(batch_texts, batch_dense_vectors, batch_sparse_vectors, batch_metadatas)

    vocab.save('data/vocab.json')

    f_rep.write(f"\nİşlem Tamamlandı! Toplam {processed_examples} adet örnek işlendi.\n")
    f_rep.close()
    print(f"\nİşlem Tamamlandı! Detaylı OOV raporu '{report_path}' dosyasına kaydedildi.")

if __name__ == '__main__':
    # Get max_examples from command line if provided
    count = 100
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    
    # We always reset for this specific architectural evaluation phase
    import_gts_examples(max_examples=count, reset=True)
