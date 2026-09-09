import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.prepare import prepare_dataset

def main():
    print("=" * 60)
    print(" BEŞ FAZLI BİRLEŞİK MÜFREDATIN DERLENMESİ (KÜLLİYAT + BEBEKLİK + EBEVEYNLİK + DPO + WIKIPEDIA)")
    print("=" * 60)

    corpus_path = 'data/raw_corpus.txt'
    infancy_path = 'data/pedagogy/infancy_dataset.jsonl'
    parenting_path = 'data/pedagogy/parenting_dataset.jsonl'
    dpo_chosen_bin = 'data/train_dpo_chosen.bin'
    wiki_bin = 'data/train_wiki.bin'
    output_bin = 'data/train.bin'

    temp_corpus_bin = 'data/train_corpus.bin'
    temp_infancy_bin = 'data/train_infancy.bin'
    temp_parenting_bin = 'data/train_parenting.bin'

    # 1. Compile Raw Corpus
    print("\n[1] Ham metin külliyatı derleniyor...")
    prepare_dataset(corpus_path, temp_corpus_bin)

    # 2. Compile Infancy Dataset
    print("\n[2] Pedagojik Bebeklik (Infancy) veri seti derleniyor...")
    prepare_dataset(infancy_path, temp_infancy_bin)

    # 3. Compile Parenting Dataset
    print("\n[3] Pedagojik Ebeveynlik (Parenting SFT) veri seti derleniyor...")
    prepare_dataset(parenting_path, temp_parenting_bin)

    # 4. Read DPO Chosen binary tokens
    print("\n[4] DPO Seçilen (Chosen) yanıt morfem akışı yükleniyor...")
    if not os.path.exists(dpo_chosen_bin):
        print(f"Hata: {dpo_chosen_bin} bulunamadı! Lütfen önce prepare_dpo_dataset.py çalıştırın.")
        return
    tokens_dpo = np.fromfile(dpo_chosen_bin, dtype=np.uint16)

    # 5. Read Wikipedia binary tokens
    print("\n[5] Wikipedia morfem akışı yükleniyor...")
    if not os.path.exists(wiki_bin):
        print(f"Hata: {wiki_bin} bulunamadı! Lütfen önce prepare_wiki_dataset.py çalıştırın.")
        return
    tokens_wiki = np.fromfile(wiki_bin, dtype=np.uint16)

    # 6. Concatenate all token streams
    print("\n[6] Tüm token akışları birleştiriliyor...")
    tokens_corpus = np.fromfile(temp_corpus_bin, dtype=np.uint16)
    tokens_infancy = np.fromfile(temp_infancy_bin, dtype=np.uint16)
    tokens_parenting = np.fromfile(temp_parenting_bin, dtype=np.uint16)
    
    combined_tokens = np.concatenate([tokens_corpus, tokens_infancy, tokens_parenting, tokens_dpo, tokens_wiki])
    combined_tokens.tofile(output_bin)
    print(f"  -> Birleştirilmiş binary token akışı kaydedildi: {output_bin}")
    print(f"     Toplam Morfem Sayısı: {len(combined_tokens)}")
    print(f"     - Külliyat: {len(tokens_corpus)}")
    print(f"     - Bebeklik: {len(tokens_infancy)}")
    print(f"     - Ebeveynlik: {len(tokens_parenting)}")
    print(f"     - DPO Chosen SFT: {len(tokens_dpo)}")
    print(f"     - Wikipedia: {len(tokens_wiki)}")

    # 7. Merge Metadata boundaries
    print("\n[7] Segmentasyon metadataları birleştiriliyor...")
    with open(temp_corpus_bin + '.meta.json', 'r', encoding='utf-8') as f:
        meta_corpus = json.load(f)
    with open(temp_infancy_bin + '.meta.json', 'r', encoding='utf-8') as f:
        meta_infancy = json.load(f)
    with open(temp_parenting_bin + '.meta.json', 'r', encoding='utf-8') as f:
        meta_parenting = json.load(f)

    # Reconstruct DPO boundaries using BOS token (2)
    dpo_boundaries = []
    record_start = 0
    for idx, token in enumerate(tokens_dpo):
        if token == 2 and idx > record_start:
            length = idx - record_start
            dpo_boundaries.append({"offset": record_start, "length": length})
            record_start = idx
    if len(tokens_dpo) > record_start:
        dpo_boundaries.append({"offset": record_start, "length": len(tokens_dpo) - record_start})

    # Reconstruct Wiki boundaries using BOS token (2)
    wiki_boundaries = []
    record_start = 0
    for idx, token in enumerate(tokens_wiki):
        if token == 2 and idx > record_start:
            length = idx - record_start
            wiki_boundaries.append({"offset": record_start, "length": length})
            record_start = idx
    if len(tokens_wiki) > record_start:
        wiki_boundaries.append({"offset": record_start, "length": len(tokens_wiki) - record_start})

    corpus_len = len(tokens_corpus)
    infancy_len = len(tokens_infancy)
    parenting_len = len(tokens_parenting)
    dpo_len = len(tokens_dpo)
    adjusted_boundaries = []
    
    # Corpus boundaries
    adjusted_boundaries.extend(meta_corpus["boundaries"])
    
    # Infancy boundaries
    for b in meta_infancy["boundaries"]:
        adjusted_boundaries.append({
            "offset": b["offset"] + corpus_len,
            "length": b["length"]
        })
        
    # Parenting boundaries
    for b in meta_parenting["boundaries"]:
        adjusted_boundaries.append({
            "offset": b["offset"] + corpus_len + infancy_len,
            "length": b["length"]
        })

    # DPO Chosen SFT boundaries
    for b in dpo_boundaries:
        adjusted_boundaries.append({
            "offset": b["offset"] + corpus_len + infancy_len + parenting_len,
            "length": b["length"]
        })

    # Wikipedia boundaries
    for b in wiki_boundaries:
        adjusted_boundaries.append({
            "offset": b["offset"] + corpus_len + infancy_len + parenting_len + dpo_len,
            "length": b["length"]
        })

    merged_meta = {
        "record_count": meta_corpus["record_count"] + meta_infancy["record_count"] + meta_parenting["record_count"] + len(dpo_boundaries) + len(wiki_boundaries),
        "boundaries": adjusted_boundaries
    }

    meta_path = output_bin + '.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)
    print(f"  -> Birleştirilmiş segmentasyon metadatası kaydedildi: {meta_path}")

    # Clean up temporary files
    for temp_file in [
        temp_corpus_bin, temp_corpus_bin + '.meta.json',
        temp_infancy_bin, temp_infancy_bin + '.meta.json',
        temp_parenting_bin, temp_parenting_bin + '.meta.json'
    ]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("\n" + "=" * 60)
    print(" BEŞ FAZLI DERLEME VE BİRLEŞTİRME İŞLEMİ BAŞARIYLA TAMAMLANDI")
    print("=" * 60)

if __name__ == '__main__':
    main()
