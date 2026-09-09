import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def main():
    print("=" * 60)
    print(" DPO TERCİH VERİ SETİ OKUMA VE KRİSTAL TOKENİZASYONU")
    print("=" * 60)

    # 1. Initialize Compiler & Tokenizer
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    print(f"Sözlük yüklendi: {len(vocab.stoi)} token.")

    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Paths
    dpo_input_path = '/Users/hakankilicaslan/Prompts/Outputs/datasets/dpo_preference_turkce.jsonl'
    output_bin_path = 'data/train_dpo_chosen.bin'
    output_jsonl_path = 'data/pedagogy/dpo_tokenized.jsonl'

    if not os.path.exists(dpo_input_path):
        print(f"Hata: DPO girdi dosyası '{dpo_input_path}' bulunamadı!")
        return

    # 3. Process records
    print(f"\nDosya okunuyor: {dpo_input_path}...")
    
    tokenized_records = []
    chosen_sft_tokens = []
    
    bos_id = vocab.encode("<BOS>")
    eos_id = vocab.encode("<EOS>")
    inst_start_id = vocab.encode("<INSTRUCTION>")
    inst_end_id = vocab.encode("</INSTRUCTION>")
    output_start_id = vocab.encode("<OUTPUT>")
    output_end_id = vocab.encode("</OUTPUT>")

    with open(dpo_input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Toplam {len(lines)} satır veri işleniyor...")
    
    processed_count = 0
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        prompt_text = data.get("prompt", "").strip()
        chosen_text = data.get("chosen", "").strip()
        rejected_text = data.get("rejected", "").strip()
        
        if not prompt_text or not chosen_text or not rejected_text:
            continue

        # Clean/strip enclosing quotes if any
        if prompt_text.startswith('"') and prompt_text.endswith('"'):
            prompt_text = prompt_text[1:-1]
            
        # Encode elements
        # Note: We encode manually to structure prompt vs output properly
        prompt_raw_ids = tokenizer.encode(prompt_text)
        # Strip BOS/EOS from raw tokenizer outputs to avoid duplicates
        if prompt_raw_ids and prompt_raw_ids[0] == bos_id:
            prompt_raw_ids = prompt_raw_ids[1:]
        if prompt_raw_ids and prompt_raw_ids[-1] == eos_id:
            prompt_raw_ids = prompt_raw_ids[:-1]
            
        chosen_raw_ids = tokenizer.encode(chosen_text)
        if chosen_raw_ids and chosen_raw_ids[0] == bos_id:
            chosen_raw_ids = chosen_raw_ids[1:]
        if chosen_raw_ids and chosen_raw_ids[-1] == eos_id:
            chosen_raw_ids = chosen_raw_ids[:-1]
            
        rejected_raw_ids = tokenizer.encode(rejected_text)
        if rejected_raw_ids and rejected_raw_ids[0] == bos_id:
            rejected_raw_ids = rejected_raw_ids[1:]
        if rejected_raw_ids and rejected_raw_ids[-1] == eos_id:
            rejected_raw_ids = rejected_raw_ids[:-1]

        # Construct SFT/DPO sequences
        prompt_ids = [bos_id, inst_start_id] + prompt_raw_ids + [inst_end_id, output_start_id]
        chosen_ids = chosen_raw_ids + [output_end_id, eos_id]
        rejected_ids = rejected_raw_ids + [output_end_id, eos_id]

        # For DPO training script (pairs)
        tokenized_records.append({
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids
        })

        # For SFT pre-training/BC (Binary chosen stream)
        sft_sequence = prompt_ids + chosen_ids
        chosen_sft_tokens.extend(sft_sequence)
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"  -> {processed_count} adet tercih çifti tokenize edildi...")

    # 4. Save vocabulary (to persist any new tokens added during DPO encoding)
    vocab.save(vocab_path)
    print(f"\nSözlük güncellendi ve kaydedildi: {len(vocab.stoi)} token.")

    # 5. Save output files
    # A. JSONL preference pairs
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for rec in tokenized_records:
            f.write(json.dumps(rec) + '\n')
    print(f"Sözcük indeksli tercih çiftleri JSONL olarak kaydedildi: {output_jsonl_path}")

    # B. SFT chosen binary token stream
    if chosen_sft_tokens:
        arr = np.array(chosen_sft_tokens, dtype=np.uint16)
        arr.tofile(output_bin_path)
        print(f"Seçilen (Chosen) yanıt morfem akışı binary olarak kaydedildi: {output_bin_path}")
        print(f"  -> Toplam SFT Morfem Sayısı: {len(chosen_sft_tokens)}")

    print("\n" + "=" * 60)
    print(" DPO VERİ HAZIRLIĞI TAMAMLANDI")
    print("=" * 60)

if __name__ == '__main__':
    main()
