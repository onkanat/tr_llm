import os
import sys
import json
import re
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def clean_markdown(text: str) -> str:
    """Removes basic markdown links and headers for cleaner tokenization."""
    if not text:
        return ""
    # Remove markdown links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove bold/italic markup: **text** or *text* -> text
    text = re.sub(r'\*\*([^*]+)\*\*|\*([^*]+)\*', r'\1\2', text)
    # Remove headers: # Header -> Header
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def main():
    print("=" * 60)
    print(" WIKIPEDIA MARKDOWN VERİ SETİ OKUMA VE KRİSTAL TOKENİZASYONU")
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
    wiki_parquet_path = '/Users/hakankilicaslan/Prompts/Outputs/datasets/open-wikipedia-markdown/tr/tr-00000.parquet'
    output_bin_path = 'data/train_wiki.bin'

    if not os.path.exists(wiki_parquet_path):
        print(f"Hata: Wikipedia Parquet dosyası '{wiki_parquet_path}' bulunamadı!")
        return

    # 3. Read Parquet and sample
    print(f"\nParquet dosyası yükleniyor: {wiki_parquet_path}...")
    df = pd.read_parquet(wiki_parquet_path)
    print(f"Toplam {len(df)} makale yüklendi.")

    # Sample the first 5,000 articles
    sample_size = 5000
    df_sample = df.head(sample_size)
    print(f"Eğitim için ilk {sample_size} makale seçildi.")

    # 4. Tokenize
    wiki_tokens = []
    processed_count = 0
    
    for idx, row in df_sample.iterrows():
        title = row.get("title", "").strip()
        markdown_text = row.get("markdown", "")
        
        # Combine title and cleaned markdown text
        article_text = f"{title}\n{clean_markdown(markdown_text)}"
        
        # Split article into paragraphs/records to avoid extremely long single sequences
        segments = [seg.strip() for seg in re.split(r'\n\s*\n+', article_text) if seg.strip()]
        
        for segment in segments:
            if len(segment) < 20:  # Skip too short segments
                continue
            token_ids = tokenizer.encode(segment)
            # Only keep segments with valid morphemes
            if len(token_ids) > 2:
                wiki_tokens.extend(token_ids)

        processed_count += 1
        if processed_count % 500 == 0:
            print(f"  -> {processed_count}/{sample_size} makale tokenize edildi...")

    # 5. Save vocabulary
    vocab.save(vocab_path)
    print(f"\nSözlük güncellendi ve kaydedildi: {len(vocab.stoi)} token.")

    # 6. Save binary tokens
    if wiki_tokens:
        arr = np.array(wiki_tokens, dtype=np.uint16)
        arr.tofile(output_bin_path)
        print(f"Wikipedia morfem akışı binary olarak kaydedildi: {output_bin_path}")
        print(f"  -> Toplam Wikipedia Morfem Sayısı: {len(wiki_tokens)}")

    print("\n" + "=" * 60)
    print(" WIKIPEDIA VERİ HAZIRLIĞI TAMAMLANDI")
    print("=" * 60)

if __name__ == '__main__':
    main()
