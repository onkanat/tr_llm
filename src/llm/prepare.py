import os
import json
import numpy as np
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

def prepare_dataset(input_filepath: str, output_filepath: str):
    """
    Ingests a raw text dataset, processes it through the Kristal Compiler,
    and saves the pure semantic 'token_vector' as a numpy array for fast LLM training.
    """
    # 1. Init Compiler
    lexicon = LexiconManager()
    # In production, load the full roots.tsv here.
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    tokenizer = KristalTokenizer(compiler, vocab)

    print(f"Preparing dataset from {input_filepath}...")
    
    # 2. Read and Tokenize
    all_token_ids = []
    if os.path.exists(input_filepath):
        with open(input_filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            all_token_ids = tokenizer.encode(text)
    
    print(f"Total morphemes encoded: {len(all_token_ids)}")
    
    # 3. Save as binary for NanoGPT speedruns
    if all_token_ids:
        # We can use uint16 since our vocab_size (20500) < 65535
        arr = np.array(all_token_ids, dtype=np.uint16)
        arr.tofile(output_filepath)
        print(f"Saved binary tokens to {output_filepath}")

if __name__ == '__main__':
    # Example usage
    # prepare_dataset("data/raw_corpus.txt", "data/train.bin")
    pass
