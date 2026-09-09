import os
import json
import re
import numpy as np
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary


def _load_raw_documents(input_filepath: str) -> list:
    with open(input_filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    if input_filepath.endswith('.jsonl') or input_filepath.endswith('.json'):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        documents = []
        for line in lines:
            if line.startswith('{') and line.endswith('}'):
                try:
                    json.loads(line)
                    documents.append(line)
                    continue
                except json.JSONDecodeError:
                    pass
            documents.append(line)
        return documents

    # Split raw text by paragraph boundaries to preserve record segmentation.
    return [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]


def prepare_dataset(input_filepath: str, output_filepath: str):
    """
    Ingests a raw text dataset, processes it through the Kristal Compiler,
    and saves the semantic token stream as a numpy array for fast LLM training.

    Additionally writes a metadata sidecar describing document boundaries.
    """
    lexicon = LexiconManager()
    lexicon_path = 'data/lexicon/roots.tsv'
    if os.path.exists(lexicon_path):
        lexicon.load_from_tsv(lexicon_path)

    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    if os.path.exists(vocab_path):
        vocab.load(vocab_path)
        print(f"Loaded existing vocabulary from {vocab_path} with {len(vocab.stoi)} tokens.")
    else:
        print("Creating a new vocabulary.")

    tokenizer = KristalTokenizer(compiler, vocab)

    print(f"Preparing dataset from {input_filepath}...")
    token_sequences = []
    boundaries = []
    offset = 0

    if os.path.exists(input_filepath):
        documents = _load_raw_documents(input_filepath)
        for document in documents:
            token_ids = tokenizer.encode(document)
            if len(token_ids) <= 2:
                continue
            token_sequences.extend(token_ids)
            boundaries.append({"offset": offset, "length": len(token_ids)})
            offset += len(token_ids)

    print(f"Processed {len(boundaries)} document segments.")
    print(f"Total morphemes encoded: {offset}")

    if token_sequences:
        arr = np.array(token_sequences, dtype=np.uint16)
        arr.tofile(output_filepath)
        print(f"Saved binary tokens to {output_filepath}")

        # Save vocabulary back to vocab.json
        vocab.save(vocab_path)
        print(f"Saved updated vocabulary to {vocab_path} with {len(vocab.stoi)} tokens.")

        meta_path = output_filepath + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as meta_file:
            json.dump({
                "record_count": len(boundaries),
                "boundaries": boundaries,
            }, meta_file, ensure_ascii=False, indent=2)
        print(f"Saved segmentation metadata to {meta_path}")


if __name__ == '__main__':
    # Example usage
    # prepare_dataset("data/raw_corpus.txt", "data/train.bin")
    pass
