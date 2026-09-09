import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm.tokenizer import Vocabulary

vocab = Vocabulary()
vocab.load('data/vocab.json')

dpo_jsonl_path = 'data/pedagogy/dpo_all_tokenized.jsonl'
with open(dpo_jsonl_path, 'r', encoding='utf-8') as f:
    for i in range(2):
        line = f.readline()
        if not line:
            break
        rec = json.loads(line)
        print(f"Record {i}:")
        print(f"  Prompt IDs length: {len(rec['prompt_ids'])}")
        print(f"  Chosen IDs length: {len(rec['chosen_ids'])}")
        print(f"  Rejected IDs length: {len(rec['rejected_ids'])}")
        
        # Decode prompt prefix
        prompt_dec = " ".join([vocab.decode(tid) for tid in rec['prompt_ids'][:50]])
        print(f"  Prompt start decoded: {prompt_dec} ...")
        
        chosen_dec = " ".join([vocab.decode(tid) for tid in rec['chosen_ids'][:50]])
        print(f"  Chosen start decoded: {chosen_dec} ...")
        
        rejected_dec = " ".join([vocab.decode(tid) for tid in rec['rejected_ids'][:50]])
        print(f"  Rejected start decoded: {rejected_dec} ...")
        print("-" * 50)
