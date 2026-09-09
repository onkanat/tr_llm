import json

dpo_jsonl_path = 'data/pedagogy/dpo_tokenized.jsonl'
lengths = []
with open(dpo_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            c_len = len(rec["prompt_ids"]) + len(rec["chosen_ids"])
            r_len = len(rec["prompt_ids"]) + len(rec["rejected_ids"])
            lengths.append(max(c_len, r_len))

lengths.sort()
print(f"Total records: {len(lengths)}")
print(f"Min length: {lengths[0]}")
print(f"Max length: {lengths[-1]}")
print(f"Median length: {lengths[len(lengths)//2]}")
for limit in [128, 256, 512, 1024, 2048]:
    count = sum(1 for l in lengths if l <= limit)
    pct = count / len(lengths) * 100
    print(f"Length <= {limit:4d}: {count:4d} records ({pct:.2f}%)")
