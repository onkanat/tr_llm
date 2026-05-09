import json
import os

def parse_gts_to_tsv(jsonl_path, tsv_path):
    roots = {} # Key: (lemma, pos), Value: attributes_set
    
    processed_count = 0
    added_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
            except:
                continue
                
            processed_count += 1
            madde = obj.get("madde", "").strip()
            
            # Skip multi-word phrases, phrases with hyphens, etc. (We want pure lemmas)
            if " " in madde or "-" in madde or "'" in madde or "^" in madde:
                continue
                
            # Determine POS and find ALL meanings
            anlamlar = obj.get("anlamlarListe", [])
            if not anlamlar: continue
            
            # A word can have multiple POS tags (Homonyms)
            word_pos_set = set()
            for anlam_obj in anlamlar:
                pos = "NOUN" # default
                if anlam_obj.get("fiil") == "1":
                    pos = "VERB"
                else:
                    ozellikler = anlam_obj.get("ozelliklerListe", [])
                    if ozellikler:
                        for oz in ozellikler:
                            tur = oz.get("tam_adi", "").lower()
                            if "sıfat" in tur: pos = "ADJ"
                            elif "zarf" in tur: pos = "ADV"
                            elif "zamir" in tur: pos = "PRON"
                            elif "bağlaç" in tur: pos = "CONJ"
                            elif "edat" in tur: pos = "POSTP"
                            elif "sayı" in tur: pos = "NUM"
                            elif "ünlem" in tur: pos = "INTERJ"
                word_pos_set.add(pos)

            # Determine attributes
            attributes = set()
            taki = obj.get("taki")
            if taki and isinstance(taki, str):
                taki = taki.strip("-")
                if len(madde) > 0 and len(taki) > 0:
                    last_char = madde[-1]
                    first_taki_char = taki[0]
                    if last_char == 'k' and first_taki_char in ['ğ', 'g']: attributes.add("VOICING")
                    elif last_char == 'p' and first_taki_char == 'b': attributes.add("VOICING")
                    elif last_char == 't' and first_taki_char == 'd': attributes.add("VOICING")
                    elif last_char == 'ç' and first_taki_char == 'c': attributes.add("VOICING")
                    
                    if len(taki) > 1 and last_char in "lrn" and len(madde) > 2:
                         if madde[-2] in "ıiuü" and first_taki_char not in "aeıioöuü":
                             attributes.add("VOWEL_DROP")

            for pos in word_pos_set:
                pure_madde = madde
                if pos == "VERB":
                    if pure_madde.endswith("mak") or pure_madde.endswith("mek"):
                        pure_madde = pure_madde[:-3]
                
                # Skip if empty or too short (unless pronoun)
                if len(pure_madde) < 2 and pos != "PRON" and pure_madde != "o":
                    continue
                
                key = (pure_madde, pos)
                if key not in roots:
                    roots[key] = set(attributes)
                    added_count += 1
                else:
                    # Merge attributes if multiple entries exist
                    roots[key].update(attributes)
                
    print(f"Processed lines: {processed_count}")
    print(f"Newly added pure roots: {added_count}")
    print(f"Total unique roots in lexicon: {len(roots)}")
    
    # Write to TSV
    with open(tsv_path, 'w', encoding='utf-8') as out_f:
        out_f.write("lemma\tpos\tattributes\n")
        # Sort by lemma then pos
        for (lemma, pos) in sorted(roots.keys()):
            attr_set = roots[(lemma, pos)]
            attr_str = ",".join(sorted(list(attr_set))) if attr_set else "-"
            out_f.write(f"{lemma}\t{pos}\t{attr_str}\n")

if __name__ == '__main__':
    parse_gts_to_tsv('data/poems/gts.json', 'data/lexicon/roots.tsv')
