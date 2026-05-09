import json
from typing import List, Dict, Optional
from src.compiler.core import CrystalCompiler

class Vocabulary:
    def __init__(self):
        # We start with some special tokens
        self.stoi = {"<UNK>": 0, "<PAD>": 1, "<BOS>": 2, "<EOS>": 3}
        self.itos = {0: "<UNK>", 1: "<PAD>", 2: "<BOS>", 3: "<EOS>"}
        self._next_id = 4

    def add_token(self, token: str):
        if token not in self.stoi:
            self.stoi[token] = self._next_id
            self.itos[self._next_id] = token
            self._next_id += 1

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<UNK>"])

    def decode(self, token_id: int) -> str:
        return self.itos.get(token_id, "<UNK>")

    def save(self, filepath: str):
        """Saves the vocabulary state to a JSON file."""
        data = {
            "stoi": self.stoi,
            "next_id": self._next_id
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """Loads the vocabulary state from a JSON file."""
        import os
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.stoi = data.get("stoi", self.stoi)
            # Reconstruct itos with integer keys
            self.itos = {int(v): k for k, v in self.stoi.items()}
            self._next_id = data.get("next_id", self._next_id)

class KristalTokenizer:
    def __init__(self, compiler: CrystalCompiler, vocab: Vocabulary):
        self.compiler = compiler
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        """
        Tokenizes text by splitting into words, running the Kristal Compiler on each,
        and returning the integer IDs for the semantic token_vector.
        """
        words = text.split()
        token_ids = [self.vocab.encode('<BOS>')]
        for word in words:
            # We strip basic punctuation, keeping apostrophes for potential future suffix parsing
            clean_word = word.strip(".,!?\"…—«»/()-;:")
            
            if not clean_word:
                continue
                
            # 1. Number Bypass
            if clean_word.replace(".", "").replace(",", "").isdigit() or clean_word.isnumeric():
                self.vocab.add_token("<NUMBER>")
                token_ids.append(self.vocab.encode("<NUMBER>"))
                continue
                
            # 2. Normal Compilation
            compile_word = clean_word.replace("'", "")
            result = self.compiler.compile(compile_word)
            
            # Use the token vector from the best analysis
            if result.get("token_vector"):
                for morpheme_id in result["token_vector"]:
                    # Dynamically add to vocab for prototype
                    self.vocab.add_token(morpheme_id)
                    token_ids.append(self.vocab.encode(morpheme_id))
            else:
                # 3. Proper Noun Bypass (If compilation fails but word is capitalized)
                if clean_word[0].isupper():
                    self.vocab.add_token("<PROPER_NOUN>")
                    token_ids.append(self.vocab.encode("<PROPER_NOUN>"))
                else:
                    # True OOV
                    print(f"Warning OOV: {clean_word!r}")
                    token_ids.append(self.vocab.encode("<UNK>"))
                    
        token_ids.append(self.vocab.encode('<EOS>'))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of integer IDs back into semantic morpheme tags.
        (Note: Converting tags back to fluent text requires running the Phonology Engine in reverse
        or using a surface generator, which is usually handled by the 'Decode' phase or directly outputting tags).
        For now, returns the space-separated morpheme IDs.
        """
        return " ".join([self.vocab.decode(tid) for tid in token_ids])
