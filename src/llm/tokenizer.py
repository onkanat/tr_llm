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
            # We strip basic punctuation and apostrophes for simplicity in this prototype
            clean_word = word.replace("'", "").strip(".,!?\"'…—«»")
            
            if not clean_word:
                continue
                
            result = self.compiler.compile(clean_word)
            # Use the token vector from the best analysis
            if result.get("token_vector"):
                for morpheme_id in result["token_vector"]:
                    # Dynamically add to vocab for prototype (in production this is fixed to 20500)
                    self.vocab.add_token(morpheme_id)
                    token_ids.append(self.vocab.encode(morpheme_id))
            else:
                # If analysis fails, log and map to UNK
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
