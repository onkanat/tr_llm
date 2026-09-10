import json
import re
import unicodedata
from typing import List, Dict
from src.compiler.core import CrystalCompiler


class Vocabulary:
    def __init__(self):
        # We start with some special tokens and prompt control markers.
        self.stoi = {
            "<UNK>": 0,
            "<PAD>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<INSTRUCTION>": 4,
            "</INSTRUCTION>": 5,
            "<INPUT>": 6,
            "</INPUT>": 7,
            "<OUTPUT>": 8,
            "</OUTPUT>": 9,
            "<NUMBER>": 10,
            "<PROPER_NOUN>": 11,
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self._next_id = 12
        self.frozen = False

    def freeze(self):
        """Freezes vocabulary so no new tokens can be dynamically added."""
        self.frozen = True

    def unfreeze(self):
        """Unfreezes vocabulary allowing new tokens to be added."""
        self.frozen = False

    def add_token(self, token: str):
        if self.frozen:
            return
        if token not in self.stoi:
            self.stoi[token] = self._next_id
            self.itos[self._next_id] = token
            self._next_id += 1

    def register_new_tokens(self, new_tokens: List[str]) -> List[int]:
        """
        Safely registers new tokens into vocabulary even if frozen,
        assigns unique IDs, and returns the assigned token IDs.
        """
        was_frozen = self.frozen
        self.unfreeze()
        added_ids = []
        for tok in new_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok not in self.stoi:
                new_id = self._next_id
                self.stoi[tok] = new_id
                self.itos[new_id] = tok
                self._next_id += 1
                added_ids.append(new_id)
            else:
                added_ids.append(self.stoi[tok])
        if was_frozen:
            self.freeze()
        return added_ids

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<UNK>"])

    def decode(self, token_id: int) -> str:
        return self.itos.get(token_id, "<UNK>")

    def save(self, filepath: str):
        """Saves the vocabulary state to a JSON file."""
        data = {
            "stoi": self.stoi,
            "next_id": self._next_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str, freeze: bool = True):
        """Loads the vocabulary state from a JSON file and freezes it by default."""
        import os
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.stoi = data.get("stoi", self.stoi)
            self.itos = {int(v): k for k, v in self.stoi.items()}
            self._next_id = data.get("next_id", self._next_id)
        if freeze:
            self.freeze()

class KristalTokenizer:
    CONTROL_TOKENS = {
        "<INSTRUCTION>", "</INSTRUCTION>",
        "<INPUT>", "</INPUT>",
        "<OUTPUT>", "</OUTPUT>",
        "<ARA>", "</ARA>",
        "<BELGE>", "</BELGE>",
    }

    def __init__(self, compiler: CrystalCompiler, vocab: Vocabulary, verbose: bool = False):
        self.compiler = compiler
        self.vocab = vocab
        self.verbose = verbose
        for token in self.CONTROL_TOKENS:
            self.vocab.add_token(token)

    def _render_structured_prompt(self, item: Dict[str, str]) -> str:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        parts = []
        if instruction:
            parts.extend(["<INSTRUCTION>", instruction, "</INSTRUCTION>"])
        if input_text or input_text == "":
            parts.extend(["<INPUT>", input_text, "</INPUT>"])
        if output_text or output_text == "":
            parts.extend(["<OUTPUT>", output_text, "</OUTPUT>"])
        return " ".join(parts).strip()

    def _normalize_text(self, text: str) -> str:
        raw = text.strip()
        raw = unicodedata.normalize('NFC', raw)
        if not raw:
            return raw

        # JSON / JSONL handling: convert structured items into prompt
        # templates.
        if raw.startswith("{") or raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return self._render_structured_prompt(parsed)
                if isinstance(parsed, list):
                    prompts = []
                    for item in parsed:
                        if isinstance(item, dict):
                            prompts.append(self._render_structured_prompt(item))
                    return "\n".join(prompts)
            except json.JSONDecodeError:
                pass

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        prompts = []
        for line in lines:
            if line.startswith("{") and line.endswith("}"):
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        prompts.append(
                            self._render_structured_prompt(item)
                        )
                        continue
                except json.JSONDecodeError:
                    pass
            prompts.append(line)
        return " \n ".join(prompts)

    def encode(self, text: str) -> List[int]:
        """
        Tokenizes text by splitting into tokens, running the Kristal
        Compiler on each, and returning integer IDs for the resulting semantic
        token vectors.
        """
        text = self._normalize_text(text)
        token_ids = [self.vocab.encode("<BOS>")]

        tokens = re.findall(r"<[^>]+>|[^\s]+", text, flags=re.UNICODE)
        for token in tokens:
            if token in self.CONTROL_TOKENS:
                token_ids.append(self.vocab.encode(token))
                continue

            clean_word = token.strip(".,!?\"…—«»/()-;:")
            if not clean_word:
                continue

            is_number = (
                clean_word.replace(".", "").replace(",", "").isdigit()
                or clean_word.isnumeric()
            )
            if is_number:
                self.vocab.add_token("<NUMBER>")
                token_ids.append(self.vocab.encode("<NUMBER>"))
                continue

            compile_word = clean_word.replace("'", "").replace("’", "")
            
            # If the clean_word is already a known token in our vocabulary (e.g. suffix tags like POSS_2SG
            # or special tokens like <UNK> and <PROPER_NOUN>), we encode it directly.
            if clean_word in self.vocab.stoi:
                token_ids.append(self.vocab.encode(clean_word))
                continue
                
            result = self.compiler.compile(compile_word)

            if result.get("token_vector"):
                for morpheme_id in result["token_vector"]:
                    if morpheme_id in self.vocab.stoi:
                        token_ids.append(self.vocab.encode(morpheme_id))
                    elif not self.vocab.frozen:
                        self.vocab.add_token(morpheme_id)
                        token_ids.append(self.vocab.encode(morpheme_id))
                    else:
                        token_ids.append(self.vocab.encode("<UNK>"))
            else:
                if clean_word[0].isupper():
                    token_ids.append(self.vocab.encode("<PROPER_NOUN>"))
                else:
                    if self.verbose:
                        print(f"Warning OOV: {clean_word!r}")
                    token_ids.append(self.vocab.encode("<UNK>"))

        token_ids.append(self.vocab.encode("<EOS>"))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a sequence of integer IDs back into semantic morpheme tags.
        """
        return " ".join([self.vocab.decode(tid) for tid in token_ids])


def get_morpheme_weight(tag: str) -> float:
    """
    Returns the architectural weight multiplier based on morpheme category.
    - LOGICAL_OPERATOR (NEG, IMPOTENTIAL_NEG, ama, fakat): 4.0
    - LEXICAL_ROOT (lowercase tag, or <PROPER_NOUN>): 3.5
    - DERIVATION_AFFIX (starts with DERIV_): 1.5
    - INFLECTION_AFFIX (starts with TENSE_, PERSON_, POSS_, CASE_, COPULA_, etc.): 0.1
    - CONTROL_TOKENS: 0.1
    """
    if tag in [
        "<BOS>", "<EOS>", "<PAD>", "<UNK>",
        "<INSTRUCTION>", "</INSTRUCTION>",
        "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>",
        "<NUMBER>", "<SYMBOL>"
    ]:
        return 0.1
    
    if tag in ("NEG", "IMPOTENTIAL_NEG", "ama", "fakat"):
        return 4.0
        
    if tag == "<PROPER_NOUN>":
        return 3.5
        
    if tag.startswith("DERIV_"):
        return 1.5
        
    inflection_prefixes = (
        "TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", 
        "PART_", "INF_", "GERUND_"
    )
    if (
        tag.startswith(inflection_prefixes) or 
        tag in ("PLURAL", "POTENTIAL")
    ):
        return 0.1
        
    return 3.5


