import csv
from typing import List, Dict, Any

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.data = None

class LexiconManager:
    def __init__(self):
        self.root = TrieNode()

    def load_from_tsv(self, filepath: str):
        """Loads lexicon from a TSV file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                lemma = row['lemma']
                # Insert original lemma
                self._insert(lemma, row)
                
                # Pre-calculate voicing if applicable for performance
                # This ensures O(k) lookup without runtime phonology checks for stems
                if row.get('attributes') == 'VOICING':
                    voiced_lemma = self._apply_voicing(lemma)
                    if voiced_lemma != lemma:
                        self._insert(voiced_lemma, row)

    def _apply_voicing(self, lemma: str) -> str:
        """Applies consonant mutation (yumuşama) to the final character."""
        if not lemma: return lemma
        last_char = lemma[-1]
        # Common Turkish voicing rules: p->b, ç->c, t->d, k->ğ (sometimes g)
        voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'} 
        if last_char in voicing_map:
            return lemma[:-1] + voicing_map[last_char]
        return lemma

    def _insert(self, word: str, data: Dict[str, Any]):
        """Inserts a word and its metadata into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.data = data

    def find_stems(self, word: str) -> List[tuple[str, Dict[str, Any]]]:
        """
        Scans prefixes of the input word and returns all possible root candidates
        along with the exact matched surface prefix.
        Complexity: O(k) where k is the length of the word.
        """
        stems = []
        node = self.root
        matched_prefix = ""
        for char in word:
            if char not in node.children:
                break
            matched_prefix += char
            node = node.children[char]
            if node.is_word:
                stems.append((matched_prefix, node.data))
        return stems
