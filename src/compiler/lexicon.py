import csv
from typing import List, Dict, Any

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.entries = [] # List of data Dicts to support homonyms

class LexiconManager:
    def __init__(self):
        self.root = TrieNode()

    def load_from_tsv(self, filepath: str):
        """Loads lexicon from a TSV file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                lemma = row['lemma']
                self._insert(lemma, row)
                lower_lemma = lemma.lower()
                if lower_lemma != lemma:
                    self._insert(lower_lemma, row)

    def _insert(self, word: str, data: Dict[str, Any]):
        """Inserts a word and its metadata into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.entries.append(data)

    def find_stems(self, word: str) -> List[tuple[str, Dict[str, Any]]]:
        """
        Mutation-aware stem finding.
        """
        stems = []
        node = self.root
        matched_prefix = ""
        # Mapping from surface character in word -> possible root character
        # e.g. if we see 'd' in word, we might be looking for root ending in 't'
        reverse_voicing_map = {'b': 'p', 'c': 'ç', 'd': 't', 'ğ': 'k', 'g': 'k'}
        
        for i, char in enumerate(word):
            # Look ahead for potential roots ending in unvoiced consonants
            unvoiced_char = reverse_voicing_map.get(char)
            
            # If we have an unvoiced candidate (like 't' for 'd'), check that path too
            if unvoiced_char and unvoiced_char in node.children:
                unvoiced_node = node.children[unvoiced_char]
                if unvoiced_node.is_word:
                    for entry in unvoiced_node.entries:
                        # Only allow if the root has VOICING attribute
                        if "VOICING" in entry.get('attributes', ''):
                            stems.append((matched_prefix + char, entry))

            # Look ahead for vowel drop candidates (direct or voiced/unvoiced)
            for v in ('ı', 'i', 'u', 'ü'):
                if v in node.children:
                    v_node = node.children[v]
                    chars_to_check = [char]
                    if unvoiced_char:
                        chars_to_check.append(unvoiced_char)
                    
                    for c in chars_to_check:
                        if c in v_node.children:
                            target_node = v_node.children[c]
                            if target_node.is_word:
                                for entry in target_node.entries:
                                    attrs = entry.get('attributes', '')
                                    if "VOWEL_DROP" in attrs:
                                        if c == unvoiced_char and "VOICING" not in attrs:
                                            continue
                                        stems.append((matched_prefix + char, entry))

            # Advance node based on direct match
            if char in node.children:
                matched_prefix += char
                node = node.children[char]
                if node.is_word:
                    for entry in node.entries:
                        stems.append((matched_prefix, entry))
            else:
                # No more possible prefix matches in Trie
                break
                    
        return stems
