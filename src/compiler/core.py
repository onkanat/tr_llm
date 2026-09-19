import json
from typing import List, Dict, Any
from src.compiler.lexicon import LexiconManager, turkish_lower
from src.compiler.morphotactics import MorphotacticsGraph, State
from src.compiler.phonology import PhonologyEngine

class CrystalCompiler:
    def __init__(self, lexicon: LexiconManager, graph: MorphotacticsGraph):
        self.lexicon = lexicon
        self.graph = graph

    def _turkish_lower(self, word: str) -> str:
        """Properly lowercases Turkish text.

        ⚠️ TEK KAYNAK: gövde `lexicon.turkish_lower`'a delege eder. Bu gövde
        ile `load_from_tsv`'nin trie anahtarı AYNI olmak zorundadır; ayrışırsa
        `İ`/`I` ile başlayan lemmalar trie'de erişilemez kalır (T-0080).
        """
        return turkish_lower(word)

    def compile(self, word: str) -> Dict[str, Any]:
        """
        Main entry point for morphological analysis.
        Returns a CrystalPack JSON dict.
        """
        valid_paths = []
        word = self._turkish_lower(word) # Turkish Case-insensitive compile

        
        # 1. Find all possible roots for the word
        stems = self.lexicon.find_stems(word)
        
        # 2. For each root, start recursive DFS pathfinding
        for matched_prefix, stem_data in stems:
            lemma = stem_data['lemma']
            pos = stem_data['pos']
            attributes = stem_data.get('attributes', '-')
            
            # Determine start state based on POS
            pos_to_state = {
                "VERB": State.VERB_ROOT,
                "NOUN": State.NOUN_ROOT,
                "ADJ":  State.ADJ_ROOT,
                "ADV":  State.ADV_ROOT,
                "NUM":  State.NOUN_ROOT,
                "PRON": State.NOUN_ROOT,
                "POSTP": State.NOUN_ROOT,
            }
            start_state = pos_to_state.get(pos, State.NOUN_ROOT)
            
            # Initial path contains the root
            initial_path = [{
                'type': 'ROOT',
                'id': lemma,
                'surface': matched_prefix,
                'pos': pos,
                'attributes': attributes
            }]
            
            # Recurse
            self._find_paths_recursive(
                target_word=word,
                current_string=matched_prefix,
                current_state=start_state,
                current_path=initial_path,
                results=valid_paths
            )

        # 3. Score and disambiguate
        scored_paths = self._score_paths(valid_paths)
        
        best_surface = ""
        token_vector = []
        needs_disambiguation = False
        
        if scored_paths:
            best_analysis = scored_paths[0]
            best_surface = best_analysis['surface_form']
            token_vector = [item['id'] for item in best_analysis['morphemes']]
            
            if len(scored_paths) > 1:
                score1 = scored_paths[0]['score']
                score2 = scored_paths[1]['score']
                if abs(score1 - score2) < 0.1:
                    needs_disambiguation = True

        return {
            "input": word,
            "language": "tr",
            "analyses": scored_paths,
            "best_surface": best_surface,
            "token_vector": token_vector,
            "needs_disambiguation": needs_disambiguation
        }

    def _find_paths_recursive(self, target_word: str, current_string: str, current_state: str, current_path: List[Dict[str, Any]], results: List[Dict[str, Any]]):
        """
        Depth-First Search for valid morphological paths.
        """
        # print(f"DFS: word='{target_word}', current='{current_string}', state='{current_state}'")
        if current_string == target_word:
            if self.graph.is_terminal(current_state):
                results.append({
                    "morphemes": list(current_path),
                    "surface_form": current_string
                })
        
        # Recursive step
        transitions = self.graph.get_valid_transitions(current_state)
        for transition in transitions:
            # Attributes for stem mutation logic
            last_morpheme = current_path[-1]
            last_attrs = last_morpheme.get('attributes', '-')
            
            mutated_stem, affix_surface = PhonologyEngine.resolve_affix(
                current_string, 
                transition.affix_template,
                last_attrs
            )
            
            new_string = mutated_stem + affix_surface

            # Beam Pruning: startswith check with mutation allowance
            is_valid_prefix = target_word.startswith(new_string)
            if not is_valid_prefix and len(target_word) >= len(new_string):
                # If target has voiced char where current has unvoiced, allow it if VOICING active
                voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'}
                last_char = new_string[-1]
                target_char = target_word[len(new_string)-1]
                if last_char in voicing_map and voicing_map[last_char] == target_char:
                     if "VOICING" in transition.attributes:
                         is_valid_prefix = True

            if not is_valid_prefix:
                continue
                
            # Avoid infinite null-loops
            if len(new_string) <= len(current_string) and transition.affix_template == "":
                if any(m['id'] == transition.affix_id for m in current_path):
                    continue

            new_path = list(current_path)
            
            # Sync surface forms if stem mutated
            if mutated_stem != current_string:
                last_morpheme_copy = dict(new_path[-1])
                # Find length of the last morpheme in mutated_stem
                last_morpheme_len = len(last_morpheme_copy['surface'])
                # If vowel drop occurred, length changed
                if len(mutated_stem) < len(current_string):
                    # Vowel drop: stem was e.g. 'burun', became 'burn'
                    # surface was 'burun', becomes 'burn'
                    last_morpheme_copy['surface'] = mutated_stem[len(mutated_stem) - last_morpheme_len + 1:] # Simplified
                else:
                    # Voicing only
                    last_morpheme_copy['surface'] = last_morpheme_copy['surface'][:-1] + mutated_stem[-1]
                new_path[-1] = last_morpheme_copy

            new_path.append({
                'type': 'AFFIX',
                'id': transition.affix_id,
                'surface': affix_surface,
                'attributes': transition.attributes
            })
            
            self._find_paths_recursive(
                target_word=target_word,
                current_string=new_string,
                current_state=transition.to_state,
                current_path=new_path,
                results=results
            )

    def _score_paths(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        ScoringModule implementation.
        Score = length_factor * (heuristic confidence)
        For now, a basic heuristic favoring fewer morphemes (simpler derivations usually preferred).
        """
        for path in paths:
            # Basic scoring: 10.0 - number of morphemes (favor simpler explanations)
            path['score'] = 10.0 - len(path['morphemes'])
            
        # Sort descending by score
        return sorted(paths, key=lambda x: x['score'], reverse=True)
