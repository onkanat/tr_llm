import json
from typing import List, Dict, Any
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import MorphotacticsGraph, State
from src.compiler.phonology import PhonologyEngine

class CrystalCompiler:
    def __init__(self, lexicon: LexiconManager, morph_graph: MorphotacticsGraph):
        self.lexicon = lexicon
        self.graph = morph_graph

    def compile(self, word: str) -> Dict[str, Any]:
        """
        Main entry point for morphological analysis.
        Returns a CrystalPack JSON dict.
        """
        valid_paths = []
        
        # 1. Find all possible roots for the word
        stems = self.lexicon.find_stems(word)
        
        # 2. For each root, start recursive DFS pathfinding
        for matched_prefix, stem_data in stems:
            lemma = stem_data['lemma']
            pos = stem_data['pos']
            
            # Determine start state based on POS
            start_state = State.VERB_ROOT if pos == 'VERB' else State.NOUN_ROOT
            
            # Initial path contains the root
            initial_path = [{
                'type': 'ROOT',
                'id': lemma,
                'surface': matched_prefix,
                'pos': pos
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
                # Basic disambiguation check: if top 2 scores are very close
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
        Early pruning: if current_string is not a prefix of target_word, stop.
        """
        print(f"DFS: current_string='{current_string}', current_state='{current_state}'")
        
        # Beam pruning check: current_string must be a prefix of target_word
        # Allow for future consonant mutation: if current_string ends in p,ç,t,k and target has b,c,d,ğ at that pos.
        is_valid_prefix = target_word.startswith(current_string)
        if not is_valid_prefix and len(target_word) >= len(current_string) and len(current_string) > 0:
            prefix_except_last = current_string[:-1]
            if target_word.startswith(prefix_except_last):
                last_char = current_string[-1]
                target_char = target_word[len(current_string)-1]
                voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'}
                if last_char in voicing_map and voicing_map[last_char] == target_char:
                    is_valid_prefix = True

        if not is_valid_prefix:
            print(f"  -> Pruned: not prefix of {target_word}")
            return
            
        # Base case / Success condition:
        # If we matched the exact target word AND we are in a valid terminal state
        if current_string == target_word:
            if self.graph.is_terminal(current_state):
                print("  -> Success! Adding to results.")
                # Save a copy of the successful path
                results.append({
                    "morphemes": list(current_path),
                    "surface_form": current_string
                })
            else:
                print(f"  -> Reached target word but state {current_state} is not terminal.")
        
        # Recursive step: Explore all valid transitions from the current state
        transitions = self.graph.get_valid_transitions(current_state)
        for transition in transitions:
            mutated_string, affix_surface = PhonologyEngine.resolve_affix(current_string, transition.affix_template)
            new_string = mutated_string + affix_surface

            
            # Prune invalid paths early
            # Allow for future consonant mutation: if new_string ends in p,ç,t,k and target has b,c,d,ğ at that pos.
            is_valid_prefix = target_word.startswith(new_string)
            if not is_valid_prefix and len(target_word) >= len(new_string):
                # Check if they match except for the last char
                prefix_except_last = new_string[:-1]
                if target_word.startswith(prefix_except_last):
                    last_char = new_string[-1]
                    target_char = target_word[len(new_string)-1]
                    voicing_map = {'p': 'b', 'ç': 'c', 't': 'd', 'k': 'ğ'}
                    if last_char in voicing_map and voicing_map[last_char] == target_char:
                        is_valid_prefix = True

            if not is_valid_prefix:
                continue
                
            new_path = list(current_path)
            
            # If the stem was mutated (e.g. ecek -> eceğ), update the surface of the last morpheme
            if mutated_string != current_string and len(new_path) > 0:
                last_morpheme = dict(new_path[-1]) # Copy it
                last_morpheme['surface'] = last_morpheme['surface'][:-1] + mutated_string[-1]
                new_path[-1] = last_morpheme

            new_path.append({
                'type': 'AFFIX',
                'id': transition.affix_id,
                'surface': affix_surface
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
