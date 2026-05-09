import sys
sys.path.append('.')
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.compiler.phonology import PhonologyEngine

lex = LexiconManager()
lex.load_from_tsv('data/lexicon/roots.tsv')
comp = CrystalCompiler(lex, build_default_graph())

original_find = comp._find_paths_recursive

def patch_find(target_word, current_string, current_state, current_path, results):
    print(f"TR: '{current_string}' | {current_state}")
    transitions = comp.graph.get_valid_transitions(current_state)
    for t in transitions:
        if current_string.startswith('boğ'):
            last_attrs = current_path[-1].get('attributes', '-')
            mut_stem, aff_surf = PhonologyEngine.resolve_affix(current_string, t.affix_template, last_attrs)
            ns = mut_stem + aff_surf
            print(f"  -> test transition {t.affix_id} ('{t.affix_template}') -> '{ns}'")
    original_find(target_word, current_string, current_state, current_path, results)

comp._find_paths_recursive = patch_find
comp.compile('boğmalısınız')
