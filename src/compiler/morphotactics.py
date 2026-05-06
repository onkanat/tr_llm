from typing import Dict, List, Set

class State:
    START = "S_START"
    VERB_ROOT = "S_VERB_ROOT"
    NOUN_ROOT = "S_NOUN_ROOT"
    VERB_POST_TENSE = "S_VERB_POST_TENSE"
    VERB_POST_COPULA = "S_VERB_POST_COPULA"
    VERB_POST_PERSON = "S_VERB_POST_PERSON"
    NOUN_POST_PLURAL = "S_NOUN_POST_PLURAL"
    NOUN_POST_POSSESSIVE = "S_NOUN_POST_POSSESSIVE"
    NOUN_POST_POSSESSIVE_3 = "S_NOUN_POST_POSSESSIVE_3"
    NOUN_POST_CASE = "S_NOUN_POST_CASE"
    STOP = "S_STOP"

class Transition:
    def __init__(self, to_state: str, affix_id: str, affix_template: str):
        self.to_state = to_state
        self.affix_id = affix_id
        self.affix_template = affix_template

class MorphotacticsGraph:
    def __init__(self):
        # A dictionary mapping from state to a list of possible transitions
        self.transitions: Dict[str, List[Transition]] = {}
        # A set of states that are considered valid ending points (terminal states)
        self.terminal_states: Set[str] = set()

    def add_transition(self, from_state: str, to_state: str, affix_id: str, affix_template: str):
        """Adds a valid transition between states with a specific affix."""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(Transition(to_state, affix_id, affix_template))

    def mark_terminal(self, state: str):
        """Marks a state as a valid end of a word."""
        self.terminal_states.add(state)

    def get_valid_transitions(self, current_state: str) -> List[Transition]:
        """Returns all valid transitions from the current state."""
        return self.transitions.get(current_state, [])

    def is_terminal(self, state: str) -> bool:
        """Checks if a state is a valid terminal state."""
        return state in self.terminal_states

def build_default_graph() -> MorphotacticsGraph:
    """Builds a basic morphotactics graph for testing and initial implementation."""
    graph = MorphotacticsGraph()

    # From START to Roots (This transition is conceptual, usually roots are derived from lexicon first)
    # We will assume Lexicon gives us the ROOT state.
    
    # Verb Path: ROOT -> TENSE -> PERSON
    # Example: gel (VERB_ROOT) -> -ecek (TENSE_FUT) -> -im (PERSON_1SG)
    graph.add_transition(State.VERB_ROOT, State.VERB_POST_TENSE, "TENSE_FUT", "(y)AcAk")
    graph.add_transition(State.VERB_ROOT, State.VERB_POST_TENSE, "TENSE_PROG", "(y)Iyor")
    graph.add_transition(State.VERB_ROOT, State.VERB_POST_TENSE, "TENSE_PAST", "DI")
    graph.add_transition(State.VERB_ROOT, State.VERB_POST_TENSE, "TENSE_EVIDENTIAL", "mIş")
    
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_PERSON, "PERSON_1SG", "(I)m")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_PERSON, "PERSON_2SG", "sIn")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_PERSON, "PERSON_3SG", "")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_PERSON, "PERSON_1PL", "(I)z")
    
    # Copula Path (Ek-fiil): TENSE -> COPULA -> PERSON
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_PAST", "DI")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_EVIDENTIAL", "mIş")
    
    graph.add_transition(State.VERB_POST_COPULA, State.VERB_POST_PERSON, "PERSON_1SG", "(I)m")
    graph.add_transition(State.VERB_POST_COPULA, State.VERB_POST_PERSON, "PERSON_2SG", "sIn")
    graph.add_transition(State.VERB_POST_COPULA, State.VERB_POST_PERSON, "PERSON_3SG", "")
    graph.add_transition(State.VERB_POST_COPULA, State.VERB_POST_PERSON, "PERSON_1PL", "(I)z")
    
    # Noun Path: ROOT -> PLURAL -> POSSESSIVE -> CASE
    # Example: kitap (NOUN_ROOT) -> -lar (PLURAL) -> -ım (POSS_1SG) -> -da (CASE_LOC)
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_PLURAL, "PLURAL", "lAr")
    
    # 1st and 2nd Person Possessives
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_POSSESSIVE, "POSS_1SG", "(I)m")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_POSSESSIVE, "POSS_1SG", "(I)m")
    
    # 3rd Person Possessive
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_POSSESSIVE_3, "POSS_3SG", "(s)I")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_POSSESSIVE_3, "POSS_3SG", "I") # lAr + I
    
    # Normal Case Transitions
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_CASE, "CASE_LOC", "DA")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_CASE, "CASE_LOC", "DA")
    graph.add_transition(State.NOUN_POST_POSSESSIVE, State.NOUN_POST_CASE, "CASE_LOC", "DA")

    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_CASE, "CASE_ABL", "DAn")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_CASE, "CASE_ABL", "DAn")
    graph.add_transition(State.NOUN_POST_POSSESSIVE, State.NOUN_POST_CASE, "CASE_ABL", "DAn")

    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_CASE, "CASE_ACC", "(y)I")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_CASE, "CASE_ACC", "(y)I")
    graph.add_transition(State.NOUN_POST_POSSESSIVE, State.NOUN_POST_CASE, "CASE_ACC", "(y)I")

    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_CASE, "CASE_DAT", "(y)A")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_CASE, "CASE_DAT", "(y)A")
    graph.add_transition(State.NOUN_POST_POSSESSIVE, State.NOUN_POST_CASE, "CASE_DAT", "(y)A")

    # Genitive Case (Tamlayan): Always takes (n)
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_CASE, "CASE_GEN", "(n)In")
    graph.add_transition(State.NOUN_POST_PLURAL, State.NOUN_POST_CASE, "CASE_GEN", "(n)In")
    graph.add_transition(State.NOUN_POST_POSSESSIVE, State.NOUN_POST_CASE, "CASE_GEN", "(n)In")
    graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, "CASE_GEN", "nIn")

    # Special Case Transitions for 3rd Person Possessive (n buffer)
    graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, "CASE_LOC", "nDA")
    graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, "CASE_ABL", "nDAn")
    graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, "CASE_ACC", "nI")
    graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, "CASE_DAT", "nA")
    
    # Participles and Gerunds (Fiilimsiler): Verb -> Noun/Stop
    graph.add_transition(State.VERB_ROOT, State.NOUN_ROOT, "PARTICIPLE_PAST", "DIk") # Sıfat-fiil: söylediği
    graph.add_transition(State.VERB_ROOT, State.NOUN_ROOT, "INFINITIVE", "mA") # İsim-fiil: etmeye
    graph.add_transition(State.VERB_ROOT, State.STOP, "GERUND_AFTER", "DIktAn") # Zarf-fiil: duyduktan
    graph.add_transition(State.VERB_ROOT, State.STOP, "GERUND_REPEATED", "(y)A") # Zarf-fiil: koşa

    # Define Terminal States
    # Roots can be terminal (e.g., "gel", "kitap")
    graph.mark_terminal(State.VERB_ROOT)
    graph.mark_terminal(State.NOUN_ROOT)
    # Most post-suffix states are terminal
    graph.mark_terminal(State.VERB_POST_TENSE)
    graph.mark_terminal(State.VERB_POST_COPULA)
    graph.mark_terminal(State.VERB_POST_PERSON)
    graph.mark_terminal(State.NOUN_POST_PLURAL)
    graph.mark_terminal(State.NOUN_POST_POSSESSIVE)
    graph.mark_terminal(State.NOUN_POST_POSSESSIVE_3)
    graph.mark_terminal(State.NOUN_POST_CASE)
    graph.mark_terminal(State.STOP)

    return graph
