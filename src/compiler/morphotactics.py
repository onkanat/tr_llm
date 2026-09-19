from typing import Dict, List, Set

class State:
    START = "S_START"
    
    # Root States
    VERB_ROOT = "S_VERB_ROOT"
    NOUN_ROOT = "S_NOUN_ROOT"
    ADJ_ROOT = "S_ADJ_ROOT"
    ADV_ROOT = "S_ADV_ROOT"
    
    # Verb Voice States (Çatı)
    VERB_VOICE_CAUSATIVE = "S_VERB_VOICE_CAUSATIVE"
    VERB_VOICE_PASSIVE = "S_VERB_VOICE_PASSIVE"
    
    # Verb Modals
    VERB_POTENTIAL = "S_VERB_POTENTIAL"
    
    # Verb Inflection States
    VERB_NEGATION = "S_VERB_NEGATION"
    VERB_POST_TENSE = "S_VERB_POST_TENSE"
    VERB_POST_COPULA = "S_VERB_POST_COPULA"
    VERB_POST_PERSON = "S_VERB_POST_PERSON"
    
    # Noun Inflection States
    NOUN_POST_PLURAL = "S_NOUN_POST_PLURAL"
    NOUN_POST_POSSESSIVE = "S_NOUN_POST_POSSESSIVE"
    NOUN_POST_POSSESSIVE_3 = "S_NOUN_POST_POSSESSIVE_3"
    NOUN_POST_CASE = "S_NOUN_POST_CASE"
    
    STOP = "S_STOP"

class Transition:
    def __init__(self, to_state: str, affix_id: str, affix_template: str, attributes: str = "-"):
        self.to_state = to_state
        self.affix_id = affix_id
        self.affix_template = affix_template
        self.attributes = attributes

class MorphotacticsGraph:
    def __init__(self):
        self.transitions: Dict[str, List[Transition]] = {}
        self.terminal_states: Set[str] = set()

    def add_transition(self, from_state: str, to_state: str, affix_id: str, affix_template: str, attributes: str = "-"):
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append(Transition(to_state, affix_id, affix_template, attributes))

    def mark_terminal(self, state: str):
        self.terminal_states.add(state)

    def get_valid_transitions(self, current_state: str) -> List[Transition]:
        return self.transitions.get(current_state, [])

    def is_terminal(self, state: str) -> bool:
        return state in self.terminal_states

def build_default_graph() -> MorphotacticsGraph:
    graph = MorphotacticsGraph()

    # --- VERB PATH (FİİL YOLU) ---
    
    # 1. Voice (Çatı)
    graph.add_transition(State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, "VOICE_CAUS_t", "t", "-")
    graph.add_transition(State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, "VOICE_CAUS_DIr", "DIr", "-")
    graph.add_transition(State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_CAUSATIVE, "VOICE_CAUS_DIr", "DIr", "-")
    
    graph.add_transition(State.VERB_ROOT, State.VERB_VOICE_PASSIVE, "VOICE_PASS_Il", "Il", "-")
    graph.add_transition(State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_PASSIVE, "VOICE_PASS_Il", "Il", "-")
    
    # 1.5 Potential / Impotential (Yeterlilik / Eksi-Yeterlilik)
    verb_pre_pot_states = [State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_PASSIVE]
    for from_state in verb_pre_pot_states:
        # Potential loops back to VERB_ROOT so it can take neg or tense normally
        graph.add_transition(from_state, State.VERB_ROOT, "POTENTIAL", "(y)Abil", "-")
        # Impotential goes straight to negation
        graph.add_transition(from_state, State.VERB_NEGATION, "IMPOTENTIAL_NEG", "(y)AmA", "-")
    
    # 2. Negation (Olumsuzluk)
    verb_pre_neg_states = [State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_PASSIVE]
    for from_state in verb_pre_neg_states:
        graph.add_transition(from_state, State.VERB_NEGATION, "NEG", "mA", "-")
    
    # 3. Tense (Zaman/Kip)
    verb_pre_tense_states = [State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_PASSIVE, State.VERB_NEGATION]
    
    tenses = [
        ("TENSE_FUT", "(y)AcAk", "VOICING"),
        ("TENSE_PROG", "(y)Iyor", "-"),
        ("TENSE_PAST", "DI", "-"),
        ("TENSE_EVIDENTIAL", "mIş", "-"),
        ("TENSE_AORIST", "(I)r", "-"),
        ("TENSE_AORIST_VOWEL", "Ar", "-"),
        ("TENSE_COND", "sA", "-"),
        ("TENSE_NECESS", "mAlI", "-"),
        ("TENSE_OPTATIVE", "(y)A", "-"),
    ]
    
    for from_state in verb_pre_tense_states:
        for tid, template, attr in tenses:
            graph.add_transition(from_state, State.VERB_POST_TENSE, tid, template, attr)

    # Special Aorist Negation (gel-mez)
    graph.add_transition(State.VERB_NEGATION, State.VERB_POST_TENSE, "TENSE_AORIST_NEG", "z", "-")

    # 4. Copula (Ek-fiil / Birleşik Zaman)
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_PAST", "(y)DI", "-")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_EVIDENTIAL", "(y)mIş", "-")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_COND", "(y)sA", "-")
    graph.add_transition(State.VERB_POST_TENSE, State.VERB_POST_COPULA, "COPULA_AORIST", "DIr", "-")
    
    # 5. Person (Şahıs)
    person_suffixes = [
        ("PERSON_1SG", "(I)m", "-"),
        ("PERSON_2SG", "sIn", "-"),
        ("PERSON_3SG", "", "-"),
        ("PERSON_1PL", "(I)z", "-"),
        ("PERSON_2PL", "sInIz", "-"),
        ("PERSON_3PL", "lAr", "-"),
    ]
    
    for from_state in [State.VERB_POST_TENSE, State.VERB_POST_COPULA]:
        for tid, template, attr in person_suffixes:
            graph.add_transition(from_state, State.VERB_POST_PERSON, tid, template, attr)

    # --- NOUN PATH (İSİM YOLU) ---
    
    graph.add_transition(State.NOUN_ROOT, State.NOUN_POST_PLURAL, "PLURAL", "lAr", "-")
    
    # Possessives
    possessives = [
        ("POSS_1SG", "(I)m", "-"),
        ("POSS_2SG", "(I)n", "-"),
        ("POSS_3SG", "(s)I", "-"),
        ("POSS_1PL", "(I)mIz", "-"),
        ("POSS_2PL", "(I)nIz", "-"),
        ("POSS_3PL", "lArI", "-"),
    ]
    
    for from_state in [State.NOUN_ROOT, State.NOUN_POST_PLURAL]:
        for tid, template, attr in possessives:
            target = State.NOUN_POST_POSSESSIVE if "3" not in tid else State.NOUN_POST_POSSESSIVE_3
            graph.add_transition(from_state, target, tid, template, attr)

    # Cases
    cases = [
        ("CASE_LOC", "DA", "-"),
        ("CASE_ABL", "DAn", "-"),
        ("CASE_ACC", "(y)I", "-"),
        ("CASE_DAT", "(y)A", "-"),
        ("CASE_GEN", "(n)In", "-"),
        ("CASE_INS", "(y)lA", "-"),
        ("CASE_EQU", "CA", "-"),
    ]
    
    for from_state in [State.NOUN_ROOT, State.NOUN_POST_PLURAL, State.NOUN_POST_POSSESSIVE]:
        for tid, template, attr in cases:
            graph.add_transition(from_state, State.NOUN_POST_CASE, tid, template, attr)
            
    # Special buffer for 3rd person possessive + case (n buffer)
    n_cases = [
        ("CASE_LOC_N", "nDA", "-"),
        ("CASE_ABL_N", "nDAn", "-"),
        ("CASE_ACC_N", "nI", "-"),
        ("CASE_DAT_N", "nA", "-"),
        ("CASE_GEN_N", "nIn", "-"),
        ("CASE_EQU_N", "nCA", "-"),
        # D3-A (T-0080): 3. tekil iyelikten sonra ARAÇ DURUMU eksikti ⇒
        # 'adıyla'/'aracıyla' ÇÖZÜLEMİYORDU (0 yol). Kimlik `CASE_INS` olarak
        # YENİDEN KULLANILIR, çünkü ekin YÜZEYİ çıplak addakiyle AYNIDIR
        # (`(y)lA`; y-buffer, n-buffer değil) — yeni bir id, sözlükte AYNI ek için
        # İKİNCİ bir morfem jetonu açardı. Yalnız bu grupta `n` öneki kuraldı;
        # ondan sapma bilinçli ve burada beyan edilmiştir.
        ("CASE_INS", "(y)lA", "-"),
    ]
    for tid, template, attr in n_cases:
        graph.add_transition(State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE, tid, template, attr)

    # Relative Suffix (Aitlik eki: evdeki, masadaki, edebiyattaki, akşamki)
    graph.add_transition(State.NOUN_POST_CASE, State.ADJ_ROOT, "REL_ki", "ki", "-")
    graph.add_transition(State.NOUN_ROOT, State.ADJ_ROOT, "REL_ki", "ki", "-")

    # Noun Copula (İsim soylu sözcüklerde ek-fiil bildirme eki)
    noun_copula_states = [State.NOUN_ROOT, State.ADJ_ROOT, State.NOUN_POST_PLURAL, State.NOUN_POST_POSSESSIVE, State.NOUN_POST_POSSESSIVE_3, State.NOUN_POST_CASE]
    for from_state in noun_copula_states:
        graph.add_transition(from_state, State.STOP, "COPULA_AORIST", "DIr", "-")
        # For past copula on nouns (e.g. güzeldi)
        graph.add_transition(from_state, State.VERB_POST_COPULA, "COPULA_PAST", "(y)DI", "-")
        graph.add_transition(from_state, State.VERB_POST_COPULA, "COPULA_EVIDENTIAL", "(y)mIş", "-")
        graph.add_transition(from_state, State.VERB_POST_COPULA, "COPULA_COND", "(y)sA", "-")

    # --- DERIVATION (TÜRETİM) ---
    
    # Verb to Noun/Stop (Fiilimsi ve İsim Yapım)
    v2n = [
        ("INF_mAk", "mAk", "-"),
        ("INF_mA", "mA", "-"),
        ("INF_Iş", "Iş", "-"),
        ("PART_An", "(y)An", "-"),
        ("PART_DIk", "DIk", "VOICING"),
    ]
    verb_pre_deriv_states = [State.VERB_ROOT, State.VERB_VOICE_CAUSATIVE, State.VERB_VOICE_PASSIVE, State.VERB_NEGATION]
    for from_state in verb_pre_deriv_states:
        for tid, template, attr in v2n:
            graph.add_transition(from_state, State.NOUN_ROOT, tid, template, attr)

    # Gerunds (Zarf-fiiller) -> They go to STOP
    gerunds = [
        ("GERUND_ArAk", "(y)ArAk", "-"),
        ("GERUND_IncA", "(y)IncA", "-"),
        ("GERUND_Ip", "(y)Ip", "-"),
        ("GERUND_mAdAn", "mAdAn", "-"),
        ("GERUND_AlI", "(y)AlI", "-"),
    ]
    for from_state in verb_pre_deriv_states:
        for tid, template, attr in gerunds:
            graph.add_transition(from_state, State.STOP, tid, template, attr)
            
    # -ken gerund (attaches to tense or noun states)
    graph.add_transition(State.VERB_POST_TENSE, State.STOP, "GERUND_KEN", "ken", "-")
    for from_state in noun_copula_states:
        graph.add_transition(from_state, State.STOP, "GERUND_KEN", "(y)ken", "-")

    # Noun to Noun / Adj
    n2nx = [
        ("DERIV_lIk", "lIk", "VOICING"),
        ("DERIV_lI", "lI", "-"),
        ("DERIV_sIz", "sIz", "-"),
        ("DERIV_CI", "CI", "-"),
    ]
    for from_state in [State.NOUN_ROOT, State.NOUN_POST_PLURAL, State.NOUN_POST_POSSESSIVE]:
        for tid, template, attr in n2nx:
            graph.add_transition(from_state, State.NOUN_ROOT, tid, template, attr)
            graph.add_transition(from_state, State.ADJ_ROOT, tid, template, attr)

    # Noun/Adj to Verb
    graph.add_transition(State.NOUN_ROOT, State.VERB_ROOT, "DERIV_lA", "lA", "-")
    graph.add_transition(State.ADJ_ROOT, State.VERB_ROOT, "DERIV_lAş", "lAş", "-")
    graph.add_transition(State.ADJ_ROOT, State.VERB_ROOT, "DERIV_lAn", "lAn", "-")

    # --- TERMINAL STATES ---
    graph.mark_terminal(State.VERB_ROOT)
    graph.mark_terminal(State.NOUN_ROOT)
    graph.mark_terminal(State.ADJ_ROOT)
    graph.mark_terminal(State.ADV_ROOT)
    graph.mark_terminal(State.VERB_POST_TENSE)
    graph.mark_terminal(State.VERB_POST_COPULA)
    graph.mark_terminal(State.VERB_POST_PERSON)
    graph.mark_terminal(State.NOUN_POST_PLURAL)
    graph.mark_terminal(State.NOUN_POST_POSSESSIVE)
    graph.mark_terminal(State.NOUN_POST_POSSESSIVE_3)
    graph.mark_terminal(State.NOUN_POST_CASE)
    graph.mark_terminal(State.STOP)

    return graph
