import unittest
from src.compiler.morphotactics import State, MorphotacticsGraph, build_default_graph

class TestMorphotacticsGraph(unittest.TestCase):
    def setUp(self):
        self.graph = build_default_graph()

    def test_terminal_states(self):
        # Verify valid endings
        self.assertTrue(self.graph.is_terminal(State.VERB_ROOT))
        self.assertTrue(self.graph.is_terminal(State.NOUN_POST_CASE))
        self.assertFalse(self.graph.is_terminal(State.START))

    def test_verb_transitions(self):
        # Get transitions from VERB_ROOT
        transitions = self.graph.get_valid_transitions(State.VERB_ROOT)
        self.assertGreater(len(transitions), 0)
        
        # Verify TENSE_FUT is an option
        tense_fut_transition = next((t for t in transitions if t.affix_id == "TENSE_FUT"), None)
        self.assertIsNotNone(tense_fut_transition)
        self.assertEqual(tense_fut_transition.to_state, State.VERB_POST_TENSE)

    def test_noun_transitions(self):
        # A noun can go to PLURAL, POSSESSIVE, or CASE directly
        transitions = self.graph.get_valid_transitions(State.NOUN_ROOT)
        target_states = [t.to_state for t in transitions]
        
        self.assertIn(State.NOUN_POST_PLURAL, target_states)
        self.assertIn(State.NOUN_POST_POSSESSIVE, target_states)
        self.assertIn(State.NOUN_POST_CASE, target_states)

    def test_invalid_transitions(self):
        # Nouns shouldn't get verb tenses
        transitions = self.graph.get_valid_transitions(State.NOUN_ROOT)
        verb_states = [State.VERB_POST_TENSE, State.VERB_POST_PERSON]
        for t in transitions:
            self.assertNotIn(t.to_state, verb_states)

    def test_sequential_transitions(self):
        # Test sequential correctness manually
        # ROOT -> PLURAL -> CASE
        plural_transitions = self.graph.get_valid_transitions(State.NOUN_ROOT)
        plural_t = next((t for t in plural_transitions if t.affix_id == "PLURAL"), None)
        self.assertIsNotNone(plural_t)
        
        case_transitions = self.graph.get_valid_transitions(plural_t.to_state)
        case_t = next((t for t in case_transitions if t.affix_id == "CASE_LOC"), None)
        self.assertIsNotNone(case_t)
        self.assertEqual(case_t.to_state, State.NOUN_POST_CASE)

if __name__ == '__main__':
    unittest.main()
