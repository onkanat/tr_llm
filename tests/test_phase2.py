import unittest
import os
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.rag.vector_memory import VectorMemory

class TestPhase2(unittest.TestCase):
    def setUp(self):
        # Setup Compiler dependencies
        self.lexicon = LexiconManager()
        self.test_tsv = "test_phase2_roots.tsv"
        with open(self.test_tsv, "w", encoding="utf-8") as f:
            f.write("lemma\tpos\tattributes\n")
            f.write("gel\tVERB\t-\n")
            f.write("kitap\tNOUN\tVOICING\n")
        self.lexicon.load_from_tsv(self.test_tsv)
        self.graph = build_default_graph()
        self.compiler = CrystalCompiler(self.lexicon, self.graph)
        
        # Setup Phase 2 components
        self.vocab = Vocabulary()
        self.tokenizer = KristalTokenizer(self.compiler, self.vocab)
        self.memory = VectorMemory(vector_size=3) # Tiny dummy vector size

    def tearDown(self):
        if os.path.exists(self.test_tsv):
            os.remove(self.test_tsv)

    def test_kristal_tokenizer(self):
        # Text: "kitaplarda geleceğim"
        # kitap + lar + da -> kitap PLURAL CASE_LOC
        # gel + ecek + im -> gel TENSE_FUT PERSON_1SG
        text = "kitaplarda geleceğim"
        encoded = self.tokenizer.encode(text)
        
        self.assertGreater(len(encoded), 0)
        
        # Decode back to tags
        decoded_tags = self.tokenizer.decode(encoded)
        self.assertIn("kitap", decoded_tags)
        self.assertIn("PLURAL", decoded_tags)
        self.assertIn("CASE_LOC", decoded_tags)
        self.assertIn("gel", decoded_tags)
        self.assertIn("TENSE_FUT", decoded_tags)

    def test_vector_memory_recall(self):
        # Add dummy context
        self.memory.add_document("Ahşap işleme teknikleri", [0.9, 0.1, 0.1], {"domain": "marangoz"})
        self.memory.add_document("Derin öğrenme modelleri", [0.1, 0.9, 0.1], {"domain": "ai"})
        self.memory.add_document("Mobilya tasarımı tarihi", [0.8, 0.2, 0.1], {"domain": "marangoz"})

        # Query vector close to 'marangoz' documents
        query_vec = [0.85, 0.15, 0.1]
        results = self.memory.recall(query_vec, top_k=2)
        
        self.assertEqual(len(results), 2)
        # Should rank "Ahşap işleme" or "Mobilya tasarımı" first
        self.assertEqual(results[0]["metadata"]["domain"], "marangoz")
        self.assertEqual(results[1]["metadata"]["domain"], "marangoz")

    def test_rag_pipeline_integration(self):
        # Simulate full pipeline: Query -> Context -> Tokenize context -> Feed to LLM
        
        # 1. Store a document
        self.memory.add_document("kitaplarda geleceğim", [1.0, 0.0, 0.0], {"doc_id": 1})
        
        # 2. Recall context
        retrieved = self.memory.recall([1.0, 0.0, 0.0], top_k=1)
        context_text = retrieved[0]["text"]
        
        # 3. CrystalCompile the context into token IDs for the LLM
        token_ids = self.tokenizer.encode(context_text)
        
        self.assertGreater(len(token_ids), 0)
        # The LLM now receives clean, semantic morpheme IDs instead of fuzzy BPE tokens.
        decoded_tags = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded_tags, "kitap PLURAL CASE_LOC gel TENSE_FUT PERSON_1SG")

if __name__ == '__main__':
    unittest.main()
