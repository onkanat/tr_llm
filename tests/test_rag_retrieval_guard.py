#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_tool import DocumentIngestionEngine
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector

class TestRAGRetrievalGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DocumentIngestionEngine(
            collection_name="test_retrieval_guard_coll",
            storage_path=None, # Ephemeral in-memory
            host=None
        )
        docs = [
            {"text": "Meşe ağacı masif mobilya ve parke üretiminde kullanılır. Aşınmaya karşı dirençlidir.", "source": "ahsap_doc.txt"},
            {"text": "Kırlangıç kuyruğu geçme çekmecelerde kullanılır.", "source": "zivana_doc.txt"}
        ]
        cls.engine.ingest_documents(docs)

    def test_matching_query_retrieval(self):
        results = self.engine.search("Meşe ağacı ile yapılan işler", top_k=2)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("meşe", top["text"].lower())
        self.assertTrue(top.get("has_root_match", False))
        self.assertGreaterEqual(top["score"], 0.40)
        self.assertIn("meşe", top.get("matching_roots", []))

    def test_unrelated_query_penalty(self):
        results = self.engine.search("Kuantum parçacık fiziği atom", top_k=2)
        self.assertGreater(len(results), 0)
        top = results[0]
        # Because neither kuantum nor parça nor fizik matches ahşap doc
        self.assertFalse(top.get("has_root_match", True))
        self.assertLess(top["score"], 0.40)

if __name__ == "__main__":
    unittest.main()
