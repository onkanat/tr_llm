#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import tempfile
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_tool import DocumentIngestionEngine

class TestRAGTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ephemeral memory engine for testing
        cls.engine = DocumentIngestionEngine(
            collection_name="test_rag_tool_coll",
            storage_path=None, # Ephemeral in-memory
            host=None
        )

    def test_chunk_text_short(self):
        text = "Kısa bir cümle."
        chunks = self.engine.chunk_text(text, chunk_size=300)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_paragraphs(self):
        text = "Birinci paragraf metni buradadır.\n\nİkinci paragraf metni burada yer almaktadır."
        chunks = self.engine.chunk_text(text, chunk_size=50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("Birinci paragraf", chunks[0])

    def test_read_txt_and_md(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Ahşap işleme teknikleri.")
            txt_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Başlık\n\nKırlangıç kuyruğu birleştirme.")
            md_path = f.name

        try:
            txt_docs = self.engine.read_file_content(txt_path)
            self.assertEqual(len(txt_docs), 1)
            self.assertEqual(txt_docs[0]["text"], "Ahşap işleme teknikleri.")

            md_docs = self.engine.read_file_content(md_path)
            self.assertEqual(len(md_docs), 1)
            self.assertIn("Kırlangıç kuyruğu", md_docs[0]["text"])
        finally:
            if os.path.exists(txt_path): os.remove(txt_path)
            if os.path.exists(md_path): os.remove(md_path)

    def test_read_json_and_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([{"text": "Birinci JSON kaydı."}, {"content": "İkinci JSON kaydı."}], f)
            json_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"text": "JSONL satırı 1"}) + "\n")
            f.write(json.dumps({"content": "JSONL satırı 2"}) + "\n")
            jsonl_path = f.name

        try:
            json_docs = self.engine.read_file_content(json_path)
            self.assertEqual(len(json_docs), 2)

            jsonl_docs = self.engine.read_file_content(jsonl_path)
            self.assertEqual(len(jsonl_docs), 2)
        finally:
            if os.path.exists(json_path): os.remove(json_path)
            if os.path.exists(jsonl_path): os.remove(jsonl_path)

    def test_ingest_and_search(self):
        docs = [
            {"text": "Zıvana dili ve zıvana yuvası mobilya çerçeve köşe bağlantılarında kullanılır.", "source": "zivana_doc.txt"},
            {"text": "Elma ve armut meyve bahçelerinde yetişen lezzetli ağaç ürünleridir.", "source": "meyve_doc.txt"}
        ]
        total_chunks = self.engine.ingest_documents(docs)
        self.assertGreaterEqual(total_chunks, 2)

        results = self.engine.search("zıvana köşesi", top_k=1)
        self.assertGreater(len(results), 0)
        self.assertIn("zıvana", results[0]["text"].lower())

if __name__ == "__main__":
    unittest.main()
