#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kristal-Vektörel Mimarisi - RAG Belge Yükleme ve Yönetim Aracı (rag_tool.py)
Test grubunun harici dökümanları (TXT, MD, JSON, CSV, PDF) kolayca vektörel belleğe
yüklemesini, arama testi yapmasını ve belleği yönetmesini sağlar.
"""

import os
import sys
import json
import glob
import re
import argparse
from typing import List, Dict, Any, Optional

# Kök dizini yola ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector

# ANSI Renk Kodları
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_GRAY = "\033[90m"


class DocumentIngestionEngine:
    """Belgeleri okuyan, parçalayan (chunking) ve vektörel belleğe aktaran motor."""

    def __init__(
        self,
        collection_name: str = "kristal_bellek",
        storage_path: str = "data/qdrant_db",
        host: Optional[str] = None,
        port: int = 6333,
        vocab_path: str = "data/vocab.json",
        lexicon_path: str = "data/lexicon/roots.tsv"
    ):
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.host = host
        self.port = port
        
        # 1. Sözlük ve Derleyici Başlatma
        self.vocab = Vocabulary()
        if os.path.exists(vocab_path):
            self.vocab.load(vocab_path)
        else:
            raise FileNotFoundError(f"Vocab dosyası bulunamadı: {vocab_path}")
            
        self.lexicon = LexiconManager()
        if os.path.exists(lexicon_path):
            self.lexicon.load_from_tsv(lexicon_path)
        else:
            raise FileNotFoundError(f"Leksikon dosyası bulunamadı: {lexicon_path}")
            
        self.compiler = CrystalCompiler(self.lexicon, build_default_graph())
        self.tokenizer = KristalTokenizer(self.compiler, self.vocab)
        
        # 2. Vektörel Bellek Başlatma
        self.memory = VectorMemory(
            collection_name=self.collection_name,
            vector_size=768,
            host=self.host,
            port=self.port,
            storage_path=self.storage_path
        )

    def read_file_content(self, file_path: str) -> List[Dict[str, str]]:
        """Verilen dosyayı uygun formata göre okuyup metin listesi döner."""
        ext = os.path.splitext(file_path)[1].lower()
        results = []
        
        # 1. TXT ve Markdown
        if ext in [".txt", ".md", ".markdown"]:
            encodings = ["utf-8", "utf-8-sig", "iso-8859-9", "windows-1254"]
            content = None
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if content:
                results.append({"text": content, "source": os.path.basename(file_path)})

        # 2. JSON ve JSONL
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, str):
                        results.append({"text": item, "source": f"{os.path.basename(file_path)}#{idx}"})
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("icerik") or item.get("metin") or item.get("output") or ""
                        if text:
                            results.append({"text": text, "source": f"{os.path.basename(file_path)}#{idx}"})
            elif isinstance(data, dict):
                text = data.get("text") or data.get("content") or data.get("icerik") or data.get("metin") or ""
                if text:
                    results.append({"text": text, "source": os.path.basename(file_path)})

        elif ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        text = item.get("text") or item.get("content") or item.get("icerik") or item.get("metin") or item.get("output") or ""
                        if text:
                            results.append({"text": text, "source": f"{os.path.basename(file_path)}:L{idx+1}"})
                    except Exception:
                        continue

        # 3. CSV
        elif ext == ".csv":
            import csv
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    row_text = " ".join([c.strip() for c in row if c.strip()])
                    if row_text:
                        results.append({"text": row_text, "source": f"{os.path.basename(file_path)}:R{idx+1}"})

        # 4. PDF (Opsiyonel kütüphane kontrolü)
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                full_text = []
                for page_idx, page in enumerate(reader.pages):
                    p_text = page.extract_text()
                    if p_text:
                        full_text.append(p_text)
                if full_text:
                    results.append({"text": "\n\n".join(full_text), "source": os.path.basename(file_path)})
            except ImportError:
                print(f"  {C_YELLOW}Uyarı: PDF okumak için 'pypdf' kütüphanesi gereklidir. Metni .txt veya .md formatında yükleyebilirsiniz.{C_RESET}")

        return results

    def chunk_text(self, text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
        """Metni anlam bütünlüğünü koruyacak şekilde paragraflara ve cümlelere böler."""
        clean_text = text.strip()
        if not clean_text:
            return []
            
        if len(clean_text) <= chunk_size:
            return [clean_text]
            
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        chunks = []
        
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para)
                continue
                
            # Paragraf çok uzunsa cümlelere ayır
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sent in sentences:
                sent = sent.strip()
                if not sent: continue
                
                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    current_chunk = (current_chunk + " " + sent).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
                    
            if current_chunk:
                chunks.append(current_chunk)
                
        # Çok kısa veya boş parçaları filtrele
        final_chunks = [c for c in chunks if len(c) >= 15]
        return final_chunks if final_chunks else [clean_text[:chunk_size]]

    def ingest_documents(self, documents: List[Dict[str, str]], title: Optional[str] = None, batch_size: int = 32) -> int:
        """Belgeleri parçalar, morfemik vektörlerini üretir ve Qdrant'a yükler."""
        total_chunks = 0
        all_texts = []
        all_metas = []
        
        for doc in documents:
            source = doc.get("source", "manual_input")
            raw_text = doc["text"]
            chunks = self.chunk_text(raw_text)
            
            for c_idx, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metas.append({
                    "source": source,
                    "title": title or source,
                    "chunk_index": c_idx,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk)
                })
                
        if not all_texts:
            return 0
            
        print(f"{C_CYAN}-> {len(all_texts)} parça morfemik olarak işleniyor ve vektörleştiriliyor...{C_RESET}")
        
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_metas = all_metas[i:i + batch_size]
            
            batch_dense = []
            batch_sparse = []
            batch_meta_with_tags = []
            
            for t, m in zip(batch_texts, batch_metas):
                token_ids = self.tokenizer.encode(t)
                tags = self.tokenizer.decode(token_ids)
                dense_v = generate_kristal_vector(token_ids, tags)
                sparse_v = generate_sparse_vector(token_ids, tags)
                
                m["crystal_tags"] = tags
                batch_dense.append(dense_v)
                batch_sparse.append(sparse_v)
                batch_meta_with_tags.append(m)
                
            self.memory.add_documents_batch(batch_texts, batch_dense, batch_sparse, batch_meta_with_tags)
            total_chunks += len(batch_texts)
            print(f"  {C_GREEN}✓{C_RESET} {total_chunks}/{len(all_texts)} parça indekslendi.")
            
        return total_chunks

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Sorguyu Kristal morfemlerine ayırıp hibrit RRF araması yapar."""
        q_ids = self.tokenizer.encode(query)
        q_tags = self.tokenizer.decode(q_ids)
        dense_q = generate_kristal_vector(q_ids, q_tags)
        sparse_q = generate_sparse_vector(q_ids, q_tags)
        
        return self.memory.hybrid_recall(dense_q, sparse_q, top_k=top_k, query_tags=q_tags)


def print_banner():
    print(f"\n{C_CYAN}{C_BOLD}======================================================================{C_RESET}")
    print(f"{C_CYAN}{C_BOLD} 🚀 KRİSTAL-VEKTÖREL RAG BELGE YÖNETİM VE YÜKLEME ARACI (rag_tool.py) {C_RESET}")
    print(f"{C_CYAN}{C_BOLD}======================================================================{C_RESET}")


def interactive_menu(engine: DocumentIngestionEngine):
    """Test grubu kullanıcıları için renkli, adımlı terminal arayüzü."""
    while True:
        print_banner()
        doc_count = engine.memory.get_document_count()
        print(f"  {C_BOLD}Bağlantı Türü:{C_RESET} {C_GREEN}{engine.memory.storage_type}{C_RESET}")
        print(f"  {C_BOLD}Aktif Koleksiyon:{C_RESET} {C_YELLOW}{engine.collection_name}{C_RESET}")
        print(f"  {C_BOLD}Bellekteki Parça/Belge Sayısı:{C_RESET} {C_MAGENTA}{doc_count}{C_RESET}")
        print("-" * 70)
        print(f"  {C_BOLD}[1]{C_RESET} 📄 Tek Dosya Yükle (.txt, .md, .json, .csv, .pdf)")
        print(f"  {C_BOLD}[2]{C_RESET} 📁 Klasör Yükle (Tüm belgeleri otomatik tara)")
        print(f"  {C_BOLD}[3]{C_RESET} ✍️  Doğrudan Metin / Not Ekle")
        print(f"  {C_BOLD}[4]{C_RESET} 🔍 RAG Arama ve Eşleşme Testi Yap")
        print(f"  {C_BOLD}[5]{C_RESET} 📊 Bellek Durumu ve Belgeleri Önizle")
        print(f"  {C_BOLD}[6]{C_RESET} 🗑️  Belleği Sıfırla (Koleksiyonu Temizle)")
        print(f"  {C_BOLD}[0]{C_RESET} 🚪 Çıkış")
        print("-" * 70)
        
        choice = input(f"{C_BOLD}Lütfen bir işlem seçin [0-6]: {C_RESET}").strip()
        
        if choice == "1":
            file_path = input(f"\n{C_BOLD}Yüklenecek dosya yolu: {C_RESET}").strip()
            if not os.path.exists(file_path):
                print(f"{C_RED}Hata: '{file_path}' dosyası bulunamadı!{C_RESET}")
                input("\nDevam etmek için Enter'a basın...")
                continue
            title = input(f"{C_BOLD}Belge için özel başlık (boş bırakılabilir): {C_RESET}").strip() or None
            docs = engine.read_file_content(file_path)
            if not docs:
                print(f"{C_RED}Hata: Dosyadan metin okunamadı.{C_RESET}")
            else:
                chunks = engine.ingest_documents(docs, title=title)
                print(f"\n{C_GREEN}{C_BOLD}✓ Başarılı:{C_RESET} Toplam {chunks} parça vektörel belleğe yüklendi.")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "2":
            dir_path = input(f"\n{C_BOLD}Taranacak klasör yolu: {C_RESET}").strip()
            if not os.path.isdir(dir_path):
                print(f"{C_RED}Hata: '{dir_path}' geçerli bir klasör değil!{C_RESET}")
                input("\nDevam etmek için Enter'a basın...")
                continue
                
            supported_exts = ["*.txt", "*.md", "*.json", "*.jsonl", "*.csv"]
            files = []
            for ext in supported_exts:
                files.extend(glob.glob(os.path.join(dir_path, "**", ext), recursive=True))
            files = list(set(files))
            
            if not files:
                print(f"{C_YELLOW}Uyarı: Klasörde desteklenen formatta dosya bulunamadı.{C_RESET}")
            else:
                print(f"{C_CYAN}{len(files)} dosya bulundu. Yükleniyor...{C_RESET}")
                all_docs = []
                for f in files:
                    all_docs.extend(engine.read_file_content(f))
                chunks = engine.ingest_documents(all_docs)
                print(f"\n{C_GREEN}{C_BOLD}✓ Başarılı:{C_RESET} {len(files)} dosyadan toplam {chunks} parça belleğe eklendi.")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "3":
            print(f"\n{C_BOLD}Eklenecek metni girin (Bitirmek için Enter'a basın):{C_RESET}")
            text = input("> ").strip()
            if not text:
                print(f"{C_YELLOW}Boş metin girildi, işlem iptal edildi.{C_RESET}")
            else:
                title = input(f"{C_BOLD}Not / Belge Başlığı: {C_RESET}").strip() or "Manuel Not"
                docs = [{"text": text, "source": "manuel_giris"}]
                chunks = engine.ingest_documents(docs, title=title)
                print(f"\n{C_GREEN}{C_BOLD}✓ Başarılı:{C_RESET} Notunuz {chunks} parça olarak belleğe eklendi.")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "4":
            query = input(f"\n{C_BOLD}Aramak istediğiniz soru veya anahtar kelimeler: {C_RESET}").strip()
            if not query: continue
            
            print(f"\n{C_CYAN}[RAG] Hibrit arama yapılıyor...{C_RESET}")
            results = engine.search(query, top_k=3)
            
            if not results:
                print(f"{C_YELLOW}Eşleşen belge bulunamadı.{C_RESET}")
            else:
                print(f"\n{C_BOLD}En İyi Eşleşen Sonuçlar:{C_RESET}")
                for idx, res in enumerate(results):
                    meta = res.get("metadata", {})
                    src = meta.get("source", "Bilinmiyor")
                    title = meta.get("title", "")
                    score = res.get("score", 0.0)
                    text = res.get("text", "")
                    tags = meta.get("crystal_tags", "")
                    
                    print(f"\n  {C_YELLOW}--- Sonuç #{idx+1} [Skor: {score:.4f}] ---{C_RESET}")
                    print(f"  {C_BOLD}Kaynak:{C_RESET} {C_MAGENTA}{src}{C_RESET} (Başlık: {title})")
                    print(f"  {C_BOLD}Metin:{C_RESET} {text}")
                    if tags:
                        print(f"  {C_GRAY}Morfemler: {tags[:120]}...{C_RESET}")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "5":
            print(f"\n{C_BOLD}--- Bellek Durumu ve Kayıtlar ---{C_RESET}")
            records = engine.memory.list_documents(limit=10)
            if not records:
                print(f"{C_YELLOW}Koleksiyon henüz boş. Dosya yükleyebilirsiniz.{C_RESET}")
            else:
                print(f"Son yüklenen {len(records)} kayıt listeleniyor:\n")
                for r in records:
                    p = r["payload"]
                    src = p.get("source", "-")
                    txt = p.get("text", "")
                    print(f"  • {C_CYAN}[ID: {r['id']}]{C_RESET} {C_MAGENTA}{src}{C_RESET}: {txt[:100]}...")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "6":
            confirm = input(f"\n{C_RED}{C_BOLD}DİKKAT:{C_RESET} '{engine.collection_name}' koleksiyonundaki TÜM belgeler silinecektir. Emin misiniz? (e/h): ").strip().lower()
            if confirm in ["e", "evet", "y", "yes"]:
                engine.memory.recreate_collection()
                print(f"{C_GREEN}Koleksiyon sıfırlandı ve temizlendi.{C_RESET}")
            else:
                print(f"{C_YELLOW}İşlem iptal edildi.{C_RESET}")
            input("\nDevam etmek için Enter'a basın...")

        elif choice == "0":
            print(f"\n{C_GREEN}Kristal-Vektörel RAG aracından çıkıldı. İyi çalışmalar!{C_RESET}\n")
            break


def main():
    parser = argparse.ArgumentParser(description="Kristal-Vektörel RAG Belge Yükleme ve Yönetim Aracı")
    subparsers = parser.add_subparsers(dest="command", help="Çalıştırılacak komut")

    # Genel Parametreler
    parser.add_argument("--collection", default="kristal_bellek", help="Qdrant koleksiyon adı")
    parser.add_argument("--storage", default="data/qdrant_db", help="Yerel gömülü Qdrant veritabanı dizini")
    parser.add_argument("--host", default=None, help="Harici Qdrant sunucu adresi (örn. localhost)")
    parser.add_argument("--port", type=int, default=6333, help="Harici Qdrant portu")

    # 1. add komutu
    add_parser = subparsers.add_parser("add", help="Belge veya metin yükle")
    add_parser.add_argument("--file", help="Yüklenecek tek dosya yolu")
    add_parser.add_argument("--dir", help="Taranacak klasör yolu")
    add_parser.add_argument("--text", help="Doğrudan eklenecek metin")
    add_parser.add_argument("--title", help="Belge başlığı (opsiyonel)")

    # 2. search komutu
    search_parser = subparsers.add_parser("search", help="RAG araması yap")
    search_parser.add_argument("query", help="Arama yapılacak soru veya ifade")
    search_parser.add_argument("--top_k", type=int, default=3, help="Dönecek sonuç sayısı")

    # 3. list komutu
    list_parser = subparsers.add_parser("list", help="Bellekteki belgeleri listele")
    list_parser.add_argument("--limit", type=int, default=15, help="Maksimum listelenecek kayıt")

    # 4. status komutu
    subparsers.add_parser("status", help="Bellek bağlantı durumunu göster")

    # 5. reset komutu
    subparsers.add_parser("reset", help="Koleksiyonu temizle ve sıfırla")

    args = parser.parse_args()

    # Eğer hiçbir argüman verilmemişse Etkileşimli Menü'yü başlat
    if not args.command:
        engine = DocumentIngestionEngine(
            collection_name=args.collection,
            storage_path=args.storage,
            host=args.host,
            port=args.port
        )
        interactive_menu(engine)
        return

    # CLI Komutları
    engine = DocumentIngestionEngine(
        collection_name=args.collection,
        storage_path=args.storage,
        host=args.host,
        port=args.port
    )

    if args.command == "add":
        if args.file:
            if not os.path.exists(args.file):
                print(f"Hata: '{args.file}' bulunamadı.")
                sys.exit(1)
            docs = engine.read_file_content(args.file)
            chunks = engine.ingest_documents(docs, title=args.title)
            print(f"Başarılı: '{args.file}' dosyasından {chunks} parça belleğe eklendi.")
        elif args.dir:
            if not os.path.isdir(args.dir):
                print(f"Hata: '{args.dir}' bir klasör değil.")
                sys.exit(1)
            supported_exts = ["*.txt", "*.md", "*.json", "*.jsonl", "*.csv"]
            files = []
            for ext in supported_exts:
                files.extend(glob.glob(os.path.join(args.dir, "**", ext), recursive=True))
            all_docs = []
            for f in set(files):
                all_docs.extend(engine.read_file_content(f))
            chunks = engine.ingest_documents(all_docs, title=args.title)
            print(f"Başarılı: {len(set(files))} dosyadan {chunks} parça belleğe eklendi.")
        elif args.text:
            docs = [{"text": args.text, "source": "cli_text"}]
            chunks = engine.ingest_documents(docs, title=args.title or "CLI Notu")
            print(f"Başarılı: Metin {chunks} parça olarak belleğe eklendi.")
        else:
            print("Hata: --file, --dir veya --text parametrelerinden biri zorunludur.")
            sys.exit(1)

    elif args.command == "search":
        print(f"Sorgu: '{args.query}' (top_k={args.top_k})")
        results = engine.search(args.query, top_k=args.top_k)
        if not results:
            print("Eşleşen belge bulunamadı.")
        else:
            for idx, r in enumerate(results):
                src = r["metadata"].get("source", "Bilinmiyor")
                print(f"[{idx+1}] Skor: {r['score']:.4f} | Kaynak: {src}\n    {r['text']}\n")

    elif args.command == "list":
        records = engine.memory.list_documents(limit=args.limit)
        print(f"Toplam Belge Sayısı: {engine.memory.get_document_count()}")
        for r in records:
            p = r["payload"]
            print(f"- [ID: {r['id']}] {p.get('source', '')}: {p.get('text', '')[:120]}...")

    elif args.command == "status":
        print(f"Koleksiyon: {engine.collection_name}")
        print(f"Depolama Türü: {engine.memory.storage_type}")
        print(f"Toplam Parça: {engine.memory.get_document_count()}")

    elif args.command == "reset":
        engine.memory.recreate_collection()
        print(f"'{engine.collection_name}' koleksiyonu başarıyla sıfırlandı.")


if __name__ == "__main__":
    main()
