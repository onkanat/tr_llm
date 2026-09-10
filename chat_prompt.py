#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
from datetime import datetime, timezone
import torch

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.compiler.decompiler import MorphemeDecompiler
from src.rag.vector_memory import VectorMemory, generate_kristal_vector, generate_sparse_vector
from src.rag.merak import CuriosityEngine
from src.llm.router import TriModalRouter
from src.rag.epistemic_agent import EpistemicCuriosityAgent
from src.gateway.agent_gateway import AgentGateway
from src.gateway.pedagogical_supervisor import PedagogicalSupervisor

# ANSI Color Codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_GRAY = "\033[90m"

def color_token(tag: str) -> str:
    """Formats the token tag with beautiful colors depending on its type."""
    if tag in ["<BOS>", "<EOS>", "<PAD>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>", "</OUTPUT>"]:
        return f"{C_RED}{C_BOLD}{tag}{C_RESET}"
    if tag == "<UNK>":
        return f"{C_RED}{C_BOLD}{tag}{C_RESET}"
    if tag == "<NUMBER>":
        return f"{C_YELLOW}{tag}{C_RESET}"
    if tag == "<PROPER_NOUN>":
        return f"{C_GREEN}{C_BOLD}{tag}{C_RESET}"
    
    # Suffixes
    suffix_prefixes = ("TENSE_", "PERSON_", "POSS_", "CASE_", "COPULA_", "PART_", "INF_", "GERUND_", "DERIV_")
    if tag.startswith(suffix_prefixes) or tag in ("PLURAL", "POTENTIAL", "NEG", "IMPOTENTIAL_NEG"):
        return f"{C_CYAN}{tag}{C_RESET}"
    
    # Standard lexical root
    return f"{C_GREEN}{tag}{C_RESET}"

def resize_state_dict(model, old_state_dict):
    """Resizes model embedding and linear heads to match the new vocabulary size."""
    new_state_dict = model.state_dict()
    for k, v in old_state_dict.items():
        if k in new_state_dict:
            if v.shape != new_state_dict[k].shape:
                if len(v.shape) == 2:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])] = v[:min(v.shape[0], new_state_dict[k].shape[0]), :min(v.shape[1], new_state_dict[k].shape[1])]
                elif len(v.shape) == 1:
                    new_state_dict[k][:min(v.shape[0], new_state_dict[k].shape[0])] = v[:min(v.shape[0], new_state_dict[k].shape[0])]
            else:
                new_state_dict[k] = v
    return new_state_dict

def generate_tokens(model, tokenizer, vocab, prompt_tokens, max_new_tokens=60, device='cpu', temperature=0.0, top_k=5, repetition_penalty=1.5, repetition_window=12):
    """Generates next tokens autoregressively from prompt_tokens and tracks entropy."""
    model.eval()
    generated = list(prompt_tokens)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    last_entropy = 0.0
    
    # Stop if we already have EOS or </OUTPUT> at the end
    if generated and generated[-1] in (eos_id, output_end_id):
        return generated, 0.0
        
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :] # focus on the last position
            
            # Measure token entropy
            probs_for_entropy = torch.softmax(logits, dim=-1)
            last_entropy = -torch.sum(probs_for_entropy * torch.log(probs_for_entropy + 1e-9)).item()
            
            # Apply repetition penalty only to output tokens within the sliding window
            output_tokens = generated[len(prompt_tokens):]
            if repetition_penalty > 1.0 and output_tokens:
                window_tokens = output_tokens[-repetition_window:]
                for token_id in set(window_tokens):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty
            
            if temperature > 0.0:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                pred_id = torch.multinomial(probs, num_samples=1).item()
            else:
                pred_id = torch.argmax(logits).item()
                
            generated.append(pred_id)
            
            # Print intermediate tokens to give a typing/streaming effect
            pred_token = vocab.decode(pred_id)
            print(f" {color_token(pred_token)}", end="", flush=True)
            
            if pred_id == eos_id or pred_id == output_end_id:
                break
    print() # New line after generation
    return generated, last_entropy


def print_help():
    print(f"\n{C_BOLD}Desteklenen Komutlar:{C_RESET}")
    print(f"  {C_YELLOW}help{C_RESET}       : Bu yardım menüsünü gösterir.")
    print(f"  {C_YELLOW}mode{C_RESET}       : Çıkarım modunu değiştirir (Normal / SFT / RAG).")
    print(f"  {C_YELLOW}params{C_RESET}     : Sıcaklık (temp) ve top_k parametrelerini ayarlar.")
    print(f"  {C_YELLOW}arena{C_RESET}      : Büyük Ajan Arenasını ve otonom süpervizörü başlatır.")
    print(f"  {C_YELLOW}q / exit{C_RESET}   : Programdan çıkar.")

def main():
    if "--gateway" in sys.argv:
        from scripts.run_agent_arena import main as run_arena_main
        run_arena_main()
        return

    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Vocabulary
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    if not os.path.exists(vocab_path):
        print(f"{C_RED}Hata: {vocab_path} bulunamadı! Lütfen önce verileri derleyin.{C_RESET}")
        return
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    
    # Load Lexicon & Compiler
    lexicon = LexiconManager()
    lexicon_path = 'data/lexicon/roots.tsv'
    if not os.path.exists(lexicon_path):
        print(f"{C_RED}Hata: {lexicon_path} bulunamadı!{C_RESET}")
        return
    lexicon.load_from_tsv(lexicon_path)
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    # Connect to VectorMemory (Persistent local data/qdrant_db or remote Qdrant)
    print(f"\n{C_MAGENTA}[RAG] Vektörel belleklere bağlanılıyor...{C_RESET}")
    memory = VectorMemory(collection_name="kristal_bellek", vector_size=768, host="localhost", port=6333, storage_path="data/qdrant_db")
    general_memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333, storage_path="data/qdrant_db")
    print(f"  {C_BOLD}Bellek Depolama:{C_RESET} {C_GREEN}{memory.storage_type}{C_RESET}")
    doc_count = memory.get_document_count()
    gen_count = general_memory.get_document_count()
    print(f"  {C_BOLD}Aktif Koleksiyonlar:{C_RESET} {C_YELLOW}{memory.collection_name}{C_RESET} ({doc_count} parça) + {C_CYAN}{general_memory.collection_name}{C_RESET} ({gen_count} parça)")
    
    # If collection is completely empty, load baseline samples
    if doc_count == 0:
        print(f"  {C_GRAY}-> Koleksiyon boş olduğundan temel bilgi belgeleri indeksleniyor...{C_RESET}")
        knowledge_texts = [
            ("Masif ahşap mobilya imalatında kereste nemi yüzde 8 ile 12 arasında olmalıdır.", "Masif ahşap mobilya imalatında kereste nemi yüzde 8 ile 12 arasında olmalıdır."),
            ("Meşe ağacı sert dokulu, yoğun lifli, neme ve aşınmaya dayanıklı bir ağaç türüdür. Masif mobilya ve parke yapımında kullanılır.", "Meşe ağacı sert dokulu, yoğun lifli, neme ve aşınmaya dayanıklı bir ağaç türüdür. Masif mobilya ve parke yapımında kullanılır."),
            ("Kırlangıç kuyruğu geçme, çekmece kasalarında ve sandık köşelerinde çekme kuvvetine karşı direnç sağlar.", "Kırlangıç kuyruğu geçme, çekmece kasalarında ve sandık köşelerinde çekme kuvvetine karşı direnç sağlar."),
            ("Lamba zıvana geçme geniş ahşap yüzeylerde, klasik gömme zıvana ise masa ve sandalye ayaklarında tercih edilir.", "Lamba zıvana geçme geniş ahşap yüzeylerde, klasik gömme zıvana ise masa ve sandalye ayaklarında tercih edilir.")
        ]
        
        guide_path = 'data/knowledge/ahsap_ve_marangozluk_rehberi.md'
        if os.path.exists(guide_path):
            try:
                with open(guide_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and not p.startswith('#')]
                for p in paragraphs:
                    knowledge_texts.append((p, p))
            except Exception as e:
                print(f"  {C_RED}Uyarı: {guide_path} yüklenirken hata: {e}{C_RESET}")
                
        batch_dense = []
        batch_sparse = []
        batch_meta = []
        loaded_texts = []
        for query_text, doc_text in knowledge_texts:
            q_ids = tokenizer.encode(query_text)
            q_tags = tokenizer.decode(q_ids)
            dense_vec = generate_kristal_vector(q_ids, q_tags)
            sparse_vec = generate_sparse_vector(q_ids, q_tags)
            
            d_ids = tokenizer.encode(doc_text)
            d_tags = tokenizer.decode(d_ids)
            
            loaded_texts.append(doc_text)
            batch_dense.append(dense_vec)
            batch_sparse.append(sparse_vec)
            batch_meta.append({
                "source": "knowledge_guide",
                "crystal_tags": d_tags,
                "token_ids": d_ids
            })
        memory.add_documents_batch(loaded_texts, batch_dense, batch_sparse, batch_meta)
        print(f"  {C_GREEN}Temel bellek kuruldu. {len(loaded_texts)} bilgi parçası indekslendi.{C_RESET}")
    
    # Load Model Weights
    model_path = 'data/kristal_model.pt'
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--model", "--model-path") and arg_idx + 1 < len(sys.argv):
            model_path = sys.argv[arg_idx + 1]

    if not os.path.exists(model_path):
        print(f"{C_RED}Hata: Eğitilmiş model dosyası '{model_path}' bulunamadı!{C_RESET}")
        return
        
    print(f"  {C_CYAN}Model Ağırlıkları Yükleniyor: {model_path}{C_RESET}")
        
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Initialize Merak Motoru & Tri-Modal Router
    curiosity_engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5).to(device)
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256, num_experts=4, top_k=2, expert_names=["grammar_core", "pedagogy", "carpenter", "legal"]).to(device)
    curiosity_engine.eval()
    router.eval()


    
    # Intro
    os.system("clear" if os.name == "posix" else "cls")
    print(f"{C_MAGENTA}{C_BOLD}" + "=" * 60)
    print("  KRİSTAL-VEKTÖREL MİMARİSİ: İNTERAKTİF CHAT-PROMPT CLI")
    print("=" * 60 + f"{C_RESET}")
    print(f"Cihaz:      {C_CYAN}{device}{C_RESET}")
    print(f"Sözlük:     {C_CYAN}{vocab_size} morfem{C_RESET}")
    print(f"Model:      {C_CYAN}{model_path}{C_RESET}")
    
    # CLI Settings
    mode = "SFT" # Default to SFT instruction follower
    temp = 0.0
    top_k = 5
    max_tokens = 60

    
    print_help()
    
    # Predefined instructions list for convenience
    sft_instructions = [
        "Kelimedeki kök morfemini bul.",
        "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
        "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
        "Kelimedeki eylemin zamanını veya kipini tespit et.",
        "Belgeye göre cevapla.",
        "Ahşap uzmanı olarak cevapla."
    ]
    
    while True:
        try:
            current_rag_doc = None
            current_rag_score = 0.0
            print(f"\n{C_GRAY}[Mod: {mode} | Temp: {temp} | Top-K: {top_k}]{C_RESET}")
            if mode == "SFT":
                print(f"{C_BOLD}Lütfen SFT Görevi Seçin veya Kendi Talimatınızı Girin:{C_RESET}")
                for idx, inst in enumerate(sft_instructions):
                    print(f"  {idx + 1}. {inst}")
                print(f"  {len(sft_instructions) + 1}. Özel Talimat Gir...")
                
                choice = input(f"{C_YELLOW}Seçiminiz (1-{len(sft_instructions)+1}) veya Komut: {C_RESET}").strip()
                
                if choice.lower() in ("q", "exit"):
                    print(f"{C_GREEN}Görüşmek üzere!{C_RESET}")
                    break
                elif choice.lower() == "help":
                    print_help()
                    continue
                elif choice.lower() == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    continue
                elif choice.lower() == "mode":
                    mode = "NORMAL"
                    print(f"\n{C_YELLOW}Normal metin tamamlama moduna geçildi.{C_RESET}")
                    print(f"{C_GRAY}Not: Model SFT/DPO ile talimat izlemeye aşırı hizalandığı için, ham metin girdiğinizde tekrara düşebilir veya <PROPER_NOUN> üretebilir. En iyi sonuçlar için SFT modunu tercih edin veya istemi SFT formatında verin.{C_RESET}")
                    continue
                elif choice.lower() in ("arena", "gateway"):
                    print(f"\n{C_MAGENTA}{C_BOLD}=== PEDAGOJİK AJAN ARENASI VE SÜPERVİZÖR MODU ==={C_RESET}")
                    gateway = AgentGateway(
                        model=model,
                        tokenizer=tokenizer,
                        decompiler=decompiler,
                        memory=memory,
                        general_memory=general_memory,
                        future_train_path="data/future_train_vector.jsonl",
                        device=str(device)
                    )
                    supervisor = PedagogicalSupervisor(gateway=gateway, retrain_threshold=5)
                    domain_choice = input(f"{C_YELLOW}Alan Seçin (1: Ahşap/Carpenter, 2: Morfoloji/Pedagogy, 3: Edebi/Literary) [Varsayılan 1]: {C_RESET}").strip()
                    dom_map = {"1": "carpenter", "2": "pedagogy", "3": "literary"}
                    sel_dom = dom_map.get(domain_choice, "carpenter")
                    print(f"\n{C_CYAN}Arena oturumu başlatılıyor ({sel_dom})...{C_RESET}")
                    supervisor.run_arena_session(domain=sel_dom, rounds=2, auto_retrain=True)
                    continue
                elif choice.lower() == "params":
                    try:
                        temp = float(input("Sıcaklık (0.0=Greedy): ").strip())
                        top_k = int(input("Top-K (Örn: 5): ").strip())
                    except ValueError:
                        print(f"{C_RED}Hatalı parametre girişi.{C_RESET}")
                    continue
                
                # Check choice
                selected_inst = ""
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(sft_instructions):
                        selected_inst = sft_instructions[choice_idx]
                    elif choice_idx == len(sft_instructions):
                        selected_inst = input("Özel Talimatınızı Girin: ").strip()
                except ValueError:
                    # Treat choice as direct command or custom instruction if not integer
                    if choice:
                        selected_inst = choice
                    else:
                        continue
                
                if not selected_inst:
                    print(f"{C_RED}Geçersiz seçim.{C_RESET}")
                    continue
                    
                if selected_inst in ["Belgeye göre cevapla.", "Ahşap uzmanı olarak cevapla."]:
                    word_input = input(f"{C_YELLOW}Sorunuzu veya Cümleyi Girin: {C_RESET}").strip()
                else:
                    word_input = input(f"{C_YELLOW}Analiz Edilecek Kelimeyi Girin: {C_RESET}").strip()
                    
                if not word_input:
                    continue
                
                # Check if it is a RAG query
                if selected_inst == "Belgeye göre cevapla.":
                    query_token_ids = tokenizer.encode(word_input)
                    query_tags = tokenizer.decode(query_token_ids)
                    clean_query_tags = query_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                    
                    dense_vec = generate_kristal_vector(query_token_ids, query_tags)
                    sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
                    
                    print(f"\n{C_CYAN}[RAG] Bellekten döküman aranıyor...{C_RESET}")
                    results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
                    
                    MIN_RAG_SCORE = 0.40
                    source_coll = memory.collection_name
                    if not (results and results[0]["score"] >= MIN_RAG_SCORE and results[0].get("has_root_match", True)):
                        gen_results = general_memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
                        if gen_results and gen_results[0]["score"] >= MIN_RAG_SCORE and gen_results[0].get("has_root_match", True):
                            results = gen_results
                            source_coll = general_memory.collection_name
                            
                    if results and results[0]["score"] >= MIN_RAG_SCORE and results[0].get("has_root_match", True):
                        doc = results[0]
                        doc_text = doc["text"]
                        doc_tags = doc["metadata"].get("crystal_tags", "")
                        print(f"  {C_BOLD}Bulunan Döküman ({C_YELLOW}{source_coll}{C_RESET}{C_BOLD}):{C_RESET} {C_GREEN}{doc_text}{C_RESET}")
                        print(f"  {C_BOLD}Döküman Morfemleri:{C_RESET} {doc_tags}")
                        matching_roots = results[0].get("matching_roots", [])
                        if matching_roots:
                            print(f"  {C_BOLD}Eşleşen Kökler:{C_RESET} {C_GREEN}{matching_roots}{C_RESET}")
                        print(f"  {C_BOLD}Eşleşme Skoru (RRF):{C_RESET} {C_YELLOW}{doc['score']:.4f}{C_RESET}")
                        clean_doc_tags = doc_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                        input_str = f"belge: {clean_doc_tags} sorgu: {clean_query_tags}"
                    else:
                        if results:
                            print(f"  {C_YELLOW}[RAG] Yetersiz Eşleşme: En yakın belgenin skoru ({results[0]['score']:.4f}) güven eşiğinin ({MIN_RAG_SCORE}) altında kaldı veya konu kökleri uyuşmadı.{C_RESET}")
                            print(f"  {C_YELLOW}-> Alakasız belge bağlama eklenmedi.{C_RESET}")
                        else:
                            print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı.{C_RESET}")
                        print(f"  {C_GRAY}[Bilgi] Veritabanında bu soruya referans olacak yeterli belge bulunamadı.{C_RESET}")
                        print(f"  {C_GRAY}-> 'rag_tool.py' ile ilgili belgeyi ekleyebilir veya Seçim 6 (Ahşap Uzmanı) ile genel model bilgisini sorgulayabilirsiniz.{C_RESET}")
                        continue
                        
                    prompt_dict = {
                        "instruction": selected_inst,
                        "input": input_str,
                        "output": ""
                    }
                else:
                    # Standard non-RAG SFT prompt
                    prompt_dict = {
                        "instruction": selected_inst,
                        "input": word_input,
                        "output": ""
                    }
                raw_prompt = json.dumps(prompt_dict)
                
            elif mode == "RAG":
                user_prompt = input(f"{C_YELLOW}RAG Sorgusu Girin (örn. meşe ağacı): {C_RESET}").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("q", "exit"):
                    print(f"{C_GREEN}Görüşmek üzere!{C_RESET}")
                    break
                elif user_prompt.lower() == "help":
                    print_help()
                    continue
                elif user_prompt.lower() == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    continue
                elif user_prompt.lower() == "mode":
                    mode = "SFT"
                    print(f"\n{C_GREEN}SFT (Talimat/Görev) moduna geçildi.{C_RESET}")
                    continue
                elif user_prompt.lower() == "params":
                    try:
                        temp = float(input("Sıcaklık (0.0=Greedy): ").strip())
                        top_k = int(input("Top-K (Örn: 5): ").strip())
                    except ValueError:
                        print(f"{C_RED}Hatalı parametre girişi.{C_RESET}")
                    continue
                
                # Connect RAG query processing
                query_token_ids = tokenizer.encode(user_prompt)
                query_tags = tokenizer.decode(query_token_ids)
                clean_query_tags = query_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                
                # 1. Merak Motoru & Shannon Entropisi
                q_tensor = torch.tensor([query_token_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    q_res = model(q_tensor, return_hidden_states=True)
                    if len(q_res) == 3:
                        q_logits, _, q_emb = q_res
                        last_h = q_emb[:, -1, :]
                    else:
                        q_logits, _ = q_res
                        last_h = torch.zeros(q_logits.shape[0], 768, device=device)
                        q_emb = torch.zeros(1, len(query_token_ids), 768, device=device)
                    h_val, needs_ret, q_merak = curiosity_engine(last_h, q_logits[:, -1, :])
                    
                if needs_ret.item():
                    print(f"  {C_MAGENTA}[Merak Motoru]{C_RESET} {C_YELLOW}Shannon Entropisi H(z)={h_val.item():.2f} > 2.50 — Epistemik açık tespit edildi! (Otonom bellek çağrısı tetiklendi){C_RESET}")
                else:
                    print(f"  {C_GRAY}[Merak Motoru] Model belirsizlik düzeyi H(z)={h_val.item():.2f} <= 2.50 (Düşük merak){C_RESET}")
                
                dense_vec = generate_kristal_vector(query_token_ids, query_tags)
                sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
                
                print(f"\n{C_CYAN}[RAG] Bellekten döküman aranıyor...{C_RESET}")
                results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
                
                MIN_RAG_SCORE = 0.40
                source_coll = memory.collection_name
                if not (results and results[0]["score"] >= MIN_RAG_SCORE and results[0].get("has_root_match", True)):
                    gen_results = general_memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
                    if gen_results and gen_results[0]["score"] >= MIN_RAG_SCORE and gen_results[0].get("has_root_match", True):
                        results = gen_results
                        source_coll = general_memory.collection_name
                        
                if results and results[0]["score"] >= MIN_RAG_SCORE and results[0].get("has_root_match", True):
                    doc = results[0]
                    current_rag_doc = doc
                    current_rag_score = float(doc["score"])
                    doc_text = doc["text"]
                    doc_tags = doc["metadata"].get("crystal_tags", "")
                    print(f"  {C_BOLD}Bulunan Döküman ({C_YELLOW}{source_coll}{C_RESET}{C_BOLD}):{C_RESET} {C_GREEN}{doc_text}{C_RESET}")
                    print(f"  {C_BOLD}Döküman Morfemleri:{C_RESET} {doc_tags}")
                    matching_roots = results[0].get("matching_roots", [])
                    if matching_roots:
                        print(f"  {C_BOLD}Eşleşen Kökler:{C_RESET} {C_GREEN}{matching_roots}{C_RESET}")
                    print(f"  {C_BOLD}Eşleşme Skoru (RRF):{C_RESET} {C_YELLOW}{doc['score']:.4f}{C_RESET}")
                    
                    # 2. Tri-Modal Router Uzman Seçimi
                    prompt_vec = q_emb.mean(dim=1)
                    rag_vec = None
                    if doc.get("metadata", {}).get("token_ids"):
                        d_ids = doc["metadata"]["token_ids"]
                        with torch.no_grad():
                            rag_vec = model.embedding(torch.tensor([d_ids], dtype=torch.long, device=device)).mean(dim=1)
                    _, router_indices, _ = router(prompt_vec, q_merak, rag_vec)
                    experts = router.get_selected_expert_names(router_indices)[0]
                    print(f"  {C_BOLD}Tri-Modal Router Uzmanları:{C_RESET} {C_CYAN}{', '.join(experts)}{C_RESET}")
                    
                    clean_doc_tags = doc_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                    input_str = f"belge: {clean_doc_tags} sorgu: {clean_query_tags}"
                else:
                    if results:
                        print(f"  {C_YELLOW}[RAG] Yetersiz Eşleşme: En yakın belgenin skoru ({results[0]['score']:.4f}) güven eşiğinin ({MIN_RAG_SCORE}) altında kaldı veya konu kökleri uyuşmadı.{C_RESET}")
                        print(f"  {C_YELLOW}-> Alakasız belge bağlama eklenmedi.{C_RESET}")
                    else:
                        print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı.{C_RESET}")
                    print(f"  {C_GRAY}[Bilgi] Veritabanında bu soruya referans olacak yeterli belge bulunamadı.{C_RESET}")
                    print(f"  {C_GRAY}-> 'rag_tool.py' ile ilgili belgeyi ekleyebilir veya 'mode' yazıp SFT moduna geçebilirsiniz.{C_RESET}")
                    continue
                    
                prompt_dict = {
                    "instruction": "Belgeye göre cevapla.",
                    "input": input_str,
                    "output": ""
                }
                raw_prompt = json.dumps(prompt_dict)
                
            else: # NORMAL Text continuation mode
                user_prompt = input(f"{C_YELLOW}Girdi Morfemleri veya Kelime Girin (SFT dışı girdi tekrara yol açabilir): {C_RESET}").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ("q", "exit"):
                    print(f"{C_GREEN}Görüşmek üzere!{C_RESET}")
                    break
                elif user_prompt.lower() == "help":
                    print_help()
                    continue
                elif user_prompt.lower() == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    continue
                elif user_prompt.lower() == "mode":
                    mode = "RAG"
                    print(f"\n{C_YELLOW}RAG (Geri Çağırma & Üretim) moduna geçildi.{C_RESET}")
                    continue
                elif user_prompt.lower() == "params":
                    try:
                        temp = float(input("Sıcaklık (0.0=Greedy): ").strip())
                        top_k = int(input("Top-K (Örn: 5): ").strip())
                    except ValueError:
                        print(f"{C_RED}Hatalı parametre girişi.{C_RESET}")
                    continue
                
                raw_prompt = user_prompt
            
            # Encode & Render
            try:
                token_ids = tokenizer.encode(raw_prompt)
            except Exception as e:
                print(f"{C_RED}Tokenizasyon Hatası: {e}{C_RESET}")
                continue
                
            # If SFT or RAG mode, we slice up to the <OUTPUT> token so model can generate output
            output_start_id = vocab.stoi.get("<OUTPUT>", -1)
            if mode in ("SFT", "RAG") and output_start_id in token_ids:
                output_idx = token_ids.index(output_start_id)
                prompt_tokens = token_ids[:output_idx + 1]
            else:
                # Normal mode: keep whole tokenized prompt except trailing <EOS> if we want continuation
                if token_ids and token_ids[-1] == vocab.stoi.get("<EOS>", -1):
                    prompt_tokens = token_ids[:-1]
                else:
                    prompt_tokens = token_ids
            
            # Print input token list
            print(f"\n{C_BOLD}Girdi Morfem Akışı:{C_RESET}")
            decoded_input = [vocab.decode(tid) for tid in prompt_tokens]
            print(" ".join([color_token(t) for t in decoded_input]))
            
            # Generate
            print(f"\n{C_BOLD}Model Çıktısı:{C_RESET}")
            # Stream/print generated tokens
            generated_ids, post_entropy = generate_tokens(model, tokenizer, vocab, prompt_tokens, 
                                                         max_new_tokens=max_tokens, device=device, 
                                                         temperature=temp, top_k=top_k)
            
            # Extract newly generated tokens
            new_tokens = generated_ids[len(prompt_tokens):]
            new_morphemes = [vocab.decode(tid) for tid in new_tokens]
            clean_morphemes = [t for t in new_morphemes if t not in ("<EOS>", "</OUTPUT>", "<PAD>", "<UNK>")]
            
            decompiled_text = ""
            if clean_morphemes:
                morphemes_str = " ".join(clean_morphemes)
                decompiled_text = decompiler.decompile_sentence(morphemes_str)
                if decompiled_text.strip():
                    print(f"\n{C_GREEN}{C_BOLD}Decompile Edilmiş Çıktı:{C_RESET} {C_BOLD}{decompiled_text}{C_RESET}")
                else:
                    print(f"\n{C_GREEN}{C_BOLD}Yazılı Çıktı:{C_RESET} {' '.join(clean_morphemes)}")
                    
            # Epistemik Değerlendirme & future_train_vector.jsonl Kaydı:
            # Model merak edip >= 0.85 uyumlu belge getirdiğinde, ancak anlayamadığında kaydeder.
            if mode == "RAG" and current_rag_doc and current_rag_score >= 0.85:
                decomp_lower = decompiled_text.lower()
                is_uncertain = (post_entropy > 2.5) or any(p in decomp_lower for p in ["bilgi yok", "bulunamaz", "bilinmiyor"]) or (new_morphemes.count("<UNK>") >= 2)
                if is_uncertain:
                    print(f"\n  {C_MAGENTA}{C_BOLD}[Epistemik Kayıt]{C_RESET} {C_YELLOW}Model belgeyi getirdi (Uyum: {current_rag_score:.4f} >= 0.85) fakat tam anlayamadı (H(z)={post_entropy:.2f}).{C_RESET}")
                    print(f"  {C_CYAN}-> Bu örnek gelecekteki model eğitimi için 'data/future_train_vector.jsonl' kütüğüne eklendi.{C_RESET}")
                    os.makedirs("data", exist_ok=True)
                    record = {
                        "instruction": "Belgeye göre cevapla.",
                        "input": raw_prompt,
                        "output": " ".join(clean_morphemes),
                        "decompiled_output": decompiled_text,
                        "rag_document": current_rag_doc["text"],
                        "similarity_score": round(current_rag_score, 4),
                        "entropy_post": round(post_entropy, 4),
                        "reason": "epistemic_gap_unresolved_high_similarity",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    with open("data/future_train_vector.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    print(f"\n  {C_GREEN}[Epistemik Onay] Model yüksek uyumlu belgeyi başarıyla çözümledi (H(z)={post_entropy:.2f} <= 2.50).{C_RESET}")
                            
        except KeyboardInterrupt:
            print(f"\n{C_GREEN}Çıkış yapılıyor...{C_RESET}")
            break
        except Exception as e:
            print(f"\n{C_RED}Hata Oluştu: {e}{C_RESET}")

if __name__ == '__main__':
    main()
