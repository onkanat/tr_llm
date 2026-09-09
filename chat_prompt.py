#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
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

    """Generates next tokens autoregressively from prompt_tokens."""
    model.eval()
    generated = list(prompt_tokens)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    
    # Stop if we already have EOS or </OUTPUT> at the end
    if generated and generated[-1] in (eos_id, output_end_id):
        return generated
        
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :] # focus on the last position
            
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
    return generated


def print_help():
    print(f"\n{C_BOLD}Desteklenen Komutlar:{C_RESET}")
    print(f"  {C_YELLOW}help{C_RESET}       : Bu yardım menüsünü gösterir.")
    print(f"  {C_YELLOW}mode{C_RESET}       : Çıkarım modunu değiştirir (Normal / SFT / RAG).")
    print(f"  {C_YELLOW}params{C_RESET}     : Sıcaklık (temp) ve top_k parametrelerini ayarlar.")
    print(f"  {C_YELLOW}clear{C_RESET}      : Ekranı temizler.")
    print(f"  {C_YELLOW}q / exit{C_RESET}   : Programdan çıkar.")

def main():
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
    
    # Connect to Qdrant/In-Memory VectorMemory
    print(f"\n{C_MAGENTA}[RAG] Vektörel belleğe bağlanılıyor...{C_RESET}")
    is_fallback = False
    try:
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333)
    except Exception as e:
        print(f"  {C_YELLOW}Uyarı: Qdrant sunucusuna bağlanılamadı. Geçici bellek (In-Memory) kuruluyor. Hata: {e}{C_RESET}")
        memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768)
        is_fallback = True
        
    if is_fallback:
        print(f"  {C_GRAY}-> Geçici belleğe örnek belgeler ve SFT verileri yükleniyor...{C_RESET}")
        sample_texts = [
            ("Okul müdürüyken okulun ek inşaatında hamallarla birlikte çalışmış.", "Okul müdürüyken okulun ek inşaatında hamallarla birlikte çalışmış.", "Belgeye göre cevapla."),
            ("Su düzeyi.", "Su düzeyi.", "Belgeye göre cevapla."),
            ("Kitap okumak insanı geliştirir.", "Kitap okumak insanı geliştirir.", "Belgeye göre cevapla.")
        ]
        chat_path = 'data/pedagogy/middle_school_chat.jsonl'
        if os.path.exists(chat_path):
            try:
                with open(chat_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        item = json.loads(line)
                        inst = item["instruction"].strip()
                        inp = item["input"].strip()
                        out = item["output"].strip()
                        query_text = inp if inp else inst
                        sample_texts.append((query_text, out, inst))
            except Exception as e:
                print(f"  {C_RED}Uyarı: {chat_path} yüklenirken hata oluştu: {e}{C_RESET}")
                
        batch_dense = []
        batch_sparse = []
        batch_meta = []
        loaded_texts = []
        for query_text, doc_text, inst in sample_texts:
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
                "domain": "fallback_data",
                "system_message": inst,
                "crystal_tags": d_tags,
                "token_ids": d_ids
            })
        memory.add_documents_batch(loaded_texts, batch_dense, batch_sparse, batch_meta)
        print(f"  {C_GREEN}Geçici bellek kuruldu. Örnek {len(loaded_texts)} belge indekslendi.{C_RESET}")
    
    # Load Model Weights
    model_path = 'data/kristal_model.pt'
    if not os.path.exists(model_path):
        print(f"{C_RED}Hata: Eğitilmiş model dosyası '{model_path}' bulunamadı!{C_RESET}")
        return
        
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    state_dict = resize_state_dict(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()


    
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
        "Belgeye göre cevapla."
    ]
    
    while True:
        try:
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
                    
                    if results:
                        doc = results[0]
                        doc_text = doc["text"]
                        doc_tags = doc["metadata"].get("crystal_tags", "")
                        sys_msg = doc["metadata"].get("system_message", selected_inst)
                        print(f"  {C_BOLD}Bulunan Döküman:{C_RESET} {C_GREEN}{doc_text}{C_RESET}")
                        print(f"  {C_BOLD}Döküman Morfemleri:{C_RESET} {doc_tags}")
                        print(f"  {C_BOLD}Sistem Mesajı (Instruction):{C_RESET} {C_CYAN}{sys_msg}{C_RESET}")
                        print(f"  {C_BOLD}Eşleşme Skoru (RRF + Ceza):{C_RESET} {C_YELLOW}{doc['score']:.4f}{C_RESET}")
                        clean_doc_tags = doc_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                    else:
                        print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı. Boş bağlam kullanılıyor.{C_RESET}")
                        clean_doc_tags = ""
                        sys_msg = selected_inst
                        
                    input_str = f"belge: {clean_doc_tags} sorgu: {clean_query_tags}"
                    prompt_dict = {
                        "instruction": sys_msg,
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
                user_prompt = input(f"{C_YELLOW}RAG Sorgusu Girin (örn. okul): {C_RESET}").strip()
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
                
                dense_vec = generate_kristal_vector(query_token_ids, query_tags)
                sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
                
                print(f"\n{C_CYAN}[RAG] Bellekten döküman aranıyor...{C_RESET}")
                results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
                
                if results:
                    doc = results[0]
                    doc_text = doc["text"]
                    doc_tags = doc["metadata"].get("crystal_tags", "")
                    sys_msg = doc["metadata"].get("system_message", "Belgeye göre cevapla.")
                    print(f"  {C_BOLD}Bulunan Döküman:{C_RESET} {C_GREEN}{doc_text}{C_RESET}")
                    print(f"  {C_BOLD}Döküman Morfemleri:{C_RESET} {doc_tags}")
                    print(f"  {C_BOLD}Sistem Mesajı (Instruction):{C_RESET} {C_CYAN}{sys_msg}{C_RESET}")
                    print(f"  {C_BOLD}Eşleşme Skoru (RRF + Ceza):{C_RESET} {C_YELLOW}{doc['score']:.4f}{C_RESET}")
                    clean_doc_tags = doc_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                else:
                    print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı. Boş bağlam kullanılıyor.{C_RESET}")
                    clean_doc_tags = ""
                    sys_msg = "Belgeye göre cevapla."
                    
                input_str = f"belge: {clean_doc_tags} sorgu: {clean_query_tags}"
                prompt_dict = {
                    "instruction": sys_msg,
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
            generated_ids = generate_tokens(model, tokenizer, vocab, prompt_tokens, 
                                            max_new_tokens=max_tokens, device=device, 
                                            temperature=temp, top_k=top_k)
            
            # Extract newly generated tokens
            new_tokens = generated_ids[len(prompt_tokens):]
            new_morphemes = [vocab.decode(tid) for tid in new_tokens]
            clean_morphemes = [t for t in new_morphemes if t not in ("<EOS>", "</OUTPUT>", "<PAD>", "<UNK>")]
            
            if clean_morphemes:
                morphemes_str = " ".join(clean_morphemes)
                decompiled_text = decompiler.decompile_sentence(morphemes_str)
                if decompiled_text.strip():
                    print(f"\n{C_GREEN}{C_BOLD}Decompile Edilmiş Çıktı:{C_RESET} {C_BOLD}{decompiled_text}{C_RESET}")
                else:
                    print(f"\n{C_GREEN}{C_BOLD}Yazılı Çıktı:{C_RESET} {' '.join(clean_morphemes)}")
                            
        except KeyboardInterrupt:
            print(f"\n{C_GREEN}Çıkış yapılıyor...{C_RESET}")
            break
        except Exception as e:
            print(f"\n{C_RED}Hata Oluştu: {e}{C_RESET}")

if __name__ == '__main__':
    main()
