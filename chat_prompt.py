#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: İNTERAKTİF ÇIKARIM VE SOHBET ARAYÜZÜ (CLI)
====================================================================
Yeni Nesil Modüler Arayüz:
  - Çok Turlu İnteraktif Sohbet (CHAT Modu)
  - Genişletilmiş SFT Müfredatı (Morfoloji, Ahşap, Türk Tarihi 1931, Fen, Edebiyat)
  - Dinamik Model Değiştirici (Temel Model <-> Marangozluk Modeli)
  - Otonom RAG ve Merak Motoru (Shannon Entropisi & Tri-Modal Router)
  - 4-Gram Bloklamalı ve Frekans Ölçekli Repetition Penalty (48 Pencere)
"""

import os
import sys
import json
import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional
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
    """Formats the token tag with distinct colors depending on its category."""
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


from src.llm.prompt_contract import resize_state_dict, render_prompt, render_example, build_rag_input


def load_model_instance(model_path: str, vocab_size: int, vocab: Vocabulary, device: torch.device) -> KristalLM:
    """Initializes and loads KristalLM checkpoint cleanly."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint'i bulunamadı (sessiz rastgele model engellendi): {model_path}")

    from scripts.train_step_b1_5_rigorous import compute_sha256

    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Model checkpoint yüklenirken hata oluştu ({model_path}): {e}") from e

    sha256_val = compute_sha256(model_path)
    print(f"[SOYAGACI] yuklenen={os.path.abspath(model_path)} sha256={sha256_val} anahtar={len(state_dict)}")

    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]

    # SOZLUK <-> CHECKPOINT TUTARLILIK KAPISI (T-0087). Gerekce OLCULDU: uyusmazlikta
    # resize_state_dict satirlari KIRPIYOR ve kosum DURMUYOR (bkz. K4 raporu).
    _emb = state_dict.get("embedding.embedding.weight")
    if _emb is None:
        raise RuntimeError(
            f"DURDURULDU: checkpoint'te 'embedding.embedding.weight' yok ({model_path}); "
            "sozluk uyumu DOGRULANAMAZ.")
    if int(_emb.shape[0]) != vocab_size:
        raise RuntimeError(
            f"DURDURULDU: sozluk/checkpoint UYUSMAZLIGI (T-0087). "
            f"checkpoint={model_path} ({int(_emb.shape[0])} satir) != sozluk "
            f"({vocab_size} giris), fark={int(_emb.shape[0]) - vocab_size}. "
            "Bu cift yurutulurse resize_state_dict satirlari KIRPAR ve kosum sessizce "
            "devam eder. Eslesen sozlugu ACIKCA verin (--vocab).")

    state_dict = resize_state_dict(model, state_dict)
    load_res = model.load_state_dict(state_dict, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(f"[SOYAGACI_UYARI] strict=False ile yüklendi: eksik={len(load_res.missing_keys)}, fazla={len(load_res.unexpected_keys)}")
    else:
        print("[SOYAGACI] strict=False ile yüklendi: tam eşleşme (0 eksik, 0 fazla).")

    model.to(device)
    model.eval()
    return model


def generate_tokens(model, tokenizer, vocab, prompt_tokens: List[int], max_new_tokens: int = 80, 
                    device: torch.device = torch.device('cpu'), temperature: float = 0.4, 
                    top_k: int = 10, repetition_penalty: float = 1.7, repetition_window: int = 48) -> Tuple[List[int], float]:
    """
    Generates next tokens autoregressively from prompt_tokens.
    Includes frequency-scaled sliding repetition penalty and 4-gram loop suppression.
    """
    model.eval()
    generated = list(prompt_tokens)
    eos_id = vocab.stoi.get("<EOS>", -1)
    output_end_id = vocab.stoi.get("</OUTPUT>", -1)
    last_entropy = 0.0
    
    if generated and generated[-1] in (eos_id, output_end_id):
        return generated, 0.0
        
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(x)
            logits = logits[0, -1, :]
            
            # Entropy measurement
            probs_for_entropy = torch.softmax(logits, dim=-1)
            last_entropy = -torch.sum(probs_for_entropy * torch.log(probs_for_entropy + 1e-9)).item()
            
            output_tokens = generated[len(prompt_tokens):]
            
            # 1. Frequency-scaled repetition penalty in sliding window
            if repetition_penalty > 1.0 and output_tokens:
                window_tokens = output_tokens[-repetition_window:]
                token_counts = {}
                for tid in window_tokens:
                    token_counts[tid] = token_counts.get(tid, 0) + 1
                    
                for token_id, count in token_counts.items():
                    factor = repetition_penalty ** count
                    if logits[token_id] > 0:
                        logits[token_id] /= factor
                    else:
                        logits[token_id] *= factor

            # 2. 4-gram repetition blocking (completely suppresses identical cyclic loops)
            if len(output_tokens) >= 3:
                ngram_ctx = tuple(output_tokens[-3:])
                seen_4grams = set()
                for idx_ng in range(len(output_tokens) - 3):
                    ctx = tuple(output_tokens[idx_ng:idx_ng + 3])
                    nxt = output_tokens[idx_ng + 3]
                    if ctx == ngram_ctx:
                        seen_4grams.add(nxt)
                for forbidden_id in seen_4grams:
                    logits[forbidden_id] = -float('inf')

            # 3. Structural control token suppression (Never emit structural delimiters in output)
            structural_suppress_tokens = [
                "<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>",
                "<INPUT>", "</INPUT>", "<OUTPUT>"
            ]
            for s_tok in structural_suppress_tokens:
                s_id = vocab.stoi.get(s_tok)
                if s_id is not None and s_id < logits.size(-1):
                    logits[s_id] = -float('inf')
            
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
            
            # Intermediate typing visualizer
            pred_token = vocab.decode(pred_id)
            print(f" {color_token(pred_token)}", end="", flush=True)
            
            if pred_id == eos_id or pred_id == output_end_id:
                break
                
    print()
    return generated, last_entropy


def normalize_sft_instruction(inst: str) -> str:
    """Normalizes informal SFT prompt into standard pedagogical imperative form."""
    inst = inst.strip()
    if not inst:
        return inst
    inst = re.sub(r'\bhakında\b', 'hakkında', inst, flags=re.IGNORECASE)
    
    imperative_endings = ("cevapla.", "açıkla.", "tespit et.", "belirt.", "yaz.", "çözümle.", "çevir.", "tanımla.")
    if any(inst.lower().endswith(end) for end in imperative_endings):
        return inst
        
    persona_match = re.search(r'^(.*?)\s*(?:hakkında\s+)?(?:uzmansın|uzmanısın|ustasısın|ustasın)[.!]?$', inst, re.IGNORECASE)
    if persona_match:
        field = persona_match.group(1).strip()
        if field:
            return f"{field} uzmanı olarak cevapla."
            
    if not inst.endswith("."):
        inst += "."
    return inst


def route_and_search_rag(memory: VectorMemory, general_memory: VectorMemory, 
                         dense_vec: list, sparse_vec: list, query_tags: str, 
                         user_query_text: str = "") -> Tuple[list, str]:
    """Intelligently routes RAG query to appropriate collection based on topic keywords."""
    carpenter_keywords = {
        "ahşap", "ağaç", "mobilya", "zıvana", "tutkal", "rende", "kereste", "çivi", 
        "kırlangıç", "lamba", "marangoz", "vernik", "cila", "gürgen", "meşe", "çam", 
        "ceviz", "pelesenk", "iskarpela", "gönye", "zımpara", "masif", "kavela"
    }
    words = set(re.findall(r'\b\w+\b', user_query_text.lower()))
    is_carpenter = bool(words.intersection(carpenter_keywords))
    
    first_mem = memory if is_carpenter else general_memory
    second_mem = general_memory if is_carpenter else memory
    
    MIN_RAG_SCORE = 0.40
    results = first_mem.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
    source_coll = first_mem.collection_name
    
    if not (results and results[0]["score"] >= MIN_RAG_SCORE and results[0].get("has_root_match", True)):
        sec_results = second_mem.hybrid_recall(dense_vec, sparse_vec, top_k=1, query_tags=query_tags)
        if sec_results and sec_results[0]["score"] >= MIN_RAG_SCORE and sec_results[0].get("has_root_match", True):
            results = sec_results
            source_coll = second_mem.collection_name
            
    return results, source_coll


def print_help():
    print(f"\n{C_BOLD}Desteklenen Komutlar:{C_RESET}")
    print(f"  {C_YELLOW}chat{C_RESET}       : Çok turlu interaktif diyalog (Chat) moduna geçer.")
    print(f"  {C_YELLOW}sft{C_RESET}        : SFT Görev & Talimat takip moduna geçer.")
    print(f"  {C_YELLOW}rag{C_RESET}        : Otonom RAG ve döküman tabanlı çıkarım moduna geçer.")
    print(f"  {C_YELLOW}model{C_RESET}      : Aktif modeli dinamik olarak değiştirir (Temel <-> Marangozluk).")
    print(f"  {C_YELLOW}params{C_RESET}     : Sıcaklık (temp), Top-K, Max Tokens ve ceza parametrelerini ayarlar.")
    print(f"  {C_YELLOW}status{C_RESET}     : Model, bellek, donanım ve parametre özetini gösterir.")
    print(f"  {C_YELLOW}history{C_RESET}    : Chat geçmişini gösterir.")
    print(f"  {C_YELLOW}reset{C_RESET}      : Chat geçmişini sıfırlar.")
    print(f"  {C_YELLOW}arena{C_RESET}      : Büyük Ajan Arenasını ve otonom süpervizörü başlatır.")
    print(f"  {C_YELLOW}clear{C_RESET}      : Terminal ekranını temizler.")
    print(f"  {C_YELLOW}q / exit{C_RESET}   : Programdan çıkar.")


def print_status(model_path: str, device: torch.device, vocab_size: int, 
                 memory: VectorMemory, general_memory: VectorMemory,
                 mode: str, temp: float, top_k: int, max_tokens: int,
                 rep_penalty: float, rep_window: int, chat_history: list):
    print(f"\n{C_MAGENTA}{C_BOLD}=== SİSTEM VE MODEL DURUM RAPORU ==={C_RESET}")
    print(f"  {C_BOLD}Aktif Model:{C_RESET}         {C_CYAN}{model_path}{C_RESET}")
    print(f"  {C_BOLD}Donanım Cihazı:{C_RESET}      {C_GREEN}{device}{C_RESET}")
    print(f"  {C_BOLD}Sözlük Büyüklüğü:{C_RESET}    {C_YELLOW}{vocab_size:,} morfem{C_RESET}")
    print(f"  {C_BOLD}Vektör Bellek (1):{C_RESET}   {memory.collection_name} ({memory.get_document_count()} döküman - {memory.storage_type})")
    print(f"  {C_BOLD}Vektör Bellek (2):{C_RESET}   {general_memory.collection_name} ({general_memory.get_document_count()} döküman - {general_memory.storage_type})")
    print(f"  {C_BOLD}Çıkarım Modu:{C_RESET}        {C_BOLD}{mode}{C_RESET}")
    print(f"  {C_BOLD}Sıcaklık (Temp):{C_RESET}     {temp} (Açık uçlu/Chat için optimize: 0.4)")
    print(f"  {C_BOLD}Top-K / Max Token:{C_RESET}   {top_k} / {max_tokens}")
    print(f"  {C_BOLD}Repetition Penalty:{C_RESET}  {rep_penalty} (Pencere: {rep_window} morfem, 4-gram bloklama aktif)")
    print(f"  {C_BOLD}Sohbet Geçmişi:{C_RESET}      {len(chat_history)} tur diyalog")
    print(f"{C_MAGENTA}====================================={C_RESET}\n")


def main():
    if "--gateway" in sys.argv:
        from scripts.run_agent_arena import main as run_arena_main
        run_arena_main()
        return

    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Vocabulary
    # FAIL-CLOSED (T-0087): varsayilan KALDIRILDI. Eski varsayilan 'data/vocab.json'
    # (31.357 giris) BAYATTI: guncel checkpoint 33.114 satirdir ve uyusmazlikta
    # resize_state_dict satirlari SESSIZCE kirpar. --vocab ACIKCA verilmelidir.
    vocab = Vocabulary()
    vocab_path = None
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--vocab" and arg_idx + 1 < len(sys.argv):
            vocab_path = sys.argv[arg_idx + 1]
    if not vocab_path:
        print(f"{C_RED}Hata: --vocab verilmedi. Varsayilan KALDIRILDI (T-0087): eski "
              f"varsayilan 'data/vocab.json' (31.357) BAYATTI ve checkpoint'i sessizce "
              f"kirpiyordu. Sozluk, checkpoint satir sayisiyla AYNI olmalidir; or. "
              f"--vocab data/rebuild/vocab_anka_r1_33114.json{C_RESET}")
        sys.exit(2)
    if not os.path.exists(vocab_path):
        print(f"{C_RED}Hata: Sözlük dosyası '{vocab_path}' bulunamadı! Lütfen önce verileri derleyin.{C_RESET}")
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
    
    # Connect to VectorMemory
    print(f"\n{C_MAGENTA}[RAG] Vektörel belleklere bağlanılıyor...{C_RESET}")
    memory = VectorMemory(collection_name="kristal_bellek", vector_size=768, host="localhost", port=6333, storage_path="data/qdrant_db")
    general_memory = VectorMemory(collection_name="simulasyon_bellek", vector_size=768, host="localhost", port=6333, storage_path="data/qdrant_db")
    
    # Initial Model Selection
    # FAIL-CLOSED (T-0087): varsayilan KALDIRILDI. Eski varsayilan silinmis Kristal
    # checkpoint zincirinin adini tasiyordu (Kristal zinciri operator karariyla silindi,
    # 18 Eyl 2026). --model/--model-path ACIKCA verilmelidir.
    model_path = None
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--model", "--model-path") and arg_idx + 1 < len(sys.argv):
            model_path = sys.argv[arg_idx + 1]

    if not model_path:
        print(f"{C_RED}Hata: --model verilmedi. Varsayilan KALDIRILDI (T-0087): eski "
              f"varsayilan silinmis bir checkpoint zincirinin adini tasiyordu. "
              f"Or. --model data/anka_a1r.pt{C_RESET}")
        sys.exit(2)
    if not os.path.exists(model_path):
        print(f"{C_RED}Hata: Eğitilmiş model dosyası '{model_path}' bulunamadı!{C_RESET}")
        return
        
    print(f"  {C_CYAN}Model Ağırlıkları Yükleniyor: {model_path}{C_RESET}")
    model = load_model_instance(model_path, vocab_size, vocab, device)

    # Initialize Merak Motoru & Tri-Modal Router
    curiosity_engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5).to(device)
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256, num_experts=4, top_k=2, 
                            expert_names=["grammar_core", "pedagogy", "carpenter", "history_literature"]).to(device)
    curiosity_engine.eval()
    router.eval()

    # CLI Settings
    mode = "SFT"  # Options: SFT, CHAT, RAG, NORMAL
    temp = 0.0    # 0.0 defaults to auto (0.0 for grammar, 0.4 for open-ended/chat)
    top_k = 10
    max_tokens = 80
    rep_penalty = 1.7
    rep_window = 48
    chat_history: List[Tuple[str, str]] = []

    # Predefined curriculum instructions list
    sft_instructions = [
        "Kelimedeki kök morfemini bul.",
        "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
        "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
        "Kelimedeki eylemin zamanını veya kipini tespit et.",
        "Belgeye göre cevapla.",
        "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "Türk edebiyatı ve kültür uzmanı olarak açıkla.",
        "Özel Talimat Gir..."
    ]

    os.system("clear" if os.name == "posix" else "cls")
    print(f"{C_MAGENTA}{C_BOLD}" + "=" * 62)
    print("  KRİSTAL-VEKTÖREL MİMARİSİ: İNTERAKTİF ÇIKARIM KONSOLU")
    print("=" * 62 + f"{C_RESET}")
    print(f"Cihaz:      {C_CYAN}{device}{C_RESET}")
    print(f"Sözlük:     {C_CYAN}{vocab_size:,} morfem{C_RESET}")
    print(f"Model:      {C_CYAN}{model_path}{C_RESET}")
    print_help()

    while True:
        try:
            current_rag_doc = None
            current_rag_score = 0.0
            
            # Print status banner
            mode_color = C_GREEN if mode == "CHAT" else (C_CYAN if mode == "SFT" else (C_YELLOW if mode == "RAG" else C_GRAY))
            print(f"\n{C_GRAY}[Mod: {mode_color}{mode}{C_GRAY} | Model: {os.path.basename(model_path)} | Temp: {temp} | Top-K: {top_k}]{C_RESET}")
            
            # Global Command Dispatcher Helper
            def handle_common_commands(cmd: str) -> Optional[bool]:
                nonlocal mode, temp, top_k, max_tokens, rep_penalty, rep_window, model, model_path, chat_history
                c_clean = cmd.lower().strip()
                if c_clean in ("q", "exit", "quit"):
                    print(f"{C_GREEN}Görüşmek üzere!{C_RESET}")
                    sys.exit(0)
                elif c_clean == "help":
                    print_help()
                    return True
                elif c_clean == "clear":
                    os.system("clear" if os.name == "posix" else "cls")
                    return True
                elif c_clean == "status":
                    print_status(model_path, device, vocab_size, memory, general_memory, mode, temp, top_k, max_tokens, rep_penalty, rep_window, chat_history)
                    return True
                elif c_clean == "history":
                    if not chat_history:
                        print(f"  {C_GRAY}Sohbet geçmişi boş.{C_RESET}")
                    else:
                        print(f"\n{C_BOLD}--- SOHBET GEÇMİŞİ ({len(chat_history)} Tur) ---{C_RESET}")
                        for h_idx, (q_h, a_h) in enumerate(chat_history, 1):
                            print(f"  {C_YELLOW}[{h_idx}] Kullanıcı:{C_RESET} {q_h}")
                            print(f"      {C_GREEN}Asistan:{C_RESET}   {a_h}")
                    return True
                elif c_clean == "reset":
                    chat_history.clear()
                    print(f"  {C_GREEN}Sohbet geçmişi sıfırlandı.{C_RESET}")
                    return True
                elif c_clean == "chat":
                    mode = "CHAT"
                    print(f"\n{C_GREEN}Çok Turlu Sohbet (CHAT) moduna geçildi.{C_RESET}")
                    return True
                elif c_clean == "sft":
                    mode = "SFT"
                    print(f"\n{C_CYAN}SFT (Talimat/Müfredat) moduna geçildi.{C_RESET}")
                    return True
                elif c_clean == "rag":
                    mode = "RAG"
                    print(f"\n{C_YELLOW}RAG (Geri Çağırma & Üretim) moduna geçildi.{C_RESET}")
                    return True
                elif c_clean == "normal":
                    mode = "NORMAL"
                    print(f"\n{C_GRAY}Normal metin tamamlama moduna geçildi.{C_RESET}")
                    return True
                elif c_clean == "mode":
                    modes = ["SFT", "CHAT", "RAG", "NORMAL"]
                    next_idx = (modes.index(mode) + 1) % len(modes)
                    mode = modes[next_idx]
                    print(f"\nMod değiştirildi: {mode_color}{mode}{C_RESET}")
                    return True
                elif c_clean.startswith("model"):
                    parts = c_clean.split(maxsplit=1)
                    new_path = ""
                    if len(parts) > 1:
                        new_path = parts[1].strip()
                    else:
                        # EMEKLİ (T-0087): menü artık SİLİNMİŞ checkpoint adlarını (Kristal
                        # zinciri, operatör kararı 18 Eyl 2026) SEÇENEK olarak sunmuyor.
                        # Eski menü "1"/"2" seçenekleri ölü yolları gösteriyor, seçilince
                        # yalnızca "bulunamadı" basıyordu ⇒ kullanıcıyı ölü yola yönlendiren
                        # CANLI bir ölü atıf sitesiydi. Tek yol: dosya yolunu AÇIKÇA sormak.
                        print(f"\n{C_BOLD}Aktif Model:{C_RESET} {C_CYAN}{model_path}{C_RESET}")
                        new_path = input(f"{C_YELLOW}Yeni model dosya yolu (boş = vazgeç): {C_RESET}").strip()
                    if new_path and os.path.exists(new_path):
                        print(f"  {C_CYAN}Model yükleniyor: {new_path}...{C_RESET}")
                        model = load_model_instance(new_path, vocab_size, vocab, device)
                        model_path = new_path
                        print(f"  {C_GREEN}Model başarıyla değiştirildi: {model_path}{C_RESET}")
                    elif new_path:
                        print(f"  {C_RED}Hata: '{new_path}' bulunamadı!{C_RESET}")
                    return True
                elif c_clean.startswith("params"):
                    parts = c_clean.split()
                    if len(parts) > 1:
                        try:
                            temp = float(parts[1])
                            if len(parts) > 2: top_k = int(parts[2])
                            if len(parts) > 3: max_tokens = int(parts[3])
                            if len(parts) > 4: rep_penalty = float(parts[4])
                            print(f"  {C_GREEN}Parametreler güncellendi: Temp={temp}, Top-K={top_k}, MaxTokens={max_tokens}, RepPenalty={rep_penalty}{C_RESET}")
                        except ValueError:
                            print(f"  {C_RED}Hatalı parametre girişi: params <temp> [top_k] [max_tokens] [rep_penalty]{C_RESET}")
                    else:
                        try:
                            t_inp = input(f"Sıcaklık (Şu an: {temp}, 0.0=Otomatik/Greedy): ").strip()
                            if t_inp: temp = float(t_inp)
                            k_inp = input(f"Top-K (Şu an: {top_k}): ").strip()
                            if k_inp: top_k = int(k_inp)
                            m_inp = input(f"Max Tokens (Şu an: {max_tokens}): ").strip()
                            if m_inp: max_tokens = int(m_inp)
                            p_inp = input(f"Repetition Penalty (Şu an: {rep_penalty}): ").strip()
                            if p_inp: rep_penalty = float(p_inp)
                            w_inp = input(f"Repetition Window (Şu an: {rep_window}): ").strip()
                            if w_inp: rep_window = int(w_inp)
                            print(f"  {C_GREEN}Parametreler güncellendi.{C_RESET}")
                        except ValueError:
                            print(f"  {C_RED}Hatalı parametre girişi.{C_RESET}")
                    return True
                elif c_clean in ("arena", "gateway"):
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
                    return True
                return False

            # =========================================================================
            # MODE 1: CHAT MODE (Multi-turn Interactive Conversation)
            # =========================================================================
            if mode == "CHAT":
                user_msg = input(f"{C_GREEN}{C_BOLD}Siz: {C_RESET}").strip()
                if not user_msg:
                    continue
                if handle_common_commands(user_msg):
                    continue

                # Build multi-turn context
                inst_text = "Yardımsever bir uzman olarak Türkçe cevapla."
                if "carpenter" in model_path:
                    # EMEKLİ (T-0087): eski koşul `"kristal_carpenter" in model_path` idi ve
                    # Kristal zinciri silindikten sonra (18 Eyl 2026) HİÇBİR yolda eşleşemez
                    # ⇒ dal ÖLÜYDÜ, marangoz talimatı sessizce hiç uygulanmıyordu. Ölü öneki
                    # (`kristal_`) kaldırmak davranışı geri getirir; yeni bir ad kuralı
                    # uydurulmadı.
                    inst_text = "Ahşap ve marangozluk uzmanı olarak cevapla."

                # Construct prompt with recent turns
                prompt_parts = [f"<INSTRUCTION> {inst_text} </INSTRUCTION>"]
                recent_turns = chat_history[-2:] if chat_history else []
                for past_q, past_a in recent_turns:
                    past_q_tags = tokenizer.decode(tokenizer.encode(past_q)).replace("<BOS>", "").replace("<EOS>", "").strip()
                    past_a_tags = tokenizer.decode(tokenizer.encode(past_a)).replace("<BOS>", "").replace("<EOS>", "").strip()
                    prompt_parts.append(f"<INPUT> {past_q_tags} </INPUT> <OUTPUT> {past_a_tags} </OUTPUT>")

                curr_q_tags = tokenizer.decode(tokenizer.encode(user_msg)).replace("<BOS>", "").replace("<EOS>", "").strip()
                prompt_parts.append(f"<INPUT> {curr_q_tags} </INPUT> <OUTPUT>")
                raw_prompt = " ".join(prompt_parts)
                is_open_ended = True

            # =========================================================================
            # MODE 2: SFT MODE (Instruction & Curriculum Tasks)
            # =========================================================================
            elif mode == "SFT":
                print(f"{C_BOLD}Lütfen SFT Görevi Seçin veya Kendi Talimatınızı Girin:{C_RESET}")
                for idx, inst in enumerate(sft_instructions):
                    print(f"  {idx + 1:2d}. {inst}")
                
                choice = input(f"{C_YELLOW}Seçiminiz (1-{len(sft_instructions)}) veya Komut: {C_RESET}").strip()
                if not choice:
                    continue
                if handle_common_commands(choice):
                    continue
                    
                selected_inst = ""
                is_open_ended = False
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(sft_instructions) - 1:
                        selected_inst = sft_instructions[choice_idx]
                        if choice_idx >= 4:  # Open-ended tasks (5-9)
                            is_open_ended = True
                    elif choice_idx == len(sft_instructions) - 1:
                        raw_custom = input(f"{C_YELLOW}Özel Talimatınızı Girin: {C_RESET}").strip()
                        if handle_common_commands(raw_custom):
                            continue
                        selected_inst = normalize_sft_instruction(raw_custom)
                        if selected_inst != raw_custom:
                            print(f"  {C_GRAY}[Format Standardizasyonu] -> {selected_inst}{C_RESET}")
                        is_open_ended = True
                except ValueError:
                    if choice:
                        selected_inst = normalize_sft_instruction(choice)
                        is_open_ended = True
                    else:
                        continue
                        
                if not selected_inst:
                    print(f"{C_RED}Geçersiz seçim.{C_RESET}")
                    continue
                    
                if is_open_ended or any(k in selected_inst.lower() for k in ["cevapla", "açıkla", "anlat", "uzmanı", "belge"]):
                    is_open_ended = True
                    word_input = input(f"{C_YELLOW}Sorunuzu veya Cümleyi Girin: {C_RESET}").strip()
                else:
                    word_input = input(f"{C_YELLOW}Analiz Edilecek Kelimeyi Girin: {C_RESET}").strip()
                    
                if not word_input:
                    print(f"  {C_YELLOW}Uyarı: Boş girdi girilemez. Lütfen bir soru veya metin yazın.{C_RESET}")
                    continue
                if handle_common_commands(word_input):
                    continue

                # EMEKLİ (T-0087): "Tekil Bilişsel Model Yönlendirmesi" bloğu KALDIRILDI.
                # Eski blok, `os.path.exists` kapısıyla `data/kristal_model.pt` dosyasına
                # bakıyor ve VARSA modeli SESSİZCE ona çeviriyordu. Kristal zinciri 18 Eyl
                # 2026'da operatör kararıyla silindi ⇒ blok zaten ölü koddur; ayrıca aynı
                # sözlükle İKİNCİ bir checkpoint'e sessizce geçmek T-0087'nin kapattığı
                # "sessiz model değiştirme" sınıfına girer. Yerine bir şey konmadı:
                # model seçimi artık YALNIZ `--model` iledir (yukarıda, fail-closed).

                # RAG Task & Knowledge Grounding
                is_knowledge_task = (selected_inst == "Belgeye göre cevapla.") or is_open_ended or any(k in selected_inst.lower() for k in ["uzmanı", "belge", "tarih", "marangoz", "fen", "edebiyat", "açıkla", "cevapla"])
                if is_knowledge_task:
                    query_token_ids = tokenizer.encode(word_input)
                    query_tags = tokenizer.decode(query_token_ids)
                    clean_query_tags = query_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                    
                    dense_vec = generate_kristal_vector(query_token_ids, query_tags)
                    sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
                    
                    print(f"\n{C_CYAN}[RAG] Bellekten döküman aranıyor...{C_RESET}")
                    results, source_coll = route_and_search_rag(memory, general_memory, dense_vec, sparse_vec, query_tags, word_input)
                    MIN_RAG_SCORE = 0.35
                    
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
                        doc_raw = doc.get("text", "")
                        input_str = build_rag_input(doc_raw, word_input)
                    else:
                        if selected_inst == "Belgeye göre cevapla.":
                            if results:
                                print(f"  {C_YELLOW}[RAG] Yetersiz Eşleşme: En yakın belgenin skoru ({results[0]['score']:.4f}) güven eşiğinin altında kaldı.{C_RESET}")
                            else:
                                print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı.{C_RESET}")
                            print(f"  {C_GRAY}[Bilgi] Veritabanında doğrudan referans belge bulunamadı; genel bilgiyle cevaplanıyor.{C_RESET}")
                        else:
                            print(f"  {C_GRAY}[RAG] Doğrudan referans belge bulunamadı; model genel parametrik bilgiyle yanıtlıyor.{C_RESET}")
                        input_str = word_input
                        
                    prompt_dict = {
                        "instruction": selected_inst,
                        "input": input_str,
                        "output": ""
                    }
                else:
                    prompt_dict = {
                        "instruction": selected_inst,
                        "input": word_input,
                        "output": ""
                    }
                raw_prompt = json.dumps(prompt_dict, ensure_ascii=False)

            # =========================================================================
            # MODE 3: RAG MODE (Autonomous Retrieval & Generation)
            # =========================================================================
            elif mode == "RAG":
                user_prompt = input(f"{C_YELLOW}RAG Sorgusu Girin: {C_RESET}").strip()
                if not user_prompt:
                    continue
                if handle_common_commands(user_prompt):
                    continue
                    
                query_token_ids = tokenizer.encode(user_prompt)
                query_tags = tokenizer.decode(query_token_ids)
                clean_query_tags = query_tags.replace("<BOS>", "").replace("<EOS>", "").strip()
                
                # Curiosity Engine check
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
                    print(f"  {C_MAGENTA}[Merak Motoru]{C_RESET} {C_YELLOW}Shannon Entropisi H(z)={h_val.item():.2f} > 2.50 — Otonom bellek çağrısı tetiklendi!{C_RESET}")
                else:
                    print(f"  {C_GRAY}[Merak Motoru] Model belirsizlik düzeyi H(z)={h_val.item():.2f} <= 2.50{C_RESET}")
                    
                dense_vec = generate_kristal_vector(query_token_ids, query_tags)
                sparse_vec = generate_sparse_vector(query_token_ids, query_tags)
                
                print(f"\n{C_CYAN}[RAG] Bellekten döküman aranıyor...{C_RESET}")
                results, source_coll = route_and_search_rag(memory, general_memory, dense_vec, sparse_vec, query_tags, user_prompt)
                MIN_RAG_SCORE = 0.40
                
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
                    
                    # Router
                    prompt_vec = q_emb.mean(dim=1)
                    rag_vec = None
                    if doc.get("metadata", {}).get("token_ids"):
                        d_ids = doc["metadata"]["token_ids"]
                        with torch.no_grad():
                            rag_vec = model.embedding(torch.tensor([d_ids], dtype=torch.long, device=device)).mean(dim=1)
                    _, router_indices, _ = router(prompt_vec, q_merak, rag_vec)
                    experts = router.get_selected_expert_names(router_indices)[0]
                    print(f"  {C_BOLD}Tri-Modal Router Uzmanları:{C_RESET} {C_CYAN}{', '.join(experts)}{C_RESET}")
                    
                    doc_raw = doc.get("text", "")
                    input_str = build_rag_input(doc_raw, user_prompt)
                else:
                    if results:
                        print(f"  {C_YELLOW}[RAG] Yetersiz Eşleşme (Skor: {results[0]['score']:.4f}). Alakasız belge bağlama eklenmedi.{C_RESET}")
                    else:
                        print(f"  {C_RED}Uyarı: Eşleşen döküman bulunamadı.{C_RESET}")
                    input_str = user_prompt
                    
                prompt_dict = {
                    "instruction": "Belgeye göre cevapla.",
                    "input": input_str,
                    "output": ""
                }
                raw_prompt = json.dumps(prompt_dict, ensure_ascii=False)
                is_open_ended = True

            # =========================================================================
            # MODE 4: NORMAL CONTINUATION
            # =========================================================================
            else:
                user_prompt = input(f"{C_YELLOW}Girdi Morfemleri veya Cümle Girin: {C_RESET}").strip()
                if not user_prompt:
                    continue
                if handle_common_commands(user_prompt):
                    continue
                raw_prompt = user_prompt
                is_open_ended = True

            # Encode prompt
            try:
                token_ids = tokenizer.encode(raw_prompt)
            except Exception as e:
                print(f"{C_RED}Tokenizasyon Hatası: {e}{C_RESET}")
                continue
                
            output_start_id = vocab.stoi.get("<OUTPUT>", -1)
            if mode in ("SFT", "CHAT", "RAG") and output_start_id in token_ids:
                output_idx = token_ids.index(output_start_id)
                prompt_tokens = token_ids[:output_idx + 1]
            else:
                if token_ids and token_ids[-1] == vocab.stoi.get("<EOS>", -1):
                    prompt_tokens = token_ids[:-1]
                else:
                    prompt_tokens = token_ids

            # Display prompt tokens
            print(f"\n{C_BOLD}Girdi Morfem Akışı:{C_RESET}")
            decoded_input = [vocab.decode(tid) for tid in prompt_tokens]
            print(" ".join([color_token(t) for t in decoded_input]))
            
            # Dynamic temperature:
            # Deterministic grammar tasks (1-4) stay at 0.0 (greedy)
            # Open-ended, Chat, and RAG tasks use temp=0.4, top_k=10 if temp == 0.0
            effective_temp = temp
            effective_top_k = top_k
            if temp == 0.0:
                if (mode == "SFT" and is_open_ended) or mode in ("CHAT", "RAG"):
                    effective_temp = 0.4
                    effective_top_k = 10
            
            # Generate
            print(f"\n{C_BOLD}Model Çıktısı:{C_RESET}")
            generated_ids, post_entropy = generate_tokens(
                model=model, tokenizer=tokenizer, vocab=vocab, prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens, device=device, temperature=effective_temp,
                top_k=effective_top_k, repetition_penalty=rep_penalty, repetition_window=rep_window
            )
            
            new_tokens = generated_ids[len(prompt_tokens):]
            new_morphemes = [vocab.decode(tid) for tid in new_tokens]
            clean_morphemes = [t for t in new_morphemes if t not in ("<EOS>", "</OUTPUT>", "<PAD>")]
            
            decompiled_text = ""
            if clean_morphemes:
                morphemes_str = " ".join(clean_morphemes)
                decompiled_text = decompiler.decompile_sentence(morphemes_str, capitalize=True)
                if decompiled_text.strip():
                    print(f"\n{C_GREEN}{C_BOLD}Decompile Edilmiş Çıktı:{C_RESET} {C_BOLD}{decompiled_text}{C_RESET}")
                else:
                    print(f"\n{C_GREEN}{C_BOLD}Yazılı Çıktı:{C_RESET} {' '.join(clean_morphemes)}")

            # Update chat history if in CHAT mode
            if mode == "CHAT" and decompiled_text.strip():
                chat_history.append((user_msg, decompiled_text.strip()))

            # Epistemic Record for RAG learning loop
            if mode == "RAG" and current_rag_doc and current_rag_score >= 0.85:
                decomp_lower = decompiled_text.lower()
                is_uncertain = (post_entropy > 2.5) or any(p in decomp_lower for p in ["bilgi yok", "bulunamaz", "bilinmiyor"]) or (new_morphemes.count("<UNK>") >= 2)
                if is_uncertain:
                    print(f"\n  {C_MAGENTA}{C_BOLD}[Epistemik Kayıt]{C_RESET} {C_YELLOW}Model belgeyi getirdi fakat tam anlayamadı (H(z)={post_entropy:.2f}).{C_RESET}")
                    print(f"  {C_CYAN}-> Bu örnek gelecekteki eğitim için 'data/future_train_vector.jsonl' kütüğüne eklendi.{C_RESET}")
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
