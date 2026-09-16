#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: BÜYÜK HEDEF BORU HATTI (GOAL PIPELINE)
==================================================================
Bu betik kullanıcının /goal talimatını baştan sona eksiksiz yürütür:
1. Temel Model için 300 Tur x 3 Tekrarlı Lise & Z Kuşağı Müfredat Arenası (Gemini/Ollama Öğretmen).
2. Biriken verilerin derlenmesi ve Full Temel Model Eğitimi (Pretrain -> SFT -> Chat -> DPO).
3. Marangozluk Modülü için 300 Adımlık Uzmanlık Arenası.
4. Biriken veri ile kristal_carpenter_model.pt modülünün yeniden eğitimi.
5. Her iki modelin çıkarım ve test doğrulaması.
"""

import os
import sys
import json
import time
import subprocess
import torch
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from src.gateway.agent_gateway import AgentGateway
from src.gateway.retrain_pipeline import RetrainPipeline
from src.gateway.pedagogical_supervisor import PedagogicalSupervisor, get_curriculum_probes, sanitize_teacher_card

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_RED = "\033[31m"


def check_frozen_save_path(save_path: str, allow_frozen_write: bool = False) -> None:
    """Belirtilen kaydetme yolunun donmuş olup olmadığını denetler.
    
    Donmuş yola yazma izni (allow_frozen_write=True) açıkça verilmemişse RuntimeError fırlatır.
    """
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(save_path) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {save_path} (allow_frozen_write=False)")


def log_phase(title: str):
    print(f"\n{C_BOLD}{C_MAGENTA}" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + f"{C_RESET}\n", flush=True)


def run_cmd(cmd_list: list, desc: str):
    print(f"{C_CYAN}[Komut Başlatılıyor]: {' '.join(cmd_list)}{C_RESET} ({desc})", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, check=True)
    dt = time.time() - t0
    print(f"{C_GREEN}[Tamamlandı]: {desc} ({dt:.1f} sn){C_RESET}\n", flush=True)
    return dt


def compile_jsonl_to_bin(
    jsonl_paths: list,
    output_bin: str,
    vocab_path: str,
    literal_entity_mode: bool,
    block_size: int = 64,
    oversample_factor: int = 1,
    allow_frozen_write: bool = False
) -> int:
    """Compiles JSONL records into uint16 binary token stream."""
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(output_bin) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {output_bin} (allow_frozen_write=False)")

    vocab = Vocabulary()
    vocab.load(vocab_path)
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=literal_entity_mode)

    all_token_ids = []
    total_records = 0
    failed_records = 0
    logged_errors = 0
    MAX_LOGGED_ERRORS = 5

    for path in jsonl_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    raw_str = json.dumps(data, ensure_ascii=False)
                    tids = tokenizer.encode(raw_str)
                    if len(tids) > 2:
                        for _ in range(oversample_factor):
                            all_token_ids.extend(tids)
                        total_records += 1
                except Exception as e:
                    failed_records += 1
                    if logged_errors < MAX_LOGGED_ERRORS:
                        sys.stderr.write(f"[HATA compile_jsonl_to_bin] Dosya: {path}, Satır: {line_idx}, Tip: {type(e).__name__}, Mesaj: {e}\n")
                        logged_errors += 1

    if not all_token_ids:
        print(f"{C_YELLOW}Uyarı: {output_bin} için veri bulunamadı!{C_RESET}")
        return 0

    import numpy as np
    arr = np.array(all_token_ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    arr.tofile(output_bin)
    
    meta = {
        "block_size": block_size,
        "vocab_size": len(vocab.stoi),
        "total_tokens": len(arr),
        "total_records": total_records,
        "failed_records": failed_records,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    with open(output_bin + ".meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"  {C_GREEN}Derleme Başarılı:{C_RESET} {output_bin} ({len(arr):,} token, {total_records} kaynak kayıt, {failed_records} başarısız kayıt)")
    return len(arr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Goal Pipeline Orchestrator")
    parser.add_argument("--base-rounds", type=int, default=300, help="Temel model arena soru sayısı")
    parser.add_argument("--base-reps", type=int, default=3, help="Temel model arena tekrar sayısı")
    parser.add_argument("--carpenter-rounds", type=int, default=300, help="Marangozluk arena soru sayısı")
    parser.add_argument("--train-steps", type=int, default=200, help="Pretrain, SFT ve Chat eğitim adım sayısı")
    parser.add_argument("--dpo-steps", type=int, default=100, help="DPO adım sayısı")
    parser.add_argument("--carpenter-steps", type=int, default=150, help="Marangozluk eğitim adım sayısı")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Donanım cihazı (mps/cpu)")
    parser.add_argument("--allow-frozen-write", action="store_true", default=False, help="Donmuş kütüklere (data/*.pt, data/*.bin vb.) yazma izni ver")
    args = parser.parse_args()

    device = args.device
    base_rounds = args.base_rounds
    base_reps = args.base_reps
    carpenter_rounds = args.carpenter_rounds
    train_steps = str(args.train_steps)
    dpo_steps = str(args.dpo_steps)
    carpenter_steps = str(args.carpenter_steps)
    allow_frozen_write = args.allow_frozen_write

    print(f"{C_BOLD}{C_GREEN}======================================================================")
    print("  KRİSTAL-VEKTÖREL: OTONOM /GOAL EĞİTİM VE UZMANLAŞMA BORU HATTI")
    print("======================================================================")
    print(f"Cihaz:                  {device}")
    print(f"Temel Model Arenası:    {base_rounds} Soru x {base_reps} Tekrar (Hedef: Lise & Z Kuşağı)")
    print(f"Full Temel Eğitim:      Pretrain ({train_steps}), SFT ({train_steps}), Chat ({train_steps}), DPO ({dpo_steps})")
    print(f"Marangozluk Arenası:    {carpenter_rounds} Soru (Hedef: Ahşap Uzmanlığı, Eğitim: {carpenter_steps} adım)")
    print("======================================================================\n" + C_RESET, flush=True)

    # -------------------------------------------------------------------------
    # AŞAMA 1: TEMEL MODEL ARENA EĞİTİMİ (LİSE & Z KUŞAĞI)
    # -------------------------------------------------------------------------
    log_phase(f"AŞAMA 1: TEMEL MODEL LİSE & Z KUŞAĞI ARENASI ({base_rounds} Tur x {base_reps} Tekrar)")
    
    gateway_base = AgentGateway.create_default(model_path="data/kristal_model.pt", device=device)
    retrain_pipeline = RetrainPipeline(device=device)
    supervisor_base = PedagogicalSupervisor(
        gateway=gateway_base,
        retrain_pipeline=retrain_pipeline,
        enrich_rag=True
    )

    base_probes = get_curriculum_probes("highschool_genz", count=base_rounds)
    print(f"Yüklenen Lise & Z Kuşağı Probları: {len(base_probes)} adet\n", flush=True)

    accumulated_base_records = []
    base_arena_archive = "data/pedagogy/arena_base_accumulated.jsonl"
    os.makedirs("data/pedagogy", exist_ok=True)

    total_base_steps = 0
    t0_base = time.time()

    for rep in range(base_reps):
        print(f"\n{C_MAGENTA}--- [Temel Model Arenası] Tekrar (Epoch) {rep + 1}/{base_reps} Başlatıldı ---{C_RESET}", flush=True)
        for idx, probe in enumerate(base_probes):
            total_base_steps += 1
            t_step0 = time.time()
            step_res = supervisor_base.execute_supervision_step(probe, auto_inject=True, auto_retrain=False)
            dt_step = time.time() - t_step0
            
            # Save enriched QA record (strictly pure declarative knowledge)
            clean_card = step_res.get("clean_card") or step_res.get("injected_text") or probe.get("knowledge_to_inject", "")
            knowledge = sanitize_teacher_card(clean_card) or probe.get("knowledge_to_inject", "")
            rec = {
                "instruction": probe.get("instruction", "Lise düzeyinde eğitim almış Z kuşağı genci olarak açıkla."),
                "input": probe["query"],
                "output": knowledge,
                "domain": "highschool_genz",
                "rep": rep + 1
            }
            with open(base_arena_archive, "a", encoding="utf-8") as af:
                af.write(json.dumps(rec, ensure_ascii=False) + "\n")
            accumulated_base_records.append(rec)

            fa = step_res["first_attempt"]
            status_tag = "Yeterli" if step_res["is_satisfactory"] else ("Zenginleştirildi (Gemini)" if step_res.get("gemini_enriched") else "Enjekte Edildi")
            q_short = (probe['query'][:36] + "..") if len(probe['query']) > 38 else probe['query']
            print(f"  [Aşama 1 | Tkr {rep + 1}/{base_reps} | Soru {idx + 1:3d}/{len(base_probes)}] {q_short:<38} | {status_tag} | H(z)={fa['entropy_post']:.2f} | RAG={fa['rag_score']:.2f} ({dt_step:.1f}s)", flush=True)

    gateway_base.close()
    dt_base_arena = time.time() - t0_base
    print(f"\n{C_GREEN}[Aşama 1 Tamamlandı] Temel model için {total_base_steps} pedagojik sınav tamamlandı ({dt_base_arena:.1f} sn).{C_RESET}")

    # -------------------------------------------------------------------------
    # AŞAMA 2: VERİ DERLEME & FULL TEMEL MODEL EĞİTİMİ (4 AŞAMA)
    # -------------------------------------------------------------------------
    log_phase("AŞAMA 2: BİRİKEN VERİLERİN DERLENMESİ VE FULL TEMEL MODEL EĞİTİMİ")

    # 1. Update SFT dataset with base arena archive
    print("1. Dengeli SFT Veri Kümesi Güncelleniyor...")
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/high_school_foundation_dataset.jsonl",
            "data/pedagogy/middle_school_chat.jsonl",
            "data/pedagogy/turk_tarihi_sft.jsonl",
            "data/pedagogy/literature_poetry_dataset.jsonl",
            "data/pedagogy/infancy_dataset.jsonl",
            "data/pedagogy/parenting_dataset.jsonl",
            base_arena_archive,
            "data/future_train_vector.jsonl"
        ],
        output_bin="data/train_balanced_sft.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64
    )

    # 2. Update Chat SFT dataset with base arena archive
    print("2. Chat SFT Veri Kümesi Güncelleniyor...")
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/chat_conversations.jsonl",
            "data/pedagogy/turk_tarihi_chat.jsonl",
            base_arena_archive,
            "data/future_train_vector.jsonl"
        ],
        output_bin="data/train_chat_balanced.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64
    )

    python_bin = sys.executable

    # Stage 1: Pretraining
    print(f"\n{C_BOLD}[2.1 / 4] Stage-1: Temel Ön Eğitim (Pretraining {train_steps} adım)...{C_RESET}")
    stage1_cmd = [python_bin, "train.py", "--device", device, "--data", "data/train.bin", "--steps", train_steps, "--from-scratch", "--save-path", "data/kristal_model.pt"]
    if allow_frozen_write:
        stage1_cmd.append("--allow-frozen-write")
    check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
    run_cmd(stage1_cmd, "Stage-1 Pretraining")

    # Stage 2: SFT Fine-Tuning
    print(f"\n{C_BOLD}[2.2 / 4] Stage-2: Dengeli SFT Eğitimi ({train_steps} adım)...{C_RESET}")
    stage2_cmd = [python_bin, "train.py", "--device", device, "--data", "data/train_balanced_sft_v2.bin", "--steps", train_steps, "--load-path", "data/kristal_model.pt", "--save-path", "data/kristal_model.pt"]
    if allow_frozen_write:
        stage2_cmd.append("--allow-frozen-write")
    check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
    run_cmd(stage2_cmd, "Stage-2 SFT")

    # Stage 3: Chat SFT
    print(f"\n{C_BOLD}[2.3 / 4] Stage-3: Chat SFT Eğitimi ({train_steps} adım)...{C_RESET}")
    stage3_cmd = [python_bin, "train.py", "--device", device, "--data", "data/train_chat_balanced.bin", "--steps", train_steps, "--load-path", "data/kristal_model.pt", "--save-path", "data/kristal_model.pt"]
    if allow_frozen_write:
        stage3_cmd.append("--allow-frozen-write")
    check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
    run_cmd(stage3_cmd, "Stage-3 Chat SFT")

    # Stage 4: DPO Alignment (CPU)
    print(f"\n{C_BOLD}[2.4 / 4] Stage-4: DPO Tercih Hizalama Eğitimi ({dpo_steps} adım)...{C_RESET}")
    import shutil
    check_frozen_save_path("data/kristal_model_sft.pt", allow_frozen_write=allow_frozen_write)
    shutil.copyfile("data/kristal_model.pt", "data/kristal_model_sft.pt")
    stage4_cmd = [python_bin, "train_dpo.py", "--steps", dpo_steps]
    if allow_frozen_write:
        stage4_cmd.append("--allow-frozen-write")
    check_frozen_save_path("data/kristal_model.pt", allow_frozen_write=allow_frozen_write)
    run_cmd(stage4_cmd, "Stage-4 DPO Alignment")

    print(f"\n{C_GREEN}Full Temel Model Eğitimi Başarıyla Tamamlandı: data/kristal_model.pt{C_RESET}\n")

    # -------------------------------------------------------------------------
    # AŞAMA 3: MARANGOZLUK MODÜLÜ ARENASI (300 ADIM)
    # -------------------------------------------------------------------------
    log_phase(f"AŞAMA 3: MARANGOZLUK UZMANLIK ARENASI ({carpenter_rounds} Adım)")

    gateway_carp = AgentGateway.create_default(model_path="data/kristal_model.pt", device=device)
    supervisor_carp = PedagogicalSupervisor(
        gateway=gateway_carp,
        retrain_pipeline=retrain_pipeline,
        enrich_rag=True
    )

    carp_probes = get_curriculum_probes("carpenter", count=carpenter_rounds)
    print(f"Yüklenen Marangozluk Probları: {len(carp_probes)} adet\n", flush=True)

    carpenter_arena_archive = "data/pedagogy/arena_carpenter_accumulated.jsonl"
    t0_carp = time.time()

    for idx, probe in enumerate(carp_probes):
        t_step0 = time.time()
        step_res = supervisor_carp.execute_supervision_step(probe, auto_inject=True, auto_retrain=False)
        dt_step = time.time() - t_step0
        
        clean_card = step_res.get("clean_card") or step_res.get("injected_text") or probe.get("knowledge_to_inject", "")
        knowledge = sanitize_teacher_card(clean_card) or probe.get("knowledge_to_inject", "")
        rec = {
            "instruction": probe.get("instruction", "Ahşap ve marangozluk uzmanı olarak cevapla."),
            "input": probe["query"],
            "output": knowledge,
            "domain": "carpenter"
        }
        with open(carpenter_arena_archive, "a", encoding="utf-8") as af:
            af.write(json.dumps(rec, ensure_ascii=False) + "\n")

        fa = step_res["first_attempt"]
        status_tag = "Yeterli" if step_res["is_satisfactory"] else ("Zenginleştirildi (Gemini)" if step_res.get("gemini_enriched") else "Enjekte Edildi")
        q_short = (probe['query'][:36] + "..") if len(probe['query']) > 38 else probe['query']
        print(f"  [Aşama 3 | Soru {idx + 1:3d}/{len(carp_probes)}] {q_short:<38} | {status_tag} | H(z)={fa['entropy_post']:.2f} | RAG={fa['rag_score']:.2f} ({dt_step:.1f}s)", flush=True)

    gateway_carp.close()
    dt_carp_arena = time.time() - t0_carp
    print(f"\n{C_GREEN}[Aşama 3 Tamamlandı] Marangozluk için {len(carp_probes)} adım arena tamamlandı ({dt_carp_arena:.1f} sn).{C_RESET}", flush=True)

    # -------------------------------------------------------------------------
    # AŞAMA 4: MARANGOZLUK MODÜLÜNÜN YENİDEN EĞİTİMİ (MPS ÜZERİNDE)
    # -------------------------------------------------------------------------
    log_phase("AŞAMA 4: MARANGOZLUK UZMANLIK MODÜLÜNÜN YENİDEN EĞİTİMİ")

    print("1. Marangozluk Uzmanlık Veri Kümesi Güncelleniyor...", flush=True)
    compile_jsonl_to_bin(
        jsonl_paths=[
            "data/pedagogy/carpenter_specialization_dataset.jsonl",
            carpenter_arena_archive
        ],
        output_bin="data/train_carpenter_specialization.bin",
        vocab_path="data/rebuild/vocab_base_32852.json",
        literal_entity_mode=True,
        block_size=64
    )

    print(f"\n2. Marangozluk Modülü Eğitiliyor ({device} üzerinde {carpenter_steps} adım)...", flush=True)
    carpenter_cmd = [
        python_bin, "train.py",
        "--device", device,
        "--base-model", "data/kristal_model.pt",
        "--output-model", "data/kristal_carpenter_model.pt",
        "--data", "data/train_carpenter_specialization.bin",
        "--steps", carpenter_steps,
        "--lr", "0.0003"
    ]
    if allow_frozen_write:
        carpenter_cmd.append("--allow-frozen-write")
    check_frozen_save_path("data/kristal_carpenter_model.pt", allow_frozen_write=allow_frozen_write)
    run_cmd(carpenter_cmd, "Carpenter Specialization Retraining")

    print(f"\n{C_GREEN}Marangozluk Modülü Başarıyla Eğitildi: data/kristal_carpenter_model.pt{C_RESET}\n", flush=True)

    # -------------------------------------------------------------------------
    # AŞAMA 5: DOĞRULAMA VE TEST ÇIKARIMLARI
    # -------------------------------------------------------------------------
    log_phase("AŞAMA 5: KAPSAMLI DOĞRULAMA VE MODEL ÇIKARIM TESTLERİ")

    print(f"{C_CYAN}Doğrulama modelleri yükleniyor...{C_RESET}", flush=True)
    test_gateway = AgentGateway.create_default(model_path="data/kristal_model.pt", device=device)
    test_queries = [
        ("Hücre teorisinin temel ilkeleri nelerdir?", "Lise düzeyinde eğitim almış Z kuşağı genci olarak açıkla."),
        ("Selam nasılsın, sınav haftasındayım çok bunaldım ne yapayım?", "Lise düzeyinde eğitim almış Z kuşağı genci olarak açıkla."),
        ("Lozan Barış Antlaşması'nın Türk milleti için önemi nedir?", "Lise tarih dersi kapsamında açıkla.")
    ]

    print(f"{C_BOLD}--- TEMEL MODEL TESTLERİ ---{C_RESET}", flush=True)
    for q, inst in test_queries:
        print(f"Soru:    {q}", flush=True)
        res = test_gateway.ask(q, instruction=inst, mode="RAG")
        print(f"Cevap:   {res['response_text']}", flush=True)
        print(f"Entropi: {res['entropy_post']:.2f} | RAG Skoru: {res['rag_score']:.3f}\n", flush=True)
    test_gateway.close()

    carp_gateway = AgentGateway.create_default(model_path="data/kristal_carpenter_model.pt", device=device)
    carp_queries = [
        ("Kırlangıç kuyruğu birleştirme nerelerde kullanılır?", "ahşap uzmanı olarak cevapla."),
        ("Lamba zıvana geçme hangi ahşap yüzeylerde uygulanır?", "ahşap uzmanı olarak cevapla.")
    ]

    print(f"{C_BOLD}--- MARANGOZLUK MODELİ TESTLERİ ---{C_RESET}", flush=True)
    for q, inst in carp_queries:
        print(f"Soru:    {q}", flush=True)
        res = carp_gateway.ask(q, instruction=inst, mode="RAG")
        print(f"Cevap:   {res['response_text']}", flush=True)
        print(f"Entropi: {res['entropy_post']:.2f} | RAG Skoru: {res['rag_score']:.3f}\n", flush=True)
    carp_gateway.close()

    print(f"\n{C_BOLD}{C_GREEN}TÜM SÜREÇ BAŞARIYLA TAMAMLANDI!{C_RESET}", flush=True)


if __name__ == "__main__":
    main()
