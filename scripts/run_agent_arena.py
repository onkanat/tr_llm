#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: AJAN ARENASI (AGENT ARENA CLI)
=========================================================
Büyük agent modelleri (Antigravity Agent, Gemini API, Ollama) ve yerel
pedagojik süpervizör ile küçük KristalLM modelini otonom eğitme ve sınama CLI aracı.

Kullanım:
  # Marangozluk alanında 2 tur pedagojik diyalog ve RAG enjeksiyonu:
  # (T-0088: --model ve --vocab ZORUNLUDUR; ikisi de checkpoint satır sayısıyla
  #  AYNI olmalıdır. Ornek yollar, T-0087'nin önerdiği GÜNCEL Anka checkpoint'i ve
  #  onun külliyat sözlüğüdür — uydurma ad değil, ölçülmüş dosyalardır.)
  ./venv/bin/python scripts/run_agent_arena.py --domain carpenter --rounds 2 \
      --model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json

  # Dilbilgisi alanında otomatik yeniden eğitim bayrağı ile:
  ./venv/bin/python scripts/run_agent_arena.py --domain pedagogy --auto-retrain \
      --model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json

  # Dış agent'lar için HTTP REST API Sunucusu olarak çalıştırma:
  ./venv/bin/python scripts/run_agent_arena.py --server --port 8080 \
      --model data/anka_a1r.pt --vocab data/rebuild/vocab_anka_r1_33114.json
"""

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.agent_gateway import AgentGateway
from src.gateway.retrain_pipeline import RetrainPipeline
from src.gateway.pedagogical_supervisor import PedagogicalSupervisor, CURRICULUM_PROBES, get_curriculum_probes

# Renk Kodları
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_GRAY = "\033[90m"


def main():
    parser = argparse.ArgumentParser(description="Kristal-Vektörel Agent Arena CLI")
    parser.add_argument("--domain", type=str, default="arena_mix", choices=["carpenter", "pedagogy", "literary", "highschool", "highschool_genz", "arena_mix", "poetry", "edebiyat", "history_1931", "turk_tarihi"], help="Eğitim/Soru alanı")
    parser.add_argument("--rounds", type=int, default=10, help="Diyalog tur sayısı")
    parser.add_argument("--repetitions", type=int, default=1, help="Müfredat üzerinden geçilecek tekrar (epoch) sayısı")
    parser.add_argument("--model", type=str, default=None, help="Sınanacak model dosya yolu — ZORUNLU (varsayilan YOK; T-0088)")
    parser.add_argument("--vocab", type=str, default=None, help="Külliyat sözlüğü — ZORUNLU; checkpoint satır sayısıyla AYNI olmalı (T-0088)")
    parser.add_argument("--auto-retrain", action="store_true", help="future_train eşiği aşıldığında modeli otomatik yeniden eğit")
    parser.add_argument("--retrain-threshold", type=int, default=5, help="Otomatik eğitim için biriken kayıt eşiği")
    parser.add_argument("--device", type=str, default="cpu", help="Hesaplama cihazı (cpu / mps)")
    parser.add_argument("--server", action="store_true", help="HTTP REST API sunucusu olarak başlat")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Sunucu adresi")
    parser.add_argument("--port", type=int, default=8080, help="Sunucu portu")
    parser.add_argument("--gemini-api-key", type=str, default=os.environ.get("GEMINI_API_KEY"), help="Google Gemini API anahtarı")
    parser.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_URL", "https://ollama.com/v1"), help="Ollama API URL")
    parser.add_argument("--ollama-model", type=str, default=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"), help="Ollama model adı")
    parser.add_argument("--ollama-api-key", type=str, default=os.environ.get("OLLAMA_API_KEY"), help="Ollama API anahtarı")
    parser.add_argument("--enrich-rag", action="store_true", default=True, help="Öğretmen modeli ile RAG bilgi kartlarını zenginleştir")
    
    args = parser.parse_args()

    # EMEKLI (T-0088): eski varsayilan `--model` degeri `data/kristal_model.pt` idi ve o
    # checkpoint SILINMISTIR. Varsayilan BASKA BIR ADA tasinmadi (T-0085 emsali: uydurma ad
    # yazilmaz) — bunun yerine yoklugu ACIKCA durdurulur. Kapi CAGRI ANINDADIR.
    #
    # OLCUM (T-0088): bu cagri yeri T-0087'DEN BERI ZATEN DURUYORDU, cunku
    # `AgentGateway.create_default` artik `vocab_path`i de ZORUNLU tutuyor ama burada
    # GECILMIYORDU (olculdu: RuntimeError "DURDURULDU: vocab_path verilmedi"). Yani
    # sozluk yolu da ayni kapidan gecer; ikisi tek yerde ve ACIKCA istenir.
    eksik = [ad for ad, deger in (("--model", args.model), ("--vocab", args.vocab)) if not deger]
    if eksik:
        parser.error(
            f"{' ve '.join(eksik)} ZORUNLUDUR. Eskiden varsayilan degerler vardi ve o "
            "artefaktlar artik YOKTUR/BAYATTIR; guncel checkpoint ve onun kulliyat sozlugunu "
            "acikca verin (T-0088). Ornek: --model data/anka_a1r.pt "
            "--vocab data/rebuild/vocab_anka_r1_33114.json"
        )
    yok = [y for y in (args.model, args.vocab) if not os.path.exists(y)]
    if yok:
        # Sessiz dusme YOK: yol verilmis ama dosya yoksa da DURULUR. Sozluk dosyasi
        # okunamazsa `Vocabulary.load` bos sozluk uretir (ayri bir sessizlik sinifi);
        # kapi onu cagri aninda keser (T-0088).
        parser.error("VERILEN YOL YOK: " + " · ".join(yok) + " (T-0088)")

    print(f"\n{C_MAGENTA}{C_BOLD}" + "=" * 65)
    print("  KRİSTAL-VEKTÖREL MİMARİSİ: OTONOM PEDAGOJİK AJAN ARENASI")
    print("=" * 65 + f"{C_RESET}")
    print(f"Model:          {C_CYAN}{args.model}{C_RESET}")
    print(f"Sözlük:         {C_CYAN}{args.vocab}{C_RESET}")
    print(f"Cihaz:          {C_CYAN}{args.device}{C_RESET}")
    print(f"Hedef Alan:     {C_YELLOW}{args.domain}{C_RESET}")
    print(f"Plan:           {C_GREEN}{args.rounds} Tur x {args.repetitions} Tekrar (Toplam: {args.rounds * args.repetitions} Sınav){C_RESET}")
    print(f"Mod:            {C_GREEN}{'REST API Sunucusu' if args.server else 'Otonom Denetim Döngüsü'}{C_RESET}")
    if args.gemini_api_key:
        print(f"Harici Öğretmen: {C_MAGENTA}Google Gemini API (gemini-2.5-flash) [Edebiyat & Bilim Uzmanı]{C_RESET}\n")
    elif args.ollama_url:
        print(f"Harici Öğretmen: {C_MAGENTA}{args.ollama_url} ({args.ollama_model}){C_RESET}\n")
    else:
        print()

    print(f"{C_BLUE}[1/3] Ajan Kapısı ve Bileşenler Yükleniyor...{C_RESET}")
    gateway = AgentGateway.create_default(model_path=args.model, vocab_path=args.vocab, device=args.device)
    retrain_pipeline = RetrainPipeline(device=args.device)
    supervisor = PedagogicalSupervisor(
        gateway=gateway,
        retrain_pipeline=retrain_pipeline,
        retrain_threshold=args.retrain_threshold,
        gemini_api_key=args.gemini_api_key,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_api_key=args.ollama_api_key,
        enrich_rag=args.enrich_rag
    )
    
    status = gateway.get_status()
    print(f"  * Bellek Durumu: {C_YELLOW}kristal_bellek{C_RESET} ({status['kristal_bellek_docs']} belge) | {C_CYAN}simulasyon_bellek{C_RESET} ({status['simulasyon_bellek_docs']} belge)")
    print(f"  * Epistemik Kütük (future_train): {C_MAGENTA}{status['epistemic_backlog_samples']} bekleyen örnek{C_RESET}")

    if args.server:
        print(f"\n{C_GREEN}[REST API] Ajan Kapısı http://{args.host}:{args.port} üzerinde dinleniyor...{C_RESET}")
        print(f"  * POST /api/query   : Soru sorma ve telemetri alma")
        print(f"  * POST /api/inject  : RAG'a yeni bilgi ekleme")
        print(f"  * GET  /api/status  : Sistem ve bellek durumu")
        print(f"  * GET  /api/backlog : future_train kayıtları")
        http_server = gateway.create_http_server(host=args.host, port=args.port)
        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Sunucu durduruldu.{C_RESET}")
            http_server.server_close()
        return

    # Otonom Arena Döngüsü
    probes = get_curriculum_probes(args.domain, count=args.rounds)
    print(f"\n{C_BLUE}[2/3] Pedagojik Süpervizör Sınavı Başlatıyor ({len(probes)} Soru, {args.repetitions} Tekrar)...{C_RESET}")

    total_probes_run = 0
    satisfactory_count = 0
    injected_count = 0

    for rep in range(args.repetitions):
        print(f"\n{C_MAGENTA}{C_BOLD}=================== TEKRAR (EPOCH) {rep + 1}/{args.repetitions} ==================={C_RESET}")
        for r_idx, probe in enumerate(probes):
            total_probes_run += 1
            print(f"\n{C_BOLD}------------------------------------------------------------{C_RESET}")
            print(f"{C_MAGENTA}{C_BOLD}[Tekrar {rep + 1}/{args.repetitions} | Tur {r_idx + 1}/{len(probes)}] Soru:{C_RESET} {C_BOLD}{probe['query']}{C_RESET}")
            
            step_res = supervisor.execute_supervision_step(
                probe,
                auto_inject=True,
                auto_retrain=args.auto_retrain
            )
            
            fa = step_res["first_attempt"]
            print(f"  {C_CYAN}Çırak Model Yanıtı:{C_RESET} {fa['response_text']}")
            print(f"  {C_GRAY}Ön-Entropi: {fa['entropy_pre']:.2f} | Son-Entropi: {fa['entropy_post']:.2f} | RAG Skoru: {fa['rag_score']:.3f}{C_RESET}")
            print(f"  {C_BOLD}Öğretmen Değerlendirmesi:{C_RESET} {C_GREEN if step_res['is_satisfactory'] else C_YELLOW}{step_res['evaluation_message']}{C_RESET}")
            
            if step_res["is_satisfactory"]:
                satisfactory_count += 1
            
            if "knowledge_injected" in step_res:
                injected_count += 1
                teacher_prov = step_res.get("teacher_provider")
                if teacher_prov:
                    print(f"  {C_MAGENTA}[Öğretmen Zenginleştirmesi]{C_RESET} Bilgi kartı {teacher_prov} tarafından derinleştirildi.")
                print(f"  {C_MAGENTA}[RAG Enjeksiyonu Yapıldı]{C_RESET} Doğru bilgi kartı '{probe.get('target_collection')}' koleksiyonuna kaydedildi.")
                print(f"  {C_BOLD}Bulunabilirlik Testi:{C_RESET} {C_GREEN if step_res.get('retrieval_verified') else C_RED}Uyum Skoru >= 0.85 Doğrulandı: {step_res.get('retrieval_verified')}{C_RESET}")
                
                ra = step_res.get("reprobe_attempt")
                if ra:
                    print(f"  {C_CYAN}Pekiştirme Sonrası Model Yanıtı:{C_RESET} {ra['response_text']}")
                    print(f"  {C_GRAY}Pekiştirme RAG Skoru: {ra['rag_score']:.3f} | Son-Entropi: {ra['entropy_post']:.2f}{C_RESET}")
                    if ra.get("future_train_recorded"):
                        print(f"  {C_YELLOW}[Epistemik Kayıt] Model yüksek uyumlu belgeyi almasına rağmen anlayamadı -> future_train_vector.jsonl'e eklendi!{C_RESET}")

            if "retrain_executed" in step_res:
                retrain_info = step_res["retrain_executed"]
                print(f"\n  {C_GREEN}{C_BOLD}[OTONOM YENİDEN EĞİTİM TAMAMLANDI]{C_RESET}")
                print(f"  Eğitilen Örnek: {retrain_info.get('samples_trained')} | Adım: {retrain_info.get('steps')} | Süre: {retrain_info.get('duration_sec')} sn")

    print(f"\n{C_BLUE}[3/3] Arena Oturumu Başarıyla Tamamlandı.{C_RESET}")
    print(f"Toplam Sınanan Soru: {total_probes_run} | Başarılı: {satisfactory_count} | Enjeksiyon Yapılan: {injected_count}")
    print(f"Güncel Epistemik Kütük Durumu: {retrain_pipeline.get_pending_count()} bekleyen örnek.\n")
    gateway.close()


if __name__ == "__main__":
    main()
