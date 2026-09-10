#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: PEDAGOJİK SÜPERVİZÖR (PEDAGOGICAL SUPERVISOR)
=======================================================================
Bu modül; Antigravity Agent, Gemini API, Ollama veya yerel sentetik müfredat
aracılığıyla küçük KristalLM modelimizi pedagojik bir sınav ve gelişim
döngüsüne (Teacher-Student Arena) tabi tutar.

Döngü Adımları:
1. Müfredat alanı (Ahşap, Morfoloji, Temel Kavram, Edebi Dil) seçilir ve soru yöneltilir.
2. Modelin cevabı, Shannon Entropisi ve RAG skoru değerlendirilir.
3. Modelde bilgi eksiği/hatası varsa doğru bilgi RAG belleğine (kristal_bellek veya simulasyon_bellek) enjekte edilir.
4. Bilginin model tarafından >= 0.85 uyumla bulunup bulunamadığı kontrol edilir.
5. Model yine de anlayamazsa `future_train_vector.jsonl` kaydı doğrulanır.
6. Biriken kayıt eşiği aşıldığında model otomatik yeniden eğitime (retrain) alınır.
"""

import os
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from src.gateway.agent_gateway import AgentGateway
from src.gateway.retrain_pipeline import RetrainPipeline


# Yerleşik Zengin Pedagojik Müfredat ve Bilgi Kartları (Antigravity Offline / Standalone Teacher)
CURRICULUM_PROBES = {
    "carpenter": [
        {
            "query": "Kırlangıç kuyruğu birleştirme nerelerde kullanılır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "Kırlangıç kuyruğu geçme, çekme kuvvetine karşı olağanüstü mukavemet gösterdiği için özellikle çekmece kasalarında, sandık köşelerinde ve masif gövde birleştirmelerinde tercih edilir.",
            "expected_keywords": ["çekmece", "sandık", "kasa", "kuvvet", "mukavemet", "köşe"]
        },
        {
            "query": "Lamba zıvana geçme hangi ahşap yüzeylerde uygulanır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "Lamba zıvana geçme tekniği, geniş masif ahşap yüzeylerde, döşeme tahtalarında, tavan kaplamalarında ve ahşap panellerde mevsimsel genleşmeyi dengelemek için uygulanır.",
            "expected_keywords": ["geniş", "yüzey", "döşeme", "tavan", "panel", "kaplama"]
        },
        {
            "query": "Ahşap kurutmada kereste nem oranı masif mobilya için yüzde kaç olmalıdır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "İç mekan masif ahşap mobilya imalatında kullanılacak kerestenin nem oranı yüzde 8 ile 12 arasında dengelenmiş olmalıdır. Bu oran ahşabın çatlamasını ve dönmesini engeller.",
            "expected_keywords": ["8", "12", "yüzde", "nem", "iç mekan", "mobilya"]
        }
    ],
    "pedagogy": [
        {
            "query": "Evden",
            "instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Türkçede -den / -dan / -ten / -tan ekleri ismin ayrılma (çıkma) hâl ekidir (CASE_ABL). Örnek: evden, sokaktan.",
            "expected_keywords": ["CASE_ABL", "ayrılma", "çıkma"]
        },
        {
            "query": "Kitaplar",
            "instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Türkçede -ler / -lar ekleri çoğul ekidir (PLURAL). Örnek: kitaplar, ağaçlar.",
            "expected_keywords": ["PLURAL", "çoğul", "evet"]
        }
    ],
    "literary": [
        {
            "query": "Yiğit duldasında ne saklanır?",
            "instruction": "Belgeye göre cevapla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Yiğit duldasında yiğit saklanır atasözü; mert ve cömert insanların himayesinde başka mert ve yetenekli kişilerin yetişip korunacağını ifade eder.",
            "expected_keywords": ["yiğit", "saklanır", "mert"]
        }
    ],
    "highschool": [
        {
            "query": "Hücre teorisinin temel ilkeleri nelerdir?",
            "instruction": "Lise biyoloji dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Tüm canlılar bir ya da birden fazla hücreden oluşur. Hücre canlılığın en temel yapısal ve işlevsel birimidir. Yeni hücreler var olan hücrelerin bölünmesiyle oluşur.",
            "expected_keywords": ["hücre", "canlı", "birim", "bölünme"]
        },
        {
            "query": "Newton'ın temel yasası F=ma neyi ifade eder?",
            "instruction": "Lise fizik dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Temel yasa (2. Yasa); bir cisme etki eden net kuvvetin, cismin kütlesi ile ivmesinin çarpımına eşit olduğunu ifade eder. Net kuvvet arttıkça cismin ivmesi doğru orantılı olarak artar.",
            "expected_keywords": ["kuvvet", "kütle", "ivme", "net"]
        },
        {
            "query": "Isı ile sıcaklık arasındaki fark nedir?",
            "instruction": "Lise fizik dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Sıcaklık, bir maddedeki moleküllerin ortalama kinetik enerjisinin bir göstergesidir ve termometre ile ölçülür. Isı ise sıcaklık farkından dolayı bir maddeden diğerine aktarılan toplam termal enerjidir.",
            "expected_keywords": ["sıcaklık", "kinetik", "enerji", "termometre", "ısı"]
        },
        {
            "query": "Lozan Barış Antlaşması'nın Türk milleti için önemi nedir?",
            "instruction": "Lise tarih dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Lozan Barış Antlaşması, Türkiye Cumhuriyeti'nin bağımsızlığını ve sınırlarını uluslararası alanda tescilleyen, kapitülasyonları kaldıran kurucu antlaşmadır.",
            "expected_keywords": ["bağımsızlık", "sınır", "kapitülasyon", "cumhuriyet"]
        }
    ],
    "arena_mix": [
        {
            "query": "Kırlangıç kuyruğu birleştirme nerelerde kullanılır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "Kırlangıç kuyruğu geçme, çekme kuvvetine karşı olağanüstü mukavemet gösterdiği için özellikle çekmece kasalarında, sandık köşelerinde ve masif gövde birleştirmelerinde tercih edilir.",
            "expected_keywords": ["çekmece", "sandık", "kasa", "kuvvet", "mukavemet", "köşe"]
        },
        {
            "query": "Lamba zıvana geçme hangi ahşap yüzeylerde uygulanır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "Lamba zıvana geçme tekniği, geniş masif ahşap yüzeylerde, döşeme tahtalarında, tavan kaplamalarında ve ahşap panellerde mevsimsel genleşmeyi dengelemek için uygulanır.",
            "expected_keywords": ["geniş", "yüzey", "döşeme", "tavan", "panel", "kaplama"]
        },
        {
            "query": "Ahşap kurutmada kereste nem oranı masif mobilya için yüzde kaç olmalıdır?",
            "instruction": "ahşap uzmanı olarak cevapla.",
            "target_collection": "kristal_bellek",
            "knowledge_to_inject": "İç mekan masif ahşap mobilya imalatında kullanılacak kerestenin nem oranı yüzde 8 ile 12 arasında dengelenmiş olmalıdır. Bu oran ahşabın çatlamasını ve dönmesini engeller.",
            "expected_keywords": ["8", "12", "yüzde", "nem", "iç mekan", "mobilya"]
        },
        {
            "query": "Evden",
            "instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Türkçede -den / -dan / -ten / -tan ekleri ismin ayrılma (çıkma) hâl ekidir (CASE_ABL). Örnek: evden, sokaktan.",
            "expected_keywords": ["CASE_ABL", "ayrılma", "çıkma"]
        },
        {
            "query": "Kitaplar",
            "instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Türkçede -ler / -lar ekleri çoğul ekidir (PLURAL). Örnek: kitaplar, ağaçlar.",
            "expected_keywords": ["PLURAL", "çoğul", "evet"]
        },
        {
            "query": "Yiğit duldasında ne saklanır?",
            "instruction": "Belgeye göre cevapla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Yiğit duldasında yiğit saklanır atasözü; mert ve cömert insanların himayesinde başka mert ve yetenekli kişilerin yetişip korunacağını ifade eder.",
            "expected_keywords": ["yiğit", "saklanır", "mert"]
        },
        {
            "query": "Hücre teorisinin temel ilkeleri nelerdir?",
            "instruction": "Lise biyoloji dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Tüm canlılar bir ya da birden fazla hücreden oluşur. Hücre canlılığın en temel yapısal ve işlevsel birimidir. Yeni hücreler var olan hücrelerin bölünmesiyle oluşur.",
            "expected_keywords": ["hücre", "canlı", "birim", "bölünme"]
        },
        {
            "query": "Newton'ın temel yasası F=ma neyi ifade eder?",
            "instruction": "Lise fizik dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Temel yasa (2. Yasa); bir cisme etki eden net kuvvetin, cismin kütlesi ile ivmesinin çarpımına eşit olduğunu ifade eder. Net kuvvet arttıkça cismin ivmesi doğru orantılı olarak artar.",
            "expected_keywords": ["kuvvet", "kütle", "ivme", "net"]
        },
        {
            "query": "Isı ile sıcaklık arasındaki fark nedir?",
            "instruction": "Lise fizik dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Sıcaklık, bir maddedeki moleküllerin ortalama kinetik enerjisinin bir göstergesidir ve termometre ile ölçülür. Isı ise sıcaklık farkından dolayı bir maddeden diğerine aktarılan toplam termal enerjidir.",
            "expected_keywords": ["sıcaklık", "kinetik", "enerji", "termometre", "ısı"]
        },
        {
            "query": "Lozan Barış Antlaşması'nın Türk milleti için önemi nedir?",
            "instruction": "Lise tarih dersi kapsamında açıkla.",
            "target_collection": "simulasyon_bellek",
            "knowledge_to_inject": "Lozan Barış Antlaşması, Türkiye Cumhuriyeti'nin bağımsızlığını ve sınırlarını uluslararası alanda tescilleyen, kapitülasyonları kaldıran kurucu antlaşmadır.",
            "expected_keywords": ["bağımsızlık", "sınır", "kapitülasyon", "cumhuriyet"]
        }
    ]
}


def get_curriculum_probes(domain: str = "arena_mix", count: int = 10) -> List[Dict[str, Any]]:
    """
    Returns up to `count` probes for the given domain.
    If count exceeds built-in CURRICULUM_PROBES, loads additional high quality probes
    from data/pedagogy/high_school_foundation_dataset.jsonl.
    """
    probes = list(CURRICULUM_PROBES.get(domain, CURRICULUM_PROBES["arena_mix"]))
    if len(probes) >= count:
        return probes[:count]
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lit_path = os.path.join(base_dir, "data", "pedagogy", "literature_poetry_dataset.jsonl")
    
    if domain in ("literary", "poetry", "edebiyat"):
        extra_sources = [
            ("simulasyon_bellek", lit_path),
            ("simulasyon_bellek", os.path.join(base_dir, "data", "pedagogy", "high_school_foundation_dataset.jsonl")),
        ]
    else:
        extra_sources = [
            ("simulasyon_bellek", os.path.join(base_dir, "data", "pedagogy", "high_school_foundation_dataset.jsonl")),
            ("simulasyon_bellek", lit_path),
            ("kristal_bellek", os.path.join(base_dir, "data", "pedagogy", "carpenter_specialization_dataset.jsonl")),
            ("simulasyon_bellek", os.path.join(base_dir, "data", "pedagogy", "middle_school_chat.jsonl")),
        ]
    seen_queries = {p["query"].strip().lower() for p in probes}
    
    import re
    stopwords = {
        "olan", "için", "gibi", "kadar", "veya", "ancak", "çünkü", "böyle", 
        "olarak", "şekilde", "vardır", "yoktur", "denir", "göre", "tarafından", 
        "bunun", "şudur", "şunlardır", "cevap", "soru", "lütfen", "açıklar"
    }

    for target_col, file_path in extra_sources:
        if len(probes) >= count:
            break
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(probes) >= count:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    q = data.get("input", "").strip()
                    if q.startswith("Soru:"):
                        q = q[5:].strip()
                    if not q or q.lower() in seen_queries or len(q) < 5:
                        continue
                    seen_queries.add(q.lower())
                    ans = data.get("output", "").strip()
                    if ans.startswith("Cevap:"):
                        ans = ans[6:].strip()
                        
                    kws = data.get("expected_keywords")
                    if not kws:
                        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}\b', ans.lower())
                        keywords = [w for w in words if w not in stopwords][:5]
                        if not keywords:
                            keywords = [w for w in re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b', ans.lower())][:3]
                    else:
                        keywords = kws
                    if not keywords:
                        continue
                        
                    probes.append({
                        "query": q,
                        "instruction": data.get("instruction", "Belirtilen konuda uzman ve bilimsel doğrulukla açıkla."),
                        "target_collection": target_col,
                        "knowledge_to_inject": ans,
                        "expected_keywords": keywords
                    })
                except Exception:
                    continue
                    
    return probes[:count]


class PedagogicalSupervisor:
    def __init__(
        self,
        gateway: AgentGateway,
        retrain_pipeline: Optional[RetrainPipeline] = None,
        retrain_threshold: int = 5,
        gemini_api_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: str = "gpt-oss:20b",
        ollama_api_key: Optional[str] = None,
        enrich_rag: bool = True
    ):
        self.gateway = gateway
        self.retrain_pipeline = retrain_pipeline or RetrainPipeline()
        self.retrain_threshold = retrain_threshold
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL")
        self.ollama_model = ollama_model
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY")
        self.enrich_rag = enrich_rag
        self.history: List[Dict[str, Any]] = []

    def call_ollama(self, prompt: str) -> Optional[str]:
        """Calls external Ollama API (supports both /v1/chat/completions and /api/generate)."""
        if not self.ollama_url:
            return None
            
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        try:
            if "/v1" in self.ollama_url:
                api_url = f"{self.ollama_url.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                }
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or msg.get("reasoning", "")
                    return content.strip() if content else None
            else:
                api_url = f"{self.ollama_url.rstrip('/')}/api/generate"
                payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "").strip()
        except Exception:
            return None

    def call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API (gemini-2.5-flash) for expert pedagogical knowledge synthesis."""
        if not self.gemini_api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "thinkingConfig": {"thinkingBudget": 0},
                "maxOutputTokens": 500,
                "temperature": 0.2
            }
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception:
            return None

    def evaluate_response_quality(
        self,
        response_text: str,
        expected_keywords: List[str]
    ) -> Tuple[bool, str]:
        """
        Heuristic / keyword and length evaluation of student model response.
        """
        resp_lower = response_text.lower()
        if not response_text.strip():
            return False, "Model boş yanıt verdi."
            
        matched = [k for k in expected_keywords if k.lower() in resp_lower]
        coverage = len(matched) / max(len(expected_keywords), 1)
        
        if coverage >= 0.35: # At least partial expected concept covered
            return True, f"Yeterli kavrayış (%{int(coverage*100)} anahtar terim eşleşmesi: {matched})."
        else:
            return False, f"Eksik/Yetersiz kavrayış (Eşleşen: {matched} / Beklenen: {expected_keywords})."

    def execute_supervision_step(
        self,
        probe: Dict[str, Any],
        auto_inject: bool = True,
        auto_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Runs a complete pedagogical supervision round:
        1. Asks question.
        2. Evaluates answer.
        3. Injects knowledge into RAG if answer is lacking or uncertain.
        4. Re-probes to test retrieval and comprehension.
        5. Checks future_train_vector.jsonl and triggers retrain if threshold reached.
        """
        query = probe["query"]
        instruction = probe.get("instruction", "Belgeye göre cevapla.")
        target_coll = probe.get("target_collection", "kristal_bellek")
        knowledge_text = probe.get("knowledge_to_inject", "")
        expected_keywords = probe.get("expected_keywords", [])
        
        step_log = {
            "query": query,
            "instruction": instruction,
            "target_collection": target_coll,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Step 1: Initial Ask
        first_attempt = self.gateway.ask(query, instruction=instruction, mode="RAG")
        step_log["first_attempt"] = first_attempt
        
        is_satisfactory, eval_msg = self.evaluate_response_quality(
            first_attempt["response_text"],
            expected_keywords
        )
        step_log["is_satisfactory"] = is_satisfactory
        step_log["evaluation_message"] = eval_msg
        
        # Step 2: Knowledge Injection & Enrichment if needed
        injection_result = None
        reprobe_attempt = None
        
        if (not is_satisfactory or first_attempt["epistemic_failure"] or first_attempt["rag_score"] < 0.85) and auto_inject and knowledge_text:
            text_to_inject = knowledge_text
            teacher_card = None
            teacher_provider = None

            if self.enrich_rag:
                enrich_prompt = (
                    f"Sen Türk dili, edebiyatı ve bilimi uzmanı bir öğretmensin. Öğrencinin şu sorusuna en fazla 2-3 cümlelik net, "
                    f"öz ve pedagojik olarak kusursuz bir bilgi kartı hazırla:\n"
                    f"Soru: {query}\n"
                    f"Temel Kavram: {knowledge_text}"
                )
                # Prioritize Gemini API if available (superior for Turkish literature/poetry)
                if self.gemini_api_key:
                    teacher_card = self.call_gemini(enrich_prompt)
                    if teacher_card and len(teacher_card.strip()) >= 20:
                        teacher_provider = "Gemini (gemini-2.5-flash)"
                        step_log["gemini_enriched"] = True

                # Fallback to Ollama if Gemini not available
                if not teacher_card and self.ollama_url:
                    teacher_card = self.call_ollama(enrich_prompt)
                    if teacher_card and len(teacher_card.strip()) >= 20:
                        teacher_provider = f"Ollama ({self.ollama_model})"
                        step_log["ollama_enriched"] = True

            # Always combine base textbook knowledge with teacher explanation
            if teacher_card:
                text_to_inject = f"{knowledge_text}\n\n[Öğretmen Açıklaması]: {teacher_card.strip()}"
            else:
                text_to_inject = knowledge_text

            injection_result = self.gateway.inject_knowledge(
                text=text_to_inject,
                target_collection=target_coll,
                metadata={"topic": query, "verified_by": teacher_provider or "pedagogical_supervisor", "base_concept": knowledge_text}
            )
            step_log["knowledge_injected"] = injection_result
            step_log["injected_text"] = text_to_inject
            step_log["teacher_provider"] = teacher_provider
            
            # Step 3: Verify Retrieval
            search_check = self.gateway.check_memory(query, target_collection=target_coll, top_k=1)
            step_log["search_check"] = search_check
            top_score = search_check[0]["score"] if search_check else 0.0
            step_log["retrieval_verified"] = bool(top_score >= 0.85)
            
            # Step 4: Re-probe student with newly available knowledge
            reprobe_attempt = self.gateway.ask(query, instruction=instruction, mode="RAG", force_rag=True)
            step_log["reprobe_attempt"] = reprobe_attempt
            
        # Step 5: Check Epistemic Backlog and Retrain if threshold reached
        backlog_count = self.retrain_pipeline.get_pending_count()
        step_log["epistemic_backlog_count"] = backlog_count
        
        retrain_result = None
        if auto_retrain and backlog_count >= self.retrain_threshold:
            retrain_result = self.retrain_pipeline.run_training(steps=50, batch_size=16)
            step_log["retrain_executed"] = retrain_result
            
        self.history.append(step_log)
        return step_log

    def run_arena_session(
        self,
        domain: str = "arena_mix",
        rounds: int = 10,
        auto_retrain: bool = False
    ) -> List[Dict[str, Any]]:
        """Runs multiple rounds across a chosen domain."""
        probes = get_curriculum_probes(domain, count=rounds)
        results = []
        
        for probe in probes:
            res = self.execute_supervision_step(probe, auto_inject=True, auto_retrain=auto_retrain)
            results.append(res)
            
        return results
