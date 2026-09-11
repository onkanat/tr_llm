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
import re
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


def sanitize_teacher_card(raw_text: Optional[str]) -> Optional[str]:
    """
    Cleans and extracts pure Turkish pedagogical knowledge from raw LLM teacher output,
    stripping internal Chain-of-Thought (CoT), English reasoning tokens, and meta-instructions.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
        
    text = raw_text.strip()
    
    # 1. Remove XML/HTML style thought blocks (<think>...</think>, <thought>...</thought>, etc.)
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Filter out English meta-reasoning and CoT preambles line by line
    cot_patterns = [
        r'^\s*we need to\b',
        r'^\s*the user says\b',
        r'^\s*the user:\b',
        r'^\s*they want\b',
        r'^\s*let\'s produce\b',
        r'^\s*here is\b',
        r'^\s*so that should\b',
        r'^\s*then the question\b',
        r'^\s*and they give\b',
        r'^\s*make it\b',
        r'^\s*something like\b',
        r'^\s*but better to be clear\b',
        r'^\s*but the question says\b',
        r'^\s*in this case\b',
        r'^\s*first, let\'s\b',
        r'^\s*to answer this\b',
        r'^\s*concise, pedagogically\b'
    ]
    cot_regex = re.compile('|'.join(cot_patterns), re.IGNORECASE)
    
    cleaned_lines = []
    for line in text.split('\n'):
        line_s = line.strip()
        if not line_s:
            continue
        if cot_regex.search(line_s):
            continue
        cleaned_lines.append(line_s)
        
    candidate = '\n'.join(cleaned_lines).strip()
    if not candidate:
        return None
        
    # 3. Check for English dominance: if candidate is mostly English, reject it
    tr_chars = set('çğıöşüÇĞİÖŞÜ')
    words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]+\b', candidate.lower())
    if not words:
        return None
        
    english_stopwords = {
        'the', 'to', 'and', 'is', 'we', 'user', 'says', 'it', 'that', 'can', 'of',
        'in', 'for', 'on', 'with', 'as', 'at', 'this', 'be', 'are', 'should', 'produced',
        'let', 'create', 'info', 'card', 'sentences', 'solution', 'write', 'they', 'want'
    }
    eng_word_count = sum(1 for w in words if w in english_stopwords)
    tr_char_count = sum(1 for c in candidate if c in tr_chars)
    
    if len(words) > 6 and (eng_word_count / len(words) > 0.20) and tr_char_count < 2:
        return None
        
    # 4. Clean leading label tags
    candidate = re.sub(r'^(Bilgi Kartı|Öğretmen Açıklaması|Özet|Cevap|Teknik Çözüm)\s*:\s*', '', candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(' "\'“’')
    
    if len(candidate) < 15:
        return None
        
    return candidate


def extract_cot_and_card(raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Separates internal Chain-of-Thought (thought_trace) from clean Turkish declarative knowledge (clean_card).
    Handles structured tags (<DUSUNCE>...</DUSUNCE>, <BILGI_KARTI>...</BILGI_KARTI>),
    standard model reasoning tags (<think>, <thought>, <reasoning>),
    and falls back to sanitize_teacher_card for declarative extraction.
    
    Returns:
        (thought_trace, clean_card)
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, None
        
    text = raw_text.strip()
    thought_trace = None
    clean_card = None
    
    # 1. Look for explicit <DUSUNCE>...</DUSUNCE> tag
    dusunce_match = re.search(r'<DUSUNCE>(.*?)</DUSUNCE>', text, flags=re.DOTALL | re.IGNORECASE)
    if dusunce_match:
        thought_trace = dusunce_match.group(1).strip()
    else:
        # Check standard reasoning tags
        think_match = re.search(r'<(think|thought|reasoning)>(.*?)</\1>', text, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            thought_trace = think_match.group(2).strip()

    # 2. Look for explicit <BILGI_KARTI>...</BILGI_KARTI> tag
    card_match = re.search(r'<BILGI_KARTI>(.*?)</BILGI_KARTI>', text, flags=re.DOTALL | re.IGNORECASE)
    if card_match:
        cand = card_match.group(1).strip()
        cand = re.sub(r'^(Bilgi Kartı|Öğretmen Açıklaması|Özet|Cevap|Teknik Çözüm)\s*:\s*', '', cand, flags=re.IGNORECASE)
        cand = cand.strip(' "\'“’')
        if len(cand) >= 15:
            clean_card = cand

    # 3. If explicit <BILGI_KARTI> not found, sanitize remaining text
    if not clean_card:
        rem = text
        rem = re.sub(r'<DUSUNCE>.*?</DUSUNCE>', '', rem, flags=re.DOTALL | re.IGNORECASE)
        rem = re.sub(r'<(think|thought|reasoning)>.*?</\1>', '', rem, flags=re.DOTALL | re.IGNORECASE)
        clean_card = sanitize_teacher_card(rem)

    # 4. If thought_trace not found yet via tags, check for preambles in raw text
    if not thought_trace:
        lines = text.split('\n')
        cot_lines = []
        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue
            if re.search(r'^\s*(we need to|the user says|the user:|they want|let\'s produce|here is|first, let\'s|to answer this)\b', line_s, re.IGNORECASE):
                cot_lines.append(line_s)
        if cot_lines:
            thought_trace = '\n'.join(cot_lines).strip()

    return thought_trace, clean_card


def record_to_cot_vault(
    query: str,
    instruction: str,
    thought_trace: str,
    final_answer: str,
    source: str = "teacher_pedagogy",
    vault_path: str = "data/pedagogy/cot_vault.jsonl"
) -> bool:
    """
    Appends an isolated reasoning record to cot_vault.jsonl.
    """
    if not thought_trace or not final_answer:
        return False
    try:
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "instruction": instruction,
            "thought_trace": thought_trace.strip(),
            "final_answer": final_answer.strip(),
            "source": source
        }
        with open(vault_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        print(f"[CoT Vault] Kayıt hatası: {e}")
        return False


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

    def call_ollama(self, prompt: str, return_raw: bool = False) -> Optional[str]:
        """Calls external Ollama API (supports both /v1/chat/completions and /api/generate) with sanitization."""
        if not self.ollama_url:
            return None
            
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        try:
            raw_text = None
            if "/v1" in self.ollama_url:
                api_url = f"{self.ollama_url.rstrip('/')}/chat/completions"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800
                }
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg = data["choices"][0]["message"]
                    # Do not fall back to reasoning if content is empty
                    raw_text = msg.get("content") or ""
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
                    raw_text = data.get("response", "")
            if return_raw:
                return raw_text
            return sanitize_teacher_card(raw_text)
        except Exception:
            return None

    def call_gemini(self, prompt: str, return_raw: bool = False) -> Optional[str]:
        """Calls Google Gemini API (gemini-2.5-flash) for expert pedagogical knowledge synthesis with sanitization."""
        if not self.gemini_api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 800,
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
                        raw_text = parts[0].get("text", "")
                        if return_raw:
                            return raw_text
                        return sanitize_teacher_card(raw_text)
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
            thought_trace = None
            teacher_provider = None

            if self.enrich_rag:
                enrich_prompt = (
                    f"Sen Türk dili, edebiyatı ve bilimi uzmanı bir öğretmensin.\n"
                    f"Şu soru için önce derinlemesine düşünce adımlarını <DUSUNCE> ... </DUSUNCE> etiketleri arasına,\n"
                    f"ardından öğrenci için en fazla 2-3 cümlelik net, öz ve pedagojik olarak kusursuz bilgi kartını <BILGI_KARTI> ... </BILGI_KARTI> etiketleri arasına yaz.\n\n"
                    f"Soru: {query}\n"
                    f"Temel Kavram: {knowledge_text}"
                )
                raw_teacher_text = None

                # Prioritize Gemini API if available (superior for Turkish literature/poetry)
                if self.gemini_api_key:
                    raw_teacher_text = self.call_gemini(enrich_prompt, return_raw=True)
                    if raw_teacher_text and len(raw_teacher_text.strip()) >= 20:
                        teacher_provider = "Gemini (gemini-2.5-flash)"
                        step_log["gemini_enriched"] = True

                # Fallback to Ollama if Gemini not available
                if not raw_teacher_text and self.ollama_url:
                    raw_teacher_text = self.call_ollama(enrich_prompt, return_raw=True)
                    if raw_teacher_text and len(raw_teacher_text.strip()) >= 20:
                        teacher_provider = f"Ollama ({self.ollama_model})"
                        step_log["ollama_enriched"] = True

                if raw_teacher_text:
                    thought_trace, clean_card = extract_cot_and_card(raw_teacher_text)
                    teacher_card = clean_card

                    # Record reasoning trace to CoT Vault and muhakeme_bellek
                    if thought_trace and len(thought_trace.strip()) >= 15:
                        step_log["thought_trace"] = thought_trace
                        record_to_cot_vault(
                            query=query,
                            instruction=instruction,
                            thought_trace=thought_trace,
                            final_answer=clean_card or knowledge_text,
                            source=teacher_provider or "pedagogical_supervisor"
                        )
                        if hasattr(self.gateway, "inject_reasoning_trace"):
                            self.gateway.inject_reasoning_trace(
                                query=query,
                                thought_text=thought_trace,
                                final_card=clean_card or knowledge_text,
                                domain=target_coll,
                                metadata={"source": teacher_provider or "pedagogical_supervisor", "topic": query}
                            )
                            step_log["reasoning_injected"] = True

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
