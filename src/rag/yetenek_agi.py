#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-2 — VEKTÖREL GEZİN v1: YETENEK AĞI (yetenek_agi)
====================================================
Kullanıcının özgün mimarisi (memory_archive/vektorel_gezgin_arch.md):
statik skill.md / istem yüklemesi REDDEDİLİR — yetenekler, özellikler ve
kısıtlar VEKTÖR olarak dinamik uzayda yaşar; sorgu YAKINLIĞI (kosinüs)
ile aktivasyon olur, ağırlıklı etki (agirlik × benzerlik) skorlanır.
Yalnız kök şeması (roots.tsv) statiktir. Bu modül o vizyonun v1'dir:

(a) YetenekKarti  — {kimlik, ad, açıklama/kapsam, etki_alanı, agirlik, meta}
(b) kayıt defteri — Qdrant 'yetenek_agi' koleksiyonu (kalıp: seed_muhakeme_bellek;
                    dense 768 Cosine + sparse IDF; idempotent upsert + read-back)
(c) AKTİVASYON MOTORU — sorgu → vektör üretimi → kosinüs yakınlığı →
                    aktif yetenek kümesi (top-k + eşik) + ağırlıklı etki
(d) sonuç        — {"aktif_yetenekler": [(ad, skor, etki), ...], "karar_notu"}

Determinizm (tohum sabit): (1) KristalTokenizer deterministiktir;
(2) generate_kristal_vector morfem-tohumları md5-tabanlı SABİT çekirdekli
rastgele üretir (hiçbir global tohum çekilmez); (3) sıralama anahtarı
(-etki, -skor, ad) dır — eşitliklerde ad kırıcı deterministiktir.
Aynı sorgu + aynı koleksiyon durumu → birebir aynı sonuç.

VG-2b v2 (2026-11-05): (1) TELEMETRİ — aktivasyon-olay kaydı JSONL
(data/eval/yetenek_telemetri.jsonl: zaman, sorgu-özeti, aktif
yetenekler+skorlar, router kararı) + SONUÇ SİNYALİ (zirve yeteneğin
router-uzman eşleşmesi: uyum +1 / uyumsuz −1 / zemin yoksa kayıt-yalnız);
(2) AĞIRLIK EVRİMİ — deterministik bantlı kural ×(1±eps), [0,5; 2,0],
idempotent payload geri-yazımı + geçmiş telemetriye; (3) DANIŞMA
kancası gateway'de OP-İN'dir (burada yalnız rapor üretir; router KARARI
DEĞİŞMEZ); (4) KISIT KARTLARI — 'kisit' tipi BASKILAYICI kartlar
(çarpan ≤ 1): etki = ağırlık × benzerlik × kısıt-çarpanı.

YETKİ (SERT KURAL): bu modül YALNIZ 'yetenek_agi' koleksiyonuna yazar;
anka_bellek / muhakeme_bellek / simulasyon_bellek DOKUNULMAZ. Farklı
koleksiyonlu bir bellek verildiğinde kurulum ValueError ile durur
(fail-closed — sessiz-yanlış-koleksiyon yazımı yok).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.rag.embedding import generate_kristal_vector, generate_sparse_vector
from src.rag.vector_memory import VectorMemory

KOLEKSIYON = "yetenek_agi"
VEKTOR_BOYUTU = 768

# Varsayılan aktivasyon eşiği (kosinüs yakınlık). KALİBRE EDİLDİ (VG-2 demo
# probu, scripts/yetenek_probe.py, 2026-11-04, 36 kart / 5 sorgu, ham skor):
# ilgili kart isabetleri 0,033–0,262 bandında (tek-nadir-kelime sorgularında
# 'zımpara'→marangoz 0,033 gibi dibin dibine), ilgisiz kuyruk ise ~0,005–0,024
# bandında. 0,02: beş isabetin TAMAMINI alır (min 0,033 > 0,02), derin
# kuyruk gürültüsünü eler; kalan gürültüyü top-k sınırlar. Yüksek eşik
# (0,15) tek-kelime sorgularda ağı KÖR ederdi — ölçümsüz eşik konmaz.
DEFAULT_ESIK = 0.02

KANCA_TOP_K = 3  # gateway gözlem kancası sorgu-başına kaç kart raporlar

# ===========================================================================
# VG-2b v2 — TELEMETRİ / AĞIRLIK EVRİMİ / KISIT SABİTLERİ (2026-11-05)
# ===========================================================================

TELEMETRI_YOLU = "data/eval/yetenek_telemetri.jsonl"
TELEMETRI_OLAY_TIPI = "aktivasyon"
TELEMETRI_EVRIM_TIPI = "agirlik_evrimi"
SORGU_OZETI_MAX = 120  # telemetride sorgu-özeti uzunluğu (karakter)

AGIRLIK_MIN = 0.5   # evrim bandı alt sınırı (bant DIŞI ağırlık fail-closed)
AGIRLIK_MAX = 2.0   # evrim bandı üst sınırı (v1 envanter max'ı ile uyumlu)
EVRIM_EPSILON = 0.05  # iş emri önerisi: uyum ×(1+eps), uyumsuz ×(1−eps)

KISIT_TIPI = "kisit"
KISIT_KOSULLARI = ("hedef_etkin", "gurultu_bandi")

# Kart kimliği → router uzman sınıfı (GERÇEK MoE uzman adları — epistemic_agent
# expert_names: grammar_core, pedagogy, carpenter, gardener, legal). Alan
# ceketleri kendi uzman sınıflarına, dil/derleyici çekirdeği grammar_core'a
# eşlenir. EŞLENMeyen kartlar (rag/router/gateway/veri/ölçüm) uzman
# karşıtlığı OLMAYAN yeteneklerdir → sinyal zemini yok (kayıt-yalnız).
KIMLIK_UZMAN_HARITASI = {
    "marangoz_ceketi": "carpenter",
    "bahcivan_ceketi": "gardener",
    "legal_ceketi": "legal",
    "pedagoji_uzmani": "pedagogy",
    "kristal_derleyici": "grammar_core",
    "morfem_cozumleyici": "grammar_core",
    "kok_sozlugu": "grammar_core",
    "morfolojik_taktikler": "grammar_core",
    "fonoloji_katmani": "grammar_core",
    "kristal_tokenizer": "grammar_core",
    "kristal_vektor_uretme": "grammar_core",
}

_KART_INDEKSI: Optional[Dict[str, Dict[str, Optional[str]]]] = None


def _kart_indeksi() -> Dict[str, Dict[str, Optional[str]]]:
    """ad → {kimlik, uzman} envanter indeksi (tembel, modül-önbellekli)."""
    global _KART_INDEKSI
    if _KART_INDEKSI is None:
        _KART_INDEKSI = {
            k.ad: {"kimlik": k.kimlik, "uzman": KIMLIK_UZMAN_HARITASI.get(k.kimlik)}
            for k in kart_envanteri()
        }
    return _KART_INDEKSI


def ad_kimligi(ad: str) -> Optional[str]:
    """Kart adı → kimlik (envanter dışı ad → None — dürüst)."""
    return _kart_indeksi().get(ad, {}).get("kimlik")


def ad_uzmani(ad: str) -> Optional[str]:
    """Kart adı → router uzman sınıfı (eşleşme yoksa None)."""
    return _kart_indeksi().get(ad, {}).get("uzman")


@dataclass(frozen=True)
class YetenekKarti:
    """Sistemin GERÇEK bir yeteneğinin vektörel kartı (uydurma yok)."""

    kimlik: str
    ad: str
    aciklama: str
    etki_alani: str
    agirlik: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def doc_text(self) -> str:
        """Embedding'e giren kanonik metin (koleksiyon şeması sabittir)."""
        return f"Yetenek: {self.ad}\nAlan: {self.etki_alani}\nKapsam: {self.aciklama}"


def nokta_kimligi(kimlik: str) -> int:
    """Kayıt-kimliği: sha256('yetenek:<kimlik>') → 2^63-mod.

    Aynı kimlik her koşumda AYNI noktaya çarpar → tohumlama idempotent
    (upsert; ikinci koşum çoğaltmaz, günceller).
    """
    return int(
        hashlib.sha256(f"yetenek:{kimlik}".encode("utf-8")).hexdigest()[:16], 16
    ) % (2 ** 63)


# ===========================================================================
# (a→b) KART ENVANTERİ — sistemin GERÇEK yetenekleri (src/ + scripts/ izinden)
# ===========================================================================

def kart_envanteri() -> List[YetenekKarti]:
    """repo envanterinden çıkarılmış gerçek yetenek kartları (36 kart).

    Her kartın meta['kanit'] alanı repoda VAR OLAN gerçek bir dosyadır
    (tests/test_yetenek_agi.py::test_her_kartin_kaniti_repo_icinde_gercek
    bu sözleşmeyi zorlar — uydurma kart yazılamaz).
    """
    K = YetenekKarti
    return [
        # --- derleyici çekirdeği ---
        K("kristal_derleyici", "Kristal Derleyici (CrystalCompiler)",
          "Türkçe yüzey kelimeyi morfem dizisine çözümleyen derleyici çekirdeği: kök bulma, ek ekleme, "
          "analiz üretimi. Kelime çözümleme ve morfolojik analiz: 'gideceğim' gibi kelimeleri kök ve ek "
          "bileşenlerine ayrıştırır.",
          "derleyici", 1.10, {"kanit": "src/compiler/core.py"}),
        K("morfem_cozumleyici", "Morfem Çözümleyici (MorphemeDecompiler)",
          "Etiket dizisini okunabilir yüzey metne geri çevirir: decompile_sentence, morfem birleştirme, "
          "ondalık/rakam blok kuralları. Token etiketlerinden Türkçe cümle üretimi; üretim hattının "
          "çıkış kapısı (chat_prompt, agent_gateway).",
          "derleyici", 1.05, {"kanit": "src/compiler/decompiler.py"}),
        K("kok_sozlugu", "Kök Sözlüğü (LexiconManager)",
          "roots.tsv temelli statik kök şeması: lemma arama, VOICING türevleri, find_stems. Vektörel "
          "Gezgin'in tek statik katmanı; 'hükmü' gibi kelimelerin kök araması buradan yürür.",
          "derleyici", 1.05, {"kanit": "src/compiler/lexicon.py"}),
        K("morfolojik_taktikler", "Morfolojik Taktikler Grafi (build_default_graph)",
          "Ek dizilim kuralları: TENSE/PERSON/POSS/CASE/COPULA şablon listeleri, morphotactics graph "
          "kurulumu. Hangi ek hangi kökten sonra gelebilir sorusunun hakem katmanı.",
          "derleyici", 1.00, {"kanit": "src/compiler/morphotactics.py"}),
        K("fonoloji_katmani", "Fonoloji Katmanı (phonology)",
          "Ünlü uyumu ve ses olayları: ünsüz yumuşaması (voicing), ünlü düşmesi, harmony kuralları. "
          "Yüzey biçimin doğallık denetimi; 'kitabı'/'kitap' ayrımı.",
          "derleyici", 1.00, {"kanit": "src/compiler/phonology.py"}),
        # --- tokenizer / vektör ---
        K("kristal_tokenizer", "Kristal Tokenizer (KristalTokenizer)",
          "Metni kristal token kimliklerine çevirir (encode) ve etiket dizisine geri açar (decode): "
          "kelime çözümleme, apostrof dalı (literal_entity_mode), UNK yönetimi. 'gideceğim kelimesini "
          "çözümle' tipi isteklerin ilk kapısı (vocab 33.114).",
          "tokenizer", 1.15, {"kanit": "src/llm/tokenizer.py"}),
        K("kristal_vektor_uretme", "Kristal Vektör Üretimi (embedding)",
          "768 boyutlu yoğun (dense) ve seyrek (sparse) vektör üretimi: generate_kristal_vector, "
          "generate_sparse_vector, negasyon kutup değişimi, konumsal ağırlık. Vektörel Gezgin'in uzay "
          "haritası; tüm aramanın taşıyıcısı.",
          "tokenizer", 1.15, {"kanit": "src/rag/embedding.py"}),
        # --- rag geri-getirim ---
        K("rag_boru_hatti", "RAG Boru Hattı (RagPipeline)",
          "Kanonik arama-koşullama-üretim hattı: build_query_vectors, retrieve_context, 0.33 eşik "
          "kapısı (RAG_MATCH_THRESHOLD), kanonik prompt zarfı, greedy_decode. Soruya belge bulup modele "
          "koşullayan ana akış.",
          "rag", 1.15, {"kanit": "src/rag/rag_pipeline.py"}),
        K("hibrit_rrf_arama", "Hibrit RRF Araması (hybrid_search)",
          "Yoğun ve seyrek vektörü Reciprocal Rank Fusion ile birleştirir: reciprocal_rank_fusion, "
          "filter_and_rank_candidates, ayırt edici kök filtresi. Geri-getirim sıralamasının füzyon katmanı.",
          "rag", 1.10, {"kanit": "src/rag/hybrid_search.py"}),
        K("vektor_bellek_yonetimi", "Vektör Bellek Yönetimi (VectorMemory)",
          "Qdrant koleksiyon yönetimi: hibrit koleksiyon kurulumu (dense 768 + sparse IDF), idempotent "
          "upsert, hybrid_recall, dense_recall, fail-closed bağlantı politikası. Bellek okuma-yazma "
          "altyapısı; 'Qdrant'ta ne var' sorusunun muhatabı.",
          "rag", 1.10, {"kanit": "src/rag/vector_memory.py"}),
        K("anka_bellek", "Anka Bellek (bildirimsel bilgi)",
          "Bildirimsel bilgi koleksiyonu: kavram tanımları, genel kültür, belgeler. Gateway inject_knowledge "
          "ve check_memory hedefi; 'nedir' sorularının ana geri-getirim denizi.",
          "rag", 1.10, {"kanit": "src/gateway/agent_gateway.py"}),
        K("muhakeme_bellek", "Muhakeme Bellek (CoT Vault)",
          "Akıl yürütme izi koleksiyonu: thought_trace ve final_answer kayıtları (cot_vault.jsonl), "
          "inject_reasoning_trace, seed_muhakeme_bellek. Nasıl düşündüm sorusunun vektörel deposu; adım "
          "adım çözüm izleri.",
          "rag", 1.10, {"kanit": "scripts/seed_muhakeme_bellek.py"}),
        K("simulasyon_bellek", "Simülasyon Bellek (genel bilgi)",
          "Genel/simülasyon bilgi koleksiyonu: gateway general_memory, senaryo ve ortam bilgileri. "
          "seed_simulasyon_bellek ile tohumlanır, ayna-kopya ile yedeklenir.",
          "rag", 1.05, {"kanit": "scripts/seed_simulasyon_bellek.py"}),
        K("epistemik_ajan", "Epistemik Ajan (EpistemicCuriosityAgent)",
          "Sorgu işleme döngüsü: entropi ölçümü, geri-getirim ihtiyacı (needs_retrieval), "
          "future_train_vector.jsonl epistemik backlog kaydı, koşullanma (conditioned). Sorgunun "
          "'bilmiyorum' demesinin dürüst kaydı.",
          "rag", 1.10, {"kanit": "src/rag/epistemic_agent.py"}),
        # --- router / merak ---
        K("uc_modal_router", "Üç-Modal Router (TriModalRouter)",
          "5 uzmanlı yönlendirici: grammar_core, pedagogy, carpenter (marangoz), gardener (bahçıvan), "
          "legal. Prompt + q_merak + ctx_rag üç kanalı top-2 uzman seçer. Merak kanalı val'e +24,75 puan "
          "katıyor (P2.8).",
          "router", 1.00, {"kanit": "src/llm/router.py"}),
        K("merak_motoru", "Merak Motoru (CuriosityEngine)",
          "Entropi tabanlı merak: Shannon H(z) eşiği tau=2.5 aşımında q_merak vektörü sentezler. Bilgi "
          "boşluğu algısı; router'ın merak kanalının kaynağı (trigger_signal).",
          "router", 1.00, {"kanit": "src/rag/merak.py"}),
        # --- alan ceketleri ---
        K("marangoz_ceketi", "Marangoz Alan Ceketi (carpenter)",
          "Ahşap ve marangozluk uzman ceketi: zımpara, rende, planya, testere, çivi, tutkal, ahşap "
          "birleştirme, mobilya, atölye, talaş. modules/marangoz*.pt MoE modülleri; router carpenter sınıfı.",
          "alan", 1.25, {"kanit": "scripts/train_t0209_marangoz.py"}),
        K("bahcivan_ceketi", "Bahçıvan Alan Ceketi (gardener)",
          "Bitki ve bahçe uzman ceketi: fidan dikme, gübre, ağaç yetiştirme, ceviz ağacı, elma, budama, "
          "sulama, hasat, toprak, sera, çiçek. anka_bahcivan.pt; router gardener sınıfı.",
          "alan", 1.25, {"kanit": "scripts/train_t0207_bahcivan.py"}),
        K("legal_ceketi", "Legal Alan Ceketi (hukuk)",
          "İş hukuku ve mevzuat uzman ceketi: iş kanunu maddesi, işçi ücret, fesih, tazminat, iş sözleşmesi, "
          "mevzuat metinleri. anka_legal_replay.pt; router legal sınıfı (P25 terfi, 8/10 hukuk sorgusu).",
          "alan", 1.25, {"kanit": "scripts/train_legal_ceket.py"}),
        K("pedagoji_uzmani", "Pedagoji Uzmanı (pedagogy)",
          "Eğitim-öğretim uzman ceketi: ders konuları, lise müfredatı, soru çözümü, veli-çocuk diyalogları, "
          "okul, öğretmen. Eğitim kartı üretimi; router pedagogy sınıfı.",
          "alan", 1.15, {"kanit": "scripts/generate_high_school_dataset.py"}),
        # --- gateway ---
        K("ajan_kapisi", "Ajan Kapısı (AgentGateway HTTP)",
          "HTTP REST sunucusu: /api/query (ask), /api/inject, /api/check, /api/status, /api/backlog. Dış "
          "ajanların küçük modele yapılandırılmış erişimi; telemetri (entropi, RAG skoru, router uzmanları).",
          "gateway", 1.10, {"kanit": "src/gateway/agent_gateway.py"}),
        K("bilgi_enjeksiyonu", "Bilgi Enjeksiyonu (inject_knowledge)",
          "Yeni belge ve akıl yürütme izi yazma: hibrit vektör üretimiyle koleksiyona upsert; fail-closed "
          "koleksiyon ad çözümü (bilinmeyen ad yazımı kapalı). Belleğe yeni bilgi öğretme kanalı.",
          "gateway", 1.05, {"kanit": "src/gateway/agent_gateway.py"}),
        K("pedagojik_supervisor", "Pedagojik Süpervizör (PedagogicalSupervisor)",
          "Yanıt kalitesi değerlendirmesi: ollama/gemini öğretmen çağrısı, CoT çıkarımı (extract_cot_and_card), "
          "cot_vault'a kayıt, öz-denetim adımları (execute_supervision_step).",
          "gateway", 1.00, {"kanit": "src/gateway/pedagogical_supervisor.py"}),
        K("otomatik_yeniden_egitim", "Otomatik Yeniden Eğitim (RetrainPipeline)",
          "Epistemik backlog'dan öğrenme: future_train_vector.jsonl kayıtlarını SFT bin'ine derler "
          "(compile_backlog_to_bin), eğitim koşar (run_training), işlenen kayıtları arşivler. Bilmediğini "
          "sonra öğrenme kanalı.",
          "gateway", 1.00, {"kanit": "src/gateway/retrain_pipeline.py"}),
        # --- veri / eğitim ---
        K("gts_tanim_cikarici", "GTS Tanım Çıkarıcısı",
          "TDK Genel Türkçe Sözlük tanımları: v12.gts.json.tar.gz'den 133.034 anlam çıkarımı "
          "(gts_definitions.txt, 128.327 satır), import_gts ile roots.tsv örtüşmesi. Kelime anlamlarının "
          "ham kaynağı.",
          "veri", 0.90, {"kanit": "scripts/extract_gts_definitions.py"}),
        K("mevzuat_getirici", "Mevzuat Getirici (fetch_mevzuat_corpus)",
          "mevzuat.gov.tr'den kanun metinleri: ölçülü aralıklı getirme, madde numaralı satır yapısı, "
          "hukuk korpusu. İş kanunu ve mevzuat maddelerinin kaynak getiricisi.",
          "veri", 0.95, {"kanit": "scripts/fetch_mevzuat_corpus.py"}),
        K("sft_veri_uretimi", "SFT Veri Üretimi",
          "Eğitim veri kümeleri: RAG/sohbet/müfredat üretimi, UNK'suz kayıt ölçümü (tur-5: 526/526), "
          "pedagoji ekleme, dengeleme. Modelin okul kitapları.",
          "veri", 0.90, {"kanit": "scripts/generate_rag_dataset.py"}),
        K("dpo_egitimi", "DPO Eğitimi (tercih optimizasyonu)",
          "Direct Preference Optimization: tercih çiftleri veri kümesi (prepare_dpo_dataset), ödül modeli "
          "olmadan politika hizalama, dpo_tokenized.jsonl.",
          "veri", 0.85, {"kanit": "train_dpo.py"}),
        K("rlhf_odullendirme", "RLHF Ödüllendirme (rlhf_reward)",
          "İnsan-geri-bildirimli ödül skoru: calculate_reward, token dizisinden ödül hesabı; üretim "
          "kalitesi sinyali.",
          "veri", 0.85, {"kanit": "src/llm/rlhf_reward.py"}),
        K("mufredat_uretici", "Müfredat Üreticisi (CurriculumGenerator)",
          "Pedagojik evre verisi: infancy, parenting, specialization evre-evre öğretim kümeleri; modelin "
          "büyüme planı.",
          "veri", 0.85, {"kanit": "src/llm/curriculum.py"}),
        K("prompt_sozlesmesi", "Prompt Sözleşmesi (prompt_contract)",
          "Checkpoint ve vocab sözleşmesi: build_rag_input, resize_state_dict; sözlük/checkpoint satır "
          "uyumu kapıları. Model-yapısı koruma katmanı.",
          "veri", 0.90, {"kanit": "src/llm/prompt_contract.py"}),
        # --- ölçüm / kapılar ---
        K("olcum_kabi", "Ölçüm Kabini (olcum_kabi)",
          "Ölçüm kabini öz-doğrulaması ve koşum kapıları: T-0098 faz 1-2; değerlendirme ölçümünün "
          "güvenilirlik denetimi.",
          "ölçüm", 0.90, {"kanit": "scripts/olcum_kabi.py"}),
        K("esik_kalibrasyonu", "Eşik Kalibrasyonu (RRF eşik ölçümü)",
          "Geri-getirim eşiğinin ölçülü kalibrasyonu: ayırım-aralığı ölçümü, 0.40 → 0.33 kararı. Eşik "
          "kapılarının sayısal gerekçesi.",
          "ölçüm", 0.90, {"kanit": "scripts/dogrulama_t0178_rrf_esik_olcum.py"}),
        K("kosullandirma_olcumu", "Koşullandırma Ölçümü (conditioning_score)",
          "Belge koşullanmasının model çıktısına etkisi: içerik kökü örtüşmesi skoru; RAG'ın işe yarayıp "
          "yaramadığının ölçüsü.",
          "ölçüm", 0.85, {"kanit": "src/eval/conditioning_score.py"}),
        K("donmus_yol_bekcisi", "Donmuş Yol Bekçisi (frozen_guard)",
          "Donmuş yol güvenliği: frozen.json tek hakikat kaynağı; yetkisiz yazma denetimi, fail-closed. "
          "Değişmemesi gerekenin değişmediğinin garantisi.",
          "ölçüm", 0.85, {"kanit": "src/llm/frozen_guard.py"}),
        K("sozluk_genisletme", "Sözlük Genişletme (expand_lexicon_and_vocab)",
          "Sözlük ve vocab büyütme: yeni kök ekleme, vocab uyumu, purify_lexicon arındırma. Bilinmeyen "
          "kelime (UNK) azaltma kanalı.",
          "ölçüm", 0.85, {"kanit": "scripts/expand_lexicon_and_vocab.py"}),
    ]


# ===========================================================================
# (c) AKTİVASYON MOTORU — saf çekirdek (Qdrant'sız, birim-test edilebilir)
# ===========================================================================

def etki_hesapla(agirlik: float, skor: float, kisit_carpani: float = 1.0) -> float:
    """Ağırlıklı etki: etki = ağırlık × benzerlik × kısıt-çarpanı (6 basamak).

    kisit_carpani default 1.0 → VG-2 v1 aritmetiği BİREBİR (çarpansız).
    Kısıt kartları çarpanı yalnız (0, 1] bandında verebilir (baskılayıcı).
    """
    return round(float(agirlik) * float(skor) * float(kisit_carpani), 6)


def aktivasyon_seti(
    adaylar: Sequence[Tuple[str, float, float]],
    esik: float = DEFAULT_ESIK,
    top_k: int = 3,
    carpanlar: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, float]]:
    """Eşik süz + etki skorla + sırala + top-k seç.

    adaylar: [(ad, benzerlik_skoru, agirlik), ...]
    Sıralama anahtarı: (-etki, -skor, ad) — ağırlıklı etki belirleyicidir;
    beraberlikte ad kırıcı determinizmi sağlar. Eşik DAHİLDİR (skor >= esik).
    carpanlar: {ad: kısıt-çarpanı} — None/boş → v1 davranışı birebir
    (etki = ağırlık × benzerlik); verilirse baskılayıcı çarpan uygulanır.
    """
    gecen: List[Tuple[str, float, float]] = []
    for ad, skor, agirlik in adaylar:
        skor = round(float(skor), 6)
        if skor >= esik:
            gecen.append((ad, skor, float(agirlik)))
    sirali = sorted(
        ((ad, skor, etki_hesapla(agirlik, skor, (carpanlar or {}).get(ad, 1.0)))
         for (ad, skor, agirlik) in gecen),
        key=lambda t: (-t[2], -t[1], t[0]),
    )
    return [(ad, skor, etki) for (ad, skor, etki) in sirali][: max(0, int(top_k))]


def karar_notu_olustur(
    aktif: List[Tuple[str, float, float]],
    aday_sayisi: int,
    esik: float,
    top_k: int,
) -> str:
    """Deterministik karar notu (görev-1d). Zaman damgası YOK."""
    if not aktif:
        return (
            f"VG-2: 0/{aday_sayisi} kart etkin — eşik={esik:g} altında aday kalmadı "
            f"(top_k={top_k}); etki = ağırlık × benzerlik."
        )
    zirve_ad, _, zirve_etki = aktif[0]
    return (
        f"VG-2: {len(aktif)}/{aday_sayisi} kart etkin (eşik={esik:g}, top_k={top_k}); "
        f"sıra: etki=ağırlık×benzerlik; zirve: {zirve_ad} (etki={zirve_etki:.6f})."
    )


# ===========================================================================
# (b+c) KAYIT DEFTERİ + AKTİVASYON MOTORU — Qdrant destekli
# ===========================================================================

class YetenekAgi:
    """'yetenek_agi' koleksiyonu üzerinde kayıt defteri + aktivasyon motoru."""

    def __init__(
        self,
        memory: Optional[VectorMemory] = None,
        tokenizer: Any = None,
        vector_size: int = VEKTOR_BOYUTU,
        host: Optional[str] = None,
        port: int = 6333,
        storage_path: Optional[str] = None,
        url: Optional[str] = None,
        timeout: float = 2.0,
    ) -> None:
        self.tokenizer = tokenizer
        if memory is None:
            memory = VectorMemory(
                collection_name=KOLEKSIYON,
                vector_size=vector_size,
                host=host,
                port=port,
                storage_path=storage_path,
                url=url,
                timeout=timeout,
            )
        if getattr(memory, "collection_name", KOLEKSIYON) != KOLEKSIYON:
            # SERT KURAL kapısı: yetenek_agi DIŞINA yazım kurulumda durur.
            raise ValueError(
                f"[YETENEK_HATA] bellek koleksiyonu '{memory.collection_name}' ≠ "
                f"'{KOLEKSIYON}' — yetenek ağı YALNIZ kendi koleksiyonuna yazar "
                f"(fail-closed)."
            )
        self.memory = memory

    # --- kayıt defteri ---

    def kart_sayisi(self) -> int:
        return self.memory.get_document_count()

    def kartlari_oku(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read-back: koleksiyondaki kart payload'ları."""
        return [kayit["payload"] for kayit in self.memory.list_documents(limit=limit)]

    def tohumla(
        self,
        kartlar: Optional[Sequence[YetenekKarti]] = None,
        parcaboyu: int = 32,
    ) -> Dict[str, int]:
        """Kart envanterini 'yetenek_agi' koleksiyonuna idempotent yazar.

        Nokta kimliği = nokta_kimligi(kart.kimlik) → ikinci koşum ÇOĞALTMAZ,
        günceller. Read-back sayısı döner ({yazilan, benzersiz, kart_sayisi}).
        """
        if self.tokenizer is None:
            raise ValueError(
                "[YETENEK_HATA] tohumlama için tokenizer gerekli (fail-closed; "
                "sessiz boş-yazım yok)."
            )
        kartlar = list(kartlar) if kartlar is not None else kart_envanteri()
        kimlikler = [k.kimlik for k in kartlar]
        if len(kimlikler) != len(set(kimlikler)):
            raise ValueError("[YETENEK_HATA] envanterde yinelenik kimlik var — tohumlanmaz.")

        simdi = datetime.now(timezone.utc).isoformat()
        dokumanlar, kimlik_noktalari = [], []
        for kart in kartlar:
            doc = kart.doc_text()
            token_ids = self.tokenizer.encode(doc)
            tags = self.tokenizer.decode(token_ids)
            meta = {
                "kimlik": kart.kimlik,
                "ad": kart.ad,
                "aciklama": kart.aciklama,
                "etki_alani": kart.etki_alani,
                "agirlik": kart.agirlik,
                "kanit": kart.meta.get("kanit", ""),
                "crystal_tags": tags,
                "token_ids": token_ids,
                "injected_by": "seed_yetenek_agi",
                "timestamp": simdi,
            }
            if kart.meta.get("tip") == KISIT_TIPI:
                # VG-2b: kısıt kartı — baskılayıcı payload alanları aktarılır
                # ve doğrulanır (yetenek kartı payload'ı v1 ile birebir kalır).
                for anahtar in ("tip", "carpan", "kosul", "hedef", "hedef_alan",
                               "bant_esik"):
                    if anahtar in kart.meta:
                        meta[anahtar] = kart.meta[anahtar]
                _kisit_payloadini_doğrula(kart.meta, kart.kimlik)
            dokumanlar.append((doc, token_ids, tags, meta))
            kimlik_noktalari.append(nokta_kimligi(kart.kimlik))

        for bas in range(0, len(dokumanlar), parcaboyu):
            dilim = dokumanlar[bas: bas + parcaboyu]
            self.memory.add_documents_batch(
                texts=[d[0] for d in dilim],
                dense_vectors=[
                    generate_kristal_vector(d[1], d[2], VEKTOR_BOYUTU) for d in dilim
                ],
                sparse_vectors=[generate_sparse_vector(d[1], d[2]) for d in dilim],
                metadatas=[d[3] for d in dilim],
                upsert=True,
                point_ids=kimlik_noktalari[bas: bas + parcaboyu],
            )
        return {
            "yazilan": len(dokumanlar),
            "benzersiz": len(set(kimlik_noktalari)),
            "kart_sayisi": self.kart_sayisi(),  # read-back doğrulama
        }

    # --- aktivasyon motoru ---

    def aktivasyon(
        self,
        sorgu: str,
        top_k: int = 3,
        esik: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Sorgu → vektör → kosinüs yakınlığı → aktif yetenek kümesi.

        Dönüş: {"aktif_yetenekler": [(ad, skor, etki), ...], "karar_notu": str}
        Determinizm: aynı sorgu + aynı koleksiyon → birebir aynı sonuç.
        """
        if self.tokenizer is None:
            raise ValueError(
                "[YETENEK_HATA] aktivasyon için tokenizer gerekli (fail-closed; "
                "sessiz boş-sonuç DÖNMEZ)."
            )
        if not sorgu or not sorgu.strip():
            raise ValueError("[YETENEK_HATA] boş sorgu aktivasyona gelmez (fail-closed).")
        token_ids = self.tokenizer.encode(sorgu)
        tags = self.tokenizer.decode(token_ids)
        dense = generate_kristal_vector(token_ids, tags, VEKTOR_BOYUTU)
        return self.aktivasyon_vektor_ile(dense, top_k=top_k, esik=esik)

    def aktivasyon_vektor_ile(
        self,
        dense: List[float],
        top_k: int = 3,
        esik: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Yoğun sorgu vektörüyle aktivasyon (tokenizer-bağımsız çekirdek yol).

        VG-2b: havuzda 'kisit' tipi nokta VARSA kısıt kartları yetenek
        adayı OLMAKTAN çıkıp BASKILAYICI kural olarak uygulanır (etki =
        ağırlık × benzerlik × kısıt-çarpanı) ve sonuç 'kisit_etkileri'
        alanı taşır. Havuzda kısıt yoksa sonuç v1 ile BİREBİRDİR
        (anahtar seti ve etki aritmetiği değişmez). Not: kısıtlar havuzdan
        okunur — koleksiyon ≤ havuz boyutu (max(50, 4×top_k)) varsayımı.
        """
        esik = DEFAULT_ESIK if esik is None else float(esik)
        havuz = max(50, int(top_k) * 4)
        ham = self.memory.dense_recall(dense, top_k=havuz)
        adaylar: List[Tuple[str, float, float]] = []
        aday_meta: List[Tuple[str, str, str, float]] = []
        kisitlar: List[Dict[str, Any]] = []
        for kayit in ham:
            meta = kayit.get("metadata") or {}
            ad = meta.get("ad")
            if ad is None:
                raise ValueError(
                    "[YETENEK_HATA] nokta payload'ında 'ad' yok — bozuk kayıt "
                    "defteri (fail-closed; sessiz atlama yok)."
                )
            if meta.get("tip") == KISIT_TIPI:
                kisit = dict(meta)
                _kisit_payloadini_doğrula(kisit, str(kisit.get("kimlik", ad)))
                kisitlar.append(kisit)
                continue  # kısıt kartı yetenek adayı DEĞİL
            skor = float(kayit.get("score", 0.0))
            kimlik = str(meta.get("kimlik") or ad)
            etki_alani = str(meta.get("etki_alani") or "")
            adaylar.append((ad, skor, float(meta.get("agirlik", 1.0))))
            aday_meta.append((ad, kimlik, etki_alani, skor))
        carpanlar = kisit_carpanlari(aday_meta, kisitlar, esik=esik) if kisitlar else None
        aktif = aktivasyon_seti(adaylar, esik=esik, top_k=top_k, carpanlar=carpanlar)
        sonuc = {
            "aktif_yetenekler": aktif,
            "karar_notu": karar_notu_olustur(aktif, aday_sayisi=len(adaylar), esik=esik, top_k=top_k),
        }
        if kisitlar:
            sonuc["kisit_etkileri"] = {
                "uygulanan": carpanlar or {},
                "kisitlar": sorted(str(k.get("kimlik", "")) for k in kisitlar),
            }
        return sonuc

    # --- VG-2b: ağırlık evrimi (GÖREV-2) ---

    def agirlik_yaz(self, kimlik: str, yeni_agirlik: float) -> Dict[str, Any]:
        """Tek kartın ağırlığını idempotent upsert ile geri yazar (GÖREV-2).

        Nokta kimliği sabittir (sha256-türeli) → ağırlık güncellemesi
        ÇOĞALTMAZ, günceller; doc/vektör BİREBİR korunur (embedding metni
        ağırlık içermez; vektörler saklı token_ids/crystal_tags'den yeniden
        üretilir — tokenizer GEREKMEZ). Bant dışı ağırlık ve bilinmeyen
        kimlik fail-closed ValueError. YALNIZ 'yetenek_agi' koleksiyonu
        (kurulumdaki ad-kapısı zorlar).
        """
        try:
            yeni = round(float(yeni_agirlik), 6)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[YETENEK_HATA] ağırlık sayısal değil: {yeni_agirlik!r}"
            ) from e
        if not (AGIRLIK_MIN <= yeni <= AGIRLIK_MAX):
            raise ValueError(
                f"[YETENEK_HATA] ağırlık {yeni} evrim bandının [{AGIRLIK_MIN}, "
                f"{AGIRLIK_MAX}] DIŞINDA — bant dışı yazım fail-closed."
            )
        kayitlar = self.memory.list_documents(limit=500)
        for kayit in kayitlar:
            payload = kayit.get("payload") or {}
            if payload.get("kimlik") != kimlik:
                continue
            eski = float(payload.get("agirlik", 1.0))
            token_ids = payload.get("token_ids")
            tags = payload.get("crystal_tags")
            if not token_ids or not tags:
                raise ValueError(
                    f"[YETENEK_HATA] kart '{kimlik}' payload'ında token_ids/"
                    f"crystal_tags yok — vektör birebir korunamaz (fail-closed)."
                )
            meta = {k: v for (k, v) in payload.items()
                    if k not in ("text", "agirlik", "id")}
            meta["agirlik"] = yeni
            nokta_id = kayit.get("id")
            if nokta_id is None:
                raise ValueError(
                    f"[YETENEK_HATA] kart '{kimlik}' nokta kimliği okunamadı "
                    f"(fail-closed — kimliksiz upsert yapılmaz)."
                )
            self.memory.add_documents_batch(
                texts=[payload.get("text", "")],
                dense_vectors=[generate_kristal_vector(token_ids, tags, VEKTOR_BOYUTU)],
                sparse_vectors=[generate_sparse_vector(token_ids, tags)],
                metadatas=[meta],
                upsert=True,
                point_ids=[int(nokta_id)],
            )
            return {
                "kimlik": kimlik,
                "eski_agirlik": eski,
                "yeni_agirlik": yeni,
                "nokta_id": int(nokta_id),
                "kart_sayisi": self.kart_sayisi(),  # read-back: çoğaltma yok
            }
        raise ValueError(
            f"[YETENEK_HATA] kart '{kimlik}' koleksiyonda yok — ağırlık "
            f"yazılamaz (fail-closed; sessiz boş-yazım yok)."
        )

    def evrim_isle(
        self,
        olaylar: Sequence[Dict[str, Any]],
        telemetri_yolu: str = TELEMETRI_YOLU,
        eps: float = EVRIM_EPSILON,
        top_k: int = KANCA_TOP_K,
        esik: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Telemetri olaylarını İŞLEYİP ağırlıkları evrimleştirir (GÖREV-2).

        Her olay: {'sorgu': str, 'router_uzmani': str|None, 'kaynak': str}.
        Döngü (canlı, deterministik — koleksiyon durumu + olay sırası sabit
        ise birebir aynı sonuç): aktivasyon → zirve→uzman sinyali → telemetri
        kaydı → sinyal ±1 ise zirve kartının ağırlığı ×(1±eps) bantla
        güncellenir (idempotent upsert) → geçmiş telemetriye yazılır.
        Kayıt-yalnız olaylar (sinyal None) ağırlığı DEĞİŞTİRMEZ.
        """
        gecmis: List[Dict[str, Any]] = []
        islenen: List[Dict[str, Any]] = []
        for olay in olaylar:
            sorgu = olay.get("sorgu")
            if not sorgu or not str(sorgu).strip():
                raise ValueError(
                    "[YETENEK_HATA] evrim olayında sorgu yok/boş — evrim "
                    "işlenmez (fail-closed)."
                )
            sorgu = str(sorgu)
            aktivasyon_sonuc = self.aktivasyon(sorgu, top_k=top_k, esik=esik)
            sinyal_bilgi = sonuc_sinyali(
                aktivasyon_sonuc["aktif_yetenekler"],
                router_uzmani=olay.get("router_uzmani"),
            )
            olay_kaydi = aktivasyon_olayi(
                sorgu, aktivasyon_sonuc,
                router_uzmani=olay.get("router_uzmani"),
                kaynak=olay.get("kaynak", "evrim"),
            )
            olay_kaydi["olay"] = TELEMETRI_OLAY_TIPI
            telemetri_ekle(olay_kaydi, telemetri_yolu)
            islenen.append(olay_kaydi)
            if sinyal_bilgi["sinyal"] in (1, -1):
                kimlik = sinyal_bilgi["zirve_kimligi"]
                yazma = self.agirlik_yaz(
                    kimlik,
                    agirlik_evrimi(self._agirlik_oku(kimlik),
                                   sinyal_bilgi["sinyal"], eps=eps),
                )
                gecmis.append({
                    "zaman": datetime.now(timezone.utc).isoformat(),
                    "kimlik": kimlik,
                    "eski_agirlik": yazma["eski_agirlik"],
                    "yeni_agirlik": yazma["yeni_agirlik"],
                    "sinyal": sinyal_bilgi["sinyal"],
                    "eps": eps,
                })
                telemetri_ekle({
                    "olay": TELEMETRI_EVRIM_TIPI,
                    "zaman": gecmis[-1]["zaman"],
                    "kaynak": olay.get("kaynak", "evrim"),
                    "kimlik": kimlik,
                    "eski_agirlik": yazma["eski_agirlik"],
                    "yeni_agirlik": yazma["yeni_agirlik"],
                    "sinyal": sinyal_bilgi["sinyal"],
                    "eps": eps,
                }, telemetri_yolu)
        return {
            "islenen": len(islenen),
            "sinyalli": len(gecmis),
            "agirlik_gecmisi": gecmis,
            "olaylar": islenen,
        }

    def _agirlik_oku(self, kimlik: str) -> float:
        """Koleksiyondan tek kartın GÜNCEL ağırlığı (evrim döngüsü içi)."""
        for p in self.kartlari_oku(limit=500):
            if p.get("kimlik") == kimlik:
                return float(p.get("agirlik", 1.0))
        raise ValueError(
            f"[YETENEK_HATA] kart '{kimlik}' koleksiyonda yok — ağırlık "
            f"okunamaz (fail-closed)."
        )


# ===========================================================================
# VG-2b (c2) AKTİVASYON MOTORU — KISITLı YOL (koleksiyonda 'kisit' nokta varsa)
# ===========================================================================

def _kisit_payloadini_doğrula(meta: Dict[str, Any], kimlik: str) -> None:
    """Kısıt nokta payload'ı sözleşmesi (fail-closed — sessiz bozuk kısıt yok).

    çarpan ∈ (0, 1] (BASKILAYICI — yükseltici kısıt kurulumu durur);
    kosul bilinen koşullardan; hedef/hedef_alan tanımlı; gurultu_bandi
    koşulunda bant_esik sayısal.
    """
    carpan = meta.get("carpan")
    if carpan is None or not (0.0 < float(carpan) <= 1.0):
        raise ValueError(
            f"[YETENEK_HATA] kısıt '{kimlik}' çarpanı ({carpan}) (0, 1] bandında "
            f"değil — kısıt kartı BASKILAYICIDIR, yükseltici çarpan kurulumu "
            f"durur (fail-closed)."
        )
    kosul = meta.get("kosul")
    if kosul not in KISIT_KOSULLARI:
        raise ValueError(
            f"[YETENEK_HATA] kısıt '{kimlik}' koşulu bilinmiyor: {kosul!r} "
            f"(geçerli: {KISIT_KOSULLARI})."
        )
    if not (meta.get("hedef") or meta.get("hedef_alan")):
        raise ValueError(
            f"[YETENEK_HATA] kısıt '{kimlik}' hedefsiz — hedef/hedef_alan zorunlu."
        )
    if kosul == "gurultu_bandi":
        try:
            float(meta["bant_esik"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"[YETENEK_HATA] kısıt '{kimlik}' gurultu_bandi koşulunda sayısal "
                f"bant_esik zorunlu (fail-closed)."
            ) from e


def _kisit_hedefini_mi(kimlik: str, etki_alani: str, kisit: Dict[str, Any]) -> bool:
    """Aday kart bu kısıtın hedefinde mi (kimlik ya da etki-alanı üzerinden)."""
    hedef = kisit.get("hedef") or []
    hedef_alan = kisit.get("hedef_alan") or []
    return kimlik in hedef or etki_alani in hedef_alan


def kisit_carpanlari(
    adaylar: Sequence[Tuple[str, str, str, float]],
    kisitlar: Sequence[Dict[str, Any]],
    esik: float = DEFAULT_ESIK,
) -> Dict[str, float]:
    """Aday başına uygulanan kısıt çarpanı (BASKILAYICI: çarpım ≤ 1).

    adaylar:  [(ad, kimlik, etki_alani, ham_skor), ...]
    kisitlar: [{'kimlik', 'carpan', 'kosul', 'hedef'/'hedef_alan', ...}, ...]
    Koşullar (gerçek, deterministik):
      - 'hedef_etkin'   : hedef kart etkinleşmeye çalışıyor (ham_skor >= esik)
                          → çarpan uygulanır (ör. donmuş şema: genişletme yarı inançla).
      - 'gurultu_bandi' : hedef kartın ham skoru kısıt bandının altında
                          (ölçülmüş gürültü bandı) → çarpılır
                          (ör. alan saflığı: ceket dışı sorguda alan kartı baskılanır).
    Çarpanlar çarpılır; sıralama kimlik anahtarlı → deterministik çarpım.
    Sadece < 1.0 sonuç dönmeye girer (uygulanmamış kısıt raporlanmaz).
    """
    carpanlar: Dict[str, float] = {}
    for (ad, kimlik, etki_alani, skor) in adaylar:
        c = 1.0
        for k in sorted(kisitlar, key=lambda x: str(x.get("kimlik", ""))):
            if not _kisit_hedefini_mi(kimlik, etki_alani, k):
                continue
            kosul = k.get("kosul")
            if kosul == "hedef_etkin" and float(skor) >= esik:
                c *= float(k["carpan"])
            elif kosul == "gurultu_bandi" and float(skor) < float(k["bant_esik"]):
                c *= float(k["carpan"])
        c = round(c, 6)
        if c < 1.0:
            carpanlar[ad] = c
    return carpanlar


# ===========================================================================
# (d2) VG-2b TELEMETRİ — aktivasyon-olay kaydı + SONUÇ SİNYALİ (GÖREV-1)
# ===========================================================================

def sonuc_sinyali(
    aktif: Sequence[Tuple[str, float, float]],
    router_uzmani: Optional[str] = None,
) -> Dict[str, Any]:
    """SONUÇ SİNYALİ: zirve yeteneğin alanı ↔ router'ın seçtiği uzman uyumu.

    - uyum    → sinyal +1 (pozitif)  → ağırlık evriminde ×(1+eps)
    - uyumsuz → sinyal −1 (negatif)  → ağırlık evriminde ×(1−eps)
    - router YOK ya da zirve kart uzman kümelerine EŞLENMİYOR → uyum/sinyal
      None (KAYIT-YALINIZ: olay telemetride kalır, ağırlık DEĞİŞMEZ).
    """
    zirve_ad = aktif[0][0] if aktif else None
    zirve_uzmani = ad_uzmani(zirve_ad) if zirve_ad else None
    uyum: Optional[bool] = None
    sinyal: Optional[int] = None
    if router_uzmani and zirve_uzmani:
        uyum = (zirve_uzmani == router_uzmani)
        sinyal = 1 if uyum else -1
    return {
        "zirve_ad": zirve_ad,
        "zirve_kimligi": ad_kimligi(zirve_ad) if zirve_ad else None,
        "zirve_uzmani": zirve_uzmani,
        "router_uzmani": router_uzmani,
        "uyum": uyum,
        "sinyal": sinyal,
    }


def aktivasyon_olayi(
    sorgu: str,
    aktivasyon_sonuc: Dict[str, Any],
    router_uzmani: Optional[str] = None,
    kaynak: str = "manuel",
) -> Dict[str, Any]:
    """Telemetri olay kaydı (JSONL satırı): zaman, sorgu-özeti, aktif
    yetenekler+skorlar, zirve/uyum/sinyal, router kararı (varsa)."""
    aktif = list(aktivasyon_sonuc.get("aktif_yetenekler") or [])
    sinyal_bilgi = sonuc_sinyali(aktif, router_uzmani=router_uzmani)
    olay = {
        "olay": TELEMETRI_OLAY_TIPI,
        "zaman": datetime.now(timezone.utc).isoformat(),
        "kaynak": kaynak,
        "sorgu_ozeti": (sorgu or "").strip()[:SORGU_OZETI_MAX],
        "aktif_yetenekler": [
            {"ad": ad, "kimlik": ad_kimligi(ad), "skor": skor, "etki": etki}
            for (ad, skor, etki) in aktif
        ],
        "zirve_kimligi": sinyal_bilgi["zirve_kimligi"],
        "zirve_uzmani": sinyal_bilgi["zirve_uzmani"],
        "router_uzmani": router_uzmani,
        "uyum": sinyal_bilgi["uyum"],
        "sinyal": sinyal_bilgi["sinyal"],
        "karar_notu": aktivasyon_sonuc.get("karar_notu", ""),
        "kisit_etkileri": aktivasyon_sonuc.get("kisit_etkileri", {}),
    }
    return olay


def telemetri_ekle(olay: Dict[str, Any], yol: str = TELEMETRI_YOLU) -> Dict[str, Any]:
    """Olayı JSONL'ye ekler ve read-back sayısıyla doğrular.

    Serileştirme sort_keys=True → aynı olay birebir aynı satır (deterministik).
    """
    satir = json.dumps(olay, ensure_ascii=False, sort_keys=True)
    klasor = os.path.dirname(os.path.abspath(yol))
    os.makedirs(klasor, exist_ok=True)
    with open(yol, "a", encoding="utf-8") as f:
        f.write(satir + "\n")
    with open(yol, encoding="utf-8") as f:
        toplam = sum(1 for _ in f)  # read-back doğrulama
    return {"yol": yol, "toplam_satir": toplam}


def telemetri_oku(yol: str = TELEMETRI_YOLU) -> List[Dict[str, Any]]:
    """Telemetri JSONL'yi olay listesi olarak okur (dosya yoksa boş)."""
    if not os.path.exists(yol):
        return []
    kayitlar = []
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if satir:
                kayitlar.append(json.loads(satir))
    return kayitlar


def danisma_raporu(
    aktivasyon_sonuc: Dict[str, Any],
    sorgu: str,
    router_uzmani: Optional[str] = None,
    kaynak: str = "gateway",
    telemetri_yolu: str = TELEMETRI_YOLU,
) -> Dict[str, Any]:
    """GÖREV-3 DANIŞMA raporu: zirve→uzman eşleşmesi + telemetri kaydı.

    GATEWAY SÖZLEŞMESİ: bu rapor YALNIZ gözlemler/zenginleştirir — router'ın
    KARARINI (active_expert/router_experts) DEĞİŞTİRMEZ (P2.8 router kanıtlı;
    karar değişikliği ayrı ölçüm ister).
    """
    olay = aktivasyon_olayi(sorgu, aktivasyon_sonuc,
                            router_uzmani=router_uzmani, kaynak=kaynak)
    yazma = telemetri_ekle(olay, telemetri_yolu)
    meta = {
        "zirve_ad": olay["aktif_yetenekler"][0]["ad"] if olay["aktif_yetenekler"] else None,
        "zirve_kimligi": olay["zirve_kimligi"],
        "zirve_uzmani": olay["zirve_uzmani"],
        "router_uzmani": olay["router_uzmani"],
        "uyum": olay["uyum"],
        "sinyal": olay["sinyal"],
    }
    return {"meta": meta, "olay": olay, "telemetri": yazma}


# ===========================================================================
# (e2) VG-2b AĞIRLIK EVRİMİ — deterministik, bantlı (GÖREV-2)
# ===========================================================================

def agirlik_evrimi(
    agirlik: float,
    sinyal: Optional[int],
    eps: float = EVRIM_EPSILON,
    alt: float = AGIRLIK_MIN,
    ust: float = AGIRLIK_MAX,
) -> float:
    """Deterministik bantlı ağırlık güncelleme kuralı.

    sinyal +1 → agirlik × (1 + eps);  sinyal −1 → agirlik × (1 − eps);
    sinyal None/0 → DEĞİŞMEZ (kayıt-yalnız). Sonuç [alt, ust] bandına
    kırpılır, 6 basamak yuvarlanır. Saf fonksiyon → yeniden-oynatma
    determinizmi (aynı girdi → aynı çıktı).
    """
    if sinyal not in (1, -1):
        return round(float(agirlik), 6)
    carpan = (1.0 + float(eps)) if sinyal > 0 else (1.0 - float(eps))
    return round(min(ust, max(alt, float(agirlik) * carpan)), 6)


def evrim_uygula(
    agirliklar: Dict[str, float],
    olaylar: Sequence[Dict[str, Any]],
    eps: float = EVRIM_EPSILON,
    alt: float = AGIRLIK_MIN,
    ust: float = AGIRLIK_MAX,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """SAF yeniden-oynatma çekirdeği: sinyalli olayları ağırlıklara uygular.

    olaylar: [{'zirve_kimligi': str, 'sinyal': +1/-1/None}, ...]
    Dönüş: (yeni ağırlıklar, geçmiş kayıtları). Geçmiş kaydı:
    {kimlik, eski_agirlik, yeni_agirlik, sinyal, eps} — telemetri JSONL'e
    yazılmak üzere. Bilinmeyen kimlik ValueError (fail-closed — telemetri
    kayıt defteri bozuksa evrim durur). Girdi sözlüğü MUTASYONA UĞRAMAZ.
    """
    yeni = dict(agirliklar)
    gecmis: List[Dict[str, Any]] = []
    for olay in olaylar:
        kimlik = olay.get("zirve_kimligi") or olay.get("kimlik")
        sinyal = olay.get("sinyal")
        if not kimlik or sinyal not in (1, -1):
            continue  # kayıt-yalnız / sinyalsiz → ağırlık DEĞİŞMEZ
        if kimlik not in yeni:
            raise ValueError(
                f"[YETENEK_HATA] evrim olayı bilinmeyen kart kimliği: {kimlik!r} "
                f"(fail-closed — bozuk telemetri defteriyle evrim işlenmez)."
            )
        eski = float(yeni[kimlik])
        yeni_deger = agirlik_evrimi(eski, sinyal, eps=eps, alt=alt, ust=ust)
        if yeni_deger != eski:
            yeni[kimlik] = yeni_deger
            gecmis.append({
                "kimlik": kimlik,
                "eski_agirlik": eski,
                "yeni_agirlik": yeni_deger,
                "sinyal": sinyal,
                "eps": eps,
            })
    return yeni, gecmis


# ===========================================================================
# (f2) VG-2b KISIT KARTLARI ENVANTERİ — GERÇEK kurallar (GÖREV-4, uydurma yok)
# ===========================================================================

def kisit_envanteri() -> List[YetenekKarti]:
    """Projedeki GERÇEK kurallardan çıkarılmış kısıt kartları (5 kart).

    Etki BASKILAYICIDIR: kısıt kartı hedefinin aktivasyon etki skorunu
    DÜŞÜRÜR (çarpan ≤ 1), asla yükseltmez. Her kartın meta['kanit'] yolu
    repoda VAR OLAN gerçek dosyadır (test_zorlar: uydurma kısıt testte ölür).
    """
    K = YetenekKarti
    return [
        # (1) İş emri örneği: donmuş sözlük 33.114 — yeni jeton yok.
        #     GERÇEK: .agent-bus/frozen.json donmuş yol desenleri — data/lexicon/**
        #     (roots.tsv: 'sözlük değişikliği tüm tokenizasyonu kaydırır'),
        #     src/compiler/**, src/llm/tokenizer.py; vocab↔checkpoint satır
        #     sözleşmesi (prompt_contract) kilitli → yeni jeton eklenemez.
        K("donmus_sozluk_33_114", "Kısıt: Donmuş Sözlük 33.114 — Yeni Jeton Yok",
          "Anka vocab'ı 33.114 jetonda donmuştur: roots.tsv, derleyici yığını ve "
          "tokenizer donmuş yol desenleridir (frozen.json); sözlük değişikliği tüm "
          "tokenizasyonu kaydırır ve vocab-checkpoint satır sözleşmesi kilitlidir. "
          "Sözlük genişletme yeteneği yeni jeton EKLEYEMEZ — arındırma dışında baskılanır.",
          "kisit", 1.0,
          {"tip": "kisit", "carpan": 0.5, "kosul": "hedef_etkin",
           "hedef": ["sozluk_genisletme"], "kanit": ".agent-bus/frozen.json"}),
        # (2) İş emri örneği: alan saflığı — ceket dışı sorguda alan kartı
        #     baskılanır. GERÇEK (ölçülmüş): VG-2 probunda gürültü bandı
        #     0,005–0,024, ilgili isabetlerin dibi 0,033 → bant 0,03.
        K("alan_safligi", "Kısıt: Alan Saflığı — Ceket Dışı Sorguda Alan Kartı Baskılanır",
          "Alan ceketleri (marangoz, bahçıvan, legal, pedagoji) yalnız kendi "
          "alanları için eğitilmiş MoE modülleridir. VG-2 probunda ölçülen gürültü "
          "bandı (ham skor 0,005–0,024) alan dışı sorguların izidir; ilgili "
          "isabetlerin dibi 0,033'tür. Gürültü bandındaki alan kartı baskılanır — "
          "ağırlığı ne olursa olsun alan dışı etkiye sokulmaz.",
          "kisit", 1.0,
          {"tip": "kisit", "carpan": 0.5, "kosul": "gurultu_bandi", "bant_esik": 0.03,
           "hedef_alan": ["alan"], "kanit": "scripts/yetenek_probe.py"}),
        # (3) GERÇEK (ölçülmüş): 'hükmü' anka vocab'ında UNK — hukuk kartı
        #     şablonlarında kullanılmaz (prepare_legal_dataset_v3 satır 90).
        K("hukum_unk_yasagi", "Kısıt: 'hükmü' UNK — Hukuk Şablonlarında Kullanılmaz",
          "Ölçülmüş UNK kuralı: 'hükmü' ve ekli biçimleri anka vocab'ında UNK "
          "kalır, hukuk kartı şablonlarında YASAKTIR ('hüküm', 'hükümleri', "
          "'metni' kullanılır). SFT veri üretimi bu kelime kısıtı altında çalışır — "
          "şablon üretimi baskılanıp ölçülmüş kelime seçimine zorlanır.",
          "kisit", 1.0,
          {"tip": "kisit", "carpan": 0.9, "kosul": "hedef_etkin",
           "hedef": ["sft_veri_uretimi"], "kanit": "scripts/prepare_legal_dataset_v3.py"}),
        # (4) GERÇEK: koleksiyon yazma yetkisi fail-closed (T-0148 5B) —
        #     bilinmeyen koleksiyon adı ValueError; yetenek ağı yalnız kendi
        #     koleksiyonuna yazar.
        K("yazma_yetkisi_fail_closed", "Kısıt: Yazma Yetkisi Fail-Closed — Bilinmeyen Koleksiyona Yazım Yok",
          "Koleksiyon yazımları ad-çözüm kapılarından geçer: bilinmeyen "
          "koleksiyon adı ValueError ile durur (T-0148 5B), yetenek ağı yalnız "
          "kendi koleksiyonuna yazar. Bilgi enjeksiyonu yeteneği bu kapıyla "
          "sınırlıdır — istediği koleksiyona yazamaz.",
          "kisit", 1.0,
          {"tip": "kisit", "carpan": 0.9, "kosul": "hedef_etkin",
           "hedef": ["bilgi_enjeksiyonu"], "kanit": "src/gateway/agent_gateway.py"}),
        # (5) GERÇEK: gateway üretim tavanı — safe_max_tokens = min(200, ...)
        #     (DoS ve VRAM kilitlenmesi önlemi, agent_gateway.py ask()).
        K("uretim_tavani_200", "Kısıt: Üretim Tavanı 200 Jeton (DoS/VRAM Kilidi)",
          "Ajan kapısının üretimi güvenli jeton tavanına kilitlidir: max_new_tokens "
          "istense de min(200, …) ile kırpılır (DoS ve VRAM kilitlenmesi önlemi). "
          "HTTP yanıt üretimi bu tavan altında kalır — sınırsız üretim vaadi baskılanır.",
          "kisit", 1.0,
          {"tip": "kisit", "carpan": 0.9, "kosul": "hedef_etkin",
           "hedef": ["ajan_kapisi"], "kanit": "src/gateway/agent_gateway.py"}),
    ]
