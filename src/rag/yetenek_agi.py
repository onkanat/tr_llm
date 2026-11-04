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

YETKİ (SERT KURAL): bu modül YALNIZ 'yetenek_agi' koleksiyonuna yazar;
anka_bellek / muhakeme_bellek / simulasyon_bellek DOKUNULMAZ. Farklı
koleksiyonlu bir bellek verildiğinde kurulum ValueError ile durur
(fail-closed — sessiz-yanlış-koleksiyon yazımı yok).
"""

import hashlib
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

def etki_hesapla(agirlik: float, skor: float) -> float:
    """Ağırlıklı etki: etki = ağırlık × benzerlik (6 basamak yuvarlanır)."""
    return round(float(agirlik) * float(skor), 6)


def aktivasyon_seti(
    adaylar: Sequence[Tuple[str, float, float]],
    esik: float = DEFAULT_ESIK,
    top_k: int = 3,
) -> List[Tuple[str, float, float]]:
    """Eşik süz + etki skorla + sırala + top-k seç.

    adaylar: [(ad, benzerlik_skoru, agirlik), ...]
    Sıralama anahtarı: (-etki, -skor, ad) — ağırlıklı etki belirleyicidir;
    beraberlikte ad kırıcı determinizmi sağlar. Eşik DAHİLDİR (skor >= esik).
    """
    gecen: List[Tuple[str, float, float]] = []
    for ad, skor, agirlik in adaylar:
        skor = round(float(skor), 6)
        if skor >= esik:
            gecen.append((ad, skor, float(agirlik)))
    sirali = sorted(
        ((ad, skor, etki_hesapla(agirlik, skor)) for (ad, skor, agirlik) in gecen),
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
        """Yoğun sorgu vektörüyle aktivasyon (tokenizer-bağımsız çekirdek yol)."""
        esik = DEFAULT_ESIK if esik is None else float(esik)
        havuz = max(50, int(top_k) * 4)
        ham = self.memory.dense_recall(dense, top_k=havuz)
        adaylar: List[Tuple[str, float, float]] = []
        for kayit in ham:
            meta = kayit.get("metadata") or {}
            ad = meta.get("ad")
            if ad is None:
                raise ValueError(
                    "[YETENEK_HATA] nokta payload'ında 'ad' yok — bozuk kayıt "
                    "defteri (fail-closed; sessiz atlama yok)."
                )
            adaylar.append((ad, float(kayit.get("score", 0.0)), float(meta.get("agirlik", 1.0))))
        aktif = aktivasyon_seti(adaylar, esik=esik, top_k=top_k)
        return {
            "aktif_yetenekler": aktif,
            "karar_notu": karar_notu_olustur(aktif, aday_sayisi=len(adaylar), esik=esik, top_k=top_k),
        }
