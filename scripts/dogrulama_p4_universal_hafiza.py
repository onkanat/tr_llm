#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-0144 — MİMARİ DOĞRULAMA PAKET-4: Üniversal Hafıza (VectorMemory yaşam-döngüsü)

İLAN: data/eval/mimari_dogrulama_p4_ilan_2026-09-27.md (koşum-ÖNCESİ damgalı).
Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK. rc ∈ {0, 2}.

Operatör kararları (27 Eyl 2026, plan onayı + AskUserQuestion):
  (1) YALNIZ DOĞRULAMA — kanonik kod (src/rag/vector_memory.py,
      src/rag/embedding.py, src/rag/rag_pipeline.py build_query_vectors)
      salt IMPORT, YAZIM YOK; recreate-yıkıcılık [83,123,126] ONARILMAZ —
      mutasyonla kanıtlanır (yalnız probe koleksiyonunda).
  (2) GEÇİCİ PROBE + SONDA SİL — yazım+recreate+delete YALNIZ
      p4_probe_bellek üstünde; koşum sonunda delete_collection ile
      kaldırılır. kristal_bellek (36 nokta) YALNIZ OKUNUR; 9 foreign
      koleksiyon DOKUNULMAZ; simulasyon_bellek KURULMAZ.

Kanonik kod IMPORT edilir (kopya YASAK):
  VectorMemory                       (src/rag/vector_memory.py)
  generate_kristal_vector            (src/rag/embedding.py)
  generate_sparse_vector             (src/rag/embedding.py)
  build_query_vectors                (src/rag/rag_pipeline.py)
  KristalTokenizer / Vocabulary      (src/llm/tokenizer.py)
  _envanter / _envanter_digest       (scripts/dogrulama_p2_rag_gezgini.py —
                                      kanonik src/** DEĞİL, betik-arası
                                      yeniden-kullanım; P3 kalıbı)

Model YÜKLEMEZ: embedding tokenizer-morfem tabanlı deterministiktir
(random.Random(seed) seed'li) — KristalLM YÜKLENMEZ; MPS YOK;
çift-eğitici YOK; saf CPU + ağ; seed=42.

Koşum sandbox DIŞI (Qdrant trafiği sandbox proxy'sinden geçmez).
"""

import argparse
import ast
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient  # noqa: E402

from src.rag.vector_memory import VectorMemory  # noqa: E402
from src.rag.embedding import (  # noqa: E402
    generate_kristal_vector,
    generate_sparse_vector,
)
from src.rag.rag_pipeline import build_query_vectors  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from scripts.dogrulama_p2_rag_gezgini import (  # noqa: E402
    _envanter,
    _envanter_digest,
)

# ---- İLAN'lı sabitler (koşum ÖNCESİ sabit; koşum-sonrası yumuşatılmaz) ----
POZITIF_N = 20
SEED = 42
KANONIK_KOLEKSIYON = "kristal_bellek"
PROBE_KOLEKSIYON = "p4_probe_bellek"
KANONIK_YASAK = frozenset({
    "kristal_bellek", "simulasyon_bellek",
})
EXPECTED_KAPILAR = frozenset({
    "K1_FAZ_A_ENVANTER", "K2_ARZ_CIPA", "K3_B1_BAGLANMA",
    "K4_B3_DETERMINIZM", "K5_B4_ADD_YOLU", "K6_B5_MUTASYON_KANITI",
    "K7_DOKUNULMAZLIK",
})
OOV_SORGU = "zzqwxx zqxwv zqqzzq"  # P2/P3'ün aynısı
PROBE_METINLER = [
    "p4_probe belge bir: univerzal hafiza yazi yolu dogrulamasi",
    "p4_probe belge iki: add_document tekil yazim kaniti",
    "p4_probe belge uc: hibrit arama pozitif kontrolu",
    "p4_probe belge dort: add_documents_batch parti yazim kaniti",
    "p4_probe belge bes: nokta sayac davranisi kaniti",
    "p4_probe belge alti: geri okuma payload birebir",
    "p4_probe belge yedi: dense recall kendi belge",
    "p4_probe belge sekiz: hibrit recall kendi belge",
]
ILANLI = {
    # ---- FAZ-A kod-envanteri (koşumsuz) ----
    "vm_qdrantclient_kurulum": 3,           # host / local-path / :memory:
    "vm_remote_timeout_sn": 2.0,            # :50
    "vm_cache_yazim_sitesi": 1,             # :73 (tek yazım)
    "vm_cache_evict_sitesi": 1,             # :44 (evict-pop)
    "vm_cache_guardi": "if not self.is_in_memory:",   # :72 — in-memory
    "vm_cache_guard_satiri": 72,            #   cache'e YAZILMAZ
    "vm_cache_yazim_satiri": 73,
    "vm_recreate_delete_satirlar": [83, 123, 126],    # P2/P3 çıpası birebir
    "vm_otomatik_kurulum_birebir": True,    # :75-77 bandı metin-teyitli
    "embedding_random_seedli": 1,           # random.Random( tek site
    "embedding_random_seedli_siz": 0,       # seed'siz random.-çağrı
    "gateway_localhost_kurulum": 2,         # :123-124 — sessiz-fallback riski
    "gateway_kanonik_koleksiyonlar": ["kristal_bellek", "simulasyon_bellek"],
    # ---- FAZ-B İLAN'lı ----
    "b1_storage_type": "remote (192.168.1.5:6333)",
    "b1_cache_anahtari": "remote:192.168.1.5:6333",
    "b1_kristal_nokta": 36,
    "envanter_digest_once": "fe7fc7649737ed35",       # P2==P3 çıpası
    "foreign_koleksiyon_once": 9,
    "simulasyon_bellek_yok": True,
    "b3_p2_birebir": 20,
    "b3_cift_kosum_birebir": 20,
    "b4_count_sekansi": [0, 3, 8, 9],
    "b4_geri_okuma_birebir": 9,
    "b4_pozitif_beklenen": 8,
    "b5_recreate_nokta": 0,
    "b5_dense_size": 384,
    "kanal_once_bayt": 0,
    "kanal_sonra_bayt": 0,
}
GERCEK_KANAL = os.path.join(REPO_ROOT, "data", "future_train_vector.jsonl")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stderr(msg: str) -> None:
    print(f"[T-0144] {msg}", file=sys.stderr)


def _sorgular(korpus_yol: str, rng: random.Random,
              adet: int) -> List[str]:
    """Korpus input'larındaki <BELGE>…</BELGE> sonrası sorgular (P2 seed'li
    seçim protokolü: random.Random(SEED).sample + sorted — P3'ün aynısı)."""
    pat = re.compile(r"<BELGE>.*?</BELGE>\s*(.+)\s*$", re.DOTALL)
    tum: List[str] = []
    with open(korpus_yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            m = pat.search(rec.get("input", ""))
            if m and m.group(1).strip():
                tum.append(m.group(1).strip())
    secili = sorted(rng.sample(range(len(tum)), min(adet, len(tum))))
    return [tum[j] for j in secili]


def _tokenizer_kur(vocab_yol: str) -> KristalTokenizer:
    """Kanonik tokenizer yükleme kalıbı (model YÜKLEMEZ — P4 salt-hafıza)."""
    vocab = Vocabulary()
    vocab.load(vocab_yol)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(
        os.path.join(REPO_ROOT, "data", "lexicon", "roots.tsv"))
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    return KristalTokenizer(compiler, vocab, literal_entity_mode=False)


def _probe_vektor(metin: str, tokenizer: KristalTokenizer):
    """Kanonik embedding zinciri: encode → tags → dense + sparse (İLAN'lı)."""
    t_ids = tokenizer.encode(metin)
    t_tags = tokenizer.decode(t_ids)
    if not t_ids:
        raise RuntimeError(f"PROBE DERLEME-BOS: {metin[:40]}")
    return (t_ids, t_tags,
            generate_kristal_vector(t_ids, t_tags),
            generate_sparse_vector(t_ids, t_tags))


def _faz_a_envanteri(repo_root: str) -> Dict[str, Any]:
    """FAZ-A: kod-envanteri (koşumsuz; grep + satır-metni; İLAN'lı)."""
    bulgular: Dict[str, Any] = {}
    vm_yol = os.path.join(repo_root, "src", "rag", "vector_memory.py")
    vm = open(vm_yol, encoding="utf-8").read()
    vm_satirlar = vm.splitlines()

    bulgular["vm_qdrantclient_kurulum"] = vm.count("QdrantClient(")
    m = re.search(r"timeout=([0-9.]+)", vm)
    bulgular["vm_remote_timeout_sn"] = float(m.group(1)) if m else None
    bulgular["vm_cache_yazim_sitesi"] = sum(
        1 for s in vm_satirlar if "_shared_clients[client_key] = (" in s)
    evict_satirlar = [i for i, s in enumerate(vm_satirlar, 1)
                      if "_shared_clients.pop(client_key" in s]
    # İLAN'lı evict-sitesi :44 (init-içi evict); close-pop :282 AYRI metot
    # (rapor-kırılımı) — T-0144 2. koşum dersi: ölçüm İLAN'lı-siteye daraltılır
    bulgular["vm_cache_evict_satirlar"] = evict_satirlar
    bulgular["vm_cache_evict_sitesi"] = sum(
        1 for i in evict_satirlar if i < 90)
    bulgular["vm_close_pop_satiri"] = (
        max(evict_satirlar) if evict_satirlar else -1)
    guard_satiri = next((i for i, s in enumerate(vm_satirlar, 1)
                         if s.strip() == ILANLI["vm_cache_guardi"]), -1)
    bulgular["vm_cache_guard_satiri"] = guard_satiri
    bulgular["vm_cache_guardi"] = (vm_satirlar[guard_satiri - 1]
                                   if guard_satiri > 0 else "YOK")
    yazim_satiri = next((i for i, s in enumerate(vm_satirlar, 1)
                         if "_shared_clients[client_key] = (" in s), -1)
    bulgular["vm_cache_yazim_satiri"] = yazim_satiri
    bulgular["vm_cache_guard_takipediyor"] = bool(
        guard_satiri > 0 and yazim_satiri == guard_satiri + 1)

    recreate_satirlar = [i for i, s in enumerate(vm_satirlar, 1)
                         if "recreate_collection" in s
                         or "delete_collection" in s]
    bulgular["vm_recreate_delete_satirlar"] = recreate_satirlar
    # otomatik-kurulum bandı :75-77 (koleksiyon-yoksa sessiz kurulum)
    ok_band = (75 <= len(vm_satirlar)
               and "collection_exists" in vm_satirlar[74]
               and "_create_hybrid_collection" in vm_satirlar[75]
               and "_next_point_id" in vm_satirlar[76])
    bulgular["vm_otomatik_kurulum_birebir"] = bool(ok_band)
    bulgular["vm_otomatik_kurulum_bandi"] = vm_satirlar[74:77]

    emb_yol = os.path.join(repo_root, "src", "rag", "embedding.py")
    emb = open(emb_yol, encoding="utf-8").read()
    bulgular["embedding_random_seedli"] = emb.count("random.Random(")
    bulgular["embedding_random_seedli_siz"] = len(
        re.findall(r"random\.(?!Random\b)\w+", emb))

    gw_yol = os.path.join(repo_root, "src", "gateway", "agent_gateway.py")
    gw = open(gw_yol, encoding="utf-8").read()
    gw_satirlar = gw.splitlines()
    bulgular["gateway_localhost_kurulum"] = gw.count('host="localhost"')
    adlar: List[str] = []
    for i, s in enumerate(gw_satirlar, 1):
        if 'host="localhost"' in s and "VectorMemory(" in s:
            adlar.append(s.split('collection_name="')[1].split('"')[0])
    bulgular["gateway_kanonik_koleksiyonlar"] = adlar

    # embedding determinizmi (saf fonksiyon — koşumsuz çift-çağrı birebir)
    v1 = generate_kristal_vector([5, 6, 7], "<PROPER_NOUN>")
    v2 = generate_kristal_vector([5, 6, 7], "<PROPER_NOUN>")
    bulgular["embedding_deterministik"] = bool(v1 == v2 and any(v1))

    # repo-içi local-storage riski (host'suz çağrıda VARSAYILAN storage_path)
    bulgular["vm_varsayilan_storage_path"] = str(
        vector_memory_defaults().get("storage_path"))
    return bulgular


def vector_memory_defaults() -> Dict[str, Any]:
    """Kanonik __init__ varsayılanları (koşumsuz; import-only)."""
    import inspect as _ins
    p = _ins.signature(VectorMemory.__init__).parameters
    return {k: v.default for k, v in p.items() if k != "self"}


def _faz_a_kapilari(faz_a: Dict[str, Any]) -> Tuple[bool,
                                                    List[Dict[str, Any]]]:
    """Kapı K1: FAZ-A İLAN'lı sayılar ↔ ölçülen (birebir)."""
    olculen = {
        "vm_qdrantclient_kurulum": faz_a["vm_qdrantclient_kurulum"],
        "vm_remote_timeout_sn": faz_a["vm_remote_timeout_sn"],
        "vm_cache_yazim_sitesi": faz_a["vm_cache_yazim_sitesi"],
        "vm_cache_evict_sitesi": faz_a["vm_cache_evict_sitesi"],
        "vm_cache_guard_satiri": faz_a["vm_cache_guard_satiri"],
        "vm_cache_yazim_satiri": faz_a["vm_cache_yazim_satiri"],
        "vm_recreate_delete_satirlar":
            faz_a["vm_recreate_delete_satirlar"],
        "vm_otomatik_kurulum_birebir":
            faz_a["vm_otomatik_kurulum_birebir"],
        "embedding_random_seedli": faz_a["embedding_random_seedli"],
        "embedding_random_seedli_siz":
            faz_a["embedding_random_seedli_siz"],
        "gateway_localhost_kurulum": faz_a["gateway_localhost_kurulum"],
        "gateway_kanonik_koleksiyonlar":
            faz_a["gateway_kanonik_koleksiyonlar"],
    }
    detay: List[Dict[str, Any]] = []
    tamam = True
    for anahtar, ilanli in ILANLI.items():
        olc = olculen.get(anahtar)
        if olc is None:
            continue
        esit = olc == ilanli
        tamam = tamam and esit
        detay.append({"olcum": anahtar, "ilanli": ilanli, "olculen": olc,
                      "durum": "GEÇTİ" if esit else "SAPMA"})
    # cache-guardı davranış-beyanı: guard satırı yazımın HEMEN üstünde
    esit = bool(faz_a["vm_cache_guard_takipediyor"])
    tamam = tamam and esit
    detay.append({"olcum": "vm_cache_guard_takipediyor", "ilanli": True,
                  "olculen": esit, "durum": "GEÇTİ" if esit else "SAPMA"})
    return tamam, detay


def _nokta_saylari(env: Dict[str, Any]) -> Dict[str, int]:
    return {k["ad"]: int(k["nokta"]) for k in env["kalemler"]}


def main() -> None:
    ayrici = argparse.ArgumentParser(
        description="T-0144 — MİMARİ DOĞRULAMA PAKET-4: Üniversal Hafıza")
    ayrici.add_argument("--host", default="192.168.1.5")
    ayrici.add_argument("--port", type=int, default=6333)
    ayrici.add_argument("--korpus",
                        default="data/realistic_rag/test_natural_150.jsonl")
    ayrici.add_argument("--vocab",
                        default="data/rebuild/vocab_anka_r1_33114.json")
    ayrici.add_argument(
        "--p2-hukum",
        default="data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json")
    ayrici.add_argument(
        "--ilan", default="data/eval/mimari_dogrulama_p4_ilan_2026-09-27.md")
    ayrici.add_argument(
        "--rapor", default="data/eval/mimari_dogrulama_p4_sonuc_2026-09-27.md")
    ayrici.add_argument("--hukum-json", default=None)
    arg = ayrici.parse_args()
    damga = _ts()
    detay: Dict[str, Any] = {}
    kapilar: Dict[str, bool] = {}

    ilan_yol = os.path.join(REPO_ROOT, arg.ilan)
    ilan_sha = _sha256(ilan_yol)
    korpus_yol = os.path.join(REPO_ROOT, arg.korpus)
    vocab_yol = os.path.join(REPO_ROOT, arg.vocab)
    p2_hukum_yol = os.path.join(REPO_ROOT, arg.p2_hukum)
    detay["ilan_sha256_kosumda"] = ilan_sha
    detay["p2_hukum_sha256_once"] = _sha256(p2_hukum_yol)

    tokenizer = _tokenizer_kur(vocab_yol)

    # ================= B2 (koşum ÖNCESİ arz-envanteri — salt-okunur) ======
    yonet_client = QdrantClient(host=arg.host, port=arg.port, timeout=10.0,
                                check_compatibility=False)
    env_once = _envanter(yonet_client)
    once_digest = _envanter_digest(env_once)
    nokta_once = _nokta_saylari(env_once)
    foreign_once = {ad: n for ad, n in nokta_once.items()
                    if ad not in KANONIK_YASAK and ad != PROBE_KOLEKSIYON}
    detay["b2_arz_once"] = {
        "adlar": env_once["adlar"],
        "foreign_sayi": len(foreign_once),
        "foreign": foreign_once,
        "kristal_bellek_var": KANONIK_KOLEKSIYON in env_once["adlar"],
        "simulasyon_bellek_yok":
            "simulasyon_bellek" not in env_once["adlar"],
        "kristal_nokta": nokta_once.get(KANONIK_KOLEKSIYON),
        "probe_once_var": PROBE_KOLEKSIYON in env_once["adlar"],
        "digest": once_digest,
    }
    kanal_once_bayt = (os.path.getsize(GERCEK_KANAL)
                       if os.path.exists(GERCEK_KANAL) else -1)
    detay["kanal_once_bayt"] = kanal_once_bayt
    kapilar["K2_ARZ_CIPA"] = bool(
        once_digest == ILANLI["envanter_digest_once"]
        and len(foreign_once) == ILANLI["foreign_koleksiyon_once"]
        and nokta_once.get(KANONIK_KOLEKSIYON)
        == ILANLI["b1_kristal_nokta"]
        and "simulasyon_bellek" not in env_once["adlar"]
        and PROBE_KOLEKSIYON not in env_once["adlar"]
        and kanal_once_bayt == ILANLI["kanal_once_bayt"])

    # ================= FAZ-A (koşumsuz kod-envanteri) ====================
    faz_a = _faz_a_envanteri(REPO_ROOT)
    detay["faz_a"] = faz_a
    k1_tam, k1_detay = _faz_a_kapilari(faz_a)
    detay["faz_a_kapi_detay"] = k1_detay
    kapilar["K1_FAZ_A_ENVANTER"] = k1_tam

    # ================= B1: bağlanma-teyidi (kanonik kurucu) ==============
    b1: Dict[str, Any] = {}
    try:
        vm = VectorMemory(KANONIK_KOLEKSIYON, 768, host=arg.host,
                          port=arg.port)
        b1["storage_type"] = str(vm.storage_type)
        b1["is_in_memory"] = bool(vm.is_in_memory)
        b1["kristal_nokta"] = int(vm.get_document_count())
        b1["cache_anahtari"] = (f"remote:{arg.host}:{arg.port}"
                                in VectorMemory._shared_clients)
        cache_once_len = len(VectorMemory._shared_clients)
        vm2 = VectorMemory(KANONIK_KOLEKSIYON, 768, host=arg.host,
                           port=arg.port)
        b1["cache_reuse"] = bool(
            len(VectorMemory._shared_clients) == cache_once_len
            and vm2.storage_type == vm.storage_type
            and vm2.is_in_memory == vm.is_in_memory
            and vm2.get_document_count() == b1["kristal_nokta"])
        detay["b1_baglanma"] = b1
        kapilar["K3_B1_BAGLANMA"] = bool(
            b1["storage_type"] == ILANLI["b1_storage_type"]
            and b1["is_in_memory"] is False
            and b1["kristal_nokta"] == ILANLI["b1_kristal_nokta"]
            and b1["cache_reuse"] is True)
    except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
        detay["b1_baglanma"] = {"istisna": f"{type(e).__name__}: {e}"}
        kapilar["K3_B1_BAGLANMA"] = False

    # ================= B3: recall-determinizm (salt-okunur) ==============
    b3: Dict[str, Any] = {}
    try:
        sorgular = _sorgular(korpus_yol, random.Random(SEED), POZITIF_N)
        b3["n"] = len(sorgular)
        with open(p2_hukum_yol, encoding="utf-8") as f:
            p2_detay = json.load(f).get("detay", {}).get("b2_pozitif", {})
        p2_skorlar = {d["sorgu"]: float(d["skor"])
                      for d in p2_detay.get("detay", []) if "skor" in d}
        b3["p2_skor_anahtari"] = len(p2_skorlar)

        def _bir_tur(bellek: VectorMemory) -> List[Dict[str, Any]]:
            cikis: List[Dict[str, Any]] = []
            for sorgu in sorgular:
                _, s_tags, s_dense, s_sparse = build_query_vectors(
                    sorgu, tokenizer)
                sonuclar = bellek.hybrid_recall(s_dense, s_sparse, top_k=1,
                                                query_tags=s_tags)
                skor = float(sonuclar[0]["score"]) if sonuclar else 0.0
                cikis.append({
                    "sorgu": sorgu[:60], "skor": round(skor, 4),
                    "kendi_belge": (str(sonuclar[0].get("text", ""))
                                    if sonuclar else ""),
                    "has_root_match": bool(sonuclar[0].get(
                        "has_root_match")) if sonuclar else None,
                })
            return cikis

        kosum1 = _bir_tur(vm)
        kosum2 = _bir_tur(vm2)
        cift_birebir = sum(
            1 for a, c in zip(kosum1, kosum2)
            if a["skor"] == c["skor"])
        b3["cift_kosum_birebir"] = cift_birebir
        karsilastirma: List[Dict[str, Any]] = []
        for a in kosum1:
            p2 = p2_skorlar.get(a["sorgu"])
            karsilastirma.append({
                "sorgu": a["sorgu"], "p4_skor": a["skor"],
                "p2_skor": p2, "birebir": (p2 is not None
                                           and round(p2, 4) == a["skor"])})
        p2_birebir = sum(1 for c in karsilastirma if c["birebir"])
        b3["p2_birebir"] = p2_birebir
        b3["p2_karsilastirma_n"] = len(karsilastirma)
        # RAPOR notu: P2 detay-sorguları [:60]-kesmeli — benzersiz-anahtar
        # sayımı §7'de beyan edilir (kesme-çakışması birebirliği bozmaz:
        # aynı-anahtar iki sorgu aynı P2 skoruyla karşılaştırılır).
        b3["benzersiz_sorgu_anahtari"] = len({s[:60] for s in sorgular})
        b3["detay"] = karsilastirma
        # OOV recall teyidi (P2/P3: skor ~0,7; RAPOR kırılımı)
        try:
            _, oov_tags, oov_dense, oov_sparse = build_query_vectors(
                OOV_SORGU, tokenizer)
            oov_sonuc = vm.hybrid_recall(oov_dense, oov_sparse, top_k=1,
                                         query_tags=oov_tags)
            b3["oov"] = {
                "skor": round(float(oov_sonuc[0]["score"]), 4) if oov_sonuc
                else None,
                "has_root_match": bool(oov_sonuc[0].get("has_root_match"))
                if oov_sonuc else None,
            }
        except Exception as e:  # noqa: BLE001 — RAPOR-düzeyi
            b3["oov"] = {"istisna": f"{type(e).__name__}: {e}"}
        detay["b3_determinizm"] = b3
        kapilar["K4_B3_DETERMINIZM"] = bool(
            b3["n"] == POZITIF_N
            and b3["cift_kosum_birebir"] == ILANLI["b3_cift_kosum_birebir"]
            and b3["p2_birebir"] == ILANLI["b3_p2_birebir"])
    except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
        detay["b3_determinizm"] = {"istisna": f"{type(e).__name__}: {e}"}
        kapilar["K4_B3_DETERMINIZM"] = False

    # ================= B4: add-yolu probe (YALNIZ probe üstünde) =========
    b4: Dict[str, Any] = {"count_sekansi": [], "guard_probe_adi": None}
    try:
        # vm (kristal_bellek) salt-okunur kullanılır — yazım guard'ı
        # _probe_yaz içinde kanonik-ada karşı uygulanır.
        probe = VectorMemory(PROBE_KOLEKSIYON, 768, host=arg.host,
                             port=arg.port)
        b4["guard_probe_adi"] = probe.collection_name
        arasekans: List[int] = [int(probe.get_document_count())]
        b4["otomatik_kurulum"] = arasekans[0] == 0  # :75-77 sessiz-kurulum
        cp: List[int] = [arasekans[0]]              # İLAN'lı [0, 3, 8, 9]
        for metin in PROBE_METINLER[:3]:            # add_document ×3
            _probe_yaz(probe, metin, tokenizer)
            arasekans.append(int(probe.get_document_count()))
        cp.append(int(probe.get_document_count()))  # checkpoint: 3
        toplu = PROBE_METINLER[3:8]                 # add_documents_batch ×5
        d_liste: List[List[float]] = []
        s_liste: List[Any] = []
        m_liste: List[Dict[str, Any]] = []
        for metin in toplu:
            t_ids, t_tags, d_vec, s_vec = _probe_vektor(metin, tokenizer)
            d_liste.append(d_vec)
            s_liste.append(s_vec)
            m_liste.append({"domain": "p4_probe",
                            "crystal_tags": t_tags, "token_ids": t_ids})
        probe.add_documents_batch(toplu, d_liste, s_liste, m_liste)
        cp.append(int(probe.get_document_count()))  # checkpoint: 8
        # DUBLÖR-teyidi: aynı metin tekrar → +1 (API-düzeyi upsert YOK)
        _probe_yaz(probe, PROBE_METINLER[0], tokenizer)
        cp.append(int(probe.get_document_count()))  # checkpoint: 9
        b4["count_sekansi"] = cp                    # İLAN'lı checkpoint-sekansı
        b4["count_ara_sekans"] = arasekans          # adım-adım (RAPOR)
        # geri-okuma birebir
        geri = probe.list_documents(20)
        geri_metin = sorted(
            str(r["payload"].get("text", "")) for r in geri)
        beklenen = sorted(PROBE_METINLER + [PROBE_METINLER[0]])
        b4["geri_okuma_sayi"] = len(geri)
        b4["geri_okuma_birebir"] = (geri_metin == beklenen
                                    and all(r["payload"].get("domain")
                                            == "p4_probe" for r in geri))
        # pozitif-kontrol: top-1 kendi-belge (8 belge × 2 yol)
        pozitif_d: List[Dict[str, Any]] = []
        dense_pozitif = 0
        hibrit_pozitif = 0
        for metin in PROBE_METINLER:
            _, q_tags, q_dense, q_sparse = build_query_vectors(
                metin, tokenizer)
            dr = probe.dense_recall(q_dense, top_k=1)
            hr = probe.hybrid_recall(q_dense, q_sparse, top_k=1,
                                     query_tags=q_tags)
            d_ok = bool(dr) and str(dr[0].get("text", "")) == metin
            h_ok = bool(hr) and str(hr[0].get("text", "")) == metin
            dense_pozitif += int(d_ok)
            hibrit_pozitif += int(h_ok)
            pozitif_d.append({"metin": metin[:50], "dense_ok": d_ok,
                              "hibrit_ok": h_ok,
                              "hibrit_skor": round(float(hr[0]["score"]), 4)
                              if hr else None,
                              "has_root_match": bool(hr[0].get(
                                  "has_root_match")) if hr else None})
        b4["pozitif_dense"] = dense_pozitif
        b4["pozitif_hibrit"] = hibrit_pozitif
        b4["detay"] = pozitif_d
        detay["b4_add_yolu"] = b4
        kapilar["K5_B4_ADD_YOLU"] = bool(
            b4["guard_probe_adi"] == PROBE_KOLEKSIYON
            and b4["count_sekansi"] == ILANLI["b4_count_sekansi"]
            and b4["geri_okuma_sayi"] == ILANLI["b4_geri_okuma_birebir"]
            and geri_metin == beklenen
            and dense_pozitif == ILANLI["b4_pozitif_beklenen"]
            and hibrit_pozitif == ILANLI["b4_pozitif_beklenen"])
    except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
        b4["istisna"] = f"{type(e).__name__}: {e}"
        detay["b4_add_yolu"] = b4
        kapilar["K5_B4_ADD_YOLU"] = False

    # ============ B5: recreate-yıkıcılık MUTASYON-kanıtı (probe) =========
    b5: Dict[str, Any] = {"explicit_recreate_nokta": None,
                          "auto_recreate_nokta": None,
                          "dense_size_after": None,
                          "reset_mesaji": None, "kaldi": None}
    try:
        # (i) explicit recreate (9 nokta) → sessiz SİLME → 0 nokta
        once_nokta_on_explicit = int(probe.get_document_count())  # 9 (B4'ten)
        buf = io.StringIO()
        with redirect_stdout(buf):
            probe.recreate_collection(768)
        b5["reset_mesaji"] = "has been reset" in buf.getvalue()
        b5["explicit_recreate_nokta"] = int(probe.get_document_count())
        b5["once_nokta_on_explicit"] = once_nokta_on_explicit
        # (ii) boyut-uyuşmazlık auto-recreate (:83 yolu): 1 belge geri-yaz →
        #      fresh VectorMemory(vector_size=384) → sessiz recreate → 0 nokta
        _probe_yaz(probe, PROBE_METINLER[1], tokenizer)
        b5["once_nokta_on_auto"] = int(probe.get_document_count())  # 1
        probe_384 = VectorMemory(PROBE_KOLEKSIYON, 384, host=arg.host,
                                 port=arg.port)
        b5["auto_recreate_nokta"] = int(probe_384.get_document_count())
        info = yonet_client.get_collection(PROBE_KOLEKSIYON)
        b5["dense_size_after"] = int(
            info.config.params.vectors["dense"].size)
        b5["yikicilik_kanitlandi"] = bool(
            b5["explicit_recreate_nokta"] == ILANLI["b5_recreate_nokta"]
            and b5["auto_recreate_nokta"] == ILANLI["b5_recreate_nokta"]
            and b5["once_nokta_on_explicit"] == len(PROBE_METINLER) + 1
            and b5["once_nokta_on_auto"] == 1
            and b5["dense_size_after"] == ILANLI["b5_dense_size"]
            and b5["reset_mesaji"])
        # (iii) temizlik: probe kaldırılır (operatör kararı: sonda sil).
        # NOT (T-0144 2. koşum dersi): VectorMemory'de delete_collection
        # METODU YOK — kanonik yol client.delete_collection (:126'daki
        # client-çağrısı; wrapper yok).
        probe_384.client.delete_collection(PROBE_KOLEKSIYON)
        b5["kaldi"] = bool(yonet_client.collection_exists(PROBE_KOLEKSIYON))
        b5["temizlik"] = not b5["kaldi"]
        detay["b5_mutasyon"] = b5
        kapilar["K6_B5_MUTASYON_KANITI"] = bool(
            b5["yikicilik_kanitlandi"] and not b5["kaldi"])
        probe_384.close()
    except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
        b5["istisna"] = f"{type(e).__name__}: {e}"
        detay["b5_mutasyon"] = b5
        kapilar["K6_B5_MUTASYON_KANITI"] = False

    # ====== B6 (RAPOR — hükme bağlanmaz): sessiz-fallback merdiveni ======
    b6: Dict[str, Any] = {}
    tmp_kok = tempfile.mkdtemp(prefix="p4_probe_local_")
    try:
        cache_once_anahtarlar = set(VectorMemory._shared_clients.keys())
        local_vm = VectorMemory(PROBE_KOLEKSIYON, 768, host="192.0.2.1",
                                port=6333, storage_path=tmp_kok)
        b6["local_storage_type"] = str(local_vm.storage_type)
        b6["local_is_in_memory"] = bool(local_vm.is_in_memory)
        b6["local_cache_yazildi"] = bool(
            "local:" in ",".join(VectorMemory._shared_clients.keys())
            and len(VectorMemory._shared_clients)
            == len(cache_once_anahtarlar) + 1)
        local_vm.close()
        b6["local_cache_cikarildi"] = bool(
            "local:" not in " ".join(VectorMemory._shared_clients.keys()))
        mem_vm = VectorMemory(PROBE_KOLEKSIYON, 768, host="192.0.2.1",
                              storage_path="")  # boş storage_path → :memory:
        b6["mem_storage_type"] = str(mem_vm.storage_type)
        b6["mem_is_in_memory"] = bool(mem_vm.is_in_memory)
        b6["mem_cache_yazilmadi"] = bool(
            len(VectorMemory._shared_clients)
            == len(cache_once_anahtarlar))
        b6["bulgu"] = ("SESSIZ_FALLBACK_CANLI — sahte-hostta remote "
                       "sessizce local-path'e (ve storage_path-yoksa "
                       ":memory:'ye) düşer; local-client cache'e YAZILIR "
                       "(gateway:123-124 storage_path'li çağrı riski)")
        detay["b6_fallback"] = b6
        mem_vm.close()
    except Exception as e:  # noqa: BLE001 — RAPOR-düzeyi (hükme bağlanmaz)
        b6["istisna"] = f"{type(e).__name__}: {e}"
        detay["b6_fallback"] = b6
    finally:
        shutil.rmtree(tmp_kok, ignore_errors=True)

    # ================= B7: dokunulmazlık (koşum SONU) ====================
    # Temizlik-sigortası (operatör kararı: sonda sil — hükümden bağımsız):
    if yonet_client.collection_exists(PROBE_KOLEKSIYON):
        try:
            _temizlik_vm = VectorMemory(PROBE_KOLEKSIYON, 768,
                                        host=arg.host, port=arg.port)
            _temizlik_vm.client.delete_collection(PROBE_KOLEKSIYON)
            _temizlik_vm.close()
            _stderr("UYARI: temizlik-sigortası devrede — koşum-içi "
                    "kaldırılmamış p4_probe_bellek silindi (kapı zaten "
                    "düşmüş olmalı; durumu B7 envanteri gösterecek)")
        except Exception as e:  # noqa: BLE001 — temizlik başarısızlığı kayda geçer
            _stderr(f"UYARI: probe-temizlik-sigortası başarısız: "
                    f"{type(e).__name__}: {e}")
    env_sonra = _envanter(yonet_client)
    sonra_digest = _envanter_digest(env_sonra)
    nokta_sonra = _nokta_saylari(env_sonra)
    foreign_sonra = {ad: n for ad, n in nokta_sonra.items()
                     if ad not in KANONIK_YASAK and ad != PROBE_KOLEKSIYON}
    kanal_sonra_bayt = (os.path.getsize(GERCEK_KANAL)
                        if os.path.exists(GERCEK_KANAL) else -1)
    p2_sha_sonra = _sha256(p2_hukum_yol)
    dokunulmaz = bool(
        sonra_digest == once_digest
        == ILANLI["envanter_digest_once"]
        and nokta_sonra.get(KANONIK_KOLEKSIYON)
        == ILANLI["b1_kristal_nokta"]
        and foreign_sonra == foreign_once
        and PROBE_KOLEKSIYON not in env_sonra["adlar"]
        and "simulasyon_bellek" not in env_sonra["adlar"]
        and kanal_sonra_bayt == ILANLI["kanal_sonra_bayt"]
        and p2_sha_sonra == detay["p2_hukum_sha256_once"])
    detay["dokunulmazlik"] = {
        "durum": "GEÇTİ" if dokunulmaz else "SAPMA",
        "once_digest": once_digest, "sonra_digest": sonra_digest,
        "foreign_once": foreign_once, "foreign_sonra": foreign_sonra,
        "kristal_once": nokta_once.get(KANONIK_KOLEKSIYON),
        "kristal_sonra": nokta_sonra.get(KANONIK_KOLEKSIYON),
        "probe_sunucuda_yok": PROBE_KOLEKSIYON not in env_sonra["adlar"],
        "simulasyon_bellek_yok":
            "simulasyon_bellek" not in env_sonra["adlar"],
        "kanal_once_bayt": kanal_once_bayt,
        "kanal_sonra_bayt": kanal_sonra_bayt,
        "p2_hukum_sha_once": detay["p2_hukum_sha256_once"],
        "p2_hukum_sha_sonra": p2_sha_sonra,
    }
    detay["faz_b_envanter_sonra"] = env_sonra
    kapilar["K7_DOKUNULMAZLIK"] = dokunulmaz

    # ---- Hizalama-ön-kontrolü (koşum-İÇİ statik; T-0143 dersi) ---------
    if set(kapilar) != EXPECTED_KAPILAR:
        _stderr(f"DUR: KAPI-ANAHTAR-HİZASIZLIĞI: beklenen "
                f"{sorted(EXPECTED_KAPILAR)} != ölçülen {sorted(kapilar)}")
        sys.exit(2)

    # ---- Hüküm (betikten) → hüküm JSON + rapor (elle sayı YOK) ----------
    hukum_json, rc = _hukum_kur(kapilar, detay, damga=_ts())
    hukum_yol = _hukum_yaz(hukum_json, arg)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha=_sha256(
        korpus_yol), vocab_sha=_sha256(vocab_yol),
        once_digest=once_digest, sonra_digest=sonra_digest)
    _stderr(f"HÜKÜM: {hukum_json['hukum']} rc={rc} → {hukum_yol}")
    yonet_client.close()
    sys.exit(rc)


def _probe_yaz(bellek: VectorMemory, metin: str,
               tokenizer: KristalTokenizer) -> None:
    """Kanonik add_document — YALNIZ probe koleksiyonunda (guard'lı)."""
    if bellek.collection_name != PROBE_KOLEKSIYON:
        raise RuntimeError(
            f"GUARD: yazım yalnız {PROBE_KOLEKSIYON} üstünde — "
            f"{bellek.collection_name} değil")
    _, _, d_vec, s_vec = _probe_vektor(metin, tokenizer)
    bellek.add_document(metin, d_vec, s_vec,
                        {"domain": "p4_probe"})


def _hukum_kur(kapilar: Dict[str, bool], detay: Dict[str, Any],
               damga: str) -> Tuple[Dict[str, Any], int]:
    """Hüküm BETİK İÇİNDE — İLAN'lı kural: tüm kapılar GEÇTİ → P4_GECTİ
    rc=0; aksi DUR rc=2."""
    tamami = all(kapilar.values())
    sonuc = "P4_GECTİ" if tamami else "DUR"
    rc = 0 if tamami else 2
    hukum_json = {
        "hukum": sonuc,
        "rc": rc,
        "damga": damga,
        "kapilar": {k: ("GEÇTİ" if v else "DÜŞTÜ")
                    for k, v in sorted(kapilar.items())},
        "detay": detay,
    }
    return hukum_json, rc


def _hukum_yaz(hukum_json: Dict[str, Any], arg: Any) -> str:
    hukum_yol = (arg.hukum_json
                 or arg.rapor.replace("_sonuc_", "_hukum_")
                 .replace(".md", ".json"))
    hukum_yol = hukum_yol if os.path.isabs(hukum_yol) \
        else os.path.join(REPO_ROOT, hukum_yol)
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2,
                  sort_keys=True)
    return hukum_yol


def _rapor_yaz(rapor_yol: str, hukum_json: Dict[str, Any], arg: Any,
               ilan_sha: str, korpus_sha: str, vocab_sha: str,
               once_digest: str, sonra_digest: str) -> None:
    """Rapor — İLAN'lı↔ölçülen tablolar; sayılar hüküm JSON'undan (YOK)."""
    d = hukum_json.get("detay", {})
    with open(rapor_yol, "w", encoding="utf-8") as f:
        f.write("# MİMARİ DOĞRULAMA PAKET-4 SONUÇ — Üniversal Hafıza "
                "(VectorMemory yaşam-döngüsü, T-0144)\n\n")
        f.write(f"**Damga:** {_ts()} · **Hüküm (BETİKTEN):** "
                f"**{hukum_json['hukum']} (rc={hukum_json['rc']})**\n\n")
        f.write("| Kapı | Durum |\n|---|---|\n")
        for k, v in hukum_json.get("kapilar", {}).items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen)\n\n")
        f.write("| Ölçüm | İLAN'lı | Ölçülen | Durum |\n"
                "|---|---|---|---|\n")
        for satir in d.get("faz_a_kapi_detay", []):
            f.write(f"| {satir['olcum']} | {satir['ilanli']} | "
                    f"{satir['olculen']} | {satir['durum']} |\n")
        c = d.get("faz_a", {})
        f.write(f"- recreate-yıkıcılık satırları (vector_memory.py): "
                f"{c.get('vm_recreate_delete_satirlar')} · otomatik-kurulum "
                f"bandı: {c.get('vm_otomatik_kurulum_bandi')}\n")
        f.write(f"- VectorMemory varsayılan storage_path: "
                f"{c.get('vm_varsayilan_storage_path')} (host'suz "
                f"çağrıda repo-içi local-storage riski — RAPOR §7)\n")
        f.write(f"- embedding deterministik (saf çift-çağrı): "
                f"{c.get('embedding_deterministik')}\n")
        b1 = d.get("b1_baglanma", {})
        f.write(f"\n## FAZ-B — VectorMemory yaşam-döngüsü koşumu\n\n")
        f.write(f"- B1 bağlanma: storage_type **{b1.get('storage_type')}** · "
                f"is_in_memory {b1.get('is_in_memory')} · kristal_bellek "
                f"**{b1.get('kristal_nokta')}** · cache-reuse "
                f"{b1.get('cache_reuse')}\n")
        b2 = d.get("b2_arz_once", {})
        f.write(f"- B2 arz-çıpa: digest **{b2.get('digest')}** · foreign "
                f"**{b2.get('foreign_sayi')}** · kristal_bellek "
                f"**{b2.get('kristal_bellek_var')}** "
                f"({b2.get('kristal_bellek_nokta')}) · simulasyon_bellek "
                f"YOK-teyit {b2.get('simulasyon_bellek_yok')}\n")
        b3 = d.get("b3_determinizm", {})
        f.write(f"- B3 determinizm: P2-birebir **{b3.get('p2_birebir')}/"
                f"{b3.get('p2_karsilastirma_n')}** + iç-çift-koşum "
                f"**{b3.get('cift_kosum_birebir')}/{b3.get('n')}** "
                f"(4 ondalık) · OOV recall: {b3.get('oov')}\n"
                f"  - NOT: P2 hüküm-detay anahtarları [:60]-kesmeli — "
                f"P2 skor-anahtarı **{b3.get('p2_skor_anahtari')}** "
                f"(20 satır) vs benzersiz-sorgu-anahtarı "
                f"**{b3.get('benzersiz_sorgu_anahtari')}** — kesme-"
                f"çakışan çiftler aynı P2-skoruyla karşılaştırılır; "
                f"birebir SATIR-düzeyi 4 ondalık (RAPOR §7 beyanı)\n")
        b4 = d.get("b4_add_yolu", {})
        f.write(f"- B4 add-yolu (probe): checkpoint-count-sekansı "
                f"**{b4.get('count_sekansi')}** (İLAN'lı "
                f"[0, 3, 8, 9] — dublör +1: API-düzeyi upsert YOK) · "
                f"ara-adım-sekansı **{b4.get('count_ara_sekans')}** "
                f"(RAPOR kırılımı) · "
                f"geri-okuma **{b4.get('geri_okuma_sayi')}** birebir · "
                f"pozitif-kontrol dense **{b4.get('pozitif_dense')}/8** + "
                f"hibrit **{b4.get('pozitif_hibrit')}/8**\n")
        b5 = d.get("b5_mutasyon", {})
        f.write(f"- B5 recreate-yıkıcılık MUTASYON-KANITI: explicit "
                f"recreate → **{b5.get('explicit_recreate_nokta')}** nokta "
                f"(9→0; reset-mesajı {b5.get('reset_mesaji')}) · boyut-"
                f"uyuşmazlık auto-recreate (:83) → "
                f"**{b5.get('auto_recreate_nokta')}** nokta (önce 1; "
                f"dense-size {b5.get('dense_size_after')}) · probe "
                f"kaldırma: kaldi={b5.get('kaldi')}\n")
        b6 = d.get("b6_fallback", {})
        f.write(f"- B6 (RAPOR) sessiz-fallback: local "
                f"**{b6.get('local_storage_type')}** (is_in_memory "
                f"{b6.get('local_is_in_memory')}; cache-yazıldı "
                f"{b6.get('local_cache_yazildi')}) · :memory: "
                f"**{b6.get('mem_storage_type')}** (is_in_memory "
                f"{b6.get('mem_is_in_memory')}; cache-yazılmadı "
                f"{b6.get('mem_cache_yazilmadi')}) · bulgu: "
                f"{b6.get('bulgu')}\n")
        dok = d.get("dokunulmazlik", {})
        f.write(f"- B7 dokunulmazlık: {dok.get('durum')} · digest "
                f"{dok.get('once_digest')}→{dok.get('sonra_digest')} · "
                f"kristal {dok.get('kristal_once')}→"
                f"{dok.get('kristal_sonra')} · probe sunucuda YOK: "
                f"{dok.get('probe_sunucuda_yok')} · kanal "
                f"{dok.get('kanal_once_bayt')}→"
                f"{dok.get('kanal_sonra_bayt')} bayt\n")
        f.write("\n## Digest tablosu\n\n| Dosya | sha256 |\n|---|---|\n")
        f.write(f"| {os.path.basename(arg.ilan)} (İLAN — koşum ÖNCESİ) | "
                f"`{ilan_sha}` |\n")
        f.write(f"| {arg.korpus} (DONMUŞ kaynak) | `{korpus_sha}` |\n")
        f.write(f"| {arg.vocab} (DONMUŞ sözlük) | `{vocab_sha}` |\n")
        f.write(f"| {arg.p2_hukum} (P2 hüküm — B3 birebir kaynağı) | "
                f"`{d.get('p2_hukum_sha256_once', '-')}` |\n")
        kanal_sha = _sha256(GERCEK_KANAL) \
            if os.path.exists(GERCEK_KANAL) else "YOK"
        f.write(f"| data/future_train_vector.jsonl (GERÇEK kanal çıpası — "
                f"0 bayt) | `{kanal_sha}` |\n")
        f.write(f"| envanter ÖNCE/SONRA digest | `{once_digest}` / "
                f"`{sonra_digest}` |\n")
        f.write(f"| dokunulmazlık | {dok.get('durum', '-')} |\n")


if __name__ == "__main__":
    main()