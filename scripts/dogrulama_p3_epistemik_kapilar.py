#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-0143 — MİMARİ DOĞRULAMA PAKET-3: Epistemik Döngü Kapıları (T-0143)

İLAN: data/eval/mimari_dogrulama_p3_ilan_2026-09-27.md (koşum-ÖNCESİ damgalı;
§7 koşum-öncesi revizyonlu — B2 hüküm-dışı rapor kırılımı; B3 force'lu OOV
hüküm kapısı). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK. rc ∈ {0, 2}.

Operatör kararları (27 Eyl 2026): (1) YALNIZ DOĞRULAMA — kanonik kod salt
IMPORT, YAZIM YOK; (2) GEÇİCİ PROBE YOLU — future_train yazımı yalnız
data/eval/p3_future_train_probe.jsonl'e; data/future_train_vector.jsonl
0-BAYT çıpası koşum-sonunda kapıyla teyit edilir. kristal_bellek (P2 arzı,
36 nokta) YALNIZ OKUNUR; 9 foreign koleksiyon DOKUNULMAZ.

Kanonik kod IMPORT edilir (kopya YASAK):
  EpistemicCuriosityAgent           (src/rag/epistemic_agent.py)
  CuriosityEngine                   (src/rag/merak.py)
  TriModalRouter                    (src/llm/router.py)
  VectorMemory                      (src/rag/vector_memory.py)
  is_context_usable / RAG_MATCH_THRESHOLD (src/rag/rag_pipeline.py)
  KristalTokenizer / Vocabulary     (src/llm/tokenizer.py)
  resize_state_dict                 (src/llm/prompt_contract.py)
  KristalLM                         (scripts/train_step_demo.py)
  _envanter / _envanter_digest
    (scripts/dogrulama_p2_rag_gezgini.py — P2 yardımcıları; kanonik src/** DEĞİL,
    kopya yazmak yerine kendi betiğimiz arası yeniden-kullanım)

Koşum sandbox DIŞI (Qdrant trafiği sandbox proxy'sinden geçmez); model saf
CPU'da; MPS YOK; çift-eğitici YOK; seed=42 (router/merak fresh-init determinizmi).
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient  # noqa: E402

from src.rag.vector_memory import VectorMemory  # noqa: E402
from src.rag.rag_pipeline import (  # noqa: E402
    is_context_usable,
    RAG_MATCH_THRESHOLD,
)
from src.rag.epistemic_agent import EpistemicCuriosityAgent  # noqa: E402
from src.rag.merak import CuriosityEngine  # noqa: E402
from src.llm.router import TriModalRouter  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from scripts.train_step_demo import KristalLM  # noqa: E402
from scripts.dogrulama_p2_rag_gezgini import (  # noqa: E402
    _envanter,
    _envanter_digest,
)

# ---- İLAN'lı sabitler (koşum ÖNCESİ sabit; koşum-sonrası yumuşatılmaz) ----
POZITIF_N = 20
SEED = 42
KANONIK_ADLAR = ("kristal_bellek", "simulasyon_bellek")
ILANLI = {
    "esik_rag_match_threshold": 0.40,
    "tau_default": 2.5,
    "similarity_threshold_default": 0.85,
    "esik_tanim_sayisi": 1,                 # AnnAssign-dahil sayaç (T-0142 §7.1)
    "esik_import_sayisi": 2,
    "merak_router_state_dict_yukleme": 0,   # eğitimsiz-projeksiyon (grep=0)
    "foreign_koleksiyon_once": 9,
    "kristal_bellek_nokta_once": 36,
    "turk_articles_nokta": 5,
    "elektor_articles_nokta": 90122,
    "gercek_kanal_once_bayt": 0,
    "gercek_kanal_sonra_bayt": 0,
    # ---- İLAN-2 (T-0147 onarım-turu; koşum-ÖNCESİ beyan) ----
    # A-3 skor-normalizasyonu (RRF_SCORE_MAX=0,75) + OOV fail-closed:
    # * OOV sorgu: ham 0,7 × 0,05 = 0,035 → normalize 0,0467 (tavan 0,075)
    #   VE koşullanma FALSE — koşum-1'deki sahte-koşullanma sınıfı (skor 0,7
    #   bandı + koşullanma TRUE) KAPANMIŞ olmalı.
    # * Kapı-D (0,85) onarım-sonrası ERİŞİLEBİLİR: pozitif-kontrol top-1 ham
    #   tavan 0,75 (koşum-1 P2 birebir-çıpa) → normalize 1,0 ≥ 0,85 →
    #   is_high en az 1/20 tetiklenir (birebir-sayı RAPOR-kaydıdır —
    #   model-entropi-tetiklenmeli kapıdan çıkarıldı; koşum-1 0/20 kapısı
    #   "0,85 > bant-maks 0,75 ulaşılamaz" GEREKÇESİYLE onarım-sonrası GEÇERSİZ).
    "b3_oov_skor_tavani": 0.075,
    "b4_is_high_beklenen_min": 1,
}
RECORD_ANAHTARLARI = frozenset({
    "instruction", "input", "output", "model_failed_output",
    "decompiled_output", "rag_document", "retrieval_collection",
    "similarity_score", "similarity_threshold", "entropy_pre",
    "entropy_post", "tau", "reason", "timestamp",
})
TELEMETRI_ANAHTARLARI = frozenset({
    "entropy_pre", "needs_retrieval", "retrieval_triggered",
    "retrieved_document", "match_score", "is_high_similarity",
    "entropy_post", "morpheme_output", "epistemic_failure",
    "future_train_recorded",
})
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
    print(f"[T-0143] {msg}", file=sys.stderr)


def _sorgular(korpus_yol: str, rng: random.Random,
              adet: int) -> Tuple[List[str], List[int]]:
    """Korpus input'larındaki <BELGE>…</BELGE> sonrası sorgular (P2 seed'li
    seçim protokolü: random.Random(SEED).sample + sorted)."""
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
    return tum, secili


def _faz_a_envanteri(repo_root: str) -> Dict[str, Any]:
    """FAZ-A: kod-envanteri (koşumsuz). AST sayaç Assign|AnnAssign İKİSİNİ
    sayar (T-0142 §7.1 dersi — type-hint'li kodda AnnAssign norm)."""
    bulgular: Dict[str, Any] = {}

    tanim_sitesi: List[str] = []
    import_sitesi: List[str] = []
    src_kok = os.path.join(repo_root, "src")
    for dirpath, _, dosyalar in os.walk(src_kok):
        for d in dosyalar:
            if not d.endswith(".py"):
                continue
            yol = os.path.join(dirpath, d)
            try:
                metin = open(yol, encoding="utf-8").read()
                agac = ast.parse(metin)
            except Exception as e:  # noqa: BLE001 — okunamayan dosya raporlanır
                _stderr(f"UYARI: AST parse edilemedi {yol}: {e}")
                continue
            for dugum in ast.walk(agac):
                if isinstance(dugum, (ast.Assign, ast.AnnAssign)):
                    hedefler = dugum.targets if isinstance(dugum, ast.Assign) \
                        else [dugum.target]
                    for hedef in hedefler:
                        if isinstance(hedef, ast.Name) \
                                and hedef.id == "RAG_MATCH_THRESHOLD":
                            tanim_sitesi.append(
                                os.path.relpath(yol, repo_root))
                if (isinstance(dugum, ast.ImportFrom) and dugum.module
                        and "rag_pipeline" in dugum.module):
                    if any(a.name == "RAG_MATCH_THRESHOLD"
                           for a in dugum.names):
                        import_sitesi.append(
                            os.path.relpath(yol, repo_root))
    bulgular["esik_tanim_sitesi"] = sorted(set(tanim_sitesi))
    bulgular["esik_tanim_sayisi"] = len(set(tanim_sitesi))
    bulgular["esik_import_sitesi"] = sorted(set(import_sitesi))
    bulgular["esik_import_sayisi"] = len(set(import_sitesi))
    bulgular["rag_match_threshold_deger"] = float(RAG_MATCH_THRESHOLD)

    param = inspect.signature(EpistemicCuriosityAgent.__init__).parameters
    bulgular["tau_default"] = float(param["tau"].default)
    bulgular["similarity_threshold_default"] = float(
        param["similarity_threshold"].default)
    bulgular["future_train_path_default"] = str(
        param["future_train_path"].default)
    bulgular["merak_tau_default"] = float(
        inspect.signature(CuriosityEngine.__init__).parameters["tau"].default)

    egitimli_yukleme: Dict[str, int] = {}
    for ad, alt in (("merak.py", "rag"), ("router.py", "llm")):
        yol = os.path.join(repo_root, "src", alt, ad)
        metin = open(yol, encoding="utf-8").read()
        say = metin.count("state_dict") + len(
            re.findall(r"\bload\b\s*\(", metin))
        egitimli_yukleme[f"src/{alt}/{ad}"] = say
    bulgular["merak_router_state_dict_yukleme"] = sum(egitimli_yukleme.values())
    bulgular["merak_router_grep_detay"] = egitimli_yukleme

    vm_yol = os.path.join(repo_root, "src", "rag", "vector_memory.py")
    recreate_satirlar: List[int] = []
    with open(vm_yol, encoding="utf-8") as f:
        for i, satir in enumerate(f, 1):
            if "recreate_collection" in satir or "delete_collection" in satir:
                recreate_satirlar.append(i)
    bulgular["recreate_satirlar_vector_memory"] = recreate_satirlar
    rp_yol = os.path.join(repo_root, "src", "rag", "rag_pipeline.py")
    rp_satirlar = open(rp_yol, encoding="utf-8").read().splitlines()
    sessiz_satirlar = [i for i in range(len(rp_satirlar) - 1)
                       if "except Exception:" in rp_satirlar[i]
                       and "return None" in rp_satirlar[i + 1]]
    bulgular["sessiz_none_satirlar_rag_pipeline"] = sessiz_satirlar

    ea_yol = os.path.join(repo_root, "src", "rag", "epistemic_agent.py")
    ea_satirlar = open(ea_yol, encoding="utf-8").read().splitlines()
    record_baslangic = -1
    for i, s in enumerate(ea_satirlar, 1):
        if "record = {" in s:
            record_baslangic = i
            break
    bulgular["record_baslangic_satiri"] = record_baslangic
    return bulgular


def _faz_a_kapilari(faz_a: Dict[str, Any]) -> Tuple[bool,
                                                    List[Dict[str, Any]]]:
    """Kapı K1: FAZ-A İLAN'lı sayılar ↔ ölçülen (birebir)."""
    olculen = {
        "esik_rag_match_threshold": faz_a["rag_match_threshold_deger"],
        "tau_default": faz_a["tau_default"],
        "similarity_threshold_default": faz_a["similarity_threshold_default"],
        "esik_tanim_sayisi": faz_a["esik_tanim_sayisi"],
        "esik_import_sayisi": faz_a["esik_import_sayisi"],
        "merak_router_state_dict_yukleme":
            faz_a["merak_router_state_dict_yukleme"],
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
    return tamam, detay


def _model_yukle(vocab_yol: str, base_ckpt: str,
                 detay: Dict[str, Any]) -> Tuple[KristalLM, Vocabulary,
                                                 KristalTokenizer, int]:
    """Kanonik yükleme kalıbı (evaluate_carpenter_anka.py kardeşi):
    kafa/sözlük uyum kapısı + KristalLM + resize_state_dict + strict=False.
    KOD YAZIMI YOK — yalnız kanonik import + kurucu-argümanları."""
    vocab = Vocabulary()
    vocab.load(vocab_yol)
    n_vocab = len(vocab.itos)
    sd = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    kafa = sd.get("lm_head.weight") if isinstance(sd, dict) else None
    if kafa is None:
        _stderr("DUR: checkpoint'te 'lm_head.weight' YOK — kimlik doğrulanamaz")
        raise SystemExit(2)
    kafa_n = int(kafa.shape[0])
    if kafa_n != n_vocab:
        _stderr(f"DUR: KAFA/SOZLUK UYUSMAZLIGI: checkpoint kafasi {kafa_n} != "
                f"sozluk {n_vocab} — boyle bir kosum checkpoint'i KIRPAR "
                f"(T-0044 sinifi)")
        raise SystemExit(2)
    model = KristalLM(vocab_size=n_vocab, n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    model.load_state_dict(resize_state_dict(model, sd), strict=False)
    model.to("cpu").eval()
    detay["model_yukleme"] = {"kafa": kafa_n, "sozluk": n_vocab,
                              "device": "cpu",
                              "base_ckpt": os.path.basename(base_ckpt)}
    lexicon = LexiconManager()
    lexicon.load_from_tsv(
        os.path.join(REPO_ROOT, "data", "lexicon", "roots.tsv"))
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    tokenizer = KristalTokenizer(compiler, vocab, literal_entity_mode=False)
    return model, vocab, tokenizer, n_vocab


def _agent_kur(model: KristalLM, tokenizer: KristalTokenizer,
               memory: VectorMemory, probe_yol: str,
               similarity_threshold: float) -> EpistemicCuriosityAgent:
    """Kanonik kurucu-argümanlarıyla agent (KOD YAZIMI YOK); seed sabit
    (router/merak fresh-init projeksiyonları — eğitilmiş ağırlık YOK)."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    return EpistemicCuriosityAgent(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        similarity_threshold=similarity_threshold,
        future_train_path=probe_yol,
        device="cpu",
    )


def _kosum_ozet(sonuc: Dict[str, Any]) -> Dict[str, Any]:
    """process_query telemetrisinin hüküm-rapor kaydı (İLAN'lı alanlar)."""
    return {
        "sorgu": str(sonuc.get("query", ""))[:60],
        "entropy_pre": round(float(sonuc.get("entropy_pre", 0.0)), 4),
        "needs_retrieval": bool(sonuc.get("needs_retrieval")),
        "retrieval_triggered": bool(sonuc.get("retrieval_triggered")),
        "match_score": round(float(sonuc.get("match_score", 0.0)), 4),
        "retrieved_document": str(sonuc.get("retrieved_document") or "")[:60],
        "kosullanma_teyit":
            len(str(sonuc.get("retrieved_document") or "")) > 0,
        "is_high_similarity": bool(sonuc.get("is_high_similarity")),
        "entropy_post": round(float(sonuc.get("entropy_post", 0.0)), 4),
        "morpheme_output": str(sonuc.get("morpheme_output", ""))[:60],
        "router_experts": sonuc.get("router_experts"),
        "epistemic_failure": bool(sonuc.get("epistemic_failure")),
        "future_train_recorded": bool(sonuc.get("future_train_recorded")),
    }


def _dur_ve_cik(kapilar: Dict[str, bool], detay: Dict[str, Any], damga: str,
                arg: Any, ilan_sha: str, korpus_sha: str, base_sha: str,
                vocab_sha: str, once_digest: Optional[str],
                sebepler: List[str]) -> None:
    """Ara-DUR yolu (fail-closed): hüküm + rapor yaz, rc=2 ile çık."""
    detay["dur_sebepleri"] = sebepler
    hukum_json, rc = _hukum_kur(kapilar, detay, damga)
    _hukum_yaz(hukum_json, arg)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
               base_sha, vocab_sha, once_digest, None)
    _stderr(f"HÜKÜM: {hukum_json['hukum']} rc={rc}")
    sys.exit(rc)


def main() -> None:
    ayrici = argparse.ArgumentParser(
        description="T-0143 — P3 Epistemik Döngü Kapıları doğrulama "
                    "(İLAN'lı, fail-closed)")
    ayrici.add_argument("--host", default="192.168.1.5")
    ayrici.add_argument("--port", type=int, default=6333)
    ayrici.add_argument("--korpus",
                        default="data/realistic_rag/test_natural_150.jsonl")
    ayrici.add_argument("--koleksiyon", default="kristal_bellek")
    ayrici.add_argument("--vocab", required=True,
                        help="Sözlük AÇIKÇA verilir (T-0087 dersi)")
    ayrici.add_argument("--base-ckpt", default="data/anka_base_v2.pt")
    ayrici.add_argument("--probe-yol",
                        default="data/eval/p3_future_train_probe.jsonl")
    ayrici.add_argument("--p2-hukum",
                        default="data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json")
    ayrici.add_argument("--ilan", required=True)
    ayrici.add_argument("--rapor", required=True)
    arg = ayrici.parse_args()

    damga = _ts()
    ilan_sha = _sha256(arg.ilan)
    korpus_sha = _sha256(arg.korpus)
    base_sha = _sha256(arg.base_ckpt)
    vocab_sha = _sha256(arg.vocab)
    kapilar: Dict[str, bool] = {}
    detay: Dict[str, Any] = {
        "damga": damga,
        "ilan": os.path.basename(arg.ilan),
        "ilan_sha256": ilan_sha,
        "korpus": arg.korpus, "korpus_sha256": korpus_sha,
        "base_ckpt": arg.base_ckpt, "base_ckpt_sha256": base_sha,
        "vocab": arg.vocab, "vocab_sha256": vocab_sha,
        "probe_yol": arg.probe_yol,
        "p2_hukum": arg.p2_hukum,
        "rag_match_threshold": RAG_MATCH_THRESHOLD,
        "kosum_sandbox_disi": True, "device": "cpu", "seed": SEED,
    }

    kanal_once_bayt = os.path.getsize(GERCEK_KANAL) \
        if os.path.exists(GERCEK_KANAL) else -1
    detay["gercek_kanal_once_bayt"] = kanal_once_bayt

    # ---- FAZ-A: kod-envanteri (koşumsuz) + kanal-çıpa
    faz_a = _faz_a_envanteri(REPO_ROOT)
    detay["faz_a"] = faz_a
    k1_gecti, k1_detay = _faz_a_kapilari(faz_a)
    kapilar["K1_FAZ_A_ENVANTER"] = k1_gecti
    detay["faz_a_kapi_detay"] = k1_detay
    if kanal_once_bayt != ILANLI["gercek_kanal_once_bayt"]:
        _stderr(f"DUR: gerçek kanal çıpası BOZULMUŞ ({kanal_once_bayt} bayt "
                f"!= 0 — koşum ÖNCESİ ihlal; P5 çıpası kayboldu)")
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha, korpus_sha,
                    base_sha, vocab_sha, None, ["kanal-cipa-bayt-sapmasi"])
        return

    # ---- FAZ-B ön-koşul: sunucu + arz-koruma (salt-okunur envanter)
    try:
        client = QdrantClient(host=arg.host, port=arg.port, timeout=5.0,
                              check_compatibility=False)
        client.get_collections()
    except Exception as e:  # noqa: BLE001 — sessiz-fallback YOK
        _stderr(f"DUR: sunucu erişilemez ({type(e).__name__}: {e}); "
                f"sessiz-fallback YAPILMAZ")
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha, korpus_sha,
                    base_sha, vocab_sha, None,
                    [f"sunucu-erisilemez: {type(e).__name__}: {e}"])
        return

    env_once = _envanter(client)
    once_digest = _envanter_digest(env_once)
    detay["faz_b_envanter_once"] = env_once
    detay["envanter_once_digest"] = once_digest
    nokta_once = {k["ad"]: k["nokta"] for k in env_once["kalemler"]}
    foreign_once = {ad: n for ad, n in nokta_once.items()
                    if ad not in KANONIK_ADLAR}
    olculen_arz = {
        "foreign_koleksiyon_once": len(foreign_once),
        "kristal_bellek_nokta_once": nokta_once.get(arg.koleksiyon, -1),
        "turk_articles_nokta": nokta_once.get("türk_articles", -1),
        "elektor_articles_nokta": nokta_once.get("elektor_articles", -1),
    }
    detay["arz_once"] = olculen_arz
    k2_gecti = all(olculen_arz[k] == ILANLI[k] for k in olculen_arz)
    kapilar["K2_ARZ_KORUNDU"] = k2_gecti

    # P2-sürekli çıpası: foreign envanteri P2 hüküm foreign_sonra ile birebir
    p2_foreign: Optional[Dict[str, int]] = None
    try:
        with open(arg.p2_hukum, encoding="utf-8") as f:
            p2 = json.load(f)
        p2_foreign = p2.get("detay", {}).get("dokunulmazlik", {}).get(
            "foreign_sonra")
    except Exception as e:  # noqa: BLE001 — okunamazsa kapı düşer (sessiz YOK)
        _stderr(f"UYARI: P2 hüküm JSON okunamadı ({type(e).__name__}: {e})")
    detay["p2_foreign_cipa"] = p2_foreign
    if p2_foreign is not None and foreign_once != p2_foreign:
        _dur_ve_cik(
            {**kapilar, "K2_ARZ_KORUNDU": False}, detay, damga, arg,
            ilan_sha, korpus_sha, base_sha, vocab_sha, once_digest,
            ["foreign-envanteri-P2-cipasiyla-uyusmuyor"])
        return

    if arg.koleksiyon not in nokta_once:
        _stderr(f"DUR: '{arg.koleksiyon}' sunucuda YOK — P2 arzı kaybolmuş "
                f"(36-nokta çıpası bozuldu)")
        _dur_ve_cik({**kapilar, "K2_ARZ_KORUNDU": False}, detay, damga,
                    arg, ilan_sha, korpus_sha, base_sha, vocab_sha,
                    once_digest, ["kanonik-koleksiyon-kayip"])
        return

    # ---- B1+B2: varsayılan agent (0,85) — 20 pozitif sorgu, force_rag=False
    model, vocab, tokenizer, n_vocab = _model_yukle(arg.vocab, arg.base_ckpt,
                                                    detay)
    memory = VectorMemory(collection_name=arg.koleksiyon, vector_size=768,
                          host=arg.host, port=arg.port, storage_path="")
    if memory.is_in_memory \
            or not str(memory.storage_type).startswith("remote"):
        _stderr(f"DUR: VectorMemory uzak sunucuya bağlanamadı "
                f"(storage_type={memory.storage_type}); sessiz-fallback YASAK")
        _dur_ve_cik({**kapilar, "K3_B1_MERAK_KAYIT_TAM": False}, detay,
                    damga, arg, ilan_sha, korpus_sha, base_sha, vocab_sha,
                    once_digest, ["vector-memory-fallback"])
        return
    sayi = memory.get_document_count()
    detay["kristal_bellek_nokta_agent"] = sayi
    if sayi != ILANLI["kristal_bellek_nokta_once"]:
        _stderr(f"DUR: '{arg.koleksiyon}' nokta sayısı {sayi} != İLAN'lı "
                f"{ILANLI['kristal_bellek_nokta_once']}")
        _dur_ve_cik({**kapilar, "K2_ARZ_KORUNDU": False}, detay, damga,
                    arg, ilan_sha, korpus_sha, base_sha, vocab_sha,
                    once_digest, ["kanonik-nokta-sayisi-sapmasi"])
        return

    tum_sorgular, secili = _sorgular(arg.korpus, random.Random(SEED),
                                     POZITIF_N)
    detay["sorgu_havuzu"] = len(tum_sorgular)
    detay["secili_sorgu"] = len(secili)

    agent_default = _agent_kur(model, tokenizer, memory, arg.probe_yol,
                               ILANLI["similarity_threshold_default"])
    b1_kayitlar: List[Dict[str, Any]] = []
    b1_istisnalar: List[Dict[str, Any]] = []
    for j in secili:
        sorgu = tum_sorgular[j]
        try:
            sonuc = agent_default.process_query(sorgu, force_rag=False)
        except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
            b1_istisnalar.append({"sorgu": sorgu[:60],
                                  "istisna": f"{type(e).__name__}: {e}"})
            continue
        b1_kayitlar.append(_kosum_ozet(sonuc))
    detay["b1_merak"] = {
        "n": POZITIF_N, "kayit_tam": len(b1_kayitlar),
        "istisna_sayisi": len(b1_istisnalar), "istisnalar": b1_istisnalar,
        "tetiklenme_kirilim": {
            "retrieval_triggered_true": sum(1 for k in b1_kayitlar
                                            if k["retrieval_triggered"]),
            "needs_retrieval_true": sum(1 for k in b1_kayitlar
                                        if k["needs_retrieval"]),
            "kosullanma_teyitli": sum(1 for k in b1_kayitlar
                                      if k["kosullanma_teyit"]),
        },
        "entropi_pre_min": min((k["entropy_pre"] for k in b1_kayitlar),
                               default=None),
        "entropi_pre_max": max((k["entropy_pre"] for k in b1_kayitlar),
                               default=None),
        "detay": b1_kayitlar,
    }
    anahtar_tam = all(TELEMETRI_ANAHTARLARI.issubset(k) for k in b1_kayitlar)
    kapilar["K3_B1_MERAK_KAYIT_TAM"] = bool(
        len(b1_kayitlar) == POZITIF_N and not b1_istisnalar and anahtar_tam)

    # ---- B2 (RAPOR — hükme bağlanmaz): triggered alt-kümesinde P2 skor-birebir
    p2_skorlar: Dict[str, float] = {}
    try:
        with open(arg.p2_hukum, encoding="utf-8") as f:
            p2_detay = json.load(f).get("detay", {}).get("b2_pozitif", {})
        p2_skorlar = {d["sorgu"]: float(d["skor"])
                      for d in p2_detay.get("detay", []) if "skor" in d}
    except Exception as e:  # noqa: BLE001 — karşılaştırma rapor-düzeyi
        _stderr(f"UYARI: P2 b2_pozitif detay okunamadı "
                f"({type(e).__name__}: {e})")
    karsilastirma: List[Dict[str, Any]] = []
    for k in b1_kayitlar:
        if not k["retrieval_triggered"]:
            continue
        p2_skor = p2_skorlar.get(k["sorgu"])
        if p2_skor is None:
            continue
        karsilastirma.append({"sorgu": k["sorgu"],
                              "p3_skor": k["match_score"],
                              "p2_skor": round(p2_skor, 4),
                              "birebir": round(p2_skor, 4)
                              == k["match_score"]})
    detay["b2_rapor_karsilastirma"] = {
        "triggered": len(karsilastirma),
        "birebir": sum(1 for c in karsilastirma if c["birebir"]),
        "detay": karsilastirma,
    }

    # ---- B3 (HÜKÜM): OOV force'lu probe — Kapı-B sahte-geçiş + Kapı-C sahte-koşullanma
    oov_sorgu = "zzqwxx zqxwv zqqzzq"
    b3: Dict[str, Any] = {"sorgu": oov_sorgu, "force_rag": True,
                          "istisna": None, "kosullanma_teyit": None,
                          "skor_bant_ici": False}
    bant = False
    kosullanma = False
    try:
        sonuc_oov = agent_default.process_query(oov_sorgu, force_rag=True)
        skor_oov = float(sonuc_oov.get("match_score", 0.0))
        kosullanma = len(str(sonuc_oov.get("retrieved_document") or "")) > 0
        b3["istisna"] = False
        b3["retrieval_triggered"] = bool(sonuc_oov.get("retrieval_triggered"))
        b3["skor"] = round(skor_oov, 4)
        b3["kosullanma_teyit"] = kosullanma
        b3["morpheme_output_ornek"] = str(
            sonuc_oov.get("morpheme_output", ""))[:80]
        b3["entropy_post"] = round(float(sonuc_oov.get("entropy_post", 0.0)), 4)
        # İLAN-2 (T-0147 A-3): bant DEĞİL TAVAN — skor ≤ 0,075 (fail-closed
        # onarım: ham 0,7 × 0,05 = 0,035 → normalize 0,0467) + koşullanma FALSE.
        tavan = (skor_oov <= ILANLI["b3_oov_skor_tavani"])
        b3["skor_tavani_ici"] = tavan
        b3["bulgu"] = ("OOV_FAIL_CLOSED_ONARIM_KANITI — skor tavan-altı ve "
                       "koşullanma kapandı"
                       if (not kosullanma and tavan) else "BEKLENMEDIK_DAVRANIS")
    except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
        b3["istisna"] = True
        b3["istisna_metin"] = f"{type(e).__name__}: {e}"
        b3["bulgu"] = "ISTISNA — canlı-döngü P2 davranışını yinelemedi"
    detay["b3_oov"] = b3
    # İLAN-2 kapı: istisna-yok + skor ≤ 0,075 + koşullanma FALSE (onarım-kanıtı;
    # koşum-1'deki "kosullanma and bant" SAHTE-koşullanma beklentisi tersine döndü).
    kapilar["K4_B3_OOV_SAHTE_KOSULLANMA"] = bool(
        b3.get("istisna") is False
        and b3.get("skor_tavani_ici")
        and b3.get("kosullanma_teyit") is False)

    # ---- B4: Kapı-D ulaşılmazlık + fail-closed probe + gerçek-kanal-0
    b4: Dict[str, Any] = {}
    b4["is_high_similarity_sayisi"] = sum(1 for k in b1_kayitlar
                                          if k["is_high_similarity"])
    b4["is_high_similarity_beklenilen"] = \
        ILANLI["b4_is_high_beklenen_min"]

    probe_yol_tam = os.path.join(REPO_ROOT, arg.probe_yol)
    probe_once_satir = 0
    if os.path.exists(probe_yol_tam):
        with open(probe_yol_tam, encoding="utf-8") as f:
            probe_once_satir = sum(1 for s in f if s.strip())
    tetiklenme = 0
    probe_istisnalar: List[Dict[str, Any]] = []
    probe_agent = _agent_kur(model, tokenizer, memory, arg.probe_yol, 0.5)
    for j in secili:
        sorgu = tum_sorgular[j]
        try:
            sonuc_p = probe_agent.process_query(sorgu, force_rag=True)
            if sonuc_p.get("future_train_recorded"):
                tetiklenme += 1
        except Exception as e:  # noqa: BLE001 — envantere kaydedilir
            probe_istisnalar.append({"sorgu": sorgu[:60],
                                     "istisna": f"{type(e).__name__}: {e}"})
    b4["probe_agent_tetiklenme"] = tetiklenme
    b4["probe_agent_istisna"] = len(probe_istisnalar)
    b4["probe_istisnalar"] = probe_istisnalar

    # deterministik şema-kanıt: kanonik yazıcı-metodu (record_to_future_train)
    probe_record = {
        "instruction": "Belgeye göre cevapla.",
        "input": "PROBE_T0143_SEMA_KANITI",
        "output": "PROBE_HEDEF",
        "model_failed_output": "PROBE_MODEL_CIKTISI",
        "decompiled_output": "PROBE_DECOMPILED",
        "rag_document": "PROBE_RAG_BELGESI",
        "retrieval_collection": arg.koleksiyon,
        "similarity_score": 0.0,
        "similarity_threshold": 0.85,
        "entropy_pre": 0.0,
        "entropy_post": 0.0,
        "tau": 2.5,
        "reason": "t0143_sema_kaniti_probe",
        "timestamp": damga,
    }
    probe_agent.record_to_future_train(probe_record)
    satirlar_probe: List[str] = []
    try:
        with open(probe_yol_tam, encoding="utf-8") as f:
            satirlar_probe = [s for s in f.read().splitlines() if s.strip()]
    except Exception as e:  # noqa: BLE001
        b4["probe_okuma_hatasi"] = f"{type(e).__name__}: {e}"
    son_satir: Dict[str, Any] = {}
    if satirlar_probe:
        try:
            son_satir = json.loads(satirlar_probe[-1])
        except Exception as e:  # noqa: BLE001
            b4["probe_json_hatasi"] = f"{type(e).__name__}: {e}"
    sema_tam = bool(son_satir) and set(son_satir.keys()).issuperset(
        RECORD_ANAHTARLARI)
    b4["probe_once_satir"] = probe_once_satir
    b4["probe_sonra_satir"] = len(satirlar_probe)
    b4["probe_sema_anahtar_sayisi"] = len(son_satir.keys()) if son_satir else 0
    b4["probe_sema_tam"] = sema_tam

    kanal_sonra_bayt = os.path.getsize(GERCEK_KANAL) \
        if os.path.exists(GERCEK_KANAL) else -1
    b4["gercek_kanal_once_bayt"] = kanal_once_bayt
    b4["gercek_kanal_sonra_bayt"] = kanal_sonra_bayt
    detay["b4_kapi_d"] = b4
    kapilar["K5_B4_KAPI_D"] = bool(
        b4["is_high_similarity_sayisi"]
        >= ILANLI["b4_is_high_beklenen_min"]
        and b4["probe_sema_tam"]
        and kanal_once_bayt == ILANLI["gercek_kanal_once_bayt"]
        and kanal_sonra_bayt == ILANLI["gercek_kanal_sonra_bayt"])

    # ---- B5: dokunulmazlık (koşum SONU) — foreign ÖNCE==SONRA + kanonik 36
    env_sonra = _envanter(client)
    sonra_digest = _envanter_digest(env_sonra)
    nokta_sonra = {k["ad"]: k["nokta"] for k in env_sonra["kalemler"]}
    foreign_sonra = {ad: n for ad, n in nokta_sonra.items()
                     if ad not in KANONIK_ADLAR}
    dokunulmaz = (foreign_once == foreign_sonra
                  and nokta_sonra.get(arg.koleksiyon)
                  == nokta_once.get(arg.koleksiyon)
                  and nokta_sonra.get(arg.koleksiyon)
                  == ILANLI["kristal_bellek_nokta_once"])
    detay["dokunulmazlik"] = {
        "once_digest": once_digest, "sonra_digest": sonra_digest,
        "foreign_once": foreign_once, "foreign_sonra": foreign_sonra,
        "kanonik_once": nokta_once.get(arg.koleksiyon),
        "kanonik_sonra": nokta_sonra.get(arg.koleksiyon),
        "durum": "GEÇTİ" if dokunulmaz else "SAPMA",
    }
    detay["faz_b_envanter_sonra"] = env_sonra
    kapilar["K6_DOKUNULMAZLIK"] = dokunulmaz

    # ---- Hüküm (betikten) → hüküm JSON + rapor (elle sayı YOK)
    hukum_json, rc = _hukum_kur(kapilar, detay, damga)
    _hukum_yaz(hukum_json, arg)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
               base_sha, vocab_sha, once_digest, sonra_digest)
    _stderr(f"HÜKÜM: {hukum_json['hukum']} rc={rc} → "
            f"{arg.rapor.replace('_sonuc_', '_hukum_').replace('.md', '.json')}")
    sys.exit(rc)


def _hukum_kur(kapilar: Dict[str, bool], detay: Dict[str, Any],
               damga: str) -> Tuple[Dict[str, Any], int]:
    """Hüküm BETİK İÇİNDE — İLAN'lı kural: tüm kapılar GEÇTİ → P3_GECTİ rc=0;
    aksi DUR rc=2."""
    tamami = all(kapilar.values())
    sonuc = "P3_GECTİ" if tamami else "DUR"
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
    hukum_yol = arg.rapor.replace("_sonuc_", "_hukum_").replace(".md", ".json")
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    return hukum_yol


def _rapor_yaz(rapor_yol: str, hukum_json: Dict[str, Any], arg: Any,
               ilan_sha: str, korpus_sha: str, base_sha: str, vocab_sha: str,
               once_digest: Optional[str],
               sonra_digest: Optional[str]) -> None:
    """Rapor — İLAN'lı↔ölçülen tablolar; sayılar hüküm JSON'undan (elle YOK)."""
    d = hukum_json.get("detay", {})
    b1 = d.get("b1_merak", {})
    with open(rapor_yol, "w", encoding="utf-8") as f:
        f.write("# MİMARİ DOĞRULAMA PAKET-3 SONUÇ — Epistemik Döngü Kapıları "
                "(T-0143)\n\n")
        f.write(f"**Damga:** {_ts()} · **Hüküm (BETİKTEN):** "
                f"**{hukum_json['hukum']} (rc={hukum_json['rc']})**\n\n")
        f.write("| Kapı | Durum |\n|---|---|\n")
        for k, v in hukum_json.get("kapilar", {}).items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen; "
                "Assign|AnnAssign sayaç)\n\n")
        f.write("| Ölçüm | İLAN'lı | Ölçülen | Durum |\n|---|---|---|---|\n")
        for satir in d.get("faz_a_kapi_detay", []):
            f.write(f"| {satir['olcum']} | {satir['ilanli']} | "
                    f"{satir['olculen']} | {satir['durum']} |\n")
        c = d.get("faz_a", {})
        f.write(f"- eşik tanım sitesi: {c.get('esik_tanim_sitesi')} · "
                f"import sitesi: {c.get('esik_import_sitesi')}\n")
        f.write(f"- recreate-yıkıcılık satırları (vector_memory.py): "
                f"{c.get('recreate_satirlar_vector_memory')} · sessiz-None "
                f"(rag_pipeline.py): "
                f"{c.get('sessiz_none_satirlar_rag_pipeline')}\n")
        f.write(f"- record-şema başlangıç satırı (epistemic_agent.py): "
                f"{c.get('record_baslangic_satiri')}\n")
        f.write("\n## FAZ-B — epistemik döngü koşumu\n\n")
        f.write(f"- B1 Kapı-A: kayıt **{b1.get('kayit_tam')}/"
                f"{b1.get('n')}** · istisna {b1.get('istisna_sayisi', '-')} · "
                f"tetiklenme: {b1.get('tetiklenme_kirilim')} · entropy_pre "
                f"min/max: {b1.get('entropi_pre_min')}/"
                f"{b1.get('entropi_pre_max')}\n")
        b2 = d.get("b2_rapor_karsilastirma", {})
        f.write(f"- B2 (RAPOR): P2-skor karşılaştırma triggered alt-kümesi: "
                f"**{b2.get('birebir')}/{b2.get('triggered')}** birebir "
                f"(4 ondalık)\n")
        b3 = d.get("b3_oov", {})
        f.write(f"- B3 OOV force'lu: istisna={b3.get('istisna')} · skor="
                f"{b3.get('skor')} · koşullanma={b3.get('kosullanma_teyit')} · "
                f"band-içi={b3.get('skor_bant_ici')} · bulgu: "
                f"{b3.get('bulgu')}\n")
        b4 = d.get("b4_kapi_d", {})
        f.write(f"- B4 Kapı-D: varsayılan is_high_similarity "
                f"**{b4.get('is_high_similarity_sayisi')}/"
                f"{b1.get('kayit_tam', 0)}** (İLAN'lı 0) · probe-agent "
                f"tetiklenme {b4.get('probe_agent_tetiklenme')} (istisna "
                f"{b4.get('probe_agent_istisna')}) · probe satır "
                f"{b4.get('probe_once_satir')}→{b4.get('probe_sonra_satir')} · "
                f"şema {b4.get('probe_sema_anahtar_sayisi')} anahtar "
                f"(tam={b4.get('probe_sema_tam')}) · gerçek kanal "
                f"{b4.get('gercek_kanal_once_bayt')}→"
                f"{b4.get('gercek_kanal_sonra_bayt')} bayt\n")
        dok = d.get("dokunulmazlik", {})
        f.write(f"- B5 dokunulmazlık: {dok.get('durum')} · kanonik nokta "
                f"{dok.get('kanonik_once')}→{dok.get('kanonik_sonra')} · "
                f"digest {dok.get('once_digest')}→{dok.get('sonra_digest')}\n")
        f.write("\n## Digest tablosu\n\n| Dosya | sha256 |\n|---|---|\n")
        f.write(f"| {os.path.basename(arg.ilan)} (İLAN — koşum ÖNCESİ) | "
                f"`{ilan_sha}` |\n")
        f.write(f"| {arg.base_ckpt} (DONMUŞ taban) | `{base_sha}` |\n")
        f.write(f"| {arg.vocab} (DONMUŞ sözlük) | `{vocab_sha}` |\n")
        f.write(f"| {arg.korpus} (DONMUŞ kaynak) | `{korpus_sha}` |\n")
        kanal_sha = _sha256(GERCEK_KANAL) \
            if os.path.exists(GERCEK_KANAL) else "YOK"
        f.write(f"| data/future_train_vector.jsonl (GERÇEK kanal çıpası — "
                f"0 bayt) | `{kanal_sha}` |\n")
        if once_digest:
            f.write(f"| envanter ÖNCE/SONRA digest | `{once_digest}` / "
                    f"`{sonra_digest}` |\n")
        f.write(f"| dokunulmazlık | {dok.get('durum', '-')} |\n")


if __name__ == "__main__":
    main()