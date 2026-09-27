#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-0145 — MİMARİ DOĞRULAMA PAKET-5: Gateway→Öğrenme Kanalı

İLAN: data/eval/mimari_dogrulama_p5_ilan_2026-09-27.md (koşum-ÖNCESİ
damgali; REVİZYON-1: koşum-öncesi keşif-ölçümü — run_training cmd-flag
kümesi [--vocab/--load-path YOK], train.py default-vocab 32.852 +
jeton-guard :203, P3-probe/replay max-jeton-id < 32.852, gw_ask
telemetri 17). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK. rc ∈ {0, 2}.

Operatör kararları (27 Eyl 2026): (1) UÇTAN-UCA KANAL — yazar
(epistemic_agent → record_to_future_train) + tüketici (retrain_pipeline:
get_pending_count + compile_backlog_to_bin + run_training İLKELİ) +
gateway-yazım-yüzeyleri (inject_knowledge/check_memory probe-instance;
inject_reasoning_trace İLKELİ) + HTTP 5-endpoint. (2) PROBE-YOLLAR —
future_train + output_bin + archive + save_path hepsi probe; GERÇEK
kanal data/future_train_vector.jsonl 0-BAYT çıpası koşum-sonunda kapı
ile teyit edilir. kristal_bellek (36) YALNIZ OKUNUR; 9 foreign
DOKUNULMAZ; muhakeme_bellek/simulasyon_bellek KURULMAZ.

Kanonik kod IMPORT edilir (kopya YASAK); model saf CPU'da; MPS YOK;
çift-eğitici YOK; seed=42. Statik hizalama-ön-kontrolü koşumdan AYRI
adımdır (--statik-ön; py_compile ayrı komut — T-0143/T-0144 dersleri).
"""

import argparse
import ast
import builtins
import hashlib
import inspect
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient  # noqa: E402

from src.rag.vector_memory import (  # noqa: E402
    VectorMemory,
    generate_kristal_vector,
    generate_sparse_vector,
)
from src.rag.epistemic_agent import EpistemicCuriosityAgent  # noqa: E402
from src.gateway.agent_gateway import AgentGateway  # noqa: E402
from src.gateway.retrain_pipeline import RetrainPipeline  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.llm.prompt_contract import render_example  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from scripts.dogrulama_p2_rag_gezgini import (  # noqa: E402
    _envanter,
    _envanter_digest,
)
from scripts.dogrulama_p3_epistemik_kapilar import (  # noqa: E402
    _sorgular,
    _model_yukle,
    _agent_kur,
)

# ---- İLAN'lı sabitler (koşum ÖNCESİ sabit; koşum-sonrası yumuşatılmaz) ----
POZITIF_N = 20
SEED = 42
KANONIK_ADLAR = ("kristal_bellek", "simulasyon_bellek")
KANONIK_KOLEKSIYON = "kristal_bellek"
PROBE_KOLEKSIYON = "p5_probe_bellek"
SAHTE_AD = "olmayan_ad_p5_x"
PROBE_FUTURE_TRAIN = "data/eval/p5_probe_future_train.jsonl"
PROBE_ARCHIVE = "data/eval/p5_probe_archive.jsonl"
PROBE_BIN = "data/eval/p5_probe_future.bin"
PROBE_SAVE = os.path.join(tempfile.gettempdir(), "data", "eval",
                          "p5_probe_anka.pt")
DEFAULT_BIN = "data/train_future_finetune.bin"
GERCEK_KANAL = "data/future_train_vector.jsonl"
GERCEK_KANAL_TAM = os.path.join(REPO_ROOT, "data",
                               "future_train_vector.jsonl")
GERCEK_KANAL_SHA = ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b9"
                    "34ca495991b7852b855")
ENVANTER_CIPA = "fe7fc7649737ed35"
REPLAY_BUFFER = os.path.join(REPO_ROOT, "data", "pedagogy",
                             "high_school_foundation_dataset.jsonl")
REPLAY_SHA = ("59e2f2786d0ec0d1574a77e5b7354e02ae957345a"
              "1b4ddce945d0db19527bdf1")
TRAIN_PY = os.path.join(REPO_ROOT, "train.py")

PROBE_DOC_1 = ("P5_PROBE_DOC: Anka mimari dogrulama T-0145 kanal yuzeyi "
               "belgesi. Kristal tokenizer ile indekslenir.")
PROBE_DOC_3 = ("P5_PROBE_DOC_HTTP: gateway yazim teyidi T-0145. "
               "Kanonik inject yolu probe koleksiyonuna yazar.")
PROBE_QUERY = "Anka T-0145 kanal yuzeyi belgesi nedir?"

EXPECTED_KAPILAR = frozenset({
    "K1_FAZ_A_ENVANTER",
    "K2_ARZ_CIPA",
    "K3_KANAL_YAZARI",
    "K4_TUKETICI_OKUMA",
    "K5_TUKETICI_DERLEME",
    "K6_RUN_TRAINING_ILKELI",
    "K7_GATEWAY_YAZIM",
    "K8_HTTP_TEYIDI",
    "K9_DOKUNULMAZLIK",
})

# FAZ-A İLAN'lı değerler (İLAN §2 tablosu; koşum ÖNCESİ sabit)
ILANLI = {
    "epi_kanal_kapisi_similarity_threshold": 0.85,
    "epi_record_yazim_sitesi": 1,
    "epi_record_sema_anahtar_sayisi": 14,
    "epi_future_train_path_paramli": True,
    "epi_os_makedirs_satiri": 180,
    "gw_ask_telemetri_anahtar_sayisi": 17,
    "gw_inject_default_koleksiyon": "kristal_bellek",
    "gw_target_collection_sahete_secici": True,
    "gw_inject_reasoning_koleksiyon": "muhakeme_bellek",
    "gw_http_endpoint_kumesi": ["/api/backlog", "/api/check",
                                "/api/inject", "/api/query",
                                "/api/status"],
    "gw_localhost_kurulum": 2,
    "retrain_zorunlu_fail_closed": 3,
    "retrain_cmd_flag_kumesi": ["--batch-size", "--data", "--device",
                                "--lr", "--save-path", "--steps"],
    "retrain_cmd_vocab_yok": True,
    "retrain_cmd_load_path_yok": True,
    "retrain_archive_success_gate": True,
    "retrain_archive_sifirlama_modu": "w",
    "retrain_replay_buffer_yolu":
        "data/pedagogy/high_school_foundation_dataset.jsonl",
    "retrain_replay_samples_default": 25,
    "retrain_output_bin_default": "data/train_future_finetune.bin",
    "retrain_oversample_factor_default": 20,
    "train_py_default_vocab": "data/rebuild/vocab_base_32852.json",
    "train_py_jeton_guard": True,
    "train_py_vocab_flag_destekli": True,
    "supervisor_retrain_threshold": 5,
    "merak_router_state_dict_yukleme": 0,
    "gercek_kanal_once_bayt": 0,
    "envanter_digest_once": ENVANTER_CIPA,
}

RECORD_ANAHTARLARI = frozenset({
    "instruction", "input", "output", "model_failed_output",
    "decompiled_output", "rag_document", "retrieval_collection",
    "similarity_score", "similarity_threshold", "entropy_pre",
    "entropy_post", "tau", "reason", "timestamp",
})

TELEMETRI_ASK = frozenset({
    "query", "instruction", "mode", "response_text", "morphemes",
    "entropy_pre", "entropy_post", "needs_retrieval", "rag_document",
    "source_collection", "rag_score", "is_high_similarity",
    "router_experts", "epistemic_failure", "future_train_recorded",
    "future_train_path", "timestamp",
})

CIPALAR = (
    "data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json",
    "data/eval/mimari_dogrulama_p3_hukum_2026-09-27.json",
    "data/eval/mimari_dogrulama_p4_hukum_2026-09-27.json",
)


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def _dosya_sha(yol: str) -> Optional[str]:
    if not os.path.exists(yol):
        return None
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stderr(msg: str) -> None:
    print("[T-0145] " + msg, file=sys.stderr, flush=True)


def _satirlar(metin: str, desen: str) -> List[int]:
    sonuc: List[int] = []
    for i, satir in enumerate(metin.splitlines(), 1):
        if re.search(desen, satir):
            sonuc.append(i)
    return sonuc


def _kapilar_anahtarlari(metin: str) -> List[str]:
    """AST: kapilar["K..."] atamaları + _dur_ve_cik dict-literal
    anahtarları (statik hizalama-ön-kontrolü — koşumdan AYRI)."""
    bulunan: List[str] = []
    agac = ast.parse(metin)
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Assign):
            hedef = dugum.targets[0]
            if isinstance(hedef, ast.Subscript):
                kapici = hedef.value
                anahtar = hedef.slice
                if isinstance(kapici, ast.Name) \
                        and kapici.id == "kapilar" \
                        and isinstance(anahtar, ast.Constant):
                    k = anahtar.value
                    if isinstance(k, str) and k.startswith("K"):
                        bulunan.append(k)
        if isinstance(dugum, ast.Call):
            if isinstance(dugum.func, ast.Name) \
                    and dugum.func.id == "_dur_ve_cik":
                for arguman in dugum.args:
                    if isinstance(arguman, ast.Dict):
                        for k_dugum in arguman.keys:
                            if isinstance(k_dugum, ast.Constant):
                                k = k_dugum.value
                                if isinstance(k, str) and k.startswith("K"):
                                    bulunan.append(k)
    return sorted(set(bulunan))


def _tanimsiz_sabitler(metin: str) -> List[str]:
    """Tanımsız-ad taraması (py_compile NameError'ı YAKALAMAZ —
    T-0144 dersi; koşum-1 ek dersi: yalnız ALL-CAPS değil,
    KÜÇÜK-AD modül-alias'ları da (np) kaçırır → genel tarama)."""
    tanimli = set(dir(builtins))
    agac = ast.parse(metin)
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Assign):
            for hedef in dugum.targets:
                if isinstance(hedef, ast.Name) and hedef.id.isupper():
                    tanimli.add(hedef.id)
        if isinstance(dugum, ast.AnnAssign) \
                and isinstance(dugum.target, ast.Name) \
                and dugum.target.id.isupper():
            tanimli.add(dugum.target.id)
        if isinstance(dugum, ast.ImportFrom):
            for a in dugum.names:
                if a.asname:
                    tanimli.add(a.asname)
                else:
                    tanimli.add(a.name)
        if isinstance(dugum, ast.Import):
            for a in dugum.names:
                tanimli.add((a.asname or a.name).split(".")[0])
        # her Store-bağlamı ad (atama, for-hedefi, with-as, walrus)
        if isinstance(dugum, ast.Name) \
                and isinstance(dugum.ctx, ast.Store):
            tanimli.add(dugum.id)
        if isinstance(dugum, ast.arg):
            tanimli.add(dugum.arg)
        if isinstance(dugum, ast.ExceptHandler) and dugum.name:
            tanimli.add(dugum.name)
        if isinstance(dugum, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
            tanimli.add(dugum.name)
    tanimsiz: List[str] = []
    duzeltme_ekleri = {"__file__", "__name__", "__doc__"}
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Name) and isinstance(dugum.ctx, ast.Load):
            ad = dugum.id
            if ad not in tanimli and ad not in duzeltme_ekleri:
                tanimsiz.append(ad)
    return sorted(set(tanimsiz))


def _statik_on_kontrol(betik_yol: str) -> Tuple[bool, Dict[str, Any]]:
    detay: Dict[str, Any] = {}
    metin = open(betik_yol, encoding="utf-8").read()
    anahtarlar = _kapilar_anahtarlari(metin)
    detay["kapilar_atanan"] = anahtarlar
    hizali = set(anahtarlar) == set(EXPECTED_KAPILAR)
    detay["kapi_hizasi"] = hizali
    tanimsiz = _tanimsiz_sabitler(metin)
    detay["tanimsiz_sabitler"] = tanimsiz
    detay["tanimsiz_sabit_temiz"] = len(tanimsiz) == 0
    # modül-seviyesi import'lar bu noktaya gelindiğinde zaten koştu —
    # ImportError olsaydı betik argparse'a varmadan traceback'le ölürdü
    detay["import_teyit"] = True
    tamam = hizali and detay["tanimsiz_sabit_temiz"]
    return tamam, detay


def _ask_anahtar_sayisi(gw_kaynak: str) -> int:
    """ask() dönüş-dict'inin anahtar sayısı (AST)."""
    agac = ast.parse(gw_kaynak)
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.FunctionDef) and dugum.name == "ask":
            for alt in ast.walk(dugum):
                if isinstance(alt, ast.Return) \
                        and isinstance(alt.value, ast.Dict):
                    return len(alt.value.keys)
    return -1


def _record_sema_sayisi(ea_kaynak: str) -> int:
    """record = {...} dict'inin anahtar sayısı (AST; 14-anahtar şeması)."""
    agac = ast.parse(ea_kaynak)
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Dict):
            anahtarlar: List[str] = []
            for k in dugum.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    anahtarlar.append(k.value)
            if "model_failed_output" in anahtarlar \
                    and "decompiled_output" in anahtarlar \
                    and "rag_document" in anahtarlar:
                return len(dugum.keys)
    return -1


def _cmd_flag_kumesi(rp_kaynak: str) -> List[str]:
    """run_training cmd-listesi flag'leri (AST — İLAN'lı sıralı küme)."""
    agac = ast.parse(rp_kaynak)
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.FunctionDef) \
                and dugum.name == "run_training":
            for alt in ast.walk(dugum):
                if isinstance(alt, ast.List) and alt.elts:
                    degerler: List[str] = []
                    for e in alt.elts:
                        if isinstance(e, ast.Constant) \
                                and isinstance(e.value, str):
                            degerler.append(e.value)
                    if "--data" in degerler:
                        bayraklar = [d for d in degerler
                                     if d.startswith("--")]
                        return sorted(bayraklar)
    return []


def _arsiv_gate(rp_kaynak: str) -> Tuple[Optional[int], Optional[int]]:
    """run_training içinde returncode-kapısı satırı + archive çağrı
    satırı (success-gate: çağrı satırı > rc-satırı)."""
    agac = ast.parse(rp_kaynak)
    rc_satiri: Optional[int] = None
    cagri_satiri: Optional[int] = None
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.FunctionDef) \
                and dugum.name == "run_training":
            for alt in ast.walk(dugum):
                if rc_satiri is None and isinstance(alt, ast.If):
                    if "returncode" in ast.dump(alt.test):
                        rc_satiri = alt.lineno
                if isinstance(alt, ast.Expr) \
                        and isinstance(alt.value, ast.Call):
                    cagri = alt.value.func
                    if isinstance(cagri, ast.Attribute) \
                            and cagri.attr == "_archive_processed_records":
                        cagri_satiri = alt.lineno
    return rc_satiri, cagri_satiri


def _faz_a_envanteri() -> Dict[str, Any]:
    """FAZ-A: kod-envanteri (koşumsuz) — İLAN'lı satırlar (grep/AST/
    inspect). Salt-okunur okuma; kanonik kod DOKUNULMAZ."""
    b: Dict[str, Any] = {}
    ea_yol = os.path.join(REPO_ROOT, "src", "rag", "epistemic_agent.py")
    gw_yol = os.path.join(REPO_ROOT, "src", "gateway",
                          "agent_gateway.py")
    rp_yol = os.path.join(REPO_ROOT, "src", "gateway",
                          "retrain_pipeline.py")
    ps_yol = os.path.join(REPO_ROOT, "src", "gateway",
                          "pedagogical_supervisor.py")
    ea = open(ea_yol, encoding="utf-8").read()
    gw = open(gw_yol, encoding="utf-8").read()
    rp = open(rp_yol, encoding="utf-8").read()
    ps = open(ps_yol, encoding="utf-8").read()
    tr = open(TRAIN_PY, encoding="utf-8").read()

    # --- epistemic_agent (yazar) ---
    eparam = inspect.signature(EpistemicCuriosityAgent.__init__).parameters
    b["epi_kanal_kapisi_similarity_threshold"] = float(
        eparam["similarity_threshold"].default)
    b["epi_record_yazim_sitesi"] = ea.count("self.record_to_future_train(")
    b["epi_record_sema_anahtar_sayisi"] = _record_sema_sayisi(ea)
    b["epi_future_train_path_paramli"] = "future_train_path" in eparam
    makedirs_satirlar = _satirlar(ea, r"os\.makedirs\(")
    b["epi_makedirs_satirlar"] = makedirs_satirlar
    b["epi_os_makedirs_satiri"] = makedirs_satirlar[0] \
        if makedirs_satirlar else -1
    b["epi_open_a_satirlar"] = _satirlar(
        ea, r'open\(self\.future_train_path, "a"')

    # --- agent_gateway (gövde) ---
    b["gw_ask_telemetri_anahtar_sayisi"] = _ask_anahtar_sayisi(gw)
    m = re.search(r'def inject_knowledge\([^)]*'
                  r'target_collection: str = "([^"]+)"', gw, re.DOTALL)
    b["gw_inject_default_koleksiyon"] = m.group(1) if m else ""
    b["gw_target_collection_sahete_secici"] = (
        gw.count('self.memory if target_collection == "kristal_bellek" '
                 'else') == 2)
    b["gw_inject_reasoning_paylasim_client"] = (
        "client = self.memory.client" in gw)
    m_f = re.search(r"def inject_reasoning_trace\(.*?(?=\n    def )",
                    gw, re.DOTALL)
    govde_i = m_f.group(0) if m_f else ""
    m3 = re.search(r'collection_name="([a-z_]+)"', govde_i)
    b["gw_inject_reasoning_koleksiyon"] = m3.group(1) if m3 else ""
    endpointler = re.findall(r'"(/api/[a-z]+)"', gw)
    b["gw_http_endpoint_kumesi"] = sorted(set(endpointler))
    b["gw_localhost_kurulum"] = gw.count('host="localhost"')

    # --- retrain_pipeline (tüketici) ---
    b["retrain_zorunlu_fail_closed"] = rp.count("self._zorunlu(")
    b["retrain_cmd_flag_kumesi"] = _cmd_flag_kumesi(rp)
    b["retrain_cmd_vocab_yok"] = ("--vocab"
                                  not in b["retrain_cmd_flag_kumesi"])
    b["retrain_cmd_load_path_yok"] = ("--load-path"
                                      not in b["retrain_cmd_flag_kumesi"])
    rc_satir, cagri_satir = _arsiv_gate(rp)
    b["retrain_archive_rc_satiri"] = rc_satir
    b["retrain_archive_cagri_satiri"] = cagri_satir
    b["retrain_archive_success_gate"] = bool(
        rc_satir is not None and cagri_satir is not None
        and cagri_satir > rc_satir)
    open_w = _satirlar(rp, r'open\(self\.future_train_path, "w"')
    b["retrain_archive_w_satirlar"] = open_w
    b["retrain_archive_sifirlama_modu"] = "w" if open_w else ""
    replay_satirlar = _satirlar(rp,
                                r"high_school_foundation_dataset\.jsonl")
    b["retrain_replay_satirlar"] = replay_satirlar
    b["retrain_replay_buffer_yolu"] = (
        "data/pedagogy/high_school_foundation_dataset.jsonl"
        if replay_satirlar else "")
    cparam = inspect.signature(
        RetrainPipeline.compile_backlog_to_bin).parameters
    b["retrain_replay_samples_default"] = int(
        cparam["replay_samples"].default)
    b["retrain_oversample_factor_default"] = int(
        cparam["oversample_factor"].default)
    iparam = inspect.signature(RetrainPipeline.__init__).parameters
    b["retrain_output_bin_default"] = str(iparam["output_bin_path"].default)

    # --- train.py (statik, salt-okunur) ---
    m2 = re.search(r"vocab_path = '(data/rebuild/[^']+)'", tr)
    b["train_py_default_vocab"] = m2.group(1) if m2 else ""
    guard_satirlar = _satirlar(tr, r"_jeton_max >= vocab_size")
    b["train_py_jeton_guard_satirlar"] = guard_satirlar
    guard_tam = False
    if guard_satirlar:
        satir_listesi = tr.splitlines()
        baslangic = guard_satirlar[0] - 1
        pencere = satir_listesi[baslangic:baslangic + 5]
        guard_tam = any("RuntimeError" in s for s in pencere)
    b["train_py_jeton_guard"] = guard_tam
    b["train_py_vocab_flag_destekli"] = 'arg == "--vocab"' in tr

    # --- pedagogical_supervisor ---
    esik_satirlar = _satirlar(ps, r"retrain_threshold")
    b["supervisor_retrain_satirlar"] = esik_satirlar
    m4 = re.search(r"retrain_threshold(?::\s*int)?\s*=\s*(\d+)", ps)
    b["supervisor_retrain_threshold"] = int(m4.group(1)) if m4 else -1

    # --- merak/router eğitimsiz-projeksiyon (P3 çıpası) ---
    egitimli = 0
    for alt_klasor, ad in (("rag", "merak.py"), ("llm", "router.py")):
        yol = os.path.join(REPO_ROOT, "src", alt_klasor, ad)
        metin_m = open(yol, encoding="utf-8").read()
        egitimli = egitimli + metin_m.count("state_dict")
    b["merak_router_state_dict_yukleme"] = egitimli
    return b


def _faz_a_kapilari(
    faz_a: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Kapı K1: FAZ-A İLAN'lı ↔ ölçülen (birebir, 28 satır)."""
    tablo: List[Dict[str, Any]] = []
    tamam = True
    for anahtar in ILANLI:
        ilanli_deger = ILANLI[anahtar]
        olculen = faz_a.get(anahtar)
        if isinstance(ilanli_deger, float) \
                and isinstance(olculen, float):
            esit = abs(olculen - ilanli_deger) < 1e-9
        else:
            esit = olculen == ilanli_deger
        tamam = tamam and esit
        durum = "GEÇTİ" if esit else "SAPMA"
        tablo.append({"olcum": anahtar, "ilanli": ilanli_deger,
                      "olculen": olculen, "durum": durum})
    return tamam, tablo


def _hukum_kur(kapilar: Dict[str, bool], detay: Dict[str, Any],
               damga: str) -> Tuple[Dict[str, Any], int]:
    """Hüküm BETİK İÇİNDE — İLAN'lı kural: 9 kapının TAMAMI GEÇTİ →
    P5_GECTİ rc=0; aksi her dal → DUR rc=2. Ölçülmemiş kapı = DÜŞTÜ
    sayılmaz, ÖLÇÜLMEDİ işaretlenir (hüküm yine DUR — fail-closed)."""
    tamami = True
    for kapı_adi in EXPECTED_KAPILAR:
        if not kapilar.get(kapı_adi, False):
            tamami = False
    sonuc_hukum = "P5_GECTİ" if tamami else "DUR"
    rc = 0 if tamami else 2
    kapilar_gorunum: Dict[str, str] = {}
    for kapı_adi in sorted(EXPECTED_KAPILAR):
        if kapı_adi in kapilar:
            kapilar_gorunum[kapı_adi] = "GEÇTİ" \
                if kapilar[kapı_adi] else "DÜŞTÜ"
        else:
            kapilar_gorunum[kapı_adi] = "ÖLÇÜLMEDİ"
    hukum_json: Dict[str, Any] = {
        "hukum": sonuc_hukum,
        "rc": rc,
        "damga": damga,
        "kapilar": kapilar_gorunum,
        "detay": detay,
    }
    return hukum_json, rc


def _hukum_yaz(hukum_json: Dict[str, Any], arg: Any) -> str:
    hukum_yol = arg.hukum_json
    if not hukum_yol:
        hukum_yol = arg.rapor.replace("_sonuc_", "_hukum_")
        hukum_yol = hukum_yol.replace(".md", ".json")
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2,
                  sort_keys=True)
    return hukum_yol


def _dur_ve_cik(kapilar: Dict[str, bool], detay: Dict[str, Any],
                damga: str, arg: Any, ilan_sha: str,
                sebepler: List[str]) -> None:
    """Ara-DUR yolu (fail-closed): hüküm + rapor yazılır, rc=2 çıkar."""
    detay["dur_sebepleri"] = sebepler
    hukum_json, rc = _hukum_kur(kapilar, detay, damga)
    hukum_yol = _hukum_yaz(hukum_json, arg)
    hukum_sha = _sha256(hukum_yol)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, hukum_yol,
               hukum_sha)
    _stderr("HÜKÜM: " + hukum_json["hukum"] + " rc=" + str(rc)
            + " → " + hukum_yol)
    sys.exit(rc)


def _rapor_yaz(rapor_yol: str, hukum_json: Dict[str, Any], arg: Any,
               ilan_sha: str, hukum_yol: str,
               hukum_sha: str = "") -> None:
    """Rapor — İLAN'lı↔ölçülen tablolar; sayılar hüküm JSON'undan."""
    d = hukum_json.get("detay", {})
    b1 = d.get("b1_kanal_yazari", {})
    b3 = d.get("b3_tuketici_okuma", {})
    b4 = d.get("b4_tuketici_derleme", {})
    b5 = d.get("b5_run_training", {})
    b6 = d.get("b6_gateway_yazim", {})
    b7 = d.get("b7_http", {})
    b8 = d.get("b8_dokunulmazlik", {})
    meta_b4 = b4.get("meta", {})
    satirlar: List[str] = []
    satirlar.append("# MİMARİ DOĞRULAMA PAKET-5 SONUÇ — "
                    "Gateway→Öğrenme Kanalı (T-0145)")
    satirlar.append("")
    satirlar.append("**Damga:** " + _ts()
                    + " · **Hüküm (BETİKTEN):** **"
                    + hukum_json["hukum"] + " (rc="
                    + str(hukum_json["rc"]) + ")**")
    satirlar.append("")
    satirlar.append("| Kapı | Durum |")
    satirlar.append("|---|---|")
    for k in sorted(hukum_json["kapilar"].keys()):
        satirlar.append("| " + k + " | "
                        + hukum_json["kapilar"][k] + " |")
    satirlar.append("")
    satirlar.append("## FAZ-A — kod-envanteri (İLAN'lı ↔ ölçülen)")
    satirlar.append("")
    satirlar.append("| Ölçüm | İLAN'lı | Ölçülen | Durum |")
    satirlar.append("|---|---|---|---|")
    for satir in d.get("faz_a_tablo", []):
        satirlar.append("| " + str(satir["olcum"]) + " | "
                        + str(satir["ilanli"]) + " | "
                        + str(satir["olculen"]) + " | "
                        + str(satir["durum"]) + " |")
    satirlar.append("")
    satirlar.append("## FAZ-B — kanal koşumu")
    satirlar.append("")
    b1_s = ("- B1 canlı-döngü: ask **" + str(b1.get("ask_kayit")) + "/"
            + str(b1.get("ask_n")) + "** · istisna "
            + str(b1.get("ask_istisna")) + " · is_high_similarity **"
            + str(b1.get("is_high_similarity_sayisi")) + "/"
            + str(b1.get("ask_kayit")) + "** (İLAN'lı 0) · ask-17-"
            + "anahtar=" + str(b1.get("ask_kume_tam")))
    satirlar.append(b1_s)
    b1_p = ("- B1 probe-yazım: tetiklenme **" + str(b1.get("recorded_n"))
            + "** (istisna " + str(b1.get("probe_istisna")) + ") · satır "
            + str(b1.get("probe_once_satir")) + "→"
            + str(b1.get("son_satir_sayisi")) + " · şema-tam "
            + str(b1.get("sema_tam")) + "/"
            + str(b1.get("son_satir_sayisi")) + " · append-teyit="
            + str(b1.get("append_teyit")))
    satirlar.append(b1_p)
    b3_s = ("- B3 tüketici-okuma: probe **" + str(b3.get("probe_sayisi"))
            + "** (beklenen " + str(b3.get("probe_beklenen"))
            + ") · gerçek-yol **" + str(b3.get("gercek_sayisi")) + "**")
    satirlar.append(b3_s)
    b4_s = ("- B4 tüketici-derleme: .bin boyut " + str(b4.get("bin_boyut"))
            + " (uint16=" + str(b4.get("uint16")) + ") · meta backlog "
            + str(meta_b4.get("backlog_samples")) + " · total "
            + str(meta_b4.get("total_samples")) + " · oversample "
            + str(meta_b4.get("oversample_factor")) + " · toplam jeton "
            + str(b4.get("toplam_jeton")) + " · **max-jeton-id "
            + str(b4.get("max_jeton_id")) + "** (< 32.852: "
            + str(b4.get("max_jeton_alti")) + ") · geri-tokenize="
            + str(b4.get("geri_tokenize")))
    satirlar.append(b4_s)
    b5_s = ("- B5 run_training İLKELİ: status **"
            + str(b5.get("run_training_status")) + "** · samples "
            + str(b5.get("samples_trained")) + " · süre "
            + str(b5.get("duration_sec")) + " sn · arşiv "
            + str(b5.get("arsiv_once_satir")) + "→"
            + str(b5.get("arsiv_sonra_satir")) + " satır (taşıma-teyit="
            + str(b5.get("arsiv_tasima_teyit")) + ") · probe-future "
            + str(b5.get("probe_sonra_bayt")) + " bayt · save .pt="
            + str(b5.get("save_var")) + " · default-bin "
            + ("DOKUNULMADI (mtime/digest-çıpa öncesi==sonrası; "
               "önceden-var=" + str(b5.get("default_bin_sonra"))
               + " — koşum yazmadı)"
               if b5.get("default_bin_mtime_once")
               == b5.get("default_bin_mtime_sonra")
               and b5.get("default_bin_digest_once")
               == b5.get("default_bin_digest_sonra")
               else "DEĞİŞTİ"))
    satirlar.append(b5_s)
    b6_s = ("- B6 gateway-yazım (probe): inject "
            + str(b6.get("inject1_collection")) + " count "
            + str(b6.get("inject1_count")) + " · geri-okuma skor "
            + str(b6.get("geri_okuma_skor")) + " · **sahte-seçici: "
            "target '" + str(b6.get("sahte_ad")) + "' → probe-count "
            + str(b6.get("inject2_count_probe")) + " · sahte-ad "
            "sunucuda YOK=" + str(b6.get("sahte_ad_sunucuda_yok"))
            + "** · check top-1 skor " + str(b6.get("check_top1_skor")))
    satirlar.append(b6_s)
    b7_s = ("- B7 HTTP: port " + str(b7.get("port")) + " · 5-endpoint "
            "kodlar [" + str(b7.get("status_kod")) + ", "
            + str(b7.get("backlog_kod")) + ", " + str(b7.get("check_kod"))
            + ", " + str(b7.get("inject_kod")) + ", "
            + str(b7.get("query_kod")) + "] · backlog-samples "
            + str(b7.get("status_backlog_samples")) + " · "
            "query-telemetri " + str(b7.get("query_anahtar_sayisi"))
            + "-anahtar")
    satirlar.append(b7_s)
    b8_s = ("- B8 dokunulmazlık: " + str(b8.get("durum")) + " · kristal "
            + str(b8.get("kristal_once")) + "→"
            + str(b8.get("kristal_sonra")) + " · digest "
            + str(b8.get("digest_once")) + "→"
            + str(b8.get("digest_sonra")) + " · kanal "
            + str(b8.get("kanal_once_bayt")) + "→"
            + str(b8.get("kanal_sonra_bayt")) + " bayt")
    satirlar.append(b8_s)
    satirlar.append("")
    satirlar.append("## Digest tablosu")
    satirlar.append("")
    satirlar.append("| Dosya | sha256 |")
    satirlar.append("|---|---|")
    for satir in d.get("digest_tablo", []):
        satirlar.append(satir)
    if hukum_sha:
        satirlar.append("| " + os.path.basename(hukum_yol)
                        + " (HÜKÜM JSON — BETİKTEN) | `" + hukum_sha
                        + "` |")
    satirlar.append("")
    with open(rapor_yol, "w", encoding="utf-8") as f:
        f.write("\n".join(satirlar) + "\n")


def main() -> None:
    ayrici = argparse.ArgumentParser(
        description="T-0145 — P5 Gateway→Öğrenme Kanalı doğrulama "
                    "(İLAN'lı, fail-closed)")
    ayrici.add_argument("--statik-ön", action="store_true",
                        dest="statik_on",
                        help="koşumdan AYRI statik ön-kontrol modu")
    ayrici.add_argument("--host", default="192.168.1.5")
    ayrici.add_argument("--port", type=int, default=6333)
    ayrici.add_argument(
        "--korpus",
        default="data/realistic_rag/test_natural_150.jsonl")
    ayrici.add_argument("--vocab", required=True,
                        help="Sözlük AÇIKÇA verilir (T-0087 dersi)")
    ayrici.add_argument("--base-ckpt", default="data/anka_base_v2.pt")
    ayrici.add_argument("--ilan", required=True)
    ayrici.add_argument("--rapor", required=True)
    ayrici.add_argument("--hukum-json", default="")
    arg = ayrici.parse_args()

    betik_yol = os.path.abspath(__file__)

    if arg.statik_on:
        tamam, detay_s = _statik_on_kontrol(betik_yol)
        _stderr("STATIK-ÖN: " + ("GECTI" if tamam else "DUR")
                + " (kapı-hizası=" + str(detay_s["kapi_hizasi"])
                + ", tanımsız-sabit="
                + str(detay_s["tanimsiz_sabit_temiz"]) + ", import="
                + str(detay_s["import_teyit"]) + ")")
        sys.exit(0 if tamam else 2)

    os.chdir(REPO_ROOT)
    damga = _ts()
    ilan_sha = _sha256(arg.ilan)
    korpus_sha = _sha256(arg.korpus)
    vocab_sha = _sha256(arg.vocab)
    base_sha = _sha256(arg.base_ckpt)
    betik_sha = _sha256(betik_yol)
    kapilar: Dict[str, bool] = {}
    detay: Dict[str, Any] = {
        "damga": damga,
        "ilan": os.path.basename(arg.ilan),
        "ilan_sha256": ilan_sha,
        "betik": os.path.basename(betik_yol),
        "betik_sha256": betik_sha,
        "korpus": arg.korpus, "korpus_sha256": korpus_sha,
        "vocab": arg.vocab, "vocab_sha256": vocab_sha,
        "base_ckpt": arg.base_ckpt, "base_ckpt_sha256": base_sha,
        "probe_yollar": {
            "future_train": PROBE_FUTURE_TRAIN,
            "archive": PROBE_ARCHIVE,
            "bin": PROBE_BIN,
            "save": PROBE_SAVE,
            "koleksiyon": PROBE_KOLEKSIYON,
        },
        "kosum_sandbox_disi": True, "device": "cpu", "seed": SEED,
    }

    # ---- koşum-ÖNCESİ: GERÇEK kanal çıpası 0 bayt ----
    if os.path.exists(GERCEK_KANAL_TAM):
        kanal_once_bayt = os.path.getsize(GERCEK_KANAL_TAM)
    else:
        kanal_once_bayt = -1
    detay["gercek_kanal_once_bayt"] = kanal_once_bayt
    detay["gercek_kanal_once_sha256"] = _dosya_sha(GERCEK_KANAL_TAM)
    if kanal_once_bayt != 0:
        _stderr("DUR: GERÇEK kanal çıpası BOZULMUŞ ("
                + str(kanal_once_bayt) + " bayt != 0 — koşum ÖNCESİ ihlal)")
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["kanal-cipa-bayt-sapmasi"])
        return

    # ---- FAZ-A: kod-envanteri (koşumsuz) ----
    faz_a = _faz_a_envanteri()
    detay["faz_a"] = faz_a

    # ---- sunucu (fail-closed; sessiz-fallback YOK) ----
    try:
        client = QdrantClient(host=arg.host, port=arg.port, timeout=5.0,
                              check_compatibility=False)
        client.get_collections()
    except Exception as e:  # noqa: BLE001
        _stderr("DUR: sunucu erişilemez (" + type(e).__name__ + ": "
                + str(e) + "); sessiz-fallback YAPILMAZ")
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["sunucu-erisilemez: " + type(e).__name__ + ": "
                     + str(e)])
        return

    env_once = _envanter(client)
    digest_once = _envanter_digest(env_once)
    detay["faz_b_envanter_once"] = env_once
    detay["envanter_once_digest"] = digest_once
    nokta_once_map: Dict[str, Any] = {}
    for kalem in env_once["kalemler"]:
        nokta_once_map[kalem["ad"]] = kalem["nokta"]

    # ---- K1: FAZ-A İLAN'lı birebir (kod-satırları + kanal + digest) ----
    faz_a["gercek_kanal_once_bayt"] = kanal_once_bayt
    faz_a["envanter_digest_once"] = digest_once
    k1_tam, k1_tablo = _faz_a_kapilari(faz_a)
    detay["faz_a_tablo"] = k1_tablo
    kapilar["K1_FAZ_A_ENVANTER"] = k1_tam

    # ---- K2: arz-çıpa (koşum ÖNCESİ) ----
    foreign_once = {}
    for ad, n in nokta_once_map.items():
        if ad not in KANONIK_ADLAR:
            foreign_once[ad] = n
    cipa_once: Dict[str, Optional[str]] = {}
    for yol_ in CIPALAR:
        cipa_once[yol_] = _dosya_sha(os.path.join(REPO_ROOT, yol_))
    replay_sha = _sha256(REPLAY_BUFFER)
    kanal_sha_once = _dosya_sha(GERCEK_KANAL_TAM)
    b2: Dict[str, Any] = {
        "digest_once": digest_once,
        "digest_ilanli": ENVANTER_CIPA,
        "foreign_once": foreign_once,
        "kristal_bellek_once": nokta_once_map.get(KANONIK_KOLEKSIYON),
        "probe_once": nokta_once_map.get(PROBE_KOLEKSIYON, "YOK"),
        "muhakeme_once": nokta_once_map.get("muhakeme_bellek", "YOK"),
        "simulasyon_once": nokta_once_map.get("simulasyon_bellek",
                                              "YOK"),
        "gercek_kanal_once_bayt": kanal_once_bayt,
        "gercek_kanal_once_sha256": kanal_sha_once,
        "replay_buffer_sha256": replay_sha,
        "cipa_shalar_once": cipa_once,
    }
    detay["b2_arz_cipa"] = b2
    k2_tam = bool(
        digest_once == ENVANTER_CIPA
        and len(foreign_once) == 9
        and nokta_once_map.get(KANONIK_KOLEKSIYON) == 36
        and PROBE_KOLEKSIYON not in nokta_once_map
        and "muhakeme_bellek" not in nokta_once_map
        and "simulasyon_bellek" not in nokta_once_map
        and kanal_once_bayt == 0
        and kanal_sha_once == GERCEK_KANAL_SHA
        and replay_sha == REPLAY_SHA
        and all(cipa_once.values()))
    kapilar["K2_ARZ_CIPA"] = k2_tam
    if not k2_tam:
        _stderr("DUR: B2 arz-çıpa DÜŞTÜ (digest/arz/çıpa sapması) — "
                "koşum ÖNCESİ durulur")
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["b2-arz-cipa-dustu"])
        return

    # ---- model yükle (kanonik P3 kalıbı; DONMUŞ salt-okunur) ----
    model, vocab, tokenizer, n_vocab = _model_yukle(arg.vocab,
                                                    arg.base_ckpt, detay)
    detay["n_vocab"] = n_vocab

    # ---- kanonik VM (kristal_bellek YALNIZ OKUNUR) ----
    memory = VectorMemory(collection_name=KANONIK_KOLEKSIYON,
                          vector_size=768, host=arg.host, port=arg.port,
                          storage_path="")
    if memory.is_in_memory \
            or not str(memory.storage_type).startswith("remote"):
        _stderr("DUR: VectorMemory uzak sunucuya bağlanamadı "
                "(storage_type=" + str(memory.storage_type)
                + "); sessiz-fallback YASAK")
        kapilar["K3_KANAL_YAZARI"] = False
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["vector-memory-fallback"])
        return
    kristal_sayi = memory.get_document_count()
    detay["kristal_bellek_nokta_agent"] = kristal_sayi
    if kristal_sayi != 36:
        _stderr("DUR: '" + KANONIK_KOLEKSIYON + "' nokta sayısı "
                + str(kristal_sayi) + " != İLAN'lı 36")
        kapilar["K2_ARZ_CIPA"] = False
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["kanonik-nokta-sayisi-sapmasi"])
        return

    # ---- B1 kanal-yazarı ----
    tum_sorgular, secili = _sorgular(arg.korpus, random.Random(SEED),
                                     POZITIF_N)
    detay["sorgu_havuzu"] = len(tum_sorgular)
    detay["secili_sorgu"] = len(secili)

    # (i) canlı-döngü: kanonik auto-build gateway (tau=2,5; sim=0,85)
    gateway_default = AgentGateway(
        model=model, tokenizer=tokenizer, memory=memory,
        future_train_path=PROBE_FUTURE_TRAIN, device="cpu")
    b1_satirlar: List[Dict[str, Any]] = []
    b1_istisna: List[Dict[str, Any]] = []
    ask_recorded = 0
    ask_kume_tam = True
    for j in secili:
        sorgu = tum_sorgular[j]
        try:
            sonuc = gateway_default.ask(sorgu)
            ask_kume_tam = ask_kume_tam \
                and (set(sonuc.keys()) == TELEMETRI_ASK)
            skor = round(float(sonuc.get("rag_score", 0.0)), 4)
            e_pre = round(float(sonuc.get("entropy_pre", 0.0)), 4)
            e_post = round(float(sonuc.get("entropy_post", 0.0)), 4)
            b1_satirlar.append({
                "sorgu": sorgu[:60],
                "rag_score": skor,
                "is_high_similarity": bool(
                    sonuc.get("is_high_similarity")),
                "future_train_recorded": bool(
                    sonuc.get("future_train_recorded")),
                "needs_retrieval": bool(sonuc.get("needs_retrieval")),
                "entropy_pre": e_pre, "entropy_post": e_post,
            })
            if sonuc.get("future_train_recorded"):
                ask_recorded = ask_recorded + 1
        except Exception as e:  # noqa: BLE001 — istisna kapı düşürür
            b1_istisna.append({"sorgu": sorgu[:60],
                               "istisna": type(e).__name__ + ": "
                                          + str(e)})

    # (ii) probe-agent (kanonik kurucu sim=0,5 — P3 kalıbı)
    probe_yol_tam = os.path.join(REPO_ROOT, PROBE_FUTURE_TRAIN)
    probe_once_satir = 0
    if os.path.exists(probe_yol_tam):
        with open(probe_yol_tam, encoding="utf-8") as f:
            for satir in f:
                if satir.strip():
                    probe_once_satir = probe_once_satir + 1
    probe_agent = _agent_kur(model, tokenizer, memory,
                             PROBE_FUTURE_TRAIN, 0.5)
    probe_recorded = 0
    probe_istisna: List[Dict[str, Any]] = []
    for j in secili:
        sorgu = tum_sorgular[j]
        try:
            sonuc_p = probe_agent.process_query(sorgu, force_rag=True)
            if sonuc_p.get("future_train_recorded"):
                probe_recorded = probe_recorded + 1
        except Exception as e:  # noqa: BLE001
            probe_istisna.append({"sorgu": sorgu[:60],
                                  "istisna": type(e).__name__ + ": "
                                             + str(e)})

    # append-davranışı: aynı kayıt ikinci kez → satır +1 (ezme YOK)
    probe_record: Dict[str, Any] = {
        "instruction": "Belgeye göre cevapla.",
        "input": "PROBE_T0145_SEMA_KANITI",
        "output": "PROBE_HEDEF",
        "model_failed_output": "PROBE_MODEL_CIKTISI",
        "decompiled_output": "PROBE_DECOMPILED",
        "rag_document": "PROBE_RAG_BELGESI",
        "retrieval_collection": KANONIK_KOLEKSIYON,
        "similarity_score": 0.0,
        "similarity_threshold": 0.85,
        "entropy_pre": 0.0,
        "entropy_post": 0.0,
        "tau": 2.5,
        "reason": "t0145_sema_kaniti_probe",
        "timestamp": damga,
    }
    probe_agent.record_to_future_train(probe_record)

    satirlar_probe: List[str] = []
    if os.path.exists(probe_yol_tam):
        with open(probe_yol_tam, encoding="utf-8") as f:
            for satir in f:
                if satir.strip():
                    satirlar_probe.append(satir)
    sema_tam = 0
    for satir in satirlar_probe:
        try:
            rec = json.loads(satir)
            if set(rec.keys()) == set(RECORD_ANAHTARLARI):
                sema_tam = sema_tam + 1
        except Exception:  # noqa: BLE001
            pass
    beklenen_satir = probe_once_satir + ask_recorded + probe_recorded + 1
    is_high_sayisi = 0
    for k in b1_satirlar:
        if k["is_high_similarity"]:
            is_high_sayisi = is_high_sayisi + 1
    detay["b1_kanal_yazari"] = {
        "ask_n": POZITIF_N,
        "ask_kayit": len(b1_satirlar),
        "ask_istisna": len(b1_istisna),
        "ask_istisnalar": b1_istisna,
        "ask_kume_tam": ask_kume_tam,
        "is_high_similarity_sayisi": is_high_sayisi,
        "ask_recorded_n": ask_recorded,
        "probe_once_satir": probe_once_satir,
        "recorded_n": probe_recorded,
        "probe_istisna": len(probe_istisna),
        "probe_istisnalar": probe_istisna,
        "beklenen_satir": beklenen_satir,
        "son_satir_sayisi": len(satirlar_probe),
        "sema_tam": sema_tam,
        "append_teyit": len(satirlar_probe) == beklenen_satir,
        "telemetri_kirilim": b1_satirlar,
    }
    kapilar["K3_KANAL_YAZARI"] = bool(
        len(b1_satirlar) == POZITIF_N
        and len(b1_istisna) == 0
        and ask_kume_tam
        and is_high_sayisi == 0
        and probe_recorded >= 1
        and len(probe_istisna) == 0
        and len(satirlar_probe) == beklenen_satir
        and sema_tam == len(satirlar_probe))


    # ---- B3 tüketici-okuma ----
    pipeline_probe = RetrainPipeline(
        future_train_path=PROBE_FUTURE_TRAIN,
        archive_path=PROBE_ARCHIVE,
        vocab_path=arg.vocab,
        model_path=arg.base_ckpt,
        output_bin_path=PROBE_BIN,
        save_path=PROBE_SAVE,
        device="cpu")
    pipeline_gercek = RetrainPipeline(
        future_train_path=GERCEK_KANAL,
        archive_path=PROBE_ARCHIVE,
        vocab_path=arg.vocab,
        model_path=arg.base_ckpt,
        output_bin_path=PROBE_BIN,
        save_path=PROBE_SAVE,
        device="cpu")
    probe_sayisi = pipeline_probe.get_pending_count()
    gercek_sayisi = pipeline_gercek.get_pending_count()
    detay["b3_tuketici_okuma"] = {
        "probe_sayisi": probe_sayisi,
        "probe_beklenen": len(satirlar_probe),
        "gercek_sayisi": gercek_sayisi,
    }
    kapilar["K4_TUKETICI_OKUMA"] = bool(
        probe_sayisi == len(satirlar_probe)
        and probe_sayisi > 0
        and gercek_sayisi == 0)

    # ---- B4 tüketici-derleme ----
    b4: Dict[str, Any] = {}
    try:
        bin_yol, backlog = pipeline_probe.compile_backlog_to_bin()
    except Exception as e:  # noqa: BLE001 — istisna kapı düşürür
        b4["istisna"] = type(e).__name__ + ": " + str(e)
        detay["b4_tuketici_derleme"] = b4
        kapilar["K5_TUKETICI_DERLEME"] = False
        _stderr("UYARI: compile_backlog_to_bin istisna — B4 düştü; "
                "kalan kapılar koşulur (fail-closed hüküm)")
        bin_yol = ""
        backlog = -1
        meta: Dict[str, Any] = {}
        arr = np.array([], dtype=np.uint16)
        max_jeton = -1
        geri_tok = False
        bin_boyut = -1
    if bin_yol:
        bin_tam = os.path.join(REPO_ROOT, bin_yol)
        if os.path.exists(bin_tam):
            bin_boyut = os.path.getsize(bin_tam)
        else:
            bin_boyut = -1
        meta_yol = bin_tam + ".meta.json"
        if os.path.exists(meta_yol):
            with open(meta_yol, encoding="utf-8") as mf:
                meta = json.load(mf)
        else:
            meta = {}
        if bin_boyut > 0:
            arr = np.fromfile(bin_tam, dtype=np.uint16)
        else:
            arr = np.array([], dtype=np.uint16)
        max_jeton = -1
        if len(arr) > 0:
            max_jeton = int(arr.max())
        # geri-tokenize: retrain'in İÇ tokenizer'ı default-kur (lexicon
        # default yol) — [[indeks-yazan-ile-arayan-normalizasyonu-
        # ayni-olmali]]: yazan kurulumla aynı modda doğrula
        geri_tok = False
        ilk_uzunluk = 0
        lexicon_b4 = LexiconManager()
        lexicon_b4.load_from_tsv("data/lexicon/roots.tsv")
        compiler_b4 = CrystalCompiler(lexicon_b4, build_default_graph())
        tokenizer_b4 = KristalTokenizer(compiler_b4, vocab)
        for satir in satirlar_probe:
            try:
                rec0 = json.loads(satir)
            except Exception:  # noqa: BLE001
                continue
            raw0 = render_example(
                rec0.get("instruction", "Belgeye göre cevapla."),
                rec0.get("input", ""),
                rec0.get("output", ""))
            tids0 = tokenizer_b4.encode(raw0)
            if len(tids0) > 2:
                ilk_uzunluk = len(tids0)
                if len(arr) >= len(tids0):
                    bas_parca = [int(x) for x in arr[:len(tids0)]]
                    geri_tok = bas_parca == list(tids0)
                break
        b4["ilk_kayit_uzunluk"] = ilk_uzunluk

    toplam_jeton = int(len(arr))
    max_jeton_alti = bool(0 <= max_jeton < 32852)
    default_bin_once = os.path.exists(os.path.join(REPO_ROOT, DEFAULT_BIN))
    default_bin_yol_tam = os.path.join(REPO_ROOT, DEFAULT_BIN)
    default_bin_mtime_once = (os.path.getmtime(default_bin_yol_tam)
                              if os.path.exists(default_bin_yol_tam)
                              else -1.0)
    default_bin_digest_once = (_dosya_sha(default_bin_yol_tam)
                               if os.path.exists(default_bin_yol_tam)
                               else "")
    default_bin_meta_yol_tam = default_bin_yol_tam + ".meta.json"
    default_bin_meta_mtime_once = (
        os.path.getmtime(default_bin_meta_yol_tam)
        if os.path.exists(default_bin_meta_yol_tam) else -1.0)
    b4["bin_yol"] = bin_yol
    b4["backlog_samples"] = backlog
    b4["bin_boyut"] = bin_boyut
    b4["uint16"] = bool(bin_boyut > 0 and bin_boyut % 2 == 0)
    b4["meta"] = meta
    b4["toplam_jeton"] = toplam_jeton
    b4["max_jeton_id"] = max_jeton
    b4["max_jeton_alti"] = max_jeton_alti
    b4["geri_tokenize"] = geri_tok
    b4["default_bin_once"] = default_bin_once
    b4["default_bin_mtime_once"] = default_bin_mtime_once
    b4["default_bin_digest_once"] = default_bin_digest_once
    b4["default_bin_meta_mtime_once"] = default_bin_meta_mtime_once
    detay["b4_tuketici_derleme"] = b4
    kapilar["K5_TUKETICI_DERLEME"] = bool(
        bin_boyut > 0
        and bin_boyut % 2 == 0
        and meta.get("block_size") == 64
        and meta.get("backlog_samples") == len(satirlar_probe)
        and meta.get("total_samples") == backlog + 25
        and meta.get("oversample_factor") == 20
        and geri_tok)

    # ---- B5 run_training İLKELİ ----
    save_dizin = os.path.dirname(os.path.abspath(PROBE_SAVE))
    os.makedirs(save_dizin, exist_ok=True)
    arsiv_yol_tam = os.path.join(REPO_ROOT, PROBE_ARCHIVE)
    arsiv_once_satir = 0
    if os.path.exists(arsiv_yol_tam):
        with open(arsiv_yol_tam, encoding="utf-8") as f:
            for satir in f:
                if satir.strip():
                    arsiv_once_satir = arsiv_once_satir + 1
    son_satir = satirlar_probe[-1] if satirlar_probe else ""
    b5: Dict[str, Any] = {"arsiv_once_satir": arsiv_once_satir}
    try:
        run_sonuc = pipeline_probe.run_training(steps=10)
    except Exception as e:  # noqa: BLE001 — kanonik İLKELİ istisnası
        run_sonuc = {"status": "istisna",
                     "error": type(e).__name__ + ": " + str(e)}
    b5["run_training_status"] = run_sonuc.get("status")
    b5["samples_trained"] = run_sonuc.get("samples_trained")
    b5["duration_sec"] = run_sonuc.get("duration_sec")
    log_deger = run_sonuc.get("log_snippet") \
        or run_sonuc.get("error") or ""
    b5["log_snippet"] = str(log_deger)[-800:]
    if os.path.exists(probe_yol_tam):
        probe_sonra_bayt = os.path.getsize(probe_yol_tam)
    else:
        probe_sonra_bayt = -1
    arsiv_sonra_satir = 0
    arsiv_geri_uyum = False
    arsiv_satirlar: List[str] = []
    if os.path.exists(arsiv_yol_tam):
        with open(arsiv_yol_tam, encoding="utf-8") as f:
            for satir in f:
                if satir.strip():
                    arsiv_satirlar.append(satir)
        arsiv_sonra_satir = len(arsiv_satirlar)
        if arsiv_satirlar and son_satir:
            arsiv_geri_uyum = arsiv_satirlar[-1].strip() \
                == son_satir.strip()
    save_var = os.path.exists(PROBE_SAVE)
    if os.path.exists(GERCEK_KANAL_TAM):
        kanal_b5_bayt = os.path.getsize(GERCEK_KANAL_TAM)
    else:
        kanal_b5_bayt = -1
    default_bin_sonra = os.path.exists(default_bin_yol_tam)
    default_bin_mtime_sonra = (os.path.getmtime(default_bin_yol_tam)
                               if default_bin_sonra else -1.0)
    default_bin_digest_sonra = (_dosya_sha(default_bin_yol_tam)
                                if default_bin_sonra else "")
    default_bin_meta_sonra = os.path.exists(default_bin_meta_yol_tam)
    default_bin_meta_mtime_sonra = (
        os.path.getmtime(default_bin_meta_yol_tam)
        if default_bin_meta_sonra else -1.0)
    b5["arsiv_sonra_satir"] = arsiv_sonra_satir
    b5["arsiv_beklenen"] = arsiv_once_satir + len(satirlar_probe)
    b5["arsiv_tasima_teyit"] = arsiv_geri_uyum
    b5["probe_sonra_bayt"] = probe_sonra_bayt
    b5["save_var"] = save_var
    b5["save_yol"] = PROBE_SAVE
    b5["kanal_b5_bayt"] = kanal_b5_bayt
    b5["default_bin_sonra"] = default_bin_sonra
    b5["default_bin_mtime_once"] = default_bin_mtime_once
    b5["default_bin_mtime_sonra"] = default_bin_mtime_sonra
    b5["default_bin_digest_once"] = default_bin_digest_once
    b5["default_bin_digest_sonra"] = default_bin_digest_sonra
    b5["default_bin_meta_mtime_once"] = default_bin_meta_mtime_once
    b5["default_bin_meta_mtime_sonra"] = default_bin_meta_mtime_sonra
    detay["b5_run_training"] = b5
    kapilar["K6_RUN_TRAINING_ILKELI"] = bool(
        b5["run_training_status"] == "success"
        and probe_sonra_bayt == 0
        and arsiv_sonra_satir == arsiv_once_satir + len(satirlar_probe)
        and arsiv_geri_uyum
        and save_var
        and kanal_b5_bayt == 0
        and default_bin_mtime_sonra == default_bin_mtime_once
        and default_bin_digest_sonra == default_bin_digest_once
        and default_bin_meta_mtime_sonra == default_bin_meta_mtime_once)


    # ---- B6 gateway-yazım yüzeyleri (yalnız probe) ----
    probe_vm = VectorMemory(collection_name=PROBE_KOLEKSIYON,
                            vector_size=768, host=arg.host,
                            port=arg.port, storage_path="")
    if probe_vm.is_in_memory \
            or not str(probe_vm.storage_type).startswith("remote"):
        _stderr("DUR: probe VectorMemory uzak sunucuya bağlanamadı — "
                "sessiz-fallback YASAK")
        kapilar["K7_GATEWAY_YAZIM"] = False
        _dur_ve_cik(kapilar, detay, damga, arg, ilan_sha,
                    ["probe-vm-fallback"])
        return
    gateway_probe = AgentGateway(
        model=model, tokenizer=tokenizer, memory=memory,
        general_memory=probe_vm, epistemic_agent=probe_agent,
        future_train_path=PROBE_FUTURE_TRAIN, device="cpu")
    b6: Dict[str, Any] = {}
    try:
        inj1 = gateway_probe.inject_knowledge(
            PROBE_DOC_1, target_collection=PROBE_KOLEKSIYON,
            metadata={"domain": "p5_probe"})
        count1 = probe_vm.get_document_count()
        t_ids = tokenizer.encode(PROBE_DOC_1)
        t_tags = tokenizer.decode(t_ids)
        dense_v = generate_kristal_vector(t_ids, t_tags)
        sparse_v = generate_sparse_vector(t_ids, t_tags)
        geri1 = probe_vm.hybrid_recall(dense_v, sparse_v, top_k=1,
                                       query_tags=t_tags)
        geri1_tam = False
        geri1_skor: Optional[float] = None
        if len(geri1) > 0:
            birinci = geri1[0]
            meta_d = birinci.get("metadata")
            if not isinstance(meta_d, dict):
                meta_d = {}
            geri1_skor = round(float(birinci.get("score", 0.0)), 4)
            geri1_tam = bool(
                geri1_skor > 0
                and "P5_PROBE_DOC" in str(birinci.get("text", "")))
            geri1_tam = geri1_tam and (
                meta_d.get("injected_by") == "agent_gateway")
        inj2 = gateway_probe.inject_knowledge(
            PROBE_DOC_1, target_collection=SAHTE_AD,
            metadata={"domain": "p5_probe"})
        count2 = probe_vm.get_document_count()
        sahte_yok = not client.collection_exists(SAHTE_AD)
        check_list = gateway_probe.check_memory(
            PROBE_QUERY, target_collection=PROBE_KOLEKSIYON, top_k=3)
        check_top1: Dict[str, Any] = {}
        if len(check_list) > 0:
            check_top1 = check_list[0]
        check_skor = round(float(check_top1.get("score", 0.0)), 4) \
            if check_top1 else 0.0
        check_text_tam = bool(
            "P5_PROBE_DOC" in str(check_top1.get("text", ""))) \
            if check_top1 else False
        b6 = {
            "inject1_status": inj1.get("status"),
            "inject1_collection": inj1.get("collection"),
            "inject1_crystal_tags": str(inj1.get("crystal_tags", ""))[:80],
            "inject1_count": count1,
            "geri_okuma_tam": geri1_tam,
            "geri_okuma_skor": geri1_skor,
            "sahte_ad": SAHTE_AD,
            "inject2_returned_collection": inj2.get("collection"),
            "inject2_count_probe": count2,
            "inject2_count_beklenen": count1 + 1,
            "sahte_ad_sunucuda_yok": sahte_yok,
            "check_sonuclar": len(check_list),
            "check_top1_skor": check_skor,
            "check_top1_text_tam": check_text_tam,
        }
        detay["b6_gateway_yazim"] = b6
        kapilar["K7_GATEWAY_YAZIM"] = bool(
            inj1.get("status") == "success"
            and inj1.get("collection") == PROBE_KOLEKSIYON
            and count1 == 1
            and geri1_tam
            and count2 == count1 + 1
            and inj2.get("collection") == SAHTE_AD
            and sahte_yok
            and len(check_list) > 0
            and check_skor > 0
            and check_text_tam)
    except Exception as e:  # noqa: BLE001 — B6 istisnası kapı düşürür
        detay["b6_gateway_yazim"] = {"istisna": type(e).__name__ + ": "
                                                  + str(e)}
        kapilar["K7_GATEWAY_YAZIM"] = False

    # ---- B7 HTTP-teyidi (sandbox DIŞI; ephemeral port) ----
    b7: Dict[str, Any] = {}
    server: Any = None
    try:
        server = gateway_probe.create_http_server("127.0.0.1", 0)
        port = int(server.server_address[1])
        worker = threading.Thread(target=server.serve_forever,
                                  kwargs={"poll_interval": 0.05},
                                  daemon=True)
        worker.start()
        base_url = "http://127.0.0.1:" + str(port)
        zaman_asimi = 600

        def _get(yol_: str) -> Tuple[int, Any]:
            yanit = urllib.request.urlopen(base_url + yol_,
                                           timeout=zaman_asimi)
            govde = yanit.read().decode("utf-8")
            yanit.close()
            return 200, json.loads(govde)

        def _post(yol_: str, yuk: Dict[str, Any]) -> Tuple[int, Any]:
            veri = json.dumps(yuk, ensure_ascii=False).encode("utf-8")
            istek = urllib.request.Request(
                base_url + yol_, data=veri,
                headers={"Content-Type": "application/json"},
                method="POST")
            yanit = urllib.request.urlopen(istek, timeout=zaman_asimi)
            govde = yanit.read().decode("utf-8")
            yanit.close()
            return 200, json.loads(govde)

        kod_s, status_s = _get("/api/status")
        kod_b, backlog_l = _get("/api/backlog")
        kod_c, check_r = _post("/api/check",
                               {"query": PROBE_QUERY,
                                "collection": PROBE_KOLEKSIYON,
                                "top_k": 3})
        count_once_b7 = probe_vm.get_document_count()
        kod_i, inj_r = _post("/api/inject",
                             {"text": PROBE_DOC_3,
                              "collection": PROBE_KOLEKSIYON,
                              "metadata": {"domain": "p5_probe"}})
        count3 = probe_vm.get_document_count()
        kod_q, query_r = _post("/api/query",
                               {"query": tum_sorgular[secili[0]]})
        check_top1_b7 = 0.0
        if isinstance(check_r, dict) and check_r.get("results"):
            check_top1_b7 = round(
                float(check_r["results"][0].get("score", 0.0)), 4)
        query_anahtar = -1
        if isinstance(query_r, dict):
            query_anahtar = len(query_r.keys())
        b7 = {
            "port": port,
            "status_kod": kod_s,
            "status_online": status_s.get("status"),
            "status_backlog_samples": status_s.get(
                "epistemic_backlog_samples"),
            "status_future_train_path": status_s.get("future_train_path"),
            "backlog_kod": kod_b,
            "backlog_uzunluk": (len(backlog_l)
                                if isinstance(backlog_l, list) else -1),
            "check_kod": kod_c,
            "check_top1_skor": check_top1_b7,
            "inject_kod": kod_i,
            "inject_status": inj_r.get("status"),
            "inject_collection": inj_r.get("collection"),
            "inject_count_probe_once": count_once_b7,
            "inject_total": inj_r.get("total_documents"),
            "inject_count_probe_sonra": count3,
            "query_kod": kod_q,
            "query_anahtar_sayisi": query_anahtar,
            "query_future_train_path": (query_r.get("future_train_path")
                                        if isinstance(query_r, dict)
                                        else None),
            "query_is_high_similarity": (query_r.get("is_high_similarity")
                                         if isinstance(query_r, dict)
                                         else None),
        }
        server.shutdown()
        server.server_close()
        detay["b7_http"] = b7
        kapilar["K8_HTTP_TEYIDI"] = bool(
            kod_s == 200
            and b7["status_online"] == "online"
            and b7["status_backlog_samples"] == 0
            and kod_b == 200
            and b7["backlog_uzunluk"] == 0
            and kod_c == 200
            and b7["check_top1_skor"] > 0
            and kod_i == 200
            and b7["inject_status"] == "success"
            and b7["inject_collection"] == PROBE_KOLEKSIYON
            and b7["inject_total"] == count3
            and kod_q == 200
            and b7["query_anahtar_sayisi"] == 17)
        # RAPOR-kırılımı (hüküm-dışı): /api/query yanıtı 17-anahtar
        # ama future_train_path alanı TAŞIMIYOR (null) — kanonik
        # HTTP-yanıt-şekli; soy-beyanı yalnız /api/status verir.
    except Exception as e:  # noqa: BLE001 — HTTP istisnası kapı düşürür
        detay["b7_http"] = {"istisna": type(e).__name__ + ": " + str(e)}
        kapilar["K8_HTTP_TEYIDI"] = False
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:  # noqa: BLE001
                pass

    # ---- temizlik: probe koleksiyonu kaldır (kanonik client-yolu) ----
    temizlik: Dict[str, Any] = {"yol": "client.delete_collection"}
    try:
        if client.collection_exists(PROBE_KOLEKSIYON):
            client.delete_collection(PROBE_KOLEKSIYON)
        temizlik["kaldi"] = client.collection_exists(PROBE_KOLEKSIYON)
    except Exception as e:  # noqa: BLE001
        temizlik["istisna"] = type(e).__name__ + ": " + str(e)
        temizlik["kaldi"] = True
    detay["temizlik"] = temizlik


    # ---- B8 dokunulmazlık (koşum SONU) ----
    env_sonra = _envanter(client)
    digest_sonra = _envanter_digest(env_sonra)
    nokta_sonra_map: Dict[str, Any] = {}
    for kalem in env_sonra["kalemler"]:
        nokta_sonra_map[kalem["ad"]] = kalem["nokta"]
    foreign_sonra = {}
    for ad, n in nokta_sonra_map.items():
        if ad not in KANONIK_ADLAR:
            foreign_sonra[ad] = n
    kanal_sonra_bayt = os.path.getsize(GERCEK_KANAL_TAM) \
        if os.path.exists(GERCEK_KANAL_TAM) else -1
    kanal_sonra_sha = _dosya_sha(GERCEK_KANAL_TAM)
    default_bin_meta_sonra = os.path.exists(
        os.path.join(REPO_ROOT, DEFAULT_BIN + ".meta.json"))
    cipa_sonra: Dict[str, Optional[str]] = {}
    for yol_ in CIPALAR:
        cipa_sonra[yol_] = _dosya_sha(os.path.join(REPO_ROOT, yol_))
    b8: Dict[str, Any] = {
        "digest_once": digest_once,
        "digest_sonra": digest_sonra,
        "digest_ilanli": ENVANTER_CIPA,
        "kristal_once": nokta_once_map.get(KANONIK_KOLEKSIYON),
        "kristal_sonra": nokta_sonra_map.get(KANONIK_KOLEKSIYON),
        "foreign_once": foreign_once,
        "foreign_sonra": foreign_sonra,
        "probe_sonra": nokta_sonra_map.get(PROBE_KOLEKSIYON, "YOK"),
        "muhakeme_sonra": nokta_sonra_map.get("muhakeme_bellek", "YOK"),
        "simulasyon_sonra": nokta_sonra_map.get("simulasyon_bellek",
                                                "YOK"),
        "kanal_once_bayt": kanal_once_bayt,
        "kanal_sonra_bayt": kanal_sonra_bayt,
        "kanal_sha256": kanal_sonra_sha,
        "default_bin_sonra": default_bin_sonra,
        "default_bin_meta_sonra": default_bin_meta_sonra,
        "default_bin_mtime_once_sonra":
            [default_bin_mtime_once, default_bin_mtime_sonra],
        "default_bin_digest_once_sonra":
            [str(default_bin_digest_once)[:16],
             str(default_bin_digest_sonra)[:16]],
        "default_bin_meta_mtime_once_sonra":
            [default_bin_meta_mtime_once, default_bin_meta_mtime_sonra],
        "cipa_shalar_once": cipa_once,
        "cipa_shalar_sonra": cipa_sonra,
    }
    detay["b8_dokunulmazlik"] = b8
    b8_durum = "GEÇTİ" if (
        digest_once == digest_sonra
        and nokta_once_map.get(KANONIK_KOLEKSIYON) == 36
        and nokta_sonra_map.get(KANONIK_KOLEKSIYON) == 36
        and foreign_once == foreign_sonra
        and PROBE_KOLEKSIYON not in nokta_sonra_map
        and "muhakeme_bellek" not in nokta_sonra_map
        and "simulasyon_bellek" not in nokta_sonra_map
        and kanal_once_bayt == 0
        and kanal_sonra_bayt == 0
        and kanal_sonra_sha == GERCEK_KANAL_SHA
        and default_bin_mtime_sonra == default_bin_mtime_once
        and default_bin_digest_sonra == default_bin_digest_once
        and default_bin_meta_mtime_sonra == default_bin_meta_mtime_once
        and cipa_once == cipa_sonra
    ) else "SAPMA"
    b8["durum"] = b8_durum
    kapilar["K9_DOKUNULMAZLIK"] = bool(b8_durum == "GEÇTİ")

    # ---- digest-cetveli (rapora) ----
    hukum_json_pre, _rc_pre = _hukum_kur(kapilar, detay, damga)
    digest_tablo: List[str] = []
    digest_tablo.append("| " + os.path.basename(arg.ilan)
                        + " (İLAN — koşum ÖNCESİ, REVİZYON-1) | `"
                        + ilan_sha + "` |")
    digest_tablo.append("| " + os.path.basename(betik_yol)
                        + " (koşulan betik) | `" + betik_sha + "` |")
    digest_tablo.append("| " + arg.base_ckpt + " (DONMUŞ taban) | `"
                        + base_sha + "` |")
    digest_tablo.append("| " + arg.vocab + " (DONMUŞ sözlük) | `"
                        + vocab_sha + "` |")
    digest_tablo.append("| " + arg.korpus + " (DONMUŞ kaynak) | `"
                        + korpus_sha + "` |")
    digest_tablo.append("| replay-buffer (DONMUŞ) | `" + replay_sha
                        + "` |")
    digest_tablo.append("| data/future_train_vector.jsonl (GERÇEK kanal"
                        " — 0 bayt) | `" + str(kanal_sonra_sha) + "` |")
    digest_tablo.append("| envanter ÖNCE/SONRA digest | `"
                        + digest_once + "` / `" + digest_sonra + "` |")
    for yol_ in CIPALAR:
        digest_tablo.append("| " + yol_ + " | `"
                            + str(cipa_sonra[yol_]) + "` |")
    detay["digest_tablo"] = digest_tablo

    # ---- Hüküm (betikten) → hüküm JSON + rapor ----
    hukum_json, rc = _hukum_kur(kapilar, detay, damga)
    hukum_yol = _hukum_yaz(hukum_json, arg)
    hukum_sha = _sha256(hukum_yol)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, hukum_yol,
               hukum_sha)
    _stderr("HÜKÜM: " + hukum_json["hukum"] + " rc=" + str(rc)
            + " → " + hukum_yol)
    sys.exit(rc)


if __name__ == "__main__":
    main()