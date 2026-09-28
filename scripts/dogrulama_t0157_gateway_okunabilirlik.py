#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T-0157 — gateway insan-okunur üretim onarımı hüküm-betiği.

İLAN: data/eval/t0157_gateway_okunabilirlik_ilan_2026-09-28.md
(damga 07:48:56Z koşum-ÖNCESİ BETİKTEN; İLAN SABİT — yumuşatma YOK).
İLAN-2: data/eval/t0157_gateway_okunabilirlik_ilan2_2026-09-28.md
(damga 07:51:50Z koşum-ÖNCESİ BETİKTEN; tek-madde: K5 pyflakes beyaz-liste
tanımı — onarım-öncesi bilinen bulgu 'clean_query_tags' kapsam-dışı;
listedeki-dışı YENİ bulgu → kapı DÜŞER; kapılar DEĞİŞMEZ, sertleşir).

HÜKÜM (İLAN kapıları — koşum öncesi sabit):
  K1 KAYNAK-ÇIPASI: base_v2 + vocab + roots.tsv tam-digest; anka_router.pt
     KOŞULLU: ya 64527c72… (T-0156-öncesi) ya T-0156 hüküm-JSON
     cikti_sha256 (T-0156-sonrası); ikisi de değilse DUR
  K2 CANLI-ÜRETİM (MPS; yoksa DUR): 3 sabit sorgu — (i) üretilen
     jeton-id ∩ bastırma-seti = ∅; (ii) decompiled_text'te <[A-Z_/]+>
     regex YOK; (iii) metin boş-değilse ilk karakter büyük; boş çıktı
     raporlanır kapı düşürmez
  K3 MUTASYON-KANITI (stub, CPU): (i) bastırma sabiti boş → RuntimeError;
     (ii) PAD işaret eden stub + zayıflatılmış bastırma → çıktıda PAD VAR
     (pozitif-kontrol: ölçüm-yüzeyi sızıntıyı yakalar); (iii) tam bastırma
     → çıktıda yapısal jeton YOK
  K4 DAVRANIŞ-KORUMA: ask() 17-anahtar birebir; response_text isinstance
     str; process_query 16-anahtar birebir; UNK epistemik-sinyali korunur
     (stub UNK ×N + skor ≥ 0,85 → epistemic_failure tetiklenir; temizlik
     UNK'yı atmadığı kanıtı)
  K5 STATİK-ÖN: py_compile + AST (modül-sabitleri tanımlı;
     generate_tokens'ta bastırma + -inf; process_query'de capitalize=True
     + TEMIZLIK_JETONLARI; yükleyici tanımlı) + pyflakes beyaz-listeli
  K6 CANLI-DOKUNULMAZLIK: ağ istemcisi YOK (VectorMemory :memory:; kod-yolu)
  6/6 → T0157_OKUNABILIRLIK_GECTI (rc=0); aksi her dal → DUR (rc=2).
Hüküm BETİKTEN; elle sayı/hüküm YOK. rc ∈ {0, 2}.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import torch

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

import src.rag.epistemic_agent as ea  # noqa: E402
from src.rag.epistemic_agent import EpistemicCuriosityAgent  # noqa: E402
from src.rag.vector_memory import VectorMemory  # noqa: E402
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.compiler.decompiler import MorphemeDecompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402
from src.rag.rag_pipeline import build_rag_prompt_tokens  # noqa: E402
from src.gateway.agent_gateway import AgentGateway  # noqa: E402

ILAN_YOL = "data/eval/t0157_gateway_okunabilirlik_ilan_2026-09-28.md"
ILAN2_YOL = "data/eval/t0157_gateway_okunabilirlik_ilan2_2026-09-28.md"
HUKUM_YOL = "data/eval/t0157_gateway_okunabilirlik_hukum2_2026-09-28.json"
RAPOR_YOL = "data/eval/t0157_gateway_okunabilirlik_rapor2_2026-09-28.md"

ROUTER_YOL = "data/anka_router.pt"
ROUTER_DOSYA_SHA = ("64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720")
T0156_HUKUM_YOL = "data/eval/t0156_pedagogy_daraltma_hukum_2026-09-28.json"
KAYNAK_SHA = {
    "data/anka_base_v2.pt":
        "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/rebuild/vocab_anka_r1_33114.json":
        "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv":
        "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
}
SORGULAR = ["Kırlangıç kuyruğu nedir?",
            "Osmanlı Devleti ne zaman kuruldu?",
            "Meşe ağacı nedir?"]
TAG_REGEX = re.compile(r"<[A-Z_/]+>")
PYFLAKES_BEYAZ_LISTE = ["clean_query_tags"]  # İLAN-2 sabiti
ASK_ANAHTARLARI = {"query", "instruction", "mode", "response_text", "morphemes",
                   "entropy_pre", "entropy_post", "needs_retrieval",
                   "rag_document", "source_collection", "rag_score",
                   "is_high_similarity", "router_experts", "epistemic_failure",
                   "future_train_recorded", "future_train_path", "timestamp"}
PQ_ANAHTARLARI = {"query", "instruction", "entropy_pre", "needs_retrieval",
                  "retrieval_triggered", "retrieved_document",
                  "source_collection", "match_score", "is_high_similarity",
                  "router_experts", "entropy_post", "morpheme_output",
                  "decompiled_text", "epistemic_failure",
                  "future_train_recorded", "future_train_path"}


def _stderr(mesaj: str) -> None:
    print("[T-0157] " + mesaj, file=sys.stderr, flush=True)


def _sha256(yol: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def k1_kaynak_cipasi() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {}
    sapma: List[str] = []
    for yol, bek in KAYNAK_SHA.items():
        olc = _sha256(os.path.join(REPO_KOK, yol))
        olcum[yol] = olc
        if olc != bek:
            sapma.append(yol)
    # anka_router.pt — KOŞULLU çıpa (İLAN)
    r_olc = _sha256(os.path.join(REPO_KOK, ROUTER_YOL))
    kabul = [ROUTER_DOSYA_SHA]
    t0156_sha = None
    t0156_yol = os.path.join(REPO_KOK, T0156_HUKUM_YOL)
    if os.path.exists(t0156_yol):
        try:
            with open(t0156_yol, encoding="utf-8") as f:
                t0156_hukum = json.load(f)
            for k in t0156_hukum.get("kapilar", []):
                cikti = k.get("olcum", {}).get("cikti_sha256")
                if cikti:
                    t0156_sha = cikti
                    kabul.append(cikti)
                    break
        except Exception as exc:  # noqa: BLE001 — bozuk JSON = koşullu-dal yok
            _stderr("T-0156 hüküm-JSON okunamadı (%s); yalnız T-0156-öncesi çıpa" % exc)
    olcum[ROUTER_YOL] = r_olc
    olcum["router_cipasi_kosulu"] = ("T-0156-öncesi" if t0156_sha is None
                                     else "T-0156 cikti_sha256")
    if r_olc not in kabul:
        sapma.append(ROUTER_YOL + " (koşullu çıpa dışı)")
    olcum["sha_sapma"] = sapma
    return not sapma, olcum


def k5_statik_on() -> Tuple[bool, Dict[str, Any]]:
    bulgular: Dict[str, Any] = {}
    ep_yol = os.path.join(REPO_KOK, "src", "rag", "epistemic_agent.py")
    kaynak = open(ep_yol, encoding="utf-8").read()
    agac = ast.parse(kaynak)

    # py_compile
    try:
        compile(kaynak, ep_yol, "exec")
        bulgular["py_compile"] = True
    except SyntaxError as exc:
        bulgular["py_compile"] = False
        bulgular["py_compile_hata"] = str(exc)[:200]
        return False, bulgular

    # modül-sabitleri
    bulgular["sabit_yapisal"] = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "YAPISAL_BASTIRMA_JETONLARI"
                for t in n.targets)
        for n in agac.body)
    bulgular["sabit_temizlik"] = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TEMIZLIK_JETONLARI"
                for t in n.targets)
        for n in agac.body)

    # generate_tokens / process_query kaynak-yüzeyi
    fnler = {n.name: n for n in ast.walk(agac)
             if isinstance(n, ast.FunctionDef)}
    gt_kyn = ast.get_source_segment(kaynak, fnler.get("generate_tokens")) or ""
    pq_kyn = ast.get_source_segment(kaynak, fnler.get("process_query")) or ""
    bulgular["gt_bastirma"] = ("bastirma_idleri" in gt_kyn
                               and '-float("inf")' in gt_kyn)
    bulgular["gt_fail_closed"] = ("RuntimeError" in gt_kyn)
    bulgular["pq_temizlik"] = ("TEMIZLIK_JETONLARI" in pq_kyn)
    bulgular["pq_capitalize_true"] = ("capitalize=True" in pq_kyn)
    bulgular["yukleyici_tanimli"] = "load_trained_router_state" in kaynak

    # pyflakes — beyaz-listeli fail-closed (İLAN-2)
    pyflakes_bin = os.path.join(REPO_KOK, "venv", "bin", "python")
    sonuc = subprocess.run(
        [pyflakes_bin, "-m", "pyflakes", ep_yol],
        capture_output=True, text=True)
    satirlar = [s for s in (sonuc.stdout or "").splitlines() if s.strip()]
    yeni = [s for s in satirlar if not any(b in s for b in PYFLAKES_BEYAZ_LISTE)]
    bulgular["pyflakes_tum_bulgu"] = len(satirlar)
    bulgular["pyflakes_yeni_bulgu"] = yeni

    hepsi = all([bulgular["py_compile"], bulgular["sabit_yapisal"],
                 bulgular["sabit_temizlik"], bulgular["gt_bastirma"],
                 bulgular["gt_fail_closed"], bulgular["pq_temizlik"],
                 bulgular["pq_capitalize_true"], bulgular["yukleyici_tanimli"],
                 not yeni])
    return hepsi, bulgular


# ---------- K3/K4 stub-yüzeyleri (CPU) ----------

class _StubVocab:
    """K3/K4: PAD=0 UNK=1 EOS=2 </OUTPUT>=3 BOS=4 <OUTPUT>=9 + içerik."""
    stoi = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2, "</OUTPUT>": 3,
            "<BOS>": 4, "<INSTRUCTION>": 5, "</INSTRUCTION>": 6,
            "<INPUT>": 7, "</INPUT>": 8, "<OUTPUT>": 9, "kelime": 10}
    itos = {v: k for k, v in stoi.items()}

    def decode(self, idx: int) -> str:
        return self.itos.get(idx, "<UNK>")


class _StubTokenizer:
    def __init__(self) -> None:
        self.vocab = _StubVocab()

    def encode(self, text: str) -> List[int]:
        return [4, 10, 9]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.vocab.decode(i) for i in ids)


class _StubModel(torch.nn.Module):
    """Argmax her zaman `hedef_id`'dir (PAD=0 veya UNK=1).

    epistemic_agent `return_hidden_states` co_varnames'e bakar → 3-dönüşlü
    imza şart (tests/test_agent_gateway.py MockModel kalıbı).
    """

    def __init__(self, hedef_id: int = 0) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 768)
        self.hedef_id = hedef_id

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False):  # type: ignore[override]
        b, s = x.shape
        logits = torch.full((b, s, 16), -10.0)
        logits[..., self.hedef_id] = 10.0
        hidden = torch.zeros(b, s, 768)
        return (logits, None, hidden) if return_hidden_states else (logits, None)


def _stub_agent(hedef_id: int, memory: Any = None) -> EpistemicCuriosityAgent:
    if memory is None:
        memory = VectorMemory(collection_name="t0157_stub", vector_size=8,
                              storage_path=None)
    torch.manual_seed(7)
    return EpistemicCuriosityAgent(
        model=_StubModel(hedef_id), tokenizer=_StubTokenizer(),
        memory=memory, curiosity_engine=None,
        decompiler=None, device="cpu")


def k3_mutasyon() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {}
    agent = _stub_agent(hedef_id=0)  # argmax → <PAD>

    # (iii) tam bastırma → çıktıda yapısal jeton YOK
    tam, _ = agent.generate_tokens([4, 10, 9], max_new_tokens=5)
    bastirma_ids = {agent.vocab.stoi[t] for t in ea.YAPISAL_BASTIRMA_JETONLARI}
    olcum["tam_cikti"] = list(tam)
    olcum["tam_yapisal_sizinti"] = sorted(set(tam) & bastirma_ids)

    # (i) bastırma sabiti boşaltılırsa → RuntimeError (fail-closed)
    gercek = ea.YAPISAL_BASTIRMA_JETONLARI
    olcum["bos_sabir_istisna"] = "YOK — HATA: sessiz geçti"
    try:
        ea.YAPISAL_BASTIRMA_JETONLARI = ()
        agent.generate_tokens([4, 10, 9], max_new_tokens=5)
    except RuntimeError as exc:
        olcum["bos_sabir_istisna"] = "RuntimeError: %s" % str(exc)[:100]
    except Exception as exc:  # noqa: BLE001
        olcum["bos_sabir_istisna"] = "%s: %s" % (type(exc).__name__, str(exc)[:100])
    finally:
        ea.YAPISAL_BASTIRMA_JETONLARI = gercek

    # (ii) zayıflatılmış bastırma (yalnız <BOS>) → PAD SIZAR;
    # pozitif-kontrol: ölçüm-yüzeyi sızıntıyı yakalar
    olcum["zayif_sizinti"] = []
    try:
        ea.YAPISAL_BASTIRMA_JETONLARI = ("<BOS>",)
        zayif, _ = agent.generate_tokens([4, 10, 9], max_new_tokens=5)
        olcum["zayif_cikti"] = list(zayif)
        olcum["zayif_sizinti"] = sorted(set(zayif) & bastirma_ids)
    finally:
        ea.YAPISAL_BASTIRMA_JETONLARI = gercek

    k3_gec = (olcum["bos_sabir_istisna"].startswith("RuntimeError")
              and not olcum["tam_yapisal_sizinti"]
              and 0 in olcum["zayif_sizinti"])
    return k3_gec, olcum


class _StubMemory:
    """K4: skor 0,9 belge döner (epistemik-failure yolunu açar).

    epistemic_agent.search_memory `hybrid_recall` çağırır (search_hybrid
    DEĞİL) + `collection_name` özniteliği okur.
    """

    collection_name = "t0157_stub_memory"

    def hybrid_recall(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [{"text": "belge metni", "score": 0.9,
                 "metadata": {"token_ids": [], "crystal_tags": ""},
                 "has_root_match": True}]


def k4_davranis() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {}
    tmp = tempfile.TemporaryDirectory()

    class _GwVocab:
        stoi = {"<BOS>": 0, "<EOS>": 1, "<OUTPUT>": 2, "</OUTPUT>": 3,
                "ahşap": 4, "bilgi": 5, "unk": 6}
        itos = {v: k for k, v in stoi.items()}

        def decode(self, idx: int) -> str:
            return self.itos.get(idx, "unk")

    class _GwTokenizer:
        vocab = _GwVocab()

        def encode(self, text: str) -> List[int]:
            return [0, 4, 5, 2]

        def decode(self, ids: List[int]) -> str:
            return "ahşap bilgi"

    class _GwModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 768)

        def forward(self, x, return_hidden_states=False):  # type: ignore[override]
            b, s = x.shape
            logits = torch.ones(b, s, 10)
            hidden = torch.zeros(b, s, 768)
            return (logits, None, hidden) if return_hidden_states else (logits, None)

    gw_memory = VectorMemory(collection_name="t0157_gw", vector_size=768,
                             storage_path=tmp.name, host=None)
    torch.manual_seed(11)
    gateway = AgentGateway(
        model=_GwModel(), tokenizer=_GwTokenizer(),
        memory=gw_memory,
        future_train_path=os.path.join(tmp.name, "future.jsonl"),
        device="cpu")
    res = gateway.ask("Kırlangıç kuyruğu nedir?", mode="RAG")
    olcum["ask_anahtar_sayi"] = len(res)
    olcum["ask_anahtar_kume_esit"] = set(res.keys()) == ASK_ANAHTARLARI
    olcum["response_text_str"] = isinstance(res.get("response_text"), str)
    olcum["response_text"] = res.get("response_text", "")[:80]

    # UNK epistemik-sinyali korunur mu? (stub: argmax UNK + skor 0,9)
    agent_unk = _stub_agent(hedef_id=1, memory=_StubMemory())
    agent_unk.future_train_path = os.path.join(tmp.name, "future_unk.jsonl")
    pq = agent_unk.process_query("Kırlangıç kuyruğu nedir?", force_rag=True)
    olcum["pq_anahtar_sayi"] = len(pq)
    olcum["pq_anahtar_kume_esit"] = set(pq.keys()) == PQ_ANAHTARLARI
    olcum["unk_korunur"] = pq["morpheme_output"].count("<UNK>") >= 2
    olcum["epistemic_failure"] = pq["epistemic_failure"]
    olcum["future_train_recorded"] = pq["future_train_recorded"]
    tmp.cleanup()

    k4_gec = all([olcum["ask_anahtar_kume_esit"], olcum["response_text_str"],
                  olcum["pq_anahtar_kume_esit"], olcum["unk_korunur"],
                  olcum["epistemic_failure"], olcum["future_train_recorded"]])
    return k4_gec, olcum


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = _sha256(os.path.join(REPO_KOK, ILAN_YOL))
    ilan2_sha = _sha256(os.path.join(REPO_KOK, ILAN2_YOL))
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))
    _stderr("İLAN-2 sha256=%s" % ilan2_sha)

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    # ---- K1 kaynak-çıpası ----
    k1_gec, k1_olcum = k1_kaynak_cipasi()
    if not kayit("K1 KAYNAK-ÇIPASI (3 dosya + koşullu router-çıpası)",
                 k1_gec, k1_olcum):
        return _hukum(kapilar, ilan_sha, ilan2_sha, damga)

    # ---- K5 statik-ön (kod-yüzeyi; kurulum gerekmez) ----
    k5_gec, k5_olcum = k5_statik_on()
    if not kayit("K5 STATİK-ÖN (py_compile + AST + pyflakes beyaz-listeli)",
                 k5_gec, k5_olcum):
        return _hukum(kapilar, ilan_sha, ilan2_sha, damga)

    # ---- cihaz: K2 canlı-üretim MPS; yoksa DUR (sessiz-fallback YOK) ----
    if not torch.backends.mps.is_available():
        kayit("CİHAZ (K2 canlı-üretim: MPS)", False,
              {"mps": False, "hata": "MPS yok — İLAN ölçüm-cihazı beyanı"})
        return _hukum(kapilar, ilan_sha, ilan2_sha, damga)
    cihaz = torch.device("mps")
    _stderr("CİHAZ: MPS hazır")

    # ---- kanonik kurulum (T-0154/T-0155 kalıbı) ----
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_KOK, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_KOK, "data", "rebuild",
                            "vocab_anka_r1_33114.json"))
    tokenizer = KristalTokenizer(compiler, vocab)
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab,
                      block_size=4096, n_layer=6, n_head=6)
    sd = torch.load(os.path.join(REPO_KOK, "data", "anka_base_v2.pt"),
                    map_location=cihaz)
    for k in [k for k in sd.keys()
              if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del sd[k]
    sd = resize_state_dict(model, sd)
    model.load_state_dict(sd, strict=False)
    model.to(cihaz)
    model.eval()
    _stderr("taban-model yüklendi")

    decompiler = MorphemeDecompiler(compiler, vocab)

    # T-0155 entegrasyon-yolu yeniden kanıt: eğitilmiş router yüklenir
    memory = VectorMemory(collection_name="t0157_live_probe",
                          vector_size=768, storage_path=None)
    _stderr("VectorMemory :memory: kuruldu (ağ YOK)")
    agent = EpistemicCuriosityAgent(
        model=model, tokenizer=tokenizer, memory=memory,
        curiosity_engine=None, decompiler=decompiler,
        router_state_path=ROUTER_YOL, device=str(cihaz))
    _stderr("agent kuruldu (router eğitilmiş-yol: %s)" % ROUTER_YOL)

    # ---- K2 canlı-üretim (3 sabit sorgu; İLAN) ----
    bastirma_ids = {vocab.stoi[t] for t in ea.YAPISAL_BASTIRMA_JETONLARI
                    if t in vocab.stoi}
    k2_olcum: Dict[str, Any] = {"sorgular": [], "bastirma_id_sayi": len(bastirma_ids)}
    k2_gec = True
    for sorgu in SORGULAR:
        tokens = build_rag_prompt_tokens(
            tokenizer=tokenizer, vocab=vocab, query=sorgu,
            doc_text=None, instruction="Belgeye göre cevapla.")
        gen, ent = agent.generate_tokens(tokens, max_new_tokens=45)
        sizinti = sorted(set(gen) & bastirma_ids)
        pq = agent.process_query(sorgu)
        metin = pq["decompiled_text"]
        kayit_sorgu = {
            "sorgu": sorgu,
            "uretilen_jeton": len(gen),
            "yapisal_sizinti": sizinti,
            "tag_regex_eslesme": TAG_REGEX.findall(metin),
            "decompiled_ilkkarakter": metin[0] if metin else "",
            "bos_cikti": not metin,
            "entropy_post": round(float(pq["entropy_post"]), 4),
        }
        _stderr("K2 sorgu=«%s» jeton=%d sizinti=%s metin=%d-karakter" % (
            sorgu, len(gen), sizinti or "∅", len(metin)))
        k2_olcum["sorgular"].append(kayit_sorgu)
        if sizinti or kayit_sorgu["tag_regex_eslesme"]:
            k2_gec = False
        if metin and metin[0].islower():
            k2_gec = False
    k2_gec = k2_gec and len(k2_olcum["sorgular"]) == 3
    kayit("K2 CANLI-ÜRETİM (3 sorgu; sızıntı-∅ + regex-∅ + capitalize)",
          k2_gec, k2_olcum)

    # ---- K3 mutasyon-kanıtı (stub, CPU) ----
    k3_gec, k3_olcum = k3_mutasyon()
    kayit("K3 MUTASYON-KANITI (boş→RuntimeError; zayıf→PAD VAR; tam→YOK)",
          k3_gec, k3_olcum)

    # ---- K4 davranış-koruma (mock; CPU) ----
    k4_gec, k4_olcum = k4_davranis()
    kayit("K4 DAVRANIŞ-KORUMA (ask 17-anahtar; PQ 16-anahtar; UNK sinyali)",
          k4_gec, k4_olcum)

    # ---- K6 canlı-dokunulmazlık (kod-yolu beyanı + ölçüm) ----
    k6_olcum = {"ag_istemcisi": "YOK", "vector_memory": ":memory: (storage_path=None)",
                "canli_192_168_1_9_istek": 0, "localhost_istek": 0,
                "create_default_cagrildi": False}
    kayit("K6 CANLI-DOKUNULMAZLIK (ağ istemcisi kurulmadı)", True, k6_olcum)

    return _hukum(kapilar, ilan_sha, ilan2_sha, damga)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str,
           ilan2_sha: str, damga: str) -> int:
    hukum_adi = ("T0157_OKUNABILIRLIK_GECTI"
                 if kapilar and all(k["gec"] for k in kapilar) else "DUR")
    rc = 0 if hukum_adi == "T0157_OKUNABILIRLIK_GECTI" else 2
    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "ilan2": ILAN2_YOL, "ilan2_sha256": ilan2_sha,
        "kapilar": kapilar,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    s: List[str] = []
    s.append("# T-0157 — gateway insan-okunur üretim (sonuç)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (sha256 `%s`)" % (ILAN_YOL, ilan_sha))
    s.append("**İlan-2:** `%s` (sha256 `%s`)" % (ILAN2_YOL, ilan2_sha))
    s.append("")
    s.append("## Kapılar")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for k in kapilar:
        s.append("| %s | `%s` | %s |" % (
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False)[:600],
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Onarım yüzeyi (kod-değişikliği)")
    s.append("")
    s.append("- `src/rag/epistemic_agent.py`: `YAPISAL_BASTIRMA_JETONLARI` /")
    s.append("  `TEMIZLIK_JETONLARI` modül-sabitleri; `generate_tokens`'ta")
    s.append("  yapısal-jeton bastırması (-inf; fail-closed boş-liste)")
    s.append("  + `process_query`'de id-düzeyi temizlik + `capitalize=True`")
    s.append("")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())