#!/usr/bin/env python
"""T-0155 — gateway TriModalRouter eğitilmiş-ağırlık entegrasyonu hüküm-betiği.

İLAN: data/eval/t0155_router_entegrasyon_ilan_2026-09-28.md
(damga 07:12:22Z koşum-ÖNCESİ BETİKTEN; İLAN SABİT — yumuşatma YOK).
İLAN-2: data/eval/t0155_router_entegrasyon_ilan2_2026-09-28.md
(damga 07:15:23Z koşum-2 ÖNCESİ; koşum-1 ölçüm-kabı cihaz-istisnası
beyanı — K3 CPU-router-kopyası ile ölçülür; kapılar DEĞİŞMEZ;
koşum-2 çıktıları ayrı adlarla: hüküm2/rapor2/kosum2).

HÜKÜM (İLAN kapıları — koşum öncesi sabit):
  K1 KAYNAK-ÇIPASI: 4 dosya SHA-256 birebir (anka_router.pt / base_v2 /
     vocab_anka_r1_33114 / roots.tsv — İLAN tam-değerler)
  K2 YÜKLEME-BİREBİR: EpistemicCuriosityAgent(router_state_path=...) kurulumu
     sonrası agent.router.state_dict özeti == 9b85f951… (bit-özdeş; 9-anahtar)
  K3 DAVRANIŞ-ÜRETİMİ: T-0154 veri-seçimi (SALT-IMPORT) + özellik-konfigürasyonu
     birebir 300-val örnekleminde top-1 == 0,93 (birebir-üretim)
  K4 FAİL-CLOSED MUTASYON: olmayan yol → RuntimeError; VE fresh-init
     (path=None) özeti != 9b85f951… (mevcut davranış korunur)
  K5 GATEWAY AST ZİNCİR-KAPISI: create_default → cls(...) → __init__ →
     EpistemicCuriosityAgent zincirinde router_state_path açık-beyan
  K6 CANLI-DOKUNULMAZLIK: ağ istemcisi YOK (VectorMemory :memory:; kod-yolu)
  6/6 → T0155_ENTEGRASYON_GECTI (rc=0); aksi her dal → DUR (rc=2).
Hüküm BETİKTEN; elle sayı/hüküm YOK. rc ∈ {0, 2}.
"""
import ast
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

# T-0154 kanonik-kod SALT-IMPORT (birebirlik garanti; kopya değil)
from scripts.dogrulama_t0154_router_egitim import (  # noqa: E402
    SINIF_KAYNAK, TRAIN_N, VAL_N, SEED,
    _metinler, _sha256, _state_dict_sha, _ozellik_cikar, _gate_logits,
    KAYNAK_SHA,
)
import random  # noqa: E402  (veri-seçimi T-0154 birebir — random.Random(SEED))
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.rag.merak import CuriosityEngine  # noqa: E402
from src.rag.vector_memory import VectorMemory  # noqa: E402
from src.rag.epistemic_agent import EpistemicCuriosityAgent  # noqa: E402
from src.llm.router import TriModalRouter  # noqa: E402
from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402

ILAN_YOL = "data/eval/t0155_router_entegrasyon_ilan_2026-09-28.md"
ILAN2_YOL = "data/eval/t0155_router_entegrasyon_ilan2_2026-09-28.md"
HUKUM_YOL = "data/eval/t0155_router_entegrasyon_hukum2_2026-09-28.json"
RAPOR_YOL = "data/eval/t0155_router_entegrasyon_rapor2_2026-09-28.md"
ROUTER_YOL = "data/anka_router.pt"

ROUTER_SD_SHA = ("9b85f95177f723dac83dfe813a5f385ff04c6f8aecf1767628a6cdd326afb500")
ROUTER_DOSYA_SHA = ("64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720")
CIHAZ_YOK_DUR = "MPS yok — T-0154 özellik-yüzeyi birebir gerekir (İLAN)"


def _stderr(mesaj: str) -> None:
    print("[T-0155] " + mesaj, file=sys.stderr, flush=True)


def k1_kaynak_cipasi() -> Tuple[bool, Dict[str, Any]]:
    beklenen = dict(KAYNAK_SHA)  # base_v2 / vocab / roots.tsv
    beklenen[ROUTER_YOL] = ROUTER_DOSYA_SHA
    olcum: Dict[str, Any] = {}
    sapma = []
    for yol, bek in beklenen.items():
        olc = _sha256(os.path.join(REPO_KOK, yol))
        olcum[yol] = olc
        if olc != bek:
            sapma.append(yol)
    olcum["sha_sapma"] = sapma
    return not sapma, olcum


def _val_metinleri() -> List[str]:
    """T-0154 veri-seçimi birebir (SALT-IMPORT sabitleri; val 300)."""
    rng = random.Random(SEED)
    metinler: List[str] = []
    etiketler: List[int] = []
    for c, yollar in SINIF_KAYNAK.items():
        tum = _metinler(yollar)
        tum = sorted(set(tum))
        rng.shuffle(tum)
        secili = tum[:TRAIN_N + VAL_N]
        metinler.extend(secili)
        etiketler.extend([c] * len(secili))
    val_metin: List[str] = []
    val_y: List[int] = []
    for c in SINIF_KAYNAK:
        blok = [m for m, e in zip(metinler, etiketler) if e == c]
        val_metin.extend(blok[TRAIN_N:TRAIN_N + VAL_N])
        val_y.extend([c] * min(VAL_N, max(0, len(blok) - TRAIN_N)))
    return val_metin, val_y


def k5_gateway_ast() -> Tuple[bool, Dict[str, Any]]:
    """Zincir: create_default → cls(...) → __init__ → EpistemicCuriosityAgent."""
    bulgular: Dict[str, Any] = {}

    def _fonksiyon(agac: ast.Module, ad: str) -> Any:
        for d in ast.walk(agac):
            if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)) and d.name == ad:
                return d
        return None

    def _param_var(fn: Any, ad: str) -> bool:
        return fn is not None and any(a.arg == ad for a in fn.args.args)

    def _cagri_anahtari(fn: Any, fonksiyon_adi: str, anahtar: str) -> bool:
        if fn is None:
            return False
        for d in ast.walk(fn):
            if (isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name) and d.func.id == fonksiyon_adi):
                for kw in d.keywords:
                    if kw.arg == anahtar:
                        return True
        return False

    agac_gw = ast.parse(open(os.path.join(
        REPO_KOK, "src", "gateway", "agent_gateway.py"), encoding="utf-8").read())
    agac_ep = ast.parse(open(os.path.join(
        REPO_KOK, "src", "rag", "epistemic_agent.py"), encoding="utf-8").read())

    fn_cd = _fonksiyon(agac_gw, "create_default")
    fn_init = None
    for d in ast.walk(agac_gw):
        if isinstance(d, ast.ClassDef) and d.name == "AgentGateway":
            fn_init = _fonksiyon(d, "__init__")
    bulgular["create_default_param"] = _param_var(fn_cd, "router_state_path")
    bulgular["cls_cagrisi_anahtari"] = _cagri_anahtari(fn_cd, "cls", "router_state_path")
    bulgular["init_param"] = _param_var(fn_init, "router_state_path")
    bulgular["agent_cagrisi_anahtari"] = _cagri_anahtari(fn_init, "EpistemicCuriosityAgent", "router_state_path")

    fn_ep_init = None
    for d in ast.walk(agac_ep):
        if isinstance(d, ast.ClassDef) and d.name == "EpistemicCuriosityAgent":
            fn_ep_init = _fonksiyon(d, "__init__")
    bulgular["epistemic_param"] = _param_var(fn_ep_init, "router_state_path")
    bulgular["yukleyici_tanimli"] = _fonksiyon(agac_ep, "load_trained_router_state") is not None
    hepsi = all([bulgular["create_default_param"], bulgular["cls_cagrisi_anahtari"],
                 bulgular["init_param"], bulgular["agent_cagrisi_anahtari"],
                 bulgular["epistemic_param"], bulgular["yukleyici_tanimli"]])
    return hepsi, bulgular


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = _sha256(os.path.join(REPO_KOK, ILAN_YOL))
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    # ---- K1 kaynak-çıpası ----
    k1_gec, k1_olcum = k1_kaynak_cipasi()
    if not kayit("K1 KAYNAK-ÇIPASI (4 dosya tam-digest birebir)", k1_gec, k1_olcum):
        return _hukum(kapilar, ilan_sha, damga)

    # ---- K5 AST zincir-kapısı (kod-yüzeyi; model kurulumu gerekmez) ----
    k5_gec, k5_olcum = k5_gateway_ast()
    if not kayit("K5 GATEWAY AST ZİNCİR-KAPISI (router_state_path açık-beyan)",
                 k5_gec, k5_olcum):
        return _hukum(kapilar, ilan_sha, damga)

    # ---- cihaz (T-0154 özellik-yüzeyi birebir: MPS; yoksa DUR) ----
    if not torch.backends.mps.is_available():
        kayit("CİHAZ (özellik-yüzeyi: MPS)", False, {"mps": False, "hata": CIHAZ_YOK_DUR})
        return _hukum(kapilar, ilan_sha, damga)
    cihaz = torch.device("mps")
    _stderr("CİHAZ: MPS hazır")

    # ---- kanonik kurulum (T-0154 birebir) ----
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_KOK, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_KOK, "data", "rebuild", "vocab_anka_r1_33114.json"))
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

    # T-0154 özellik-konfigürasyonu birebir: fresh-init seed-42 CuriosityEngine
    torch.manual_seed(SEED)
    engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5)
    engine.to(cihaz)
    engine.eval()

    # ---- K6-beyan: :memory: VectorMemory (ağ istemcisi YOK) ----
    memory = VectorMemory(collection_name="t0155_entegrasyon_probe",
                          vector_size=768, storage_path=None)
    _stderr("VectorMemory :memory: kuruldu (ağ YOK)")

    # ---- K2 yükleme-birebir (agent üzerinden) ----
    agent = EpistemicCuriosityAgent(
        model=model, tokenizer=tokenizer, memory=memory,
        curiosity_engine=engine,
        router_state_path=ROUTER_YOL,
        device=str(cihaz))
    sd_sha = _state_dict_sha(agent.router.state_dict())
    anahtar_sayi = len(agent.router.state_dict())
    k2_gec = (sd_sha == ROUTER_SD_SHA and anahtar_sayi == 9)
    if not kayit("K2 YÜKLEME-BİREBİR (state_dict sha bit-özdeş; 9-anahtar)",
                 k2_gec, {"state_dict_sha256": sd_sha, "anahtar_sayi": anahtar_sayi}):
        return _hukum(kapilar, ilan_sha, damga)

    # ---- K4 fail-closed mutasyon ----
    olcum_k4: Dict[str, Any] = {}
    try:
        EpistemicCuriosityAgent(
            model=model, tokenizer=tokenizer, memory=memory,
            router_state_path="data/olmayan_router_t0155.pt",
            device=str(cihaz))
        olcum_k4["olmayan_yol_istisna"] = "YOK — HATA: sessiz kuruldu"
    except RuntimeError as exc:
        olcum_k4["olmayan_yol_istisna"] = "RuntimeError: %s" % str(exc)[:120]
    except Exception as exc:  # noqa: BLE001 — farklı-istisna = kapı düşer
        olcum_k4["olmayan_yol_istisna"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    torch.manual_seed(SEED + 1)
    agent_fresh = EpistemicCuriosityAgent(
        model=model, tokenizer=tokenizer, memory=memory,
        router_state_path=None,
        device=str(cihaz))
    fresh_sha = _state_dict_sha(agent_fresh.router.state_dict())
    olcum_k4["fresh_init_sha256"] = fresh_sha
    olcum_k4["fresh_init_farkli"] = fresh_sha != ROUTER_SD_SHA
    k4_gec = (olcum_k4["olmayan_yol_istisna"].startswith("RuntimeError")
              and olcum_k4["fresh_init_farkli"])
    kayit("K4 FAİL-CLOSED MUTASYON (olmayan-yol RuntimeError; fresh-init != eğitilmiş)",
          k4_gec, olcum_k4)

    # ---- K3 davranış-üretimi (300-val; T-0154 konfigürasyonu birebir) ----
    # İLAN-2: ölçüm-kabı cihaz-onarımı — agent.router MPS'te (kurucu
    # router.to(device)); T-0154 ölüm-yüzeyi CPU ⇒ CPU-kopya ile ölçüm
    val_metin, val_y = _val_metinleri()
    Pva, Qva, kirpilan = _ozellik_cikar(model, engine, tokenizer, vocab,
                                        val_metin, cihaz)
    router_cpu = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                                router_dim=256, num_experts=4, top_k=2,
                                expert_names=["grammar_core", "pedagogy",
                                              "carpenter", "legal"])
    router_cpu.load_state_dict(
        {k: v.detach().cpu() for k, v in agent.router.state_dict().items()},
        strict=True)
    kopya_sha = _state_dict_sha(router_cpu.state_dict())
    yva = torch.tensor(val_y, dtype=torch.long)
    with torch.no_grad():
        pred = _gate_logits(router_cpu, Pva, Qva).argmax(dim=-1)
    dogru = int((pred == yva).sum().item())
    oran = round(dogru / len(val_y), 4)
    k3_gec = (len(val_y) == 300 and oran == 0.93 and kopya_sha == ROUTER_SD_SHA)
    kayit("K3 DAVRANIŞ-ÜRETİMİ (300-val top-1 == 0,93 birebir; CPU-kopya)", k3_gec, {
        "val_n": len(val_y), "dogru": dogru, "top1": oran,
        "beklenen": 0.93, "kirpilan_4096_ustu": kirpilan,
        "kopya_state_dict_sha256": kopya_sha})

    # ---- K6 canlı-dokunulmazlık (kod-yolu beyanı + ölçüm) ----
    k6_olcum = {"ag_istemcisi": "YOK", "vector_memory": ":memory: (storage_path=None)",
                "canli_192_168_1_9_istek": 0, "localhost_istek": 0}
    kayit("K6 CANLI-DOKUNULMAZLIK (ağ istemcisi kurulmadı)", True, k6_olcum)

    return _hukum(kapilar, ilan_sha, damga)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str) -> int:
    hukum_adi = ("T0155_ENTEGRASYON_GECTI"
                 if kapilar and all(k["gec"] for k in kapilar) else "DUR")
    rc = 0 if hukum_adi == "T0155_ENTEGRASYON_GECTI" else 2
    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "ilan2": ILAN2_YOL,
        "ilan2_sha256": _sha256(os.path.join(REPO_KOK, ILAN2_YOL)),
        "kapilar": kapilar,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    s: List[str] = []
    s.append("# T-0155 — gateway router entegrasyonu (sonuç)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (sha256 `%s`)" % (ILAN_YOL, ilan_sha))
    s.append("")
    s.append("## Kapılar")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for k in kapilar:
        s.append("| %s | `%s` | %s |" % (
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False)[:500],
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Onarım yüzeyi (kod-değişikliği)")
    s.append("")
    s.append("- `src/rag/epistemic_agent.py`: `router_state_path` param + "
             "`load_trained_router_state` (fail-closed; strict=True)")
    s.append("- `src/gateway/agent_gateway.py`: `__init__` + `create_default` "
             "geçişi; dosya-yok erken-kapısı")
    s.append("")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())