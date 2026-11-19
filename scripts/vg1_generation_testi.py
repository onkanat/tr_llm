#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VG-1.2 — DONMUŞ SORGU KOŞULLU ÜRETİM TESTİ (gateway akışı, öncesi/sonrası)
=========================================================================
Canlı gateway yığınını (baslat_canli_gateway.py ile AYNI: model anka_taban_pure,
router v3 + sd-sha çıpası, gardener/carpenter/legal uzmanları, QDRANT_URL)
kurar ve İKİ donmuş hukuk sorgusunu `ask(force_rag=True, temperature=0)` ile
sorar. Telemetri (retrieval belgesi, kaynak koleksiyon, skor, koşullanma,
üretilen metin) JSON'a yazılır — öncesi/sonrası fark tablosunun üretim ayağı.

- EKİM ÖNCESİ koşum: QDRANT_EXTRA_COLLECTIONS AYARLANMAZ → zincir bugünkü.
- EKİM SONRASI koşum: QDRANT_EXTRA_COLLECTIONS=gts_bellek,kanun_madde →
  canlı zincir aux koleksiyonları da sorgular (VG-1 yaması).

Kullanım:
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/vg1_generation_testi.py \
        --label oncesi --out data/eval/vg1_uretim_oncesi.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gateway.agent_gateway import AgentGateway

# DONMUŞ SORGULAR (VG-1.2 — değiştirilmez; vg1_retrieval_olcum.py ile aynı)
DONMUS_SORGULAR = [
    "İş sözleşmesi feshinde ihbar süresi nedir?",
    "4857'de haftalık normal çalışma süresi hangi maddede düzenlenmiştir?",
]

BEKLENEN_ROUTER_SD_SHA = "4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5"


def _sd_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build_expert_model_paths() -> dict:
    expert_model_paths = {}
    for ad, yol in (("gardener", "data/anka_bahcivan.pt"),
                    ("carpenter", "data/anka_marangoz.pt"),
                    ("legal", "data/anka_legal_replay.pt")):
        if Path(yol).exists():
            expert_model_paths[ad] = yol
    return expert_model_paths


def main() -> int:
    ayarla = argparse.ArgumentParser(description="VG-1 donmuş sorgu koşullu üretim testi (gateway akışı)")
    ayarla.add_argument("--label", type=str, required=True)
    ayarla.add_argument("--out", type=str, required=True)
    ayarla.add_argument("--model", type=str, default="data/anka_taban_pure.pt")
    ayarla.add_argument("--router", type=str, default="data/anka_router_v3.pt")
    ayarla.add_argument("--qdrant-url", type=str, default=None)
    ayarla.add_argument("--max-new-tokens", type=int, default=100)
    args = ayarla.parse_args()

    cozulen_url = args.qdrant_url or os.environ.get("QDRANT_URL")
    aux_env = os.environ.get("QDRANT_EXTRA_COLLECTIONS", "")
    print(f"[URETIM] evre: {args.label} | QDRANT_URL={cozulen_url!r} | "
          f"QDRANT_EXTRA_COLLECTIONS={aux_env!r}")

    gw = AgentGateway.create_default(
        model_path=args.model,
        vocab_path="data/rebuild/vocab_anka_r1_33114.json",
        router_state_path=args.router,
        expert_model_paths=build_expert_model_paths(),
        device="mps",
        qdrant_url=cozulen_url,
    )
    sd_sha = _sd_sha(gw.epistemic_agent.router.state_dict())
    sha_uyum = sd_sha == BEKLENEN_ROUTER_SD_SHA
    print(f"[URETIM] router_sd_sha256={sd_sha[:16]}… eşleşme={sha_uyum}")

    rapor = {
        "evre": args.label,
        "zaman_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qdrant_url": cozulen_url,
        "aux_koleksiyonlar_env": aux_env,
        "model": args.model,
        "router": args.router,
        "router_sd_sha_uyum": sha_uyum,
        "max_new_tokens": args.max_new_tokens,
        "sorgular": {},
    }

    for sorgu in DONMUS_SORGULAR:
        t0 = time.time()
        res = gw.ask(
            sorgu,
            instruction="Belgeye göre cevapla.",
            mode="RAG",
            force_rag=True,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
        )
        sure = time.time() - t0
        rapor["sorgular"][sorgu] = {
            "sure_s": round(sure, 1),
            "response_text": res.get("response_text", ""),
            "morphemes": res.get("morphemes", ""),
            "rag_document": res.get("rag_document", ""),
            "source_collection": res.get("source_collection", ""),
            "rag_score": res.get("rag_score", 0.0),
            "conditioned": res.get("conditioned", False),
            "needs_retrieval": res.get("needs_retrieval", False),
            "retrieval_triggered": res.get("retrieval_triggered", False),
            "entropy_pre": res.get("entropy_pre", 0.0),
            "entropy_post": res.get("entropy_post", 0.0),
            "router_experts": res.get("router_experts", []),
            "active_expert": res.get("active_expert", None),
            "epistemic_failure": res.get("epistemic_failure", False),
            "future_train_recorded": res.get("future_train_recorded", False),
        }
        print(f"[URETIM] {sorgu!r} → kaynak={res.get('source_collection')!r} "
              f"skor={res.get('rag_score'):.4f} koşullanma={res.get('conditioned')} ({sure:.1f}s)")
        print(f"  yanıt: {res.get('response_text', '')[:300]!r}")

    gw.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print(f"[URETIM] yazıldı: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
