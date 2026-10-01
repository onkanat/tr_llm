"""Canlı gateway başlatıcı — kalıcı sürüm (T-0160/TUR-X A1; T-0158 heredoc'tan dosyalaşma).

Kod-dışı köprü; üretim-kod DOKUNULMAZ. Operatör kararı: bellek CANLI Qdrant
192.168.1.9:6333 (anka_bellek/simulasyon_bellek). ``create_default`` host'u
``localhost`` SABİT yazıyor (agent_gateway.py:147/:148; yerelde 6333 kapalı
→ T-0148 4A fail-closed); modül-düzeyi import olduğundan (:33) çağrı-yüzeyleri
(:147/:148/:334) aynı global'i okur → host'u canlıya çeviren sarmalayıcı yeterli.

Kullanım:
    venv/bin/python scripts/baslat_canli_gateway.py [--port 8080]

DOKUNULMAZ UYARI: /api/inject canlı anka_bellek'e YAZAR (36-point kanonik
koleksiyon) — smoke-test'te kullanılmamalıdır (T-0158).
"""
import argparse
import hashlib
import sys
from pathlib import Path

# scripts/ altından doğrudan çalıştırmada sys.path[0]=scripts/ → repo-kökü eklenmez;
# `import src...` için kökü ekle (başlatma-denemesi-1 ModuleNotFoundError, T-0160 İLAN-2).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.gateway.agent_gateway as ag  # noqa: E402

CANLI_HOST = "192.168.1.9"
BEKLENEN_ROUTER_SD_SHA = "a001207058be949ba5932ada04c5058fd88d99770caff0c2e0accbb291a1c649"


class CanliVectorMemory(ag.VectorMemory):
    """Modül-global sarmalayıcı: localhost sabitini CANLI_HOST'a çevirir."""

    def __init__(self, *args, **kwargs):
        kwargs["host"] = CANLI_HOST
        super().__init__(*args, **kwargs)


def _sd_sha(sd):
    """K8-tarifi (T-0156 betiği:120-128): anahtar-adı + shape + baytlar, sıralı."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    ayarla = argparse.ArgumentParser(description="Canlı gateway başlatıcı (192.168.1.9 köprüsü)")
    ayarla.add_argument("--port", type=int, default=8080)
    ayarla.add_argument("--model", type=str, default="data/anka_t0203_polish.pt",
                        help="Yüklenecek model checkpoint yolu (varsayılan: anka_t0203_polish.pt)")
    ayarla.add_argument("--router", type=str, default="data/anka_router_v2.pt",
                        help="Yüklenecek router checkpoint yolu (varsayılan: anka_router_v2.pt)")
    args = ayarla.parse_args()

    ag.VectorMemory = CanliVectorMemory

    # T-0208: Bahçıvan uzman modelini MoE sözlüğüne bağla
    expert_model_paths = {}
    bahcivan_path = "data/anka_bahcivan.pt"
    if Path(bahcivan_path).exists():
        expert_model_paths["gardener"] = bahcivan_path

    gw = ag.AgentGateway.create_default(
        model_path=args.model,
        vocab_path="data/rebuild/vocab_anka_r1_33114.json",
        router_state_path=args.router,
        expert_model_paths=expert_model_paths,
        device="mps",
    )
    # T-0158 koşum-2 düzeltmesi: router sahibi epistemic_agent (gw.router YOK,
    # agent_gateway.py:62 yalnız self.epistemic_agent).
    sd_sha = _sd_sha(gw.epistemic_agent.router.state_dict())
    print("[CANLI-GW] router_sd_sha256 =", sd_sha, flush=True)
    print("[CANLI-GW] beklenen", BEKLENEN_ROUTER_SD_SHA[:8], "→ EŞLEŞME:", sd_sha == BEKLENEN_ROUTER_SD_SHA, flush=True)
    if sd_sha != BEKLENEN_ROUTER_SD_SHA:
        raise RuntimeError(
            "DURDURULDU: router sd-sha çıpası uymadi (%s != %s); "
            "sessiz-yanlis-agirlik baslatma engellendi (T-0160 fail-closed)" % (sd_sha, BEKLENEN_ROUTER_SD_SHA)
        )
    print("[CANLI-GW] anka_bellek storage:", gw.memory.storage_type, "| koleksiyon:", gw.memory.collection_name, flush=True)
    print("[CANLI-GW] simulasyon_bellek storage:", gw.general_memory.storage_type, flush=True)
    srv = gw.create_http_server("127.0.0.1", args.port)
    print("[CANLI-GW] http://127.0.0.1:%d DINLEMEDE — serve_forever" % args.port, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()