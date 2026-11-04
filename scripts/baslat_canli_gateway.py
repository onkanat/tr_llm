"""Canlı gateway başlatıcı — kalıcı sürüm (T-0160/TUR-X A1; T-0158 heredoc'tan dosyalaşma).

Kod-dışı köprü; üretim-kod DOKUNULMAZ. Operatör kararı: bellek CANLI Qdrant
delinux01.local:6333 (anka_bellek/simulasyon_bellek). ``create_default`` host'u
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
import os
import sys
from pathlib import Path

# scripts/ altından doğrudan çalıştırmada sys.path[0]=scripts/ → repo-kökü eklenmez;
# `import src...` için kökü ekle (başlatma-denemesi-1 ModuleNotFoundError, T-0160 İLAN-2).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.gateway.agent_gateway as ag  # noqa: E402

CANLI_HOST = "delinux01.local"
# P25 TERFİ (2026-11-04, kullanıcı talimatı): router v2 → v3 (legal sınıfı
# 0→60 örnek; hukuk probunda 0/10 → 8/10 legal; diğer 4 sınıf v2-paritesi 1,0;
# hüküm: data/eval/p25_router_v3_hukum_2026-11-04.json — P25_ROUTER_V3_LEGAL_GECTI).
DEFAULT_ROUTER_PATH = "data/anka_router_v3.pt"
# K8-tarifi state_dict SHA çıpası (K4 determinizm kapısındaki değerle birebir):
# 4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5
BEKLENEN_ROUTER_SD_SHA = "4016b210f5a777b0d2daf5ffe4be92d91019b78eaa3709f9c877e2468f1701b5"


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


def build_expert_model_paths() -> dict:
    """Canlı yığın uzman-çözümü (MoE sözlüğü): gardener + carpenter (T-0209,
    DEĞİŞMEZ) + legal (P25 terfi, 2026-11-04 kullanıcı talimatı).

    AÇIK UYARI (hüküm — bu uyarı KAPANMAZ): legal ceketi anka_legal_replay.pt
    P2.5 değerlendirme kapılarından GEÇMEDİ — KALDI 0/3 (ΔCE −0,306/−0,50;
    kanonik-20 0/20; OOD 0,365/0,40; bkz. data/eval/p25_router_v3_hukum_2026-11-04.json
    ve CHANGELOG 1.17.0). Yığına bağlanması KULLANICI TALİMATIDIR; router v3
    kendisi GEÇTİ (8/10 legal, 4 sınıf v2-paritesi 1,0). Ölçülen tuzak: bu
    anahtar olmasa v3 router 'legal'i seçer ama üretim sessizce TABANA düşer
    (epistemic_agent.py:459) — anahtar burada EKLENİR, sessiz düşüş kapanır.
    """
    expert_model_paths = {}
    bahcivan_path = "data/anka_bahcivan.pt"
    if Path(bahcivan_path).exists():
        expert_model_paths["gardener"] = bahcivan_path

    marangoz_path = "data/anka_marangoz.pt"
    if Path(marangoz_path).exists():
        expert_model_paths["carpenter"] = marangoz_path

    legal_path = "data/anka_legal_replay.pt"
    if Path(legal_path).exists():
        expert_model_paths["legal"] = legal_path

    return expert_model_paths


def main():
    ayarla = argparse.ArgumentParser(description="Canlı gateway başlatıcı (delinux01.local köprüsü)")
    ayarla.add_argument("--host", type=str, default="0.0.0.0",
                        help="Dinlenecek ağ arayüzü IP adresi (varsayılan: 0.0.0.0)")
    ayarla.add_argument("--port", type=int, default=8080)
    ayarla.add_argument("--model", type=str, default="data/anka_taban_pure.pt",
                        help="Yüklenecek saf taban model checkpoint yolu (varsayılan: anka_taban_pure.pt)")
    ayarla.add_argument("--router", type=str, default=DEFAULT_ROUTER_PATH,
                        help="Yüklenecek router checkpoint yolu (varsayılan: anka_router_v3.pt — P25 terfi)")
    ayarla.add_argument("--qdrant-url", type=str, default=None,
                        help="Qdrant URL'i (ör. http://192.168.1.9:6333); yoksa QDRANT_URL env, "
                             "o da yoksa eski CanliVectorMemory köprüsü (delinux01.local) geçerli")
    args = ayarla.parse_args()

    # P2.1: create_default artık QDRANT_URL arg/env'ini çözer — sabit-köprü
    # (CanliVectorMemory) yalnız URL verilmediğinde devrede kalır (geriye-uyum).
    cozulen_url = args.qdrant_url or os.environ.get("QDRANT_URL")
    if not cozulen_url:
        ag.VectorMemory = CanliVectorMemory

    # T-0209: Bahçıvan + Marangoz uzman ceketleri (DEĞİŞMEZ) VE P25 terfi:
    # legal uzmanı (anka_legal_replay.pt) MoE sözlüğüne bağlanır — uyarı ve
    # gerekçe build_expert_model_paths() docstring'inde AÇIK tutulur.
    expert_model_paths = build_expert_model_paths()

    gw = ag.AgentGateway.create_default(
        model_path=args.model,
        vocab_path="data/rebuild/vocab_anka_r1_33114.json",
        router_state_path=args.router,
        expert_model_paths=expert_model_paths,
        device="mps",
        qdrant_url=cozulen_url,
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
    # VG-2 gözlem kancası (2026-11-04): YALNIZ gözlern — ask() yanıtlarına
    # sorgu-başına aktif yetenekleri yazar; yönlendirme/üretim DEĞİŞMEZ.
    # Yalnız çözülmüş QDRANT_URL'de denenir (köprü modunda kapalı kalır);
    # yetenek_agi tohumlanmamışsa/sunucu yanıt vermezse KAPALI raporlanır.
    if cozulen_url:
        try:
            from src.rag.yetenek_agi import YetenekAgi
            vg2_hedef = ag.resolve_qdrant_url(cozulen_url) or ("localhost", 6333)
            gw.yetenek_agi = YetenekAgi(host=vg2_hedef[0], port=vg2_hedef[1],
                                        timeout=30.0, tokenizer=gw.tokenizer)
            print(f"[VG2] yetenek-ağı gözlemi AÇIK: {gw.yetenek_agi.kart_sayisi()} kart", flush=True)
        except Exception as e:  # noqa: BLE001 — gözlem kancası başlatmayı DURDURMAZ
            print(f"[VG2] yetenek-ağı gözlemi KAPALI: {type(e).__name__}: {e}", flush=True)
    else:
        print("[VG2] yetenek-ağı gözlemi KAPALI (köprü modunda denenmez)", flush=True)
    srv = gw.create_http_server(args.host, args.port)
    print("[CANLI-GW] http://%s:%d DINLEMEDE — serve_forever" % (args.host, args.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()