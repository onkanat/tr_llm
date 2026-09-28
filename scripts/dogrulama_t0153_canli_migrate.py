#!/usr/bin/env python
"""T-0153 — canlı-migrate kristal_bellek → anka_bellek (192.168.1.9:6333) hüküm-betiği.

İLAN: data/eval/anka_bellek_canli_migrate_ilan_2026-09-28.md
(koşum-öncesi damgalı, 06:43:18Z; İLAN SABİT — yumuşatma YOK).

HÜKÜM (İLAN kapıları — koşum öncesi sabit):
  K1 B0 + ÇIPA-ŞEMA: anka_bellek YOK; kristal_bellek 36-point green;
     şema: dense 768 Cosine (ad "dense") + sparse "sparse"/idf +
     on_disk_payload=true
  K2 ARŞİV: silme-öncesi 36 nokta tam-arşivi (id+payload+dense+sparse)
     jsonl + SHA-256
  K3 KOPYA-BİREBİR: anka_bellek şema-birebir kurulur (ham client) +
     36 nokta kopyası; doğrulama id-kümesi + payload + dense + sparse
     nokta-başına birebir (36/36)
  K4 SELF-RETRIEVAL: nokta-1 dense sorgusu → top-1 kendisi
  K5 DOKUNULMAZLIK: 9 foreign ad+nokta ÖNCE==SONRA
  K6 ESKİ-SİLME (fail-closed sıra; yalnız K1-K5 sonrası):
     VectorMemory.delete_collection wrapper; sonrası kristal_bellek YOK +
     anka_bellek 36-point green
  6/6 → T0153_MIGRATE_GECTI (rc=0); aksi her dal → DUR (rc=2).
Hüküm BETİKTEN; elle sayı/hüküm YOK. rc ∈ {0, 2}.

Geri-alma (İLAN beyanı): K1-K5 arası başarısızlıkta bu koşumun kurduğu
anka_bellek BETİKÇE silinir (eski koleksiyon DOKUNULMAZ kalır). Silme
(K6) yalnız tüm kapılar geçtikten sonra çalışır.
"""
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from qdrant_client import QdrantClient, models  # noqa: E402

ILAN_YOL = "data/eval/anka_bellek_canli_migrate_ilan_2026-09-28.md"
ESKI_AD = "kristal_bellek"
YENI_AD = "anka_bellek"
SUNUCU = "192.168.1.9"
PORT = 6333
ARSHIV_YOL = "data/eval/anka_bellek_canli_migrate_eski_nokta_arshivi_2026-09-28.jsonl"
HUKUM_YOL = "data/eval/anka_bellek_canli_migrate_hukum_2026-09-28.json"
RAPOR_YOL = "data/eval/anka_bellek_canli_migrate_rapor_2026-09-28.md"

ILANLI_SEMA = {
    "dense_size": 768,
    "dense_distance": "Cosine",
    "dense_ad": "dense",
    "sparse_ad": "sparse",
    "sparse_modifier": "IDF",
    "on_disk_payload": True,
    "nokta": 36,
}


def _stderr(mesaj: str) -> None:
    print("[T-0153] " + mesaj, file=sys.stderr, flush=True)


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _envanter(client: QdrantClient) -> Dict[str, int]:
    return {c.name: int(client.count(collection_name=c.name).count)
            for c in sorted(client.get_collections().collections,
                            key=lambda c: c.name)}


def _nokta_oku(client: QdrantClient, ad: str) -> List[Dict[str, Any]]:
    """Koleksiyondaki TÜM noktalar (id + payload + dense + sparse)."""
    noktalar: List[Dict[str, Any]] = []
    ofset = None
    while True:
        kayitlar, ofset = client.scroll(
            collection_name=ad, limit=100, offset=ofset,
            with_payload=True, with_vectors=True)
        for r in kayitlar:
            dense = list(r.vector["dense"])
            sparse = r.vector["sparse"]
            noktalar.append({
                "id": r.id,
                "payload": r.payload,
                "dense": dense,
                "sparse_indices": list(sparse.indices),
                "sparse_values": list(sparse.values),
            })
        if ofset is None:
            break
    noktalar.sort(key=lambda n: n["id"])
    return noktalar


def _sil_yeni(client: QdrantClient) -> None:
    """Geri-alma: bu koşumun kurduğu anka_bellek'i sil (eski DOKUNULMAZ)."""
    try:
        if client.collection_exists(YENI_AD):
            client.delete_collection(YENI_AD)
            _stderr("geri-alma: anka_bellek silindi (eski koleksiyon dokunulmaz)")
    except Exception as exc:  # noqa: BLE001 — geri-alma hatası görünür
        _stderr("geri-alma HATA: %s: %s" % (type(exc).__name__, exc))


def k1_b0_sema(client: QdrantClient) -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {}
    yeni_var = client.collection_exists(YENI_AD)
    olcum["anka_bellek_var"] = yeni_var
    if yeni_var:
        return False, olcum
    if not client.collection_exists(ESKI_AD):
        olcum["kristal_bellek_var"] = False
        return False, olcum
    bilgi = client.get_collection(ESKI_AD)
    v = bilgi.config.params.vectors
    sparse = bilgi.config.params.sparse_vectors or {}
    olcum.update({
        "status": str(bilgi.status),
        "nokta": int(bilgi.points_count or 0),
        "dense_ad": "dense" if isinstance(v, dict) and "dense" in v else "?",
        "dense_size": int(v["dense"].size) if isinstance(v, dict) and "dense" in v else -1,
        "dense_distance": str(v["dense"].distance) if isinstance(v, dict) and "dense" in v else "?",
        "sparse_adlar": list(sparse.keys()),
        "sparse_modifier": str(list(sparse.values())[0].modifier) if sparse else None,
        "on_disk_payload": bool(bilgi.config.params.on_disk_payload),
    })
    gec = ("GREEN" in str(olcum["status"]).upper()
           and olcum["nokta"] == ILANLI_SEMA["nokta"]
           and olcum["dense_ad"] == ILANLI_SEMA["dense_ad"]
           and olcum["dense_size"] == ILANLI_SEMA["dense_size"]
           and "COSINE" in olcum["dense_distance"].upper()
           and olcum["sparse_adlar"] == [ILANLI_SEMA["sparse_ad"]]
           and "IDF" in str(olcum["sparse_modifier"]).upper()
           and olcum["on_disk_payload"] == ILANLI_SEMA["on_disk_payload"])
    return gec, olcum


def k2_arshiv(noktalar: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    yol = os.path.join(REPO_KOK, ARSHIV_YOL)
    with open(yol, "w", encoding="utf-8") as f:
        for n in noktalar:
            f.write(json.dumps(n, ensure_ascii=False, sort_keys=True) + "\n")
    return len(noktalar) == ILANLI_SEMA["nokta"], {
        "dosya": ARSHIV_YOL, "nokta": len(noktalar), "sha256": _sha256(yol)}


def k3_kopya(client: QdrantClient, eski: List[Dict[str, Any]]
             ) -> Tuple[bool, Dict[str, Any]]:
    # şema-birebir kurulum (ham client; VM kurucu on_disk_payload ayarlamaz)
    client.create_collection(
        collection_name=YENI_AD,
        vectors_config={
            ILANLI_SEMA["dense_ad"]: models.VectorParams(
                size=ILANLI_SEMA["dense_size"],
                distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            ILANLI_SEMA["sparse_ad"]: models.SparseVectorParams(
                modifier=models.Modifier.IDF),
        },
        on_disk_payload=True,
    )
    points = [models.PointStruct(
        id=n["id"], payload=n["payload"],
        vector={
            ILANLI_SEMA["dense_ad"]: n["dense"],
            ILANLI_SEMA["sparse_ad"]: models.SparseVector(
                indices=n["sparse_indices"], values=n["sparse_values"]),
        }) for n in eski]
    client.upsert(collection_name=YENI_AD, points=points, wait=True)
    # doğrulama: yeni koleksiyon yeniden-okunur, birebir karşılaştırılır
    yeni = _nokta_oku(client, YENI_AD)
    olcum: Dict[str, Any] = {"eski_n": len(eski), "yeni_n": len(yeni)}
    if len(yeni) != len(eski):
        return False, olcum
    eski_map = {n["id"]: n for n in eski}
    uyusmayan: List[Any] = []
    for n in yeni:
        e = eski_map.get(n["id"])
        if e is None:
            uyusmayan.append({"id": n["id"], "sebep": "id_yok_eskide"})
            continue
        if e["payload"] != n["payload"]:
            uyusmayan.append({"id": n["id"], "sebep": "payload"})
        if e["dense"] != n["dense"]:
            uyusmayan.append({"id": n["id"], "sebep": "dense"})
        if e["sparse_indices"] != n["sparse_indices"] or e["sparse_values"] != n["sparse_values"]:
            uyusmayan.append({"id": n["id"], "sebep": "sparse"})
    olcum["uyusmayan"] = uyusmayan
    return not uyusmayan, olcum


def k4_self_retrieval(client: QdrantClient, eski: List[Dict[str, Any]]
                      ) -> Tuple[bool, Dict[str, Any]]:
    hedef = min(eski, key=lambda n: n["id"])
    sonuc = client.query_points(
        collection_name=YENI_AD, query=hedef["dense"],
        using=ILANLI_SEMA["dense_ad"], limit=1, with_payload=False)
    idler = [p.id for p in sonuc.points]
    return idler == [hedef["id"]], {"sorgu_id": hedef["id"], "top1": idler}


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_sha = _sha256(os.path.join(REPO_KOK, ILAN_YOL))
    _stderr("İLAN sha256=%s damga=%s" % (ilan_sha, damga))

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Any) -> bool:
        kapilar.append({"ayrac": ayrac, "olcum": olcum, "gec": bool(gec)})
        _stderr("%s → %s" % (ayrac, "GEÇTİ" if gec else "DÜŞTÜ"))
        return bool(gec)

    client: Any = None
    try:
        client = QdrantClient(host=SUNUCU, port=PORT, timeout=10.0,
                              check_compatibility=False)
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 — fail-closed
        kayit("K1 B0 + ÇIPA-ŞEMA", False, {"sunucu_hatasi": "%s: %s" % (type(exc).__name__, exc)})
        return _hukum(kapilar, ilan_sha, damga, client)

    env_once = _envanter(client)

    # K1
    k1_gec, k1_olcum = k1_b0_sema(client)
    if not kayit("K1 B0 + ÇIPA-ŞEMA", k1_gec, k1_olcum):
        return _hukum(kapilar, ilan_sha, damga, client)

    # K2 (arşiv; eski koleksiyon salt-okuma)
    eski = _nokta_oku(client, ESKI_AD)
    k2_gec, k2_olcum = k2_arshiv(eski)
    if not kayit("K2 ARŞİV", k2_gec, k2_olcum):
        return _hukum(kapilar, ilan_sha, damga, client)

    # K3 (kurulum + kopya + birebir; başarısızlıkta geri-alma)
    try:
        k3_gec, k3_olcum = k3_kopya(client, eski)
    except Exception as exc:  # noqa: BLE001
        k3_gec, k3_olcum = False, {"istisna": "%s: %s" % (type(exc).__name__, exc)}
    if not kayit("K3 KOPYA-BİREBİR (36/36)", k3_gec, k3_olcum):
        _sil_yeni(client)
        return _hukum(kapilar, ilan_sha, damga, client)

    # K4
    try:
        k4_gec, k4_olcum = k4_self_retrieval(client, eski)
    except Exception as exc:  # noqa: BLE001
        k4_gec, k4_olcum = False, {"istisna": "%s: %s" % (type(exc).__name__, exc)}
    if not kayit("K4 SELF-RETRIEVAL", k4_gec, k4_olcum):
        _sil_yeni(client)
        return _hukum(kapilar, ilan_sha, damga, client)

    # K5 (foreign dokunulmazlık)
    env_sonra = _envanter(client)
    foreign_once = {a: n for a, n in env_once.items() if a not in (ESKI_AD, YENI_AD)}
    foreign_sonra = {a: n for a, n in env_sonra.items() if a not in (ESKI_AD, YENI_AD)}
    k5_gec = foreign_once == foreign_sonra
    kayit("K5 DOKUNULMAZLIK (9 foreign SABİT)", k5_gec,
          {"once": foreign_once, "sonra": foreign_sonra})

    # K6 (fail-closed sıra: yalnız K1-K5 TAMAM ise)
    k6_gec, k6_olcum = False, {}
    if all(k["gec"] for k in kapilar):
        try:
            from src.rag.vector_memory import VectorMemory
            vm = VectorMemory(collection_name=YENI_AD, vector_size=768,
                              host=SUNUCU, port=PORT, storage_path="")
            silindi = vm.delete_collection(ESKI_AD)
            vm.close()
            eski_yok = not client.collection_exists(ESKI_AD)
            yeni_bilgi = client.get_collection(YENI_AD)
            k6_olcum = {
                "wrapper_sildi": bool(silindi), "kristal_bellek_yok": eski_yok,
                "anka_status": str(yeni_bilgi.status),
                "anka_nokta": int(yeni_bilgi.points_count or 0),
            }
            k6_gec = bool(silindi and eski_yok
                          and "GREEN" in str(yeni_bilgi.status).upper()
                          and int(yeni_bilgi.points_count or 0) == ILANLI_SEMA["nokta"])
        except Exception as exc:  # noqa: BLE001
            k6_olcum = {"istisna": "%s: %s" % (type(exc).__name__, exc)}
    else:
        k6_olcum = {"atlandi": "K1-K5 tamam DEĞİL — eski koleksiyon DOKUNULMAZ"}
    kayit("K6 ESKİ-SİLME (wrapper; fail-closed)", k6_gec, k6_olcum)

    return _hukum(kapilar, ilan_sha, damga, client)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str,
           client: Any) -> int:
    hukum_adi = ("T0153_MIGRATE_GECTI" if kapilar and all(k["gec"] for k in kapilar)
                else "DUR")
    rc = 0 if hukum_adi == "T0153_MIGRATE_GECTI" else 2
    son_env = _envanter(client) if client is not None else {}

    hukum_json = {
        "hukum": hukum_adi, "rc": rc, "damga": damga,
        "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
        "ilan": ILAN_YOL, "ilan_sha256": ilan_sha,
        "kapilar": kapilar,
        "son_envanter": son_env,
    }
    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    s: List[str] = []
    s.append("# T-0153 — canlı-migrate kristal_bellek → anka_bellek (sonuç)")
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
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False)[:600],
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Son canlı envanter (koleksiyon → nokta)")
    s.append("")
    s.append("| Koleksiyon | Nokta |")
    s.append("|---|---|")
    for ad, n in sorted(son_env.items()):
        s.append("| %s | %d |" % (ad, n))
    s.append("")
    s.append("## Arşiv")
    s.append("")
    s.append("Silme-öncesi 36 nokta tam-arşivi: `%s` (sha256 hüküm-JSON K2 içinde)."
             % ARSHIV_YOL)
    s.append("")
    s.append("## Kaynak-düzeltmesi (İLAN'da koşum-öncesi beyanlı)")
    s.append("")
    s.append("RAPOR2 §7'nin `data/pedagogy_canonical/**` kaynak-beyanı yanlıştı;")
    s.append("gerçek kaynak P2'de `data/realistic_rag/test_natural_150.jsonl` idi.")
    s.append("Yöntem nokta-kopyasıdır (T-0150 core.py onarımı sonrası")
    s.append("yeniden-derleme sapma riski; birebirlik K3'te ölçüldü).")
    s.append("")
    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())