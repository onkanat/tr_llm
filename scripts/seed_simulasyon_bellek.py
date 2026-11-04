#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P26 — simulasyon_bellek EKİM BETİĞİ (export JSONL → Qdrant geri-yükleme)
========================================================================
data/simulasyon_bellek_export.jsonl'deki 4.073 noktayı (10 Eyl 2026 yedeği)
QDRANT_URL'deki (varsayılan localhost:6333) 'simulasyon_bellek' koleksiyonuna
geri yazar. BU BİR GERİ-YÜKLEME'dir: gömme (re-embed) YOK — export'taki
HAZIR dense/sparse vektörleri ve payload'ları AYNEN upsert edilir.

- Nokta KİMLİĞİ: export kaydında 'id' varsa AYNISI (ölçüldü: 4.073 kayıtta
  da var — int, 1..4073, benzersiz); 'id' yoksa tam-kayıt (kanonik JSON)
  sha256'in 2^63-mod türevi (P2.1 nokta-kimliği kalıbı).
- Koleksiyon şeması: sync_and_backup_simulasyon_bellek.py'deki ORİJİNAL şema —
  adlı 'dense' 768 Cosine + 'sparse' IDF (on_disk_payload true). VectorMemory
  koleksiyonu yoksa birebir bu şemayı yaratır; varsa boyut uyuşmazlığında
  fail-closed (sessiz yıkım YOK). anka_bellek ile aynı dense boyutu/ölçümü.
- IDEMPOTENT: upsert + sabit nokta kimliği → tekrar koşum ÇOĞALTMAZ,
  aynı kayıtları GÜNCELLER; sayı değişmez.
- YAZMA YETKİSİ YALNIZ 'simulasyon_bellek' (SERT KURAL, koleksiyon adı
  betikte SABİTTİR; anka_bellek/muhakeme_bellek DOKUNULMAZ). Betik bu iki
  koleksiyonun nokta sayısını yalnız OKUYUP değişmediğini KANITLAR.

Kullanım:
    venv/bin/python scripts/seed_simulasyon_bellek.py                       # env/default
    QDRANT_URL=http://192.168.1.9:6333 venv/bin/python scripts/seed_simulasyon_bellek.py
    venv/bin/python scripts/seed_simulasyon_bellek.py --url http://192.168.1.9:6333
    venv/bin/python scripts/seed_simulasyon_bellek.py --dry-run             # bağlantısız sayım
    venv/bin/python scripts/seed_simulasyon_bellek.py --limit 64            # smoke/probe
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http import models

from src.rag.vector_memory import VectorMemory, resolve_qdrant_url

VARSAYILAN_URL = "http://localhost:6333"
KOLEKSIYON = "simulasyon_bellek"  # SERT KURAL: yazma yetkisi yalnız bu koleksiyon
DOKUNULMAZ = ["anka_bellek", "muhakeme_bellek"]  # salt-okuma kanıtı; YAZMA YASAK
VEKTOR_BOYUTU = 768  # orijinal şema: dense 768 Cosine (sync betiği, satır 99)
PARCA_BOYU = 64


def nokta_kimligi(rec: dict) -> int:
    """Export kaydında 'id' varsa AYNISI; yoksa tam-kayıt sha256 → 2^63-mod
    (P2.1: birebir aynı satır yeniden eklenirse AYNI noktaya güncelleme)."""
    rid = rec.get("id")
    if rid is not None:
        return int(rid)
    kanonik = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(kanonik.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 63)


def kayitlari_yukle(export_path: str) -> list:
    kayitlar = []
    with open(export_path, "r", encoding="utf-8") as f:
        for i, satir in enumerate(f):
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            vecs = rec.get("vectors") or {}
            dense = vecs.get("dense")
            sparse = vecs.get("sparse")
            if not isinstance(dense, list) or len(dense) != VEKTOR_BOYUTU:
                raise ValueError(
                    f"[EKIM_HATA] satır {i + 1}: dense vektör yok/boyut {len(dense) if isinstance(dense, list) else 'YOK'} ≠ {VEKTOR_BOYUTU}"
                )
            if not (isinstance(sparse, dict) and "indices" in sparse and "values" in sparse):
                raise ValueError(f"[EKIM_HATA] satır {i + 1}: sparse vektör (indices/values) yok")
            if "text" not in (rec.get("payload") or {}):
                raise ValueError(f"[EKIM_HATA] satır {i + 1}: payload.text yok")
            kayitlar.append(rec)
    if not kayitlar:
        raise ValueError(f"[EKIM_HATA] export boş: {export_path}")
    return kayitlar


def sayi_oku(bellek: VectorMemory, koleksiyon: str):
    """Salt-okuma sayacı (yazma YOK). Koleksiyon yoksa None."""
    try:
        if not bellek.client.collection_exists(koleksiyon):
            return None
        return bellek.client.count(collection_name=koleksiyon).count
    except Exception as e:  # noqa: BLE001 — kanıt hücresi, hata raporlanır
        return f"HATA:{type(e).__name__}"


def main() -> int:
    ayarla = argparse.ArgumentParser(
        description="simulasyon_bellek_export.jsonl → simulasyon_bellek ekim betiği "
                    "(idempotent geri-yükleme; gömme YOK, hazır vektörler AYNEN yazılır)")
    ayarla.add_argument("--url", type=str, default=None,
                        help="Qdrant URL'i (ör. http://192.168.1.9:6333); yoksa QDRANT_URL env, o da yoksa localhost:6333")
    ayarla.add_argument("--export", type=str, default="data/simulasyon_bellek_export.jsonl")
    ayarla.add_argument("--limit", type=int, default=None, help="İlk N kayıt (smoke/probe)")
    ayarla.add_argument("--timeout", type=float, default=30.0,
                        help="Uzak Qdrant istemci zaman aşımı saniyesi (varsayılan 30)")
    ayarla.add_argument("--dry-run", action="store_true",
                        help="Bağlantı KURMAZ: kayıt/benzersiz-id sayısını ve şemayı basar")
    args = ayarla.parse_args()

    print(f"[EKIM] koleksiyon: {KOLEKSIYON} (SERT KURAL: yazma yetkisi yalnız bu; "
          f"{', '.join(DOKUNULMAZ)} dokunulmaz)")
    print(f"[EKIM] export: {args.export}")

    kayitlar = kayitlari_yukle(args.export)
    if args.limit:
        kayitlar = kayitlar[: args.limit]
    print(f"[EKIM] export kaydı: {len(kayitlar)}")

    kimlikler = [nokta_kimligi(r) for r in kayitlar]
    benzersiz = len(set(kimlikler))
    print(f"[EKIM] benzersiz nokta kimliği: {benzersiz}/{len(kayitlar)}")
    if benzersiz != len(kayitlar):
        print("[EKIM][UYARI] yinelenik kimlik var — upsert bunları TEK noktada birleştirir")

    if args.dry_run:
        print("[EKIM] --dry-run: bağlantı KURULMADI, yazma YOK")
        for i, rec in enumerate(kayitlar[:3]):
            p = rec["payload"]
            print(f"  örnek[{i}] id={kimlikler[i]} dense={len(rec['vectors']['dense'])} "
                  f"sparse_nnz={len(rec['vectors']['sparse']['indices'])} "
                  f"domain={p.get('domain', p.get('v', '?'))!r} text={p['text'][:50]!r}")
        return 0

    # URL çözümü: --url > QDRANT_URL env > localhost:6333
    hedef = resolve_qdrant_url(args.url)
    if hedef is None:
        hedef = ("localhost", 6333)
    hedef_url = args.url or __import__("os").environ.get("QDRANT_URL", VARSAYILAN_URL)
    print(f"[EKIM] hedef Qdrant: {hedef[0]}:{hedef[1]} (url={hedef_url!r})")

    # Bağlantı: fail-closed (allow_fallback default False). Uzak Qdrant LAN'da
    # aralıklı erişilebilir olabilir (ölçülen kalıp) → KONTROLLÜ yeniden-dene
    # (toplam 3 deneme, 5s aralık); koleksiyon zaten VAR (0 nokta) — __init__
    # yalnız boyut uyuşmazlığında fail-closed olur, yıkım YOK.
    bellek = None
    son_hata = None
    for deneme in range(1, 4):
        try:
            bellek = VectorMemory(
                collection_name=KOLEKSIYON,
                vector_size=VEKTOR_BOYUTU,
                host=hedef[0],
                port=hedef[1],
                timeout=args.timeout,
            )
            break
        except Exception as e:  # noqa: BLE001 — hata mesajı raporlanır
            son_hata = e
            print(f"[EKIM][DENEME {deneme}/3] bağlantı hatası: {type(e).__name__}: {e}")
            if deneme < 3:
                time.sleep(5)
    if bellek is None:
        print(f"[EKIM][HATA] 3 denemede bağlanılamadı: {hedef[0]}:{hedef[1]} — DUR (uydurma yazma YOK)")
        return 1
    print(f"[EKIM] bağlandı: {bellek.storage_type}")

    # Ekim-öncesi durum: hedef + dokunulmaz koleksiyonlar (salt-okuma)
    onceki = bellek.get_document_count()
    dokunulmaz_once = {ad: sayi_oku(bellek, ad) for ad in DOKUNULMAZ}
    print(f"[EKIM] ekim-ÖNCESİ {KOLEKSIYON}: {onceki} nokta")
    print(f"[EKIM] ekim-ÖNCESİ dokunulmaz: {dokunulmaz_once}")

    # Upsert (idempotent): hazır vektörler AYNEN — add_documents_batch payload'a
    # yalnız meta['text'] ekler; export payload'ında text zaten var (aynı değer).
    bas_zaman = time.time()
    toplam_yazilan = 0
    for bas in range(0, len(kayitlar), PARCA_BOYU):
        dilim = kayitlar[bas: bas + PARCA_BOYU]
        bellek.add_documents_batch(
            texts=[r["payload"]["text"] for r in dilim],
            dense_vectors=[r["vectors"]["dense"] for r in dilim],
            sparse_vectors=[models.SparseVector(
                indices=r["vectors"]["sparse"]["indices"],
                values=r["vectors"]["sparse"]["values"],
            ) for r in dilim],
            metadatas=[r["payload"] for r in dilim],
            upsert=True,
            point_ids=[nokta_kimligi(r) for r in dilim],
        )
        toplam_yazilan += len(dilim)
        print(f"  [EKIM] upsert {bas + len(dilim)}/{len(kayitlar)} "
              f"({time.time() - bas_zaman:.1f}s)")
    sure = time.time() - bas_zaman

    # Read-back doğrulama
    sonraki = bellek.get_document_count()
    print(f"[EKIM] ekim-SONRASI nokta sayısı: {sonraki} (öncesi: {onceki}, {sure:.1f}s)")

    # 3 örnek kaydı id ile geri oku — payload'ı export ile birebir karşılaştır
    ornek_yerler = [0, len(kayitlar) // 2, len(kayitlar) - 1]
    ornek_kayitlar = [kayitlar[i] for i in ornek_yerler]
    ornek_idler = [nokta_kimligi(r) for r in ornek_kayitlar]
    geri = bellek.client.retrieve(
        collection_name=KOLEKSIYON, ids=ornek_idler, with_payload=True, with_vectors=False
    )
    geri_map = {p.id: p.payload for p in geri}
    eslesen = 0
    for idx, rec in zip(ornek_idler, ornek_kayitlar):
        p = geri_map.get(idx)
        if p is None:
            print(f"  [KANIT] id={idx} GERİ OKUNAMADI — DOĞRULAMA BAŞARISIZ")
            continue
        ayni = p == rec["payload"]
        eslesen += 1 if ayni else 0
        fark = "" if ayni else f" | FARKLI: {sorted(set(p) ^ set(rec['payload']))}"
        print(f"  [KANIT] id={idx} payload birebir={'EVET' if ayni else 'HAYIR'} "
              f"domain={rec['payload'].get('domain', rec['payload'].get('v', '?'))!r} "
              f"text={rec['payload']['text'][:60]!r}{fark}")

    dokunulmaz_sonra = {ad: sayi_oku(bellek, ad) for ad in DOKUNULMAZ}
    print(f"[EKIM] ekim-SONRASI dokunulmaz: {dokunulmaz_sonra} "
          f"(değişim {'YOK' if dokunulmaz_once == dokunulmaz_sonra else 'VAR!!'})")

    bellek.close()

    if sonraki < benzersiz:
        print(f"[EKIM][HATA] read-back {sonraki} < beklenen {benzersiz} — DOĞRULAMA BAŞARISIZ")
        return 1
    if eslesen != len(ornek_idler):
        print(f"[EKIM][HATA] örnek read-back {eslesen}/{len(ornek_idler)} eşleşti — DOĞRULAMA BAŞARISIZ")
        return 1
    if dokunulmaz_once != dokunulmaz_sonra:
        print("[EKIM][HATA] dokunulmaz koleksiyon sayısı DEĞİŞTİ — KURAL İHLALİ")
        return 1
    print(f"[EKIM] TAMAM: {toplam_yazilan} kayıt upsert ({sure:.1f}s), koleksiyonda {sonraki} nokta "
          f"(≥{benzersiz} doğrulandı), 3/3 örnek payload birebir, dokunulmaz koleksiyonlar değişmedi")
    return 0


if __name__ == "__main__":
    sys.exit(main())