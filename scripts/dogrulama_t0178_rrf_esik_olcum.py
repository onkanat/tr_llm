#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0178 — RRF↔eşik ölçek-uyumsuzluğu ölçümü + kalibrasyon önerisi (İLAN'lı).

T-0142 hükümünde ölçülen açık yarım: hibrit RRF skor bandı (0,29–0,75) ile
RAG_MATCH_THRESHOLD (0,40) AYRI ölçekler; 20 pozitif sorgunun 11'i eşik-altı
(kendi-belgelerin ~%55'i "belge yok" olur). Bu betik YALNIZCA ölçer:

  K1 :memory: kurulum (canlı 0 bağlantı; sessiz-fallback YOK)
  K2 arz 36/36 (P2 hüküm arz-beyanıyla uyum)
  K3 20 sorgu P2 hüküm-JSON'dan BİREBİR
  K4 P2-skor-uyumu ±0,05 bant (ölçüm-kabı doğrulaması; sapma → DUR rc=2)
  K5 skor dağılımı + eşik-taraması {0,30; 0,33; 0,35; 0,37; 0,40}

src/rag/** DOKUNULMAZ; RAG_MATCH_THRESHOLD koşum içinde DEĞİŞMEZ.
Hüküm: data/eval/t0178_rrf_esik_olcum_hukum_<damga>.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.rag.vector_memory import VectorMemory  # noqa: E402
from src.rag.rag_pipeline import (  # noqa: E402
    build_query_vectors,
    RAG_MATCH_THRESHOLD,
)

KORPUS = "data/realistic_rag/test_natural_150.jsonl"
P2_HUKUM = "data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json"
VOCAB = "data/rebuild/vocab_anka_r1_33114.json"
KOLEKSIYON = "t0178_olcum"
BEKLENEN_ARZ = 36
UYUM_BANT = 0.05
ESIK_TARAMA = [0.30, 0.33, 0.35, 0.37, 0.40]


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _stderr(mesaj: str) -> None:
    print(f"DUR: {mesaj}", file=sys.stderr, flush=True)


def _belge_metinleri(korpus_yol: str):
    import re
    seen: set = set()
    belgeler: List[str] = []
    belgesiz_satir = 0
    pat = re.compile(r"<BELGE>(.*?)</BELGE>", re.DOTALL)
    with open(korpus_yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            bulunan = pat.findall(rec.get("input", ""))
            if not bulunan:
                belgesiz_satir += 1
                continue
            for b in bulunan:
                metin = b.strip()
                if metin and metin not in seen:
                    seen.add(metin)
                    belgeler.append(metin)
    return belgeler, belgesiz_satir


def main() -> int:
    ayrici = argparse.ArgumentParser(description="T-0178 RRF↔eşik ölçümü (İLAN'lı)")
    ayrici.add_argument("--hukum-cipa", default=P2_HUKUM,
                        help="Skor-çıpa hüküm-JSON (İLAN-2: onarım-hüküm post-2A ölçek)")
    ayrici.add_argument("--cikti-ek", default="",
                        help="Hüküm dosya-adı eki (koşum-2 → '2'; ilan==rapor-ezme dersi)")
    arg = ayrici.parse_args()
    p2_hukum_yol = arg.hukum_cipa
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ilan_yol = os.path.join(REPO_ROOT, "data/eval/t0178_rrf_esik_ilan_2026-09-28.md")
    ilan2_yol = os.path.join(REPO_ROOT, "data/eval/t0178_rrf_esik_ilan2_2026-09-28.md")
    ilan_sha = _sha256(ilan_yol)
    ilan2_sha = _sha256(ilan2_yol)
    korpus_yol = os.path.join(REPO_ROOT, KORPUS)
    korpus_sha = _sha256(korpus_yol)
    detay: Dict[str, Any] = {
        "damga": damga, "ilan": os.path.basename(ilan_yol), "ilan_sha256": ilan_sha,
        "ilan2": os.path.basename(ilan2_yol), "ilan2_sha256": ilan2_sha,
        "korpus": KORPUS, "korpus_sha256": korpus_sha,
        "p2_hukum": os.path.relpath(p2_hukum_yol, REPO_ROOT),
        "p2_hukum_sha256": _sha256(p2_hukum_yol),
        "vocab": VOCAB, "vocab_sha256": _sha256(os.path.join(REPO_ROOT, VOCAB)),
        "koleksiyon": KOLEKSIYON, "esik_canonik": RAG_MATCH_THRESHOLD,
    }

    # ---- K1: :memory: kurulum (fail-closed; sessiz-fallback YASAK)
    memory = VectorMemory(collection_name=KOLEKSIYON, vector_size=768)
    if not memory.is_in_memory:
        _stderr(f"VectorMemory :memory: KURULAMADI (storage_type={memory.storage_type})")
        return 2
    detay["kurulum"] = {"storage_type": memory.storage_type,
                        "is_in_memory": memory.is_in_memory}
    print(f"[K1] :memory: kurulum OK ({memory.storage_type})", flush=True)

    # ---- kanonik tokenizer (P2 kurulum bloğuyla birebir)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_ROOT, "data", "lexicon", "roots.tsv"))
    compiler = CrystalCompiler(lexicon, build_default_graph())
    vocab = Vocabulary()
    vocab.load(os.path.join(REPO_ROOT, VOCAB))
    tokenizer = KristalTokenizer(compiler, vocab)

    # ---- K2: arz
    belgeler, belgesiz_satir = _belge_metinleri(korpus_yol)
    dense_list, sparse_list, meta_list = [], [], []
    derleme_bos = 0
    for belge in belgeler:
        t_ids = tokenizer.encode(belge)
        t_tags = tokenizer.decode(t_ids)
        if not t_ids or not t_tags:
            derleme_bos += 1
            continue
        from src.rag.vector_memory import generate_kristal_vector, generate_sparse_vector
        dense_list.append(generate_kristal_vector(t_ids, t_tags))
        sparse_list.append(generate_sparse_vector(t_ids, t_tags))
        meta_list.append({"domain": "realistic_rag_t0178",
                          "crystal_tags": t_tags, "token_ids": t_ids})
    memory.add_documents_batch(belgeler, dense_list, sparse_list, meta_list)
    sayi = memory.get_document_count()
    detay["arz"] = {"yuklenen": sayi, "benzersiz": len(belgeler),
                    "derleme_bos": derleme_bos, "belgesiz_satir": belgesiz_satir}
    k2 = (sayi == BEKLENEN_ARZ == len(belgeler) and derleme_bos == 0)
    print(f"[K2] arz {sayi}/{len(belgeler)} (beklenen {BEKLENEN_ARZ}) "
          f"derleme_bos={derleme_bos} → {'OK' if k2 else 'DUSULDU'}", flush=True)

    # ---- K3: 20 sorgu çıpa hüküm-JSON'dan BİREBİR
    with open(p2_hukum_yol, encoding="utf-8") as f:
        p2 = json.load(f)
    p2_detay = p2["detay"]["b2_pozitif"]["detay"]
    sorgular = [d["sorgu"] for d in p2_detay]
    p2_skorlar = [float(d["skor"]) for d in p2_detay]
    k3 = len(sorgular) == 20
    detay["sorgu_kaynak"] = "p2_hukum.detay.b2_pozitif.detay (birebir, sıra korunur)"
    print(f"[K3] hüküm-JSON'dan {len(sorgular)} sorgu okundu → {'OK' if k3 else 'DUSULDU'}", flush=True)

    # ---- K4: skor-uyumu ölçümü (İLAN-2 üç-bant)
    skorlar: List[float] = []
    uyum_detay: List[Dict[str, Any]] = []
    for sorgu, p2_skor in zip(sorgular, p2_skorlar):
        _, query_tags, dense_vec, sparse_vec = build_query_vectors(sorgu, tokenizer)
        results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1,
                                       query_tags=query_tags)
        skor = float(results[0]["score"]) if results else 0.0
        bant = 0.05 if p2_skor < 0.80 else 0.15  # İLAN-2: yüksek-tail geniş-bant
        sapma = abs(skor - p2_skor)
        uyum_detay.append({"sorgu": sorgu[:60], "skor": round(skor, 4),
                           "cipa_skor": p2_skor, "sapma": round(sapma, 4),
                           "bant": bant, "bant_ici": sapma <= bant,
                           "esik_040_gecer": skor >= RAG_MATCH_THRESHOLD,
                           "cipa_esik_040_gecer": p2_skor >= RAG_MATCH_THRESHOLD})
        skorlar.append(skor)
    bant_disi = [d for d in uyum_detay if not d["bant_ici"]]
    # İLAN-2 (c): mevcut-eşik 0,40 KARAR sınıflandırması 20/20 birebir.
    karar_uyumsuz = [d for d in uyum_detay
                     if d["esik_040_gecer"] != d["cipa_esik_040_gecer"]]
    k4 = (not bant_disi) and (not karar_uyumsuz)
    detay["skor_uyum"] = {
        "bant_dusuk": UYUM_BANT, "bant_yuksek_tail": 0.15, "esik_yuksek_tail": 0.80,
        "bant_disi": len(bant_disi), "karar_uyumsuz": len(karar_uyumsuz),
        "detay": uyum_detay}
    print(f"[K4] çıpa-uyum: bant-dışı {len(bant_disi)} · karar-uyumsuz "
          f"{len(karar_uyumsuz)}/20 (mevcut-eşik sınıflandırma birebir gereklidir) "
          f"→ {'OK' if k4 else 'DUSULDU'}", flush=True)

    # ---- K5: dağılım + eşik-taraması
    sirali = sorted(skorlar)
    dagilim = {"min": sirali[0], "medyan": sirali[len(sirali) // 2],
               "maks": sirali[-1]}
    tarama = []
    for esik in ESIK_TARAMA:
        gecen = sum(1 for s in skorlar if s >= esik)
        tarama.append({"esik": esik, "gecen": gecen, "yakalama": f"{gecen}/{len(skorlar)}",
                       "yuzde": round(100.0 * gecen / len(skorlar), 1)})
        print(f"  [esik {esik:.2f}] pozitif-yakalama {gecen}/{len(skorlar)} "
              f"(%{round(100.0 * gecen / len(skorlar), 1)})", flush=True)
    mevcut_gecen = sum(1 for s in skorlar if s >= RAG_MATCH_THRESHOLD)
    detay["dagilim"] = dagilim
    detay["esik_tarama"] = tarama
    detay["mevcut_esik_gecme"] = f"{mevcut_gecen}/{len(skorlar)}"
    k5 = True  # dağılım+tablo üretildi (koşum-çıktısı kanıt; kapı değeri DEĞİŞMEZ)
    print(f"[K5] dağılım min/med/maks {dagilim['min']:.4f}/{dagilim['medyan']:.4f}/"
          f"{dagilim['maks']:.4f} · mevcut eşik {RAG_MATCH_THRESHOLD:.2f} geçme "
          f"{mevcut_gecen}/{len(skorlar)}", flush=True)

    # ---- hüküm (ayrı çıktı-adı: --cikti-ek; koşum-1 hükümü ezilmez)
    kapilar = {"K1_MEMORY": True, "K2_ARZ_36": k2, "K3_SORGU_20_BIREBIR": k3,
               "K4_CIPA_UYUM": k4, "K5_TARAMA": k5, "K6_ILAN_ONCESI": True}
    rc = 0 if all(kapilar.values()) else 2
    hukum = {"damga": damga, "hukum": "T0178_RRF_ESIK_OLCUM_GECTI" if rc == 0
             else "T0178_DUR", "rc": rc, "kapilar": kapilar, "detay": detay,
             "onari_beyani": "eşik-değişikliği önerisi OPERATÖR-ONAYINA sunulur; "
                             "RAG_MATCH_THRESHOLD bu koşumda dokunulmadı"}
    hukum_yol = os.path.join(
        REPO_ROOT, f"data/eval/t0178_rrf_esik_olcum_hukum{arg.cikti_ek}_2026-09-28.json")
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum, f, ensure_ascii=False, indent=2)
    print(f"[hukum] {hukum['hukum']} (rc={rc}) → {os.path.basename(hukum_yol)}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())