#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-0142 — MİMARİ DOĞRULAMA PAKET-2: RAG Gezgini Dikişi (Qdrant 192.168.1.5:6333)

İLAN: data/eval/mimari_dogrulama_p2_ilan_2026-09-27.md (koşum-ÖNCESİ damgalı).
Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK. rc ∈ {0, 2}.

Operatör kararı (27 Eyl 2026): "Kendi verimizi yükle, mevcutlara dokunma" —
kanonik koleksiyon `kristal_bellek` kanonik VectorMemory ile kurulur;
data/realistic_rag/test_natural_150.jsonl (DONMUŞ salt-okunur kaynak)
benzersiz <BELGE> metinleri kanonik tokenizer ile derlenip arz edilir;
mevcut foreign koleksiyonlar DOKUNULMAZ (yalnız okuma; önce/sonra digest
dokunulmazlık kanıtı).

Kanonik kod IMPORT edilir (kopya YASAK):
  VectorMemory                      (src/rag/vector_memory.py)
  generate_kristal_vector           (src/rag/embedding.py)
  generate_sparse_vector            (src/rag/embedding.py)
  build_query_vectors               (src/rag/rag_pipeline.py)
  is_context_usable / RAG_MATCH_THRESHOLD (src/rag/rag_pipeline.py)
  KristalTokenizer / Vocabulary     (src/llm/tokenizer.py)
  LexiconManager / build_default_graph / CrystalCompiler (src/compiler/*)

Koşum sandbox DIŞI (Qdrant trafiği sandbox proxy'sinden geçmez).
"""

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from qdrant_client import QdrantClient  # noqa: E402

from src.rag.vector_memory import (  # noqa: E402
    VectorMemory,
    generate_kristal_vector,
    generate_sparse_vector,
)
from src.rag.rag_pipeline import (  # noqa: E402
    build_query_vectors,
    is_context_usable,
    RAG_MATCH_THRESHOLD,
)
from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402

# ---- İLAN'lı sabitler (koşum ÖNCESİ sabit; koşum-sonrası yumuşatılmaz) ----
POZITIF_N = 20
SEED = 42
KANONIK_ADLAR = ("kristal_bellek", "simulasyon_bellek")
ILANLI = {
    # İLAN-2 (T-0147 onarım-turu): koşum-1'de P2 'kristal_bellek'i (36 nokta)
    # kurdu — kalıcı arz P3/P4/P5'e geçti (fe7fc764… çıpası). Onarım-turu
    # koşumu PROBE-koleksiyonda (--koleksiyon p2_probe_bellek_t0147);
    # kanonik koleksiyonlar YALNIZ-OKUMA.
    "koleksiyon_sayisi_once": 10,
    "kanonik_ad_mevcut": 1,          # 1/2 — kristal_bellek VAR (36), simulasyon_bellek YOK
    "hibrit_koleksiyon": 1,          # 1/10 — kristal_bellek hibrit (P2 koşum-1 artığı)
    "tek_dense_768": 9,              # 9/10 — kristal_bellek hibrit, tek-dense sayılmaz
    "crystal_tagsli": 1,             # 1/10
    "kristal_bellek_nokta": 36,      # arz-çıpa (yalnız-okuma; dokunulmazlık-çıpası)
    "turk_articles_nokta": 5,
    "elektor_articles_nokta": 90122,
    "foreign_sema_anahtar": 6,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stderr(msg: str) -> None:
    print(f"[T-0142] {msg}", file=sys.stderr)


def _belge_metinleri(korpus_yol: str) -> Tuple[List[str], int]:
    """Korpus input'larındaki benzersiz <BELGE>…</BELGE> metinleri (sıra korunur).

    Döner: (benzersiz belgeler, belge-içermeyen satır sayısı)."""
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


def _envanter(client: QdrantClient) -> Dict[str, Any]:
    """FAZ-A: koleksiyon envanteri (salt-okunur)."""
    adlar = sorted(c.name for c in client.get_collections().collections)
    kalemler: List[Dict[str, Any]] = []
    for ad in adlar:
        info = client.get_collection(ad)
        params = info.config.params
        vectors = params.vectors
        if isinstance(vectors, dict) and "dense" in vectors:
            tek_dense = False
            dense_size = int(vectors["dense"].size)
            dense_dist = str(vectors["dense"].distance)
        else:
            tek_dense = True
            dense_size = int(vectors.size)
            dense_dist = str(vectors.distance)
        sparse_adlar = list(params.sparse_vectors.keys()) if params.sparse_vectors else []
        # payload şema örnekleme (salt-okunur scroll)
        anahtarlar: List[str] = []
        crystal_tags_var = False
        try:
            points, _ = client.scroll(
                collection_name=ad, limit=3, with_payload=True, with_vectors=False)
            for p in points:
                if p.payload:
                    anahtarlar = sorted(p.payload.keys())
                    crystal_tags_var = crystal_tags_var or ("crystal_tags" in p.payload)
                    break
        except Exception as e:  # noqa: BLE001 — envanter istisnası kaydedilir
            anahtarlar = [f"SCROLL_HATA: {type(e).__name__}"]
        kalemler.append({
            "ad": ad,
            "tek_dense": tek_dense,
            "dense_size": dense_size,
            "dense_distance": dense_dist,
            "sparse": sparse_adlar,
            "hibrit": bool(sparse_adlar),
            "nokta": int(info.points_count or 0),
            "payload_anahtar": anahtarlar,
            "crystal_tags": crystal_tags_var,
        })
    return {"adlar": adlar, "kalemler": kalemler}


def _envanter_digest(env: Dict[str, Any]) -> str:
    oz = json.dumps(
        [(k["ad"], k["nokta"]) for k in env["kalemler"]],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(oz.encode("utf-8")).hexdigest()[:16]


def _faz_a_kapilari(env: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """FAZ-A birebirlik kapısı: İLAN'lı değerler ↔ ölçülen."""
    kanonik_mevcut = sum(1 for ad in env["adlar"] if ad in KANONIK_ADLAR)
    hibrit = sum(1 for k in env["kalemler"] if k["hibrit"])
    tek768 = sum(1 for k in env["kalemler"]
                 if k["tek_dense"] and k["dense_size"] == 768 and k["dense_distance"] == "Cosine")
    ctags = sum(1 for k in env["kalemler"] if k["crystal_tags"])
    nokta = {k["ad"]: k["nokta"] for k in env["kalemler"]}
    olculen = {
        "koleksiyon_sayisi_once": len(env["adlar"]),
        "kanonik_ad_mevcut": kanonik_mevcut,
        "hibrit_koleksiyon": hibrit,
        "tek_dense_768": tek768,
        "crystal_tagsli": ctags,
        "kristal_bellek_nokta": nokta.get("kristal_bellek", -1),
        "turk_articles_nokta": nokta.get("türk_articles", -1),
        "elektor_articles_nokta": nokta.get("elektor_articles", -1),
    }
    foreign_sema = -1
    for k in env["kalemler"]:
        if k["ad"] not in KANONIK_ADLAR and k["payload_anahtar"]:
            ilk = k["payload_anahtar"][0]
            if not ilk.startswith("SCROLL_HATA"):
                foreign_sema = len(k["payload_anahtar"])
                break
    olculen["foreign_sema_anahtar"] = foreign_sema
    detay: List[Dict[str, Any]] = []
    tamam = True
    for anahtar, ilanli in ILANLI.items():
        olc = olculen.get(anahtar)
        esit = olc == ilanli
        tamam = tamam and esit
        detay.append({"olcum": anahtar, "ilanli": ilanli,
                      "olculen": olc, "durum": "GEÇTİ" if esit else "SAPMA"})
    return tamam, detay


def _faz_c_envanteri(repo_root: str) -> Dict[str, Any]:
    """FAZ-C: kod-envanteri (koşumsuz — kodda kanıt, satır-referanslı)."""
    bulgular: Dict[str, Any] = {}

    # (a) RAG_MATCH_THRESHOLD tanım+import sitesi sayımı (grep+AST; elle değil)
    tanim_sitesi: List[str] = []
    import_sitesi: List[str] = []
    for dirpath, _, dosyalar in os.walk(os.path.join(repo_root, "src")):
        for d in dosyalar:
            if not d.endswith(".py"):
                continue
            yol = os.path.join(dirpath, d)
            try:
                agac = ast.parse(open(yol, encoding="utf-8").read())
            except Exception:  # noqa: BLE001
                continue
            for dugum in ast.walk(agac):
                if isinstance(dugum, ast.Assign):
                    for hedef in dugum.targets:
                        if isinstance(hedef, ast.Name) and hedef.id == "RAG_MATCH_THRESHOLD":
                            tanim_sitesi.append(os.path.relpath(yol, repo_root))
                if (isinstance(dugum, ast.ImportFrom) and dugum.module
                        and "rag_pipeline" in dugum.module):
                    if any(a.name == "RAG_MATCH_THRESHOLD" for a in dugum.names):
                        import_sitesi.append(os.path.relpath(yol, repo_root))
    bulgular["esik_tanim_sitesi"] = sorted(tanim_sitesi)
    bulgular["esik_import_sitesi"] = sorted(import_sitesi)
    bulgular["esik_tanim_sayisi"] = len(tanim_sitesi)

    # (b) root-filtre/normalizasyon bloğu tekrar sayımı
    #     (inflection-prefix demeti + special-tokens kümesi birlikte geçen dosyalar)
    prefix_pat = re.compile(r'"TENSE_",\s*"PERSON_"')
    blok_dosyalari: List[str] = []
    arama_kokleri = [os.path.join(repo_root, "src"), os.path.join(repo_root, "scripts")]
    for kok_d in arama_kokleri:
        for dirpath, _, dosyalar in os.walk(kok_d):
            for d in dosyalar:
                if not d.endswith(".py"):
                    continue
                yol = os.path.join(dirpath, d)
                metin = open(yol, encoding="utf-8", errors="ignore").read()
                if prefix_pat.search(metin) and '"<INSTRUCTION>"' in metin:
                    blok_dosyalari.append(os.path.relpath(yol, repo_root))
    bulgular["root_filtre_blok_dosyalari"] = sorted(blok_dosyalari)

    # (c) recreate-yıkıcılık + sessiz-None satır-referansları (koşumsuz kod kanıtı)
    vm_yol = os.path.join(repo_root, "src", "rag", "vector_memory.py")
    recreate_satirlar: List[int] = []
    for i, satir in enumerate(open(vm_yol, encoding="utf-8"), 1):
        if "recreate_collection" in satir or "delete_collection" in satir:
            recreate_satirlar.append(i)
    bulgular["recreate_satirlar_vector_memory"] = recreate_satirlar
    rp_yol = os.path.join(repo_root, "src", "rag", "rag_pipeline.py")
    sessiz_satirlar: List[int] = []
    satirlar = open(rp_yol, encoding="utf-8").read().splitlines()
    for i in range(len(satirlar) - 1):
        if "except Exception:" in satirlar[i] and "return None" in satirlar[i + 1]:
            sessiz_satirlar.append(i + 1)
    bulgular["sessiz_none_satirlar_rag_pipeline"] = sessiz_satirlar
    return bulgular


def _hukum_kur(kapilar: Dict[str, bool], detay: Dict[str, Any],
               damga: str) -> Tuple[Dict[str, Any], int]:
    """Hüküm BETİK İÇİNDE — İLAN'lı kural: tüm kapılar GEÇTİ → P2_GECTİ rc=0; aksi DUR rc=2."""
    tamami = all(kapilar.values())
    sonuc = "P2_GECTİ" if tamami else "DUR"
    rc = 0 if tamami else 2
    hukum_json = {
        "hukum": sonuc,
        "rc": rc,
        "damga": damga,
        "kapilar": {k: ("GEÇTİ" if v else "DÜŞTÜ") for k, v in sorted(kapilar.items())},
        "detay": detay,
    }
    return hukum_json, rc


def _hukum_yaz(hukum_json: Dict[str, Any], arg: Any, once_digest: Optional[str]) -> str:
    hukum_yol = arg.rapor.replace("_sonuc_", "_hukum_").replace(".md", ".json")
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    return hukum_yol


def _rapor_yaz(rapor_yol: str, hukum_json: Dict[str, Any], arg: Any,
               ilan_sha: str, korpus_sha: str, hukum_yol: str,
               once_digest: Optional[str], sonra_digest: Optional[str]) -> None:
    """Rapor — İLAN'lı↔ölçülen tablolar; sayılar hüküm JSON'undan."""
    d = hukum_json.get("detay", {})
    with open(rapor_yol, "w", encoding="utf-8") as f:
        f.write("# MİMARİ DOĞRULAMA PAKET-2 SONUÇ — RAG Gezgini (Qdrant) (T-0142)\n\n")
        f.write(f"**Damga:** {_ts()} · **Hüküm (BETİKTEN):** **{hukum_json['hukum']} (rc={hukum_json['rc']})**\n\n")
        f.write("| Kapı | Durum |\n|---|---|\n")
        for k, v in hukum_json.get("kapilar", {}).items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n## FAZ-A — sunucu envanteri (İLAN'lı ↔ ölçülen)\n\n")
        f.write("| Ölçüm | İLAN'lı | Ölçülen | Durum |\n|---|---|---|---|\n")
        for satir in d.get("faz_a_kapi_detay", []):
            f.write(f"| {satir['olcum']} | {satir['ilanli']} | {satir['olculen']} | {satir['durum']} |\n")
        f.write("\n## FAZ-B — kendi-veri arzı + döngü\n\n")
        arz = d.get("arz", {})
        kur = d.get("kurulum", {})
        f.write(f"- B1 kurulum: `{kur}` · arz: **{arz.get('yuklenen')}/{arz.get('benzersiz')}**"
                f" (derleme-boş {arz.get('derleme_bos')})\n")
        poz = d.get("b2_pozitif", {})
        f.write(f"- B2 pozitif kontrol: **{poz.get('gecen', '-')}/{poz.get('n', '-')}** ·"
                f" skor min/med/max: {poz.get('skor_min')}/{poz.get('skor_medyan')}/{poz.get('skor_max')} ·"
                f" eşik-üstü (0,40): {poz.get('kullanilabilir_esik')}\n")
        neg = d.get("b3_negatif", {})
        f.write(f"- B3 OOV negatif kontrol: istisna={neg.get('istisna')} ·"
                f" sonuc_uzunluk={neg.get('sonuc_uzunluk')} · skor={neg.get('skor')} ·"
                f" has_root_match={neg.get('has_root_match')}\n")
        f.write("\n## FAZ-C — kod-envanteri (koşumsuz)\n\n")
        c = d.get("faz_c", {})
        f.write(f"- `RAG_MATCH_THRESHOLD` tanım sitesi: {c.get('esik_tanim_sitesi')}"
                f" · import sitesi: {len(c.get('esik_import_sitesi', []))}\n")
        f.write(f"- root-filtre blok tekrarı: {c.get('root_filtre_blok_dosyalari')}\n")
        f.write(f"- recreate-yıkıcılık satırları (vector_memory.py): {c.get('recreate_satirlar_vector_memory')}\n")
        f.write(f"- sessiz-None satırları (rag_pipeline.py): {c.get('sessiz_none_satirlar_rag_pipeline')}\n")
        dok = d.get("dokunulmazlik", {})
        f.write("\n## Digest tablosu\n\n")
        f.write("| Dosya | sha256 |\n|---|---|\n")
        if ilan_sha:
            f.write(f"| {os.path.basename(arg.ilan)} (İLAN — koşum ÖNCESİ) | `{ilan_sha}` |\n")
        if korpus_sha:
            f.write(f"| {arg.korpus} (DONMUŞ kaynak) | `{korpus_sha}` |\n")
        if once_digest:
            f.write(f"| envanter ÖNCE/SONRA digest | `{once_digest}` / `{sonra_digest}` |\n")
        f.write(f"| dokunulmazlık | {dok.get('durum', '-')} |\n")


def main() -> None:
    ayrici = argparse.ArgumentParser(
        description="T-0142 — P2 RAG Gezgini doğrulama (İLAN'lı, fail-closed)")
    ayrici.add_argument("--host", default="192.168.1.5")
    ayrici.add_argument("--port", type=int, default=6333)
    ayrici.add_argument("--korpus", default="data/realistic_rag/test_natural_150.jsonl")
    ayrici.add_argument("--koleksiyon", default="kristal_bellek")
    ayrici.add_argument("--vocab", required=True,
                        help="Sözlük AÇIKÇA verilir (T-0087 dersi: varsayılan KALDIRILDI)")
    ayrici.add_argument("--ilan", required=True)
    ayrici.add_argument("--rapor", required=True)
    arg = ayrici.parse_args()

    damga = _ts()
    ilan_sha = _sha256(arg.ilan)
    korpus_sha = _sha256(arg.korpus)
    detay: Dict[str, Any] = {
        "damga": damga,
        "ilan": os.path.basename(arg.ilan),
        "ilan_sha256": ilan_sha,
        "korpus": arg.korpus,
        "korpus_sha256": korpus_sha,
        "host": arg.host,
        "port": arg.port,
        "koleksiyon": arg.koleksiyon,
        "esik_rouge_040": RAG_MATCH_THRESHOLD,
    }

    # ---- FAZ-A: sunucu envanteri (salt-okunur; ham client — fail-closed)
    try:
        client = QdrantClient(host=arg.host, port=arg.port, timeout=5.0,
                              check_compatibility=False)
        client.get_collections()
    except Exception as e:  # noqa: BLE001 — sessiz-fallback YOK (vector_memory dersi)
        _stderr(f"DUR: sunucu erişilemez ({type(e).__name__}: {e}); sessiz-fallback YAPILMAZ")
        hukum_json, rc = _hukum_kur({"K1_FAZ_A_BIREBIR": False},
                                    {"sunucu_hatasi": f"{type(e).__name__}: {e}"}, damga)
        _hukum_yaz(hukum_json, arg, None)
        _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
                   "data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json", None, None)
        sys.exit(rc)
    env_once = _envanter(client)
    once_digest = _envanter_digest(env_once)
    detay["faz_a_once"] = env_once
    detay["envanter_once_digest"] = once_digest

    kapilar: Dict[str, bool] = {}
    faz_a_gecti, faz_a_detay = _faz_a_kapilari(env_once)
    kapilar["K1_FAZ_A_BIREBIR"] = faz_a_gecti
    detay["faz_a_kapi_detay"] = faz_a_detay

    # ---- FAZ-C: kod-envanteri (koşumsuz)
    detay["faz_c"] = _faz_c_envanteri(REPO_ROOT)

    # ---- B0: hedef koleksiyon yok teyidi (fail-closed)
    hedef = arg.koleksiyon
    if client.collection_exists(hedef):
        n = client.count(collection_name=hedef).count
        _stderr(f"DUR: '{hedef}' sunucuda ZATEN MEVCUT ({n} nokta); üzerine yazım İLAN'lı YOK")
        detay["b0_mevcut_nokta"] = n
        hukum_json, rc = _hukum_kur({**kapilar, "K2_B0_HEDEF_YOK": False}, detay, damga)
        _hukum_yaz(hukum_json, arg, once_digest)
        _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
                   "data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json", once_digest, None)
        sys.exit(rc)
    kapilar["K2_B0_HEDEF_YOK"] = True

    # ---- B1: kanonik kurulum + arz
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO_ROOT, "data", "lexicon", "roots.tsv"))
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load(arg.vocab)
    tokenizer = KristalTokenizer(compiler, vocab)

    belgeler, belgesiz_satir = _belge_metinleri(arg.korpus)
    detay["benzersiz_belge"] = len(belgeler)
    detay["belgesiz_satir"] = belgesiz_satir

    # Kanonik kurulum: storage_path="" → local-fallback kapatılır (sessiz düşme YOK);
    # uzak erişim ÖNCE ham client ile teyit edildi (yukarıda).
    memory = VectorMemory(collection_name=hedef, vector_size=768,
                          host=arg.host, port=arg.port, storage_path="")
    if memory.is_in_memory or not str(memory.storage_type).startswith("remote"):
        _stderr(f"DUR: VectorMemory uzak sunucuya bağlanamadı"
                f" (storage_type={memory.storage_type}); sessiz-fallback İLAN'lı YASAK")
        hukum_json, rc = _hukum_kur({**kapilar, "K3_B1_KURULUM": False}, detay, damga)
        _hukum_yaz(hukum_json, arg, once_digest)
        _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
                   "data/eval/mimari_dogrulama_p2_hukum_2026-09-27.json", once_digest, None)
        sys.exit(rc)
    coll = client.get_collection(hedef)
    vectors = coll.config.params.vectors
    dense_ok = (isinstance(vectors, dict) and "dense" in vectors
                and int(vectors["dense"].size) == 768
                and str(vectors["dense"].distance) == "Cosine")
    sparse_ok = bool(coll.config.params.sparse_vectors)
    detay["kurulum"] = {"dense_768_cosine": dense_ok, "sparse": sparse_ok,
                        "sparse_adlar": list(coll.config.params.sparse_vectors.keys())
                        if coll.config.params.sparse_vectors else [],
                        "storage_type": memory.storage_type}
    kapilar["K3_B1_KURULUM"] = bool(dense_ok and sparse_ok)

    # arz: kanonik derleme + add_documents_batch (fallback şeması rag_pipeline.py:288-292)
    dense_list: List[List[float]] = []
    sparse_list: List[Any] = []
    meta_list: List[Dict[str, Any]] = []
    derleme_bos = 0
    for belge in belgeler:
        t_ids = tokenizer.encode(belge)
        t_tags = tokenizer.decode(t_ids)
        if not t_ids or not t_tags:
            derleme_bos += 1
            continue
        dense_list.append(generate_kristal_vector(t_ids, t_tags))
        sparse_list.append(generate_sparse_vector(t_ids, t_tags))
        meta_list.append({"domain": "realistic_rag_p2",
                          "crystal_tags": t_tags, "token_ids": t_ids})
    memory.add_documents_batch(belgeler, dense_list, sparse_list, meta_list)
    sayi = memory.get_document_count()
    arz_eksiksiz = (sayi == len(belgeler)) and derleme_bos == 0
    detay["arz"] = {"yuklenen": sayi, "benzersiz": len(belgeler),
                    "derleme_bos": derleme_bos, "eksiksiz": arz_eksiksiz}
    kapilar["K4_B1_ARZ_EKSİKSİZ"] = bool(arz_eksiksiz)

    # ---- B2: pozitif kontrol (İLAN'lı 20/20)
    rng = random.Random(SEED)
    pat = re.compile(r"<BELGE>.*?</BELGE>\s*(.+)\s*$", re.DOTALL)
    sorgular: List[str] = []
    with open(arg.korpus, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir:
                continue
            rec = json.loads(satir)
            m = pat.search(rec.get("input", ""))
            if m and m.group(1).strip():
                sorgular.append(m.group(1).strip())
    secili = sorted(rng.sample(range(len(sorgular)), min(POZITIF_N, len(sorgular))))
    metin_kume = set(belgeler)
    pozitif_gecen = 0
    skorlar: List[float] = []
    kullanilabilir = 0
    pozitif_detay: List[Dict[str, Any]] = []
    for j in secili:
        sorgu = sorgular[j]
        try:
            _, query_tags, dense_vec, sparse_vec = build_query_vectors(sorgu, tokenizer)
            results = memory.hybrid_recall(dense_vec, sparse_vec, top_k=1,
                                           query_tags=query_tags)
        except Exception as e:  # noqa: BLE001 — istisna = kapı düşer
            pozitif_detay.append({"sorgu": sorgu[:60],
                                  "istisna": f"{type(e).__name__}: {e}"})
            continue
        if not results:
            pozitif_detay.append({"sorgu": sorgu[:60], "sonuc": "BOS"})
            continue
        r0 = results[0]
        skor = float(r0.get("score", 0.0))
        skorlar.append(skor)
        if is_context_usable(skor, RAG_MATCH_THRESHOLD):
            kullanilabilir += 1
        kendi = r0.get("text", "") in metin_kume
        kok = bool(r0.get("has_root_match"))
        if kendi and kok:
            pozitif_gecen += 1
        pozitif_detay.append({"sorgu": sorgu[:60], "skor": round(skor, 4),
                              "kendi_belge": kendi, "has_root_match": kok})
    detay["b2_pozitif"] = {
        "n": POZITIF_N, "gecen": pozitif_gecen,
        "skor_min": min(skorlar) if skorlar else None,
        "skor_medyan": sorted(skorlar)[len(skorlar) // 2] if skorlar else None,
        "skor_max": max(skorlar) if skorlar else None,
        "kullanilabilir_esik": f"{kullanilabilir}/{len(skorlar)}",
        "detay": pozitif_detay,
    }
    kapilar["K5_B2_POZITIF_20_20"] = (pozitif_gecen == POZITIF_N
                                      and len(secili) == POZITIF_N)

    # ---- B3: negatif kontrol (OOV sorgu; İLAN'lı: istisna YOK, davranış kaydedilir)
    oov_sorgu = "zzqwxx zqxwv zqqzzq"
    b3: Dict[str, Any] = {"sorgu": oov_sorgu}
    try:
        _, oov_tags, oov_dense, oov_sparse = build_query_vectors(oov_sorgu, tokenizer)
        oov_sonuc = memory.hybrid_recall(oov_dense, oov_sparse, top_k=1,
                                         query_tags=oov_tags)
        b3["istisna"] = False
        b3["tags"] = oov_tags
        b3["sonuc_uzunluk"] = len(oov_sonuc)
        b3["skor"] = round(float(oov_sonuc[0]["score"]), 4) if oov_sonuc else None
        b3["has_root_match"] = bool(oov_sonuc[0].get("has_root_match")) if oov_sonuc else None
    except Exception as e:  # noqa: BLE001
        b3["istisna"] = True
        b3["istisna_metin"] = f"{type(e).__name__}: {e}"
    detay["b3_negatif"] = b3
    # İLAN-2 (T-0147 A-3 onarım): OOV sorgu fail-closed — boş distinctive_roots
    # artık ATLANMAZ; ham 0,7 × 0,05 = 0,035 → normalize 0,0467 (≤ 0,075) VE
    # has_root_match=False. Koşum-1'deki sahte-koşullanma sınıfı (skor 0,7 +
    # koşullanma TRUE) kapanmış OLMAALI — bu beklenti koşum-ÖNCESİ İLAN'lıdır.
    OOV_SKOR_TAVANI = 0.075
    kapilar["K6_B3_DAVRANIS"] = (
        b3.get("istisna") is False
        and b3.get("skor") is not None
        and float(b3["skor"]) <= OOV_SKOR_TAVANI
        and b3.get("has_root_match") is False
    )

    # ---- Dokunulmazlık kapısı (koşum SONU): foreign koleksiyonlar ÖNCE==SONRA
    env_sonra = _envanter(client)
    sonra_digest = _envanter_digest(env_sonra)
    # İLAN-2 (T-0147): onarım-turu koşumu PROBE-koleksiyonda (--koleksiyon);
    # probe koşum-başında YOK (env_once'te derinden yok) → foreign-çıpa
    # hesabı hedef-probe'u DIŞLAR (koşum-1 kanonik-kurulum çıpasıyla aynı
    # foreign-9 kümesi); probe koşum-sonunda BETİKÇE silinir (aşağıda).
    foreign_once = {k["ad"]: k["nokta"] for k in env_once["kalemler"]
                    if k["ad"] not in KANONIK_ADLAR and k["ad"] != arg.koleksiyon}
    foreign_sonra = {k["ad"]: k["nokta"] for k in env_sonra["kalemler"]
                     if k["ad"] not in KANONIK_ADLAR and k["ad"] != arg.koleksiyon}
    dokunulmaz = foreign_once == foreign_sonra
    detay["dokunulmazlik"] = {
        "once_digest": once_digest,
        "sonra_digest": sonra_digest,
        "foreign_once": foreign_once,
        "foreign_sonra": foreign_sonra,
        "yeni_koleksiyon": sorted(set(env_sonra["adlar"]) - set(env_once["adlar"])),
        "durum": "GEÇTİ" if dokunulmaz else "SAPMA",
    }
    kapilar["K7_DOKUNULMAZLIK"] = dokunulmaz
    detay["faz_a_sonra"] = env_sonra

    # ---- Hüküm (betikten) → hüküm JSON + rapor (elle sayı YOK)
    hukum_json, rc = _hukum_kur(kapilar, detay, damga)
    try:
        detay["ilan_mtime"] = os.path.getmtime(arg.ilan)
    except OSError:
        pass
    hukum_yol = arg.rapor.replace("_sonuc_", "_hukum_").replace(".md", ".json")
    with open(hukum_yol, "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    _rapor_yaz(arg.rapor, hukum_json, arg, ilan_sha, korpus_sha,
               hukum_yol, once_digest, sonra_digest)
    # İLAN-2: koşum-sonu PROBE-temizliği (hüküm BETİKTEN çıktıktan SONRA;
    # rc'ye DOKUNMAZ — temizlik-arızası stderr+rapor-beyanıyla görünürlü).
    temizlik: Dict[str, Any] = {"hedef": arg.koleksiyon, "silindi": False}
    try:
        if client.collection_exists(arg.koleksiyon):
            client.delete_collection(arg.koleksiyon)
            temizlik["silindi"] = not client.collection_exists(arg.koleksiyon)
    except Exception as e:  # noqa: BLE001 — temizlik-hatası hükme bağlanmaz, görünür
        temizlik["hata"] = f"{type(e).__name__}: {e}"
    detay["probe_temizlik"] = temizlik
    _stderr(f"probe-temizlik: {temizlik}")
    _stderr(f"HÜKÜM: {hukum_json['hukum']} rc={rc} → {hukum_yol}")
    sys.exit(rc)


if __name__ == "__main__":
    main()