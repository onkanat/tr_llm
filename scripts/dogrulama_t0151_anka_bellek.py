#!/usr/bin/env python
"""T-0151 — kristal_bellek → anka_bellek koleksiyon yeniden-adlandırma hüküm-betiği.

İLAN-2: data/eval/anka_bellek_yeniden_adlandirma_ilan2_2026-09-28.md
(koşum-2; İLAN-1 + koşum-1 artefaktları DOKUNULMAZ — koşum-1 K1'i ölçüm-kabı
kendi betiğini süpürdü; İLAN-2 tek-madde: K1 istisna-beyanı ölçüm-kabı
betiğini de kapsar).

HÜKÜM (İLAN-1 kapıları + İLAN-2 revizyonu — koşum öncesi sabit):
  K1 ESKİ-AD SIFIR: grep kristal_bellek src/+scripts/+tests/ == 0
     (İSTİSNA: dogrulama_p[2345]_*.py çıpa-betikleri + BU BETİK — İLAN-2)
  K2 YENİ-AD ENVANTER: grep -o anka_bellek dosya-kırılımı == İLAN tablosu
     (7 dosya, 36 geçiş)
  K3 STATİK-ÖN: py_compile 7-dosya OK + AST tanımsız-ad 0
  K4 DAVRANIŞ: VectorMemory() default == anka_bellek;
     inject_knowledge/check_memory default == anka_bellek;
     get_status anahtarı anka_bellek_docs; :memory: koleksiyon kurulumu
     (canlı sunucuya YAZMAZ)
  K5 ÇIPA-SABİT: P2/P3/P4/P5 betik-shaları İLAN değerleriyle birebir
  K6 CANLI-DOKUNULMAZLIK: GET 192.168.1.9:6333 kristal_bellek
     points_count==36 SABİT (salt-okuma; sunucu ayakta değilse DUR)
  6/6 → T0151_GECTI (rc=0); aksi her dal → DUR (rc=2).
Hüküm BETİKTEN; elle sayı/hüküm YOK. rc ∈ {0, 2}.
"""
import ast
import builtins
import hashlib
import inspect
import json
import os
import py_compile
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Tuple

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

ILAN_YOL = "data/eval/anka_bellek_yeniden_adlandirma_ilan2_2026-09-28.md"
OLCUM_KABI = "scripts/dogrulama_t0151_anka_bellek.py"  # İLAN-2 istisna-beyanı
SUNUCU = "http://192.168.1.9:6333"

# İLAN tablosu (revizyon-1: grep -o geçiş-sayısı)
BEKLENEN_KIRILIM = {
    "src/rag/vector_memory.py": 1,
    "src/gateway/agent_gateway.py": 9,
    "src/gateway/pedagogical_supervisor.py": 11,
    "scripts/sanitize_vector_memory.py": 9,
    "scripts/sanitize_all_accumulated_datasets.py": 1,
    "scripts/run_agent_arena.py": 2,
    "tests/test_agent_gateway.py": 3,
}
BEKLENEN_TOPLAM = 36
# İLAN revizyon-2 çıpa-valueları (tam digest)
K5_SHA = {
    "scripts/dogrulama_p2_rag_gezgini.py":
        "d0d0b66e0ad0612a2bb4e4f221d65047541e0c1eca12b90d9696b9d46717a77a",
    "scripts/dogrulama_p3_epistemik_kapilar.py":
        "1249a055f5162acb352ae4a9c86334e3d70803f5cf2b3f24c125b3d45d7c8c1a",
    "scripts/dogrulama_p4_universal_hafiza.py":
        "4790e79a51d2e3f3dfb0778cc9e40b786eeb0a155d045bdbc74eb27a5951428c",
    "scripts/dogrulama_p5_ogrenme_kanali.py":
        "aee7596f6074bf9e16358e98b4da4453aede44ac46960759abea409dbb5eb65a",
}


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _stderr(mesaj: str) -> None:
    print("[T0151] " + mesaj, file=sys.stderr, flush=True)


def _grep_say(yol: str, desen: str) -> int:
    with open(yol, "r", encoding="utf-8", errors="replace") as f:
        return len(re.findall(desen, f.read()))


def k1_eski_ad() -> Dict[str, Any]:
    kalan: List[str] = []
    for kok in ("src", "scripts", "tests"):
        for dizin, _, dosyalar in os.walk(os.path.join(REPO_KOK, kok)):
            for d in dosyalar:
                if not d.endswith(".py"):
                    continue
                yol = os.path.join(dizin, d)
                if re.search(r"dogrulama_p[2345]_", d):
                    continue  # İLAN-1 İSTİSNA beyanı — geçmiş-çıpa betikleri
                if os.path.relpath(yol, REPO_KOK) == OLCUM_KABI:
                    continue  # İLAN-2 İSTİSNA beyanı — ölçüm-kabı betiğin kendisi
                if _grep_say(yol, "kristal_bellek") > 0:
                    kalan.append(os.path.relpath(yol, REPO_KOK))
    return {"kalan": kalan, "gec": not kalan}


def k2_yeni_ad() -> Tuple[bool, Dict[str, int]]:
    kirilim: Dict[str, int] = {}
    toplam = 0
    for rel, beklenen in BEKLENEN_KIRILIM.items():
        n = _grep_say(os.path.join(REPO_KOK, rel), "anka_bellek")
        kirilim[rel] = n
        toplam += n
        if n != beklenen:
            _stderr("K2 kırılım sapması: %s %d != %d" % (rel, n, beklenen))
    gec = kirilim == BEKLENEN_KIRILIM and toplam == BEKLENEN_TOPLAM
    return gec, kirilim


def k3_statik_on() -> Tuple[bool, List[str]]:
    sorunlar: List[str] = []
    for rel in BEKLENEN_KIRILIM:
        yol = os.path.join(REPO_KOK, rel)
        try:
            py_compile.compile(yol, doraise=True)
        except Exception as exc:
            sorunlar.append("py_compile %s: %s" % (rel, exc))
    # AST tanımsız-ad (iki-geçişli; Python 3.14 comprehension.target)
    for rel in BEKLENEN_KIRILIM:
        yol = os.path.join(REPO_KOK, rel)
        agac = ast.parse(open(yol, encoding="utf-8").read())
        baglanti = set(dir(builtins))

        class C(ast.NodeVisitor):
            def visit_Import(self, n):
                for a in n.names:
                    baglanti.add(a.asname or a.name.split(".")[0])

            def visit_ImportFrom(self, n):
                for a in n.names:
                    baglanti.add(a.asname or a.name)

            def visit_FunctionDef(self, n):
                baglanti.add(n.name)
                for a in n.args.args + n.args.kwonlyargs:
                    baglanti.add(a.arg)
                if n.args.vararg:
                    baglanti.add(n.args.vararg.arg)
                if n.args.kwarg:
                    baglanti.add(n.args.kwarg.arg)
                self.generic_visit(n)

            def visit_ClassDef(self, n):
                baglanti.add(n.name)
                self.generic_visit(n)

            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Store):
                    baglanti.add(n.id)
                self.generic_visit(n)

            def visit_ExceptHandler(self, n):
                if n.name:
                    baglanti.add(n.name)
                self.generic_visit(n)

            def visit_Lambda(self, n):
                for a in n.args.args:
                    baglanti.add(a.arg)
                self.generic_visit(n)

            def visit_comprehension(self, n):
                for x in ast.walk(n.target):
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                        baglanti.add(x.id)
                self.generic_visit(n)

        C().visit(agac)
        yukleme = {n.id for n in ast.walk(agac)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        hatalar = sorted(y for y in yukleme if y not in baglanti
                         and not (y.startswith("__") and y.endswith("__")))
        if hatalar:
            sorunlar.append("AST tanımsız-ad %s: %s" % (rel, hatalar))
    return (not sorunlar), sorunlar


def k4_davranis() -> Tuple[bool, Dict[str, Any]]:
    from src.rag.vector_memory import VectorMemory
    from src.gateway.agent_gateway import AgentGateway
    olcum: Dict[str, Any] = {}
    vm = VectorMemory()  # host'suz/storage'suz → :memory: (canlıya YAZMAZ)
    olcum["vector_memory_default"] = vm.collection_name
    olcum["is_in_memory"] = vm.is_in_memory
    inj_default = inspect.signature(AgentGateway.inject_knowledge).parameters[
        "target_collection"].default
    chk_default = inspect.signature(AgentGateway.check_memory).parameters[
        "target_collection"].default
    olcum["inject_default"] = inj_default
    olcum["check_default"] = chk_default
    # davranış: _resolve_target_memory üç-dallı (fail-closed) — mock-memory'lerle
    from src.rag.vector_memory import VectorMemory as VM
    m1 = VM.__new__(VM)
    m1.collection_name = "anka_bellek"
    m2 = VM.__new__(VM)
    m2.collection_name = "simulasyon_bellek"
    gw = AgentGateway.__new__(AgentGateway)
    gw.memory = m1
    gw.general_memory = m2
    cozulen = gw._resolve_target_memory("anka_bellek").collection_name
    olcum["resolve_anka"] = cozulen
    istisna = None
    try:
        gw._resolve_target_memory("olmayan_ad_t0151")
    except ValueError as exc:
        istisna = "ValueError"
    olcum["resolve_bilinmeyen_istisna"] = istisna
    # get_status anahtar-yüzeyi (memory yoksa 0-dönmeli)
    gw.memory = None
    gw.general_memory = None
    gw.device = "cpu"  # get_status str(self.device) okur — mock yüzeyi
    gw.future_train_path = "data/future_train_vector.jsonl"
    status = gw.get_status()
    olcum["status_anahtarlar"] = sorted(status.keys())
    olcum["status_anka_anahtar"] = "anka_bellek_docs" in status
    vm.close()
    gec = (olcum["vector_memory_default"] == "anka_bellek"
           and inj_default == "anka_bellek" and chk_default == "anka_bellek"
           and cozulen == "anka_bellek" and istisna == "ValueError"
           and olcum["status_anka_anahtar"])
    return gec, olcum


def k5_cipa() -> Tuple[bool, Dict[str, str]]:
    olcum: Dict[str, str] = {}
    sapma: List[str] = []
    for rel, beklenen in K5_SHA.items():
        olc = _sha256(os.path.join(REPO_KOK, rel))
        olcum[rel] = olc
        if olc != beklenen:
            sapma.append(rel)
    return (not sapma), olcum


def k6_canli() -> Tuple[bool, Dict[str, Any]]:
    olcum: Dict[str, Any] = {"sunucu": SUNUCU}
    try:
        with urllib.request.urlopen(SUNUCU + "/collections/kristal_bellek",
                                   timeout=10) as cevap:
            veri = json.loads(cevap.read().decode("utf-8"))
        sonuc = veri.get("result", {})
        olcum["status"] = sonuc.get("status")
        olcum["points_count"] = sonuc.get("points_count")
        olcum["indexed"] = sonuc.get("indexed_vectors_count")
        gec = (sonuc.get("points_count") == 36
               and sonuc.get("indexed_vectors_count") == 36)
    except Exception as exc:
        olcum["hata"] = type(exc).__name__ + ": " + str(exc)
        gec = False
    return gec, olcum


def main() -> int:
    ilan_sha = _sha256(os.path.join(REPO_KOK, ILAN_YOL))
    _stderr("İLAN-1 sha256=%s" % ilan_sha)

    kapilar: List[Dict[str, Any]] = []
    k1 = k1_eski_ad()
    kapilar.append({"ayrac": "K1 ESKİ-AD SIFIR (İSTİSNA dogrulama_p*)",
                    "olcum": k1["kalan"], "gec": k1["gec"]})
    k2_gec, k2_olcum = k2_yeni_ad()
    kapilar.append({"ayrac": "K2 YENİ-AD ENVANTER == İLAN (36 geçiş)",
                    "olcum": k2_olcum, "gec": k2_gec})
    k3_gec, k3_olcum = k3_statik_on()
    kapilar.append({"ayrac": "K3 STATİK-ÖN: py_compile + AST tanımsız-ad 0",
                    "olcum": k3_olcum, "gec": k3_gec})
    k4_gec, k4_olcum = k4_davranis()
    kapilar.append({"ayrac": "K4 DAVRANIŞ: default'lar + resolve + status "
                            "(:memory:, canlıya yazım YOK)",
                    "olcum": k4_olcum, "gec": k4_gec})
    k5_gec, k5_olcum = k5_cipa()
    kapilar.append({"ayrac": "K5 ÇIPA-SABİT: P2-P5 betik-shaları birebir",
                    "olcum": {k: v[:16] + "…" for k, v in k5_olcum.items()},
                    "gec": k5_gec})
    k6_gec, k6_olcum = k6_canli()
    kapilar.append({"ayrac": "K6 CANLI-DOKUNULMAZLIK: kristal_bellek 36 SABİT",
                    "olcum": k6_olcum, "gec": k6_gec})

    hukum_adi = "T0151_GECTI" if all(k["gec"] for k in kapilar) else "DUR"
    rc = 0 if hukum_adi == "T0151_GECTI" else 2

    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    hukum_json = ("data/eval/anka_bellek_yeniden_adlandirma_hukum2_"
                   "2026-09-28.json")
    with open(os.path.join(REPO_KOK, hukum_json), "w", encoding="utf-8") as f:
        json.dump({"hukum": hukum_adi, "rc": rc, "kapilar": kapilar,
                   "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
                   "ilan": ILAN_YOL, "ilan_sha256": ilan_sha, "damga": damga},
                  f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, hukum_json))

    rapor = ("data/eval/anka_bellek_yeniden_adlandirma_rapor2_"
             "2026-09-28.md")
    s: List[str] = []
    s.append("# T-0151 — anka_bellek yeniden-adlandırma koşumu (sonuç)")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC, `time.gmtime`) — koşum sonu" % damga)
    s.append("**İlan:** `%s` (sha256 `%s`)" % (ILAN_YOL, ilan_sha))
    s.append("")
    s.append("## Kapılar")
    s.append("")
    s.append("| Ayraç | Ölçülen | Hüküm |")
    s.append("|---|---|---|")
    for k in kapilar:
        s.append("| %s | `%s` | %s |" % (
            k["ayrac"], json.dumps(k["olcum"], ensure_ascii=False),
            "GEÇTİ" if k["gec"] else "DÜŞTÜ"))
    s.append("")
    s.append("## Tam SHA-256 digest tablosu")
    s.append("")
    s.append("| Dosya | SHA-256 |")
    s.append("|---|---|")
    for rel in BEKLENEN_KIRILIM:
        s.append("| `%s` | `%s` |" % (rel, _sha256(os.path.join(REPO_KOK, rel))))
    for rel, olc in k5_olcum.items():
        s.append("| `%s` (çıpa) | `%s` |" % (rel, olc))
    s.append("| `%s` | `%s` |" % (ILAN_YOL, ilan_sha))
    s.append("| `%s` | `%s` |" % (hukum_json, hukum_sha))
    s.append("")
    s.append("## §7 — canlı-migrate önerisi (ayrı operatör onayı)")
    s.append("")
    s.append("Bu tur sunucuya YAZMADI. Sunucu 192.168.1.9:6333'te `kristal_bellek`")
    s.append("koleksiyonu 36-point olarak duruyor (K6 kanıtı). Kod artık")
    s.append("`anka_bellek` okur → mevcut 36 kanonik-belge erişim-dışı kalır.")
    s.append("Önerilen devam-turu: (a) `anka_bellek` koleksiyonu kurulumu")
    s.append("(aynı şema: dense 768 Cosine + sparse idf + on_disk_payload),")
    s.append("(b) `data/pedagogy_canonical/**` kaynağından re-index, (c) 36/36")
    s.append("birebir kapısı, (d) eski `kristal_bellek` silme — `delete_collection`")
    s.append("wrapper (T-0148 4D) ile, silme-öncesi digest/envanter kaydıyla.")
    s.append("Canlı-sunucu YAZIMI olduğundan AYRI operatör onayı şart.")
    s.append("")
    with open(os.path.join(REPO_KOK, rapor), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum_json=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())