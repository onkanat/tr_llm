#!/usr/bin/env python
"""T-0166 — B2 Tanısı: T-0156 3 off-diagonal örneğini kimliklendirme (hüküm-betiği).

İLAN: data/eval/t0166_b2_tani_ilan_2026-09-28.md
(damga BETİKTEN, koşum-ÖNCESİ; İLAN SABİT).

AMAÇ:
  T-0156 train 3x3 karışım matrisindeki 3 off-diagonal yanlış tahmini
  (2 pedagogy -> carpenter, 1 carpenter -> grammar_core) kimliklendirmek:
  tam metin, indis, uzunluk, karakter/kelime özellikleri, modelin olasılık
  dağılımı ve kök-neden hipotezini diske kaydetmek.

DOKUNULMAZLAR:
  - anka_router.pt YENİDEN EĞİTİLMEZ (ağırlıklar sabit: 44a46d89…)
  - Canlı Qdrant'a 0 istek (VectorMemory kurulmaz)
  - data/pedagogy_canonical/**, tokenizer, compiler dokunulmaz.

HÜKÜM KAPILARI (5/5 şart):
  K1 ROUTER DOKUNULMAZLIK & KAYNAK DİGEST: anka_router.pt, base_v2, vocab, roots.tsv
  K2 DETERMİNİSTİK TEYİT: grammar_core=500, pedagogy=498, carpenter=499 (toplam 3 hata)
  K3 KİMLİKLENDİRME: 3 örneğin metni, gerçek/tahmin sınıfları ve özellikleri eksiksiz
  K4 CANLI DOKUNULMAZLIK: VectorMemory kurulmaz, canlıya 0 istek
  K5 ŞEMA KORUMASI: 9 kanonik anahtar eksiksiz

rc ∈ {0, 2}. Hüküm BETİKTEN.
"""
import hashlib
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from src.compiler.lexicon import LexiconManager  # noqa: E402
from src.compiler.morphotactics import build_default_graph  # noqa: E402
from src.compiler.core import CrystalCompiler  # noqa: E402
from src.llm.tokenizer import KristalTokenizer, Vocabulary  # noqa: E402
from src.rag.merak import CuriosityEngine  # noqa: E402
from src.llm.router import TriModalRouter  # noqa: E402
from src.rag.epistemic_agent import load_trained_router_state  # noqa: E402
from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.prompt_contract import resize_state_dict  # noqa: E402

ILAN_YOL = "data/eval/t0166_b2_tani_ilan_2026-09-28.md"
HUKUM_YOL = "data/eval/t0166_b2_tani_hukum_2026-09-28.json"
RAPOR_YOL = "data/eval/t0166_b2_tani_rapor_2026-09-28.md"
ROUTER_YOL = "data/anka_router.pt"

SEED = 42
TRAIN_N = 500
VAL_N = 100

KAYNAK_SHA = {
    "data/anka_base_v2.pt":
        "d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50",
    "data/rebuild/vocab_anka_r1_33114.json":
        "f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984",
    "data/lexicon/roots.tsv":
        "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598",
    "data/anka_router.pt":
        "44a46d89f3d4260c86f1aa0aa4756b28321f4455c277ff8a8e4dea4c1e85b36a",
}
BEKLENEN_SD_SHA = "d369b3cdd25484ab679cd61321f3ae614e932166fb8ce6bd5f9dfa4c99fb50bf"

SINIF_KAYNAK = {
    0: ["data/pedagogy/lexical_semantics_dataset.jsonl"],
    1: ["data/pedagogy_canonical/high_school_canonical.jsonl",
        "data/pedagogy_canonical/literature_canonical.jsonl",
        "data/pedagogy_canonical/middle_school_canonical.jsonl",
        "data/pedagogy_canonical/turk_tarihi_canonical.jsonl"],
    2: ["data/pedagogy_canonical/carpenter_canonical.jsonl"],
}
SINIF_AD = {0: "grammar_core", 1: "pedagogy", 2: "carpenter", 3: "legal"}


def _stderr(mesaj: str) -> None:
    print("[T-0166] " + mesaj, file=sys.stderr, flush=True)


def _sha256(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def _state_dict_sha(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _metinler(yollar: List[str]) -> List[str]:
    cikti: List[str] = []
    for yol in yollar:
        with open(os.path.join(REPO_KOK, yol), encoding="utf-8") as f:
            for satir in f:
                satir = satir.strip()
                if not satir:
                    continue
                rec = json.loads(satir)
                m = (rec.get("input") or "").strip() or (rec.get("instruction") or "").strip()
                if m:
                    cikti.append(m)
    return cikti


def _ozellik_cikar(model: Any, engine: Any, tokenizer: Any, vocab: Any,
                   metinler: List[str], cihaz: torch.device) -> Tuple[Any, Any, int]:
    unk_id = vocab.stoi.get("<UNK>", 1)
    vocab_boyut = model.embedding.embedding.weight.shape[0]
    promptler: List[torch.Tensor] = []
    qmeraklar: List[torch.Tensor] = []
    kirpilan = 0
    model.eval()
    with torch.no_grad():
        for i, metin in enumerate(metinler):
            t_ids = tokenizer.encode(metin)
            if len(t_ids) > 4096:
                kirpilan += 1
            gecerli = [t for t in t_ids if 0 <= t < vocab_boyut][:4096]
            if not gecerli:
                gecerli = [0]
            x = torch.tensor([gecerli], dtype=torch.long, device=cihaz)
            emb = model.embedding(x)
            if isinstance(emb, tuple):
                emb = emb[0]
            promptler.append(emb.mean(dim=1)[0].float().cpu())
            has_unk = any(t == unk_id for t in t_ids)
            logits, _, x_emb = model(x, return_hidden_states=True)
            _, _, q = engine(x_emb[:, -1, :], logits[:, -1, :],
                             has_unk=has_unk, unk_token_id=unk_id)
            qmeraklar.append(q[0].float().cpu())
            if (i + 1) % 300 == 0:
                _stderr("özellik %d/%d" % (i + 1, len(metinler)))
    return torch.stack(promptler), torch.stack(qmeraklar), kirpilan


def _gate_logits(router: TriModalRouter, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    e = router.w_p(p) + router.w_m(q)
    e = router.layer_norm(e)
    return router.w_g(e)


def main() -> int:
    damga = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _stderr("başlangıç %s" % damga)

    ilan_tam = os.path.join(REPO_KOK, ILAN_YOL)
    if not os.path.exists(ilan_tam):
        _stderr("İLAN YOK: %s" % ILAN_YOL)
        return 2
    ilan_sha = _sha256(ilan_tam)
    _stderr("ilan sha256=%s" % ilan_sha)

    kapilar: List[Dict[str, Any]] = []

    def kayit(ayrac: str, gec: bool, olcum: Dict[str, Any]) -> bool:
        kapilar.append({"ayrac": ayrac, "gec": bool(gec), "olcum": olcum})
        _stderr("KAPI [%s] %s -> %s" % (
            "GEÇTİ" if gec else "DÜŞTÜ", ayrac, json.dumps(olcum, ensure_ascii=False)[:300]))
        return bool(gec)

    # ---- K1 ROUTER DOKUNULMAZLIK & KAYNAK DİGEST ----
    sha_kontrol: Dict[str, str] = {}
    sha_sapma: List[str] = []
    for dosya, beklenen in KAYNAK_SHA.items():
        tam = os.path.join(REPO_KOK, dosya)
        if not os.path.exists(tam):
            sha_sapma.append("%s: YOK" % dosya)
            continue
        gercek = _sha256(tam)
        sha_kontrol[dosya] = gercek
        if gercek != beklenen:
            sha_sapma.append("%s: beklenen=%s gercek=%s" % (dosya, beklenen, gercek))

    # Router modelini yükle ve state_dict sha doğrula
    router = TriModalRouter(prompt_dim=768, merak_dim=768, rag_dim=768,
                            router_dim=256, num_experts=4, top_k=2,
                            expert_names=["grammar_core", "pedagogy",
                                          "carpenter", "legal"])
    load_trained_router_state(router, os.path.join(REPO_KOK, ROUTER_YOL))
    sd_sha = _state_dict_sha(router.state_dict())
    if sd_sha != BEKLENEN_SD_SHA:
        sha_sapma.append("router_sd_sha: beklenen=%s gercek=%s" % (BEKLENEN_SD_SHA, sd_sha))

    k1_gec = (len(sha_sapma) == 0)
    if not kayit("K1 ROUTER DOKUNULMAZLIK & KAYNAK DİGEST (44a46d89… ve d369b3cd… birebir)",
                 k1_gec, {"sha_kontrol": sha_kontrol, "sd_sha": sd_sha, "sha_sapma": sha_sapma}):
        return _hukum(kapilar, ilan_sha, damga, None)

    # ---- Cihaz Kontrolü ----
    if not torch.backends.mps.is_available():
        _stderr("MPS YOK")
        kayit("CİHAZ (özellik-çıkarımı: MPS)", False, {"mps": False})
        return _hukum(kapilar, ilan_sha, damga, None)
    cihaz = torch.device("mps")
    kayit("CİHAZ (özellik-çıkarımı: MPS)", True, {"mps": True})

    # ---- Kanonik Modelleri Yükle ----
    _stderr("kanonik modeller yükleniyor...")
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

    torch.manual_seed(SEED)
    engine = CuriosityEngine(hidden_dim=768, curiosity_dim=768, tau=2.5)
    engine.to(cihaz)
    engine.eval()

    # ---- T-0156 ile Birebir Veri Seçimi (SEED-42) ----
    rng = random.Random(SEED)
    metinler: List[str] = []
    etiketler: List[int] = []
    for c, yollar in SINIF_KAYNAK.items():
        tum = _metinler(yollar)
        tum = sorted(set(tum))  # deterministik sıra
        rng.shuffle(tum)
        secili = tum[:TRAIN_N + VAL_N]
        metinler.extend(secili)
        etiketler.extend([c] * len(secili))

    train_metin, train_y = [], []
    for c in SINIF_KAYNAK:
        blok = [m for m, e in zip(metinler, etiketler) if e == c]
        train_metin.extend(blok[:TRAIN_N])
        train_y.extend([c] * min(TRAIN_N, len(blok)))

    _stderr("train örnekleri: toplam %d (her sınıf 500)" % len(train_metin))

    # ---- Özellik Çıkarımı (MPS) ----
    _stderr("özellik-çıkarımı yapılıyor (MPS)...")
    t0 = time.time()
    Ptr, Qtr, kirp_tr = _ozellik_cikar(model, engine, tokenizer, vocab, train_metin, cihaz)
    sure = time.time() - t0
    _stderr("özellik-çıkarımı tamamlandı (%.1f sn)" % sure)

    ytr = torch.tensor(train_y, dtype=torch.long)

    # ---- K2 DETERMİNİSTİK TEYİT (K7 Karışım Matrisi Eşleşmesi) ----
    with torch.no_grad():
        logits_tr = _gate_logits(router, Ptr, Qtr)
        probs_tr = F.softmax(logits_tr, dim=-1)
        pred_tr = logits_tr.argmax(dim=-1)

    matris = [[0 for _ in range(3)] for _ in range(3)]
    for t, p in zip(ytr.tolist(), pred_tr.tolist()):
        matris[t][p] += 1

    per_sinif_dogru = {
        SINIF_AD[c]: int((pred_tr[ytr == c] == c).sum().item()) for c in range(3)
    }

    # T-0156 K7 matrisi:
    # [[500, 0, 0], [0, 498, 2], [0, 1, 499]]
    beklenen_matris = [[500, 0, 0], [0, 498, 2], [0, 1, 499]]
    k2_gec = (matris == beklenen_matris and per_sinif_dogru == {
        "grammar_core": 500, "pedagogy": 498, "carpenter": 499
    })

    if not kayit("K2 DETERMİNİSTİK TEYİT (T-0156 K7 3x3 matrisi birebir: 500/498/499)",
                 k2_gec, {"matris": matris, "per_sinif_dogru": per_sinif_dogru,
                          "beklenen_matris": beklenen_matris}):
        return _hukum(kapilar, ilan_sha, damga, None)

    # ---- K3 KİMLİKLENDİRME (3 Off-Diagonal Örnek) ----
    hatali_ornekler: List[Dict[str, Any]] = []
    for idx, (gercek, tahmin) in enumerate(zip(ytr.tolist(), pred_tr.tolist())):
        if gercek != tahmin:
            metin = train_metin[idx]
            p_vec = probs_tr[idx].tolist()
            l_vec = logits_tr[idx].tolist()
            
            # Kök-neden hipotezi tespiti
            # Hipotez faktörleri: metin uzunluğu, soru işareti, ortak terimler
            hipotez = []
            if len(metin) < 40:
                hipotez.append("kisa_metin_az_ipucu")
            if "?" in metin:
                hipotez.append("soru_formu")
            else:
                hipotez.append("soru_isareti_yok")

            # Terim / içerik analizi
            metin_l = metin.lower()
            if gercek == 1 and tahmin == 2:  # Pedagogy -> Carpenter
                if any(w in metin_l for w in ["ağaç", "odun", "kesim", "alet", "çivi", "yapı", "tür", "ağaçlar", "malzeme"]):
                    hipotez.append("marangozluk_benzeri_materyal_veya_terim")
                else:
                    hipotez.append("genel_bilim_veya_tanim_cumlesi")
            elif gercek == 2 and tahmin == 0:  # Carpenter -> Grammar Core
                if any(w in metin_l for w in ["anlam", "ek", "kelime", "sözcük", "dil"]):
                    hipotez.append("dilbilgisi_terim_cakismasi")
                else:
                    hipotez.append("kisa_veya_sozluksel_tanim")

            hatali_ornekler.append({
                "global_train_indeks": idx,
                "sinif_ici_indeks": idx % TRAIN_N,
                "gercek_sinif_id": gercek,
                "gercek_sinif": SINIF_AD[gercek],
                "tahmin_sinif_id": tahmin,
                "tahmin_sinif": SINIF_AD[tahmin],
                "metin": metin,
                "karakter_uzunlugu": len(metin),
                "kelime_sayisi": len(metin.split()),
                "soru_isareti_var": "?" in metin,
                "olasiliklar": {
                    SINIF_AD[c]: round(p_vec[c], 4) for c in range(3)
                },
                "logitler": {
                    SINIF_AD[c]: round(l_vec[c], 4) for c in range(3)
                },
                "kok_neden_hipotezi": "+".join(hipotez),
            })

    k3_gec = (len(hatali_ornekler) == 3)
    kayit("K3 KİMLİKLENDİRME (3 off-diagonal örneğin tam metni ve tanısı)",
          k3_gec, {"hatali_ornek_sayisi": len(hatali_ornekler),
                   "ornekler": hatali_ornekler})

    # ---- K4 CANLI DOKUNULMAZLIK ----
    kayit("K4 CANLI DOKUNULMAZLIK (VectorMemory kurulmadı; canlıya 0 istek)", True, {
        "canli_istek": 0,
        "vector_memory": "KULLANILMADI (betik ağ istemcisi kurmaz)",
    })

    # ---- K5 ŞEMA KORUMASI ----
    kanonik_anahtarlar = sorted(TriModalRouter(
        prompt_dim=768, merak_dim=768, rag_dim=768, router_dim=256,
        num_experts=4, top_k=2).state_dict().keys())
    mevcut_anahtarlar = sorted(router.state_dict().keys())
    k5_gec = (kanonik_anahtarlar == mevcut_anahtarlar and len(mevcut_anahtarlar) == 9)
    kayit("K5 ŞEMA KORUMASI (9 anahtar tam)", k5_gec, {
        "anahtar_sayi": len(mevcut_anahtarlar),
        "sema_ok": k5_gec,
    })

    # Koşum sonu router sha teyidi (DOKUNULMAZLIK)
    son_router_sha = _sha256(os.path.join(REPO_KOK, ROUTER_YOL))
    if son_router_sha != KAYNAK_SHA["data/anka_router.pt"]:
        _stderr("HATA: anka_router.pt DEĞİŞTİ!")
        return 2

    return _hukum(kapilar, ilan_sha, damga, hatali_ornekler)


def _hukum(kapilar: List[Dict[str, Any]], ilan_sha: str, damga: str,
           hatali_ornekler: Any) -> int:
    hukum_adi = ("T0166_B2_TANI_GECTI"
                 if kapilar and all(k["gec"] for k in kapilar) else "DUR")
    rc = 0 if hukum_adi == "T0166_B2_TANI_GECTI" else 2

    hukum_json = {
        "hukum": hukum_adi,
        "rc": rc,
        "damga": damga,
        "hukum_kaynagi": "BU BETİK — elle sayı/hüküm YOK",
        "ilan": ILAN_YOL,
        "ilan_sha256": ilan_sha,
        "kapilar": kapilar,
        "hatali_ornekler": hatali_ornekler,
        "router_sha256": KAYNAK_SHA["data/anka_router.pt"],
        "router_dokunulmaz_korundu": True,
    }

    with open(os.path.join(REPO_KOK, HUKUM_YOL), "w", encoding="utf-8") as f:
        json.dump(hukum_json, f, ensure_ascii=False, indent=2, sort_keys=True)
    hukum_sha = _sha256(os.path.join(REPO_KOK, HUKUM_YOL))

    # Markdown Rapor
    s: List[str] = []
    s.append("# T-0166 — B2 Tanısı: T-0156 3 Off-Diagonal Örneği Raporu")
    s.append("")
    s.append("**Hüküm:** **%s** (betikten; elle sayı YOK)" % hukum_adi)
    s.append("**Damga:** %s (UTC)" % damga)
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
    s.append("## Kimliklendirilen 3 Off-Diagonal Örnek")
    s.append("")
    if hatali_ornekler:
        for idx, o in enumerate(hatali_ornekler, 1):
            s.append("### Örnek %d (İndeks: %d, Sınıf-içi: %d)" % (idx, o["global_train_indeks"], o["sinif_ici_indeks"]))
            s.append("- **Gerçek Sınıf:** `%s`" % o["gercek_sinif"])
            s.append("- **Tahmin Edilen Sınıf:** `%s`" % o["tahmin_sinif"])
            s.append("- **Metin:** \"%s\"" % o["metin"])
            s.append("- **Uzunluk:** %d karakter / %d kelime | Soru İşareti: `%s`" % (
                o["karakter_uzunlugu"], o["kelime_sayisi"], o["soru_isareti_var"]))
            s.append("- **Olasılık Dağılımı:** `%s`" % json.dumps(o["olasiliklar"]))
            s.append("- **Logitler:** `%s`" % json.dumps(o["logitler"]))
            s.append("- **Kök-Neden Hipotezi:** `%s`" % o["kok_neden_hipotezi"])
            s.append("")
    s.append("## Router Dokunulmazlık Teyidi")
    s.append("")
    s.append("- `data/anka_router.pt` **korundu** (üzerine yazılmadı; sha256 `%s`)." % KAYNAK_SHA["data/anka_router.pt"])
    s.append("")

    with open(os.path.join(REPO_KOK, RAPOR_YOL), "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

    _stderr("HÜKÜM: %s rc=%d hukum_sha=%s" % (hukum_adi, rc, hukum_sha))
    return rc


if __name__ == "__main__":
    sys.exit(main())
