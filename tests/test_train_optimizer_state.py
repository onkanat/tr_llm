#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0092: AdamW momentlerinin kaydı/yüklenmesi — `train.py` optimizer yan dosyası kapıları.

ÖLÇÜLEN KUSUR (değişiklikten ÖNCE, `$TMPDIR/t0092_probe2.py`): `torch.save(model.state_dict())`
yalnız ağırlıkları yazar. 2 AdamW adımından sonra `optimizer.state` 101 kayıt taşır; model
kaydedilip YENİ bir model+optimizer ile devam edildiğinde `optimizer.state` **0** olur ⇒
momentler kayıp. Ölçülen bedel: aynı partide ilk güncelleme ||delta|| 4,735192 (momentli) →
8,194721 (momentsiz) ⇒ **1,7306x**.

Bu dosya kapıların YALNIZ DURMADIĞINI değil **AYIRT ETTİĞİNİ** de ölçer:
  * YOKLUK dalı — yan dosya yokken devam koşumu DURMAZ ama stderr'e AÇIK uyarı basar;
    `--load-optimizer` AÇIKÇA verilmişse DURUR (rc=2).
  * EŞLEŞME dalı — yan dosyadaki `model_sha256` uyuşmazsa DURUR (rc=2): yanlış ağırlığa
    moment yapıştırmak sessiz bir soyağacı bozulmasıdır.
  * GEÇEN dal (pozitif kontrol) — yan dosya VARSA ve digest UYUŞUYORSA kapı ATEŞLENMEZ.
    Yalnız "duruyor mu" bakan bir test, her şeyi durduran bir kapıyı da geçirirdi (VAKUM KAPI).

İDDİA SINIRI (beyan): bu testler optimizer durumunun **bit-özdeş** geri yüklendiğini
`$TMPDIR/t0092_probe2.py` ile gösterir; burada ise YAN DOSYA SÖZLEŞMESİ (varlık, digest
eşleşmesi, durma/geçme dalları) sınanır. Eğitim kalitesine etkisi ÖLÇÜLMEMİŞTİR ve
ölçülmüş sayılmamalıdır.
"""
import json
import subprocess
import sys
import hashlib
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
BETIK = KOK / "train.py"
VOCAB = "data/rebuild/vocab_anka_r1_33114.json"
VERI = "data/train.bin"

# Koşumu küçük tutan ortak bayrak kümesi: 2 adım, CPU, minik blok. Amaç model KALİTESİ
# değil, kayıt/yükleme SÖZLEŞMESİdir.
ORTAK = ["--device", "cpu", "--data", VERI, "--vocab", VOCAB,
         "--pretrain", "--from-scratch", "--steps", "2", "--batch-size", "1",
         "--block-size", "32", "--seed", "1234"]


def _kosum(*args: str, timeout: int = 1800) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, str(BETIK), *args],
        cwd=str(KOK), capture_output=True, text=True, timeout=timeout,
    )


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_G0_tohum_deterministik_ve_AYIRT_EDICI(tmp_path):
    """G0: `--seed` gerçekten belirleyici mi — İKİ dal birlikte (vakum kapı yasak).

    Bu test T-0092'de EKLENEN `--seed` bayrağının kapısıdır ve aslında bir **ölçüm aracı**
    kapısıdır: tohum olmadan "değişiklik davranışı değiştirdi mi?" sorusu ÖLÇÜLEMEZ
    (`KristalDataset.get_batch` global `torch.randint` kullanır; `train.py`'de hiç tohum
    kurulmuyordu). İki dal birlikte gerekir:
      * AYNI tohum ⇒ AYNI ağırlıklar (geçen dal),
      * FARKLI tohum ⇒ FARKLI ağırlıklar (ayırt eden dal) — yoksa bayrak hiçbir şey
        yapmıyor olurdu ve "deterministik" iddiası boş olurdu.

    KABUK TUZAĞI (bu görevde ölçüldü): `torch.save` bir ZIP kabıdır ve **arşiv üye adları
    dosya adını taşır** (`m.pt/data/0`). İki koşumu farklı adlarla kaydedip `cmp`/sha256
    ile karşılaştırmak SAHTE FARK üretir. Bu yüzden iki koşum **AYNI taban adla**,
    ayrı dizinlere kaydedilir.
    """
    d1, d2, d3 = tmp_path / "d1", tmp_path / "d2", tmp_path / "d3"
    for d in (d1, d2, d3):
        d.mkdir()

    def kos(dizin, tohum):
        r = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                   "--from-scratch", "--steps", "2", "--batch-size", "1",
                   "--block-size", "32", "--seed", str(tohum),
                   "--save-path", str(dizin / "m.pt"))
        assert r.returncode == 0, f"tohum {tohum} koşumu düştü:\n{r.stderr[-1200:]}"
        return _sha(dizin / "m.pt")

    a1, a2, a3 = kos(d1, 1234), kos(d2, 1234), kos(d3, 999)
    assert a1 == a2, (
        f"AYNI tohum FARKLI ağırlık üretti ⇒ `--seed` deterministik DEĞİL.\n  {a1}\n  {a2}")
    assert a1 != a3, (
        f"FARKLI tohum AYNI ağırlığı üretti ⇒ `--seed` hiçbir şey yapmıyor (VAKUM KAPI).\n"
        f"  ikisi de {a1}")


def test_G1_varsayilan_yan_dosya_uretmez(tmp_path):
    """G1: `--save-optimizer` VERİLMEZSE yan dosya OLUŞMAZ (varsayılan davranış korunur)."""
    hedef = tmp_path / "m.pt"
    res = _kosum(*ORTAK, "--save-path", str(hedef))
    assert res.returncode == 0, f"koşum düşmemeliydi:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}"
    assert hedef.exists(), "model dosyası yazılmadı"
    yan = Path(str(hedef) + ".opt.pt")
    assert not yan.exists(), (
        "G1 İHLALİ: --save-optimizer verilmediği halde yan dosya oluştu ⇒ varsayılan "
        "davranış değişti. Değişiklik geri alınmalıdır.")
    assert "AdamW momentleri" not in res.stdout, "varsayılan koşumun stdout'u yeni satır içeriyor"
    # Sıfırdan koşumda moment zaten BEKLENMEZ ⇒ uyarı basılmamalı (yanlış pozitif kapısı).
    assert "moment" not in res.stderr.lower(), (
        f"sıfırdan koşumda optimizer uyarısı basıldı (yanlış pozitif):\n{res.stderr[-800:]}")


def test_G2_yan_dosya_olur_ve_model_sha256_dogrudur(tmp_path):
    """G2: yan dosya oluşur; `model_sha256` alanı modelin GERÇEK digest'ine EŞİTTİR."""
    hedef = tmp_path / "m.pt"
    res = _kosum(*ORTAK, "--save-path", str(hedef), "--save-optimizer")
    assert res.returncode == 0, f"koşum düşmemeliydi:\n{res.stderr[-1500:]}"
    yan = Path(str(hedef) + ".opt.pt")
    assert yan.exists(), f"G2 İHLALİ: --save-optimizer verildi ama yan dosya yok:\n{res.stderr[-800:]}"
    import torch
    payload = torch.load(yan, map_location="cpu")
    assert isinstance(payload, dict) and "optimizer" in payload, "yan dosyada 'optimizer' alanı yok"
    assert "model_sha256" in payload, "yan dosyada 'model_sha256' alanı yok (eşleşme doğrulanamaz)"
    assert payload["model_sha256"] == _sha(hedef), (
        "yan dosyadaki model_sha256, model dosyasının GERÇEK digest'ine eşit değil")
    assert payload.get("adim") == 2, f"kaydedilen adım 2 olmalıydı: {payload.get('adim')}"
    assert len(payload["optimizer"].get("state", {})) > 0, (
        "kaydedilen optimizer durumu BOŞ ⇒ momentler yine kaydedilmiyor")


def test_G3_devam_kosumunda_momentler_geri_gelir(tmp_path):
    """G3: yan dosyayla devam eden koşum momentleri YÜKLER (kayıt sayısı 0 DEĞİL)."""
    m1 = tmp_path / "m1.pt"
    r1 = _kosum(*ORTAK, "--save-path", str(m1), "--save-optimizer")
    assert r1.returncode == 0, f"1. koşum düştü:\n{r1.stderr[-1500:]}"
    m2 = tmp_path / "m2.pt"
    r2 = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                "--steps", "2", "--batch-size", "1", "--block-size", "32",
                "--load-path", str(m1), "--save-path", str(m2),
                "--save-optimizer", "--load-optimizer")
    assert r2.returncode == 0, f"devam koşumu düştü:\n{r2.stdout[-1200:]}\n{r2.stderr[-1500:]}"
    assert "AdamW momentleri geri yuklendi" in r2.stdout, (
        f"momentler geri yüklenmedi:\n{r2.stdout[-1200:]}")


def test_G4a_yan_dosya_yokken_bayrak_yoksa_UYARIR_ama_DURMAZ(tmp_path):
    """G4a: devam koşumu, yan dosya yokken ve bayrak verilmemişken DEVAM eder + UYARIR.

    Bu dal bir HATA değil EKSİKLİKtir: mevcut checkpoint'ler (anka_a1.pt, anka_a1r.pt)
    moment TAŞIMAZ. Bu yüzden durdurulmaz — ama sessiz de kalınmaz.
    """
    m1 = tmp_path / "m1.pt"
    assert _kosum(*ORTAK, "--save-path", str(m1)).returncode == 0
    assert not Path(str(m1) + ".opt.pt").exists()
    m2 = tmp_path / "m2.pt"
    r2 = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                "--steps", "2", "--batch-size", "1", "--block-size", "32",
                "--load-path", str(m1), "--save-path", str(m2))
    assert r2.returncode == 0, f"YOKLUK dalı DURMAMALIYDI:\n{r2.stderr[-1200:]}"
    assert "AdamW momenti bulunamadi" in r2.stderr, (
        f"SESSİZ DÜŞME: yan dosya yok ama stderr'de uyarı yok:\n{r2.stderr[-1200:]}")
    assert "AdamW momenti bulunamadi" not in r2.stdout, "uyarı stdout'a basılmamalı"


def test_G4b_yan_dosya_yokken_load_optimizer_DURUR(tmp_path):
    """G4b: `--load-optimizer` AÇIKÇA istendi ama yan dosya yok ⇒ rc=2 ile DURUR."""
    m1 = tmp_path / "m1.pt"
    assert _kosum(*ORTAK, "--save-path", str(m1)).returncode == 0
    m2 = tmp_path / "m2.pt"
    r2 = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                "--steps", "2", "--batch-size", "1", "--block-size", "32",
                "--load-path", str(m1), "--save-path", str(m2), "--load-optimizer")
    assert r2.returncode == 2, (
        f"kapı rc=2 ile durmalıydı, rc={r2.returncode}\n--- stdout ---\n{r2.stdout[-800:]}"
        f"\n--- stderr ---\n{r2.stderr[-800:]}")
    assert "DURDURULDU" in r2.stderr and "DURDURULDU" not in r2.stdout
    assert not m2.exists(), "kapı durduğu halde model dosyası yazıldı"


def test_G5_eslesmeyen_model_sha256_DURUR(tmp_path):
    """G5: yan dosyadaki digest BAŞKA bir modele aitse DUR (sessiz yapıştırma yok).

    POZİTİF KONTROL: aynı kurulum, yalnız digest uyuşur hâliyle geçer (G3 testi). Bu test
    kapının GERÇEKTEN AYIRT ETTİĞİNİ gösterir: tek bir alan bozulunca durur.
    """
    import torch
    m1 = tmp_path / "m1.pt"
    assert _kosum(*ORTAK, "--save-path", str(m1), "--save-optimizer").returncode == 0
    yan = Path(str(m1) + ".opt.pt")
    payload = torch.load(yan, map_location="cpu")
    payload["model_sha256"] = "0" * 64          # başka bir ağırlık kümesini taklit et
    torch.save(payload, yan)

    m2 = tmp_path / "m2.pt"
    r2 = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                "--steps", "2", "--batch-size", "1", "--block-size", "32",
                "--load-path", str(m1), "--save-path", str(m2),
                "--load-optimizer")
    assert r2.returncode == 2, (
        f"EŞLEŞME KAPISI ATEŞLENMEDİ: rc={r2.returncode}\n--- stderr ---\n{r2.stderr[-1200:]}")
    assert "ESLESMIYOR" in r2.stderr, f"durma mesajı eşleşmezliği anmalı:\n{r2.stderr[-800:]}"
    assert "DURDURULDU" in r2.stderr and "DURDURULDU" not in r2.stdout


def test_G5b_model_sha256_alani_yoksa_DURUR(tmp_path):
    """G5b: alan TAMAMEN yoksa da durur — 'doğrulanamadı' sessizce 'doğru' sayılamaz."""
    import torch
    m1 = tmp_path / "m1.pt"
    assert _kosum(*ORTAK, "--save-path", str(m1), "--save-optimizer").returncode == 0
    yan = Path(str(m1) + ".opt.pt")
    payload = torch.load(yan, map_location="cpu")
    del payload["model_sha256"]
    torch.save(payload, yan)
    m2 = tmp_path / "m2.pt"
    r2 = _kosum("--device", "cpu", "--data", VERI, "--vocab", VOCAB, "--pretrain",
                "--steps", "2", "--batch-size", "1", "--block-size", "32",
                "--load-path", str(m1), "--save-path", str(m2), "--load-optimizer")
    assert r2.returncode == 2, f"alan yokken durmalıydı: rc={r2.returncode}\n{r2.stderr[-800:]}"
    assert "DURDURULDU" in r2.stderr


def test_G6a_donmus_model_yolu_DURUR(tmp_path):
    """G6a: model yolu donmuşsa (`data/*.pt`) koşum `torch.save`'a ULAŞMADAN durur.

    ÖLÇÜM KABI UYARISI: bu test YALNIZ **göreli** yolla çağırır, çünkü donmuş kapı
    göreli yollar için çalışır. **Mutlak yol kapıyı ATLAR** — bu ölçülmüş bir kusurdur,
    T-0092 raporunda "açık madde" olarak beyan edilmiştir; buraya iddia olarak
    YAZILMAZ (kusuru teste sabitlemek düzeltmeyi engeller).
    """
    hedef = "data/anka_t0092_sonda.pt"
    yan = "data/anka_t0092_sonda.pt.opt.pt"
    try:
        res = _kosum(*ORTAK, "--save-path", hedef, "--save-optimizer")
        assert res.returncode != 0, (
            f"donmuş yola yazma engellenmeliydi, rc={res.returncode}\n{res.stdout[-600:]}")
        assert "Donmuş yola yazma engellendi" in (res.stderr + res.stdout), (
            f"donmuş kapı mesajı yok:\n{res.stderr[-800:]}")
        assert hedef in (res.stderr + res.stdout), "kapı mesajı HANGİ yolu engellediğini yazmıyor"
        assert not (KOK / hedef).exists(), "donmuş model yolu OLUŞTU ⇒ kapı torch.save'dan SONRA"
        assert not (KOK / yan).exists(), "donmuş yan dosya yolu OLUŞTU"
    finally:
        for p in (KOK / hedef, KOK / yan):
            if p.exists():
                p.unlink()


def test_G6b_yalniz_yan_dosyasi_donmus_olan_yol_DURUR(tmp_path):
    """G6b: model yolu SERBEST ama yan dosyası donmuşsa da DURUR — T-0092'nin YENİ kapısı.

    `data/anka_t0092_sonda_v2` donmuş desenlerin HİÇBİRİNE girmez (uzantısız); ama
    `.opt.pt` eki onu `data/*.pt` desenine sokar ⇒ yan dosya kontrolü **erişilebilir ve
    ayırt edici** bir kapıdır, gereksiz bir tekrar DEĞİL. Kapı mesajının **yan dosya**
    yolunu anması, ateşleyen kontrolün model kontrolü değil yan dosya kontrolü olduğunu
    kanıtlar.
    """
    hedef = "data/anka_t0092_sonda_v2"
    yan = "data/anka_t0092_sonda_v2.opt.pt"
    try:
        res = _kosum(*ORTAK, "--save-path", hedef, "--save-optimizer")
        assert res.returncode != 0, (
            f"yan dosyası donmuş olan yola yazma engellenmeliydi, rc={res.returncode}\n"
            f"{res.stdout[-600:]}")
        cikti = res.stderr + res.stdout
        assert "Donmuş yola yazma engellendi" in cikti, f"donmuş kapı mesajı yok:\n{res.stderr[-800:]}"
        assert yan in cikti, (
            f"kapı MODEL yolunu engelledi, YAN DOSYA yolunu değil ⇒ yan dosya kapısı "
            f"ATEŞLENMEDİ:\n{res.stderr[-800:]}")
        assert not (KOK / hedef).exists(), "serbest sanılan model yolu OLUŞTU"
        assert not (KOK / yan).exists(), "donmuş yan dosya yolu OLUŞTU"
    finally:
        for p in (KOK / hedef, KOK / yan):
            if p.exists():
                p.unlink()
