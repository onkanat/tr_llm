"""Yetenek modülü mekanizmasının KAPILARI — her test İKİ DALLI (T-0075).

Tek dallı sınama kabul edilmez: "her şeye dur diyen sabit" de yeşil görünür. Bu yüzden her
testte (a) kapının ATEŞLEDİĞİ bozuk dal, (b) kapının GEÇTİĞİ temiz dal birlikte ölçülür.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.optim as optim

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.modules import (  # noqa: E402
    ModulSpec,
    LoRAKatmani,
    beklenen_parametre_sayisi,
    egitilebilir_parametreler,
    modul_durumu,
    modul_ekle,
    modul_katmanlari,
    modul_kaydet,
    modul_yukle,
    tabani_dondur,
)
from src.llm.tokenizer import Vocabulary  # noqa: E402

SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
SPEC = ModulSpec(ad="deneme", r=4, alpha=8)


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    v = Vocabulary()
    v.load(SOZLUK)
    return v


def _kucuk_model(vocab: Vocabulary) -> KristalLM:
    """Test modeli KASITLI küçük: 2 blok, 64 boyut => 12 hedef katman, hızlı."""
    return KristalLM(vocab_size=len(vocab.stoi), n_embd=64, vocab=vocab,
                     n_layer=2, n_head=2, dropout=0.0)


def _taban_yaz(vocab: Vocabulary, yol: str, tohum: int = 1) -> None:
    torch.manual_seed(tohum)
    m = _kucuk_model(vocab)
    torch.save(m.state_dict(), yol)


def _taban_yukle(vocab: Vocabulary, yol: str) -> KristalLM:
    m = _kucuk_model(vocab)
    m.load_state_dict(torch.load(yol, map_location="cpu"), strict=True)
    m.eval()
    return m


def _logit(m: KristalLM, vocab: Vocabulary) -> torch.Tensor:
    torch.manual_seed(99)
    x = torch.randint(0, len(vocab.stoi), (2, 16))
    with torch.no_grad():
        logits, _ = m(x)
    return logits


def test_takma_ani_tabanla_bit_ozdes(vocab: Vocabulary, tmp_path: Any) -> None:
    """SÖZLEŞME 2: B sıfır başlar => takıldığı anda çıktı tabanla ÖZDEŞ olmalı."""
    taban = str(tmp_path / "taban.pt")
    _taban_yaz(vocab, taban)
    m = _taban_yukle(vocab, taban)

    a = _logit(m, vocab)
    modul_ekle(m, SPEC)
    m.eval()
    b = _logit(m, vocab)
    assert torch.equal(a, b), "modül takıldığı anda çıktıyı DEĞİŞTİRDİ (delta sıfır değil)"

    # POZİTİF KONTROL: eşitlik testi duyarsız olmamalı — delta'yı bozunca farkı görmeli.
    with torch.no_grad():
        modul_katmanlari(m)[0][1].lora_B[0, 0] = 1e-3
    c = _logit(m, vocab)
    assert not torch.equal(a, c), "eşitlik testi DUYARSIZ: bozuk delta'yı göremedi"


def test_taban_donuk_kalir(vocab: Vocabulary, tmp_path: Any) -> None:
    """Taban parametreleri donar, yalnız modül eğitilebilir kalır; sayı formülle tutar."""
    taban = str(tmp_path / "taban.pt")
    _taban_yaz(vocab, taban)
    m = _taban_yukle(vocab, taban)
    modul_ekle(m, SPEC)

    once = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n = tabani_dondur(m)

    assert n == beklenen_parametre_sayisi(m), "eğitilebilir sayı formülle uyuşmuyor"
    assert n < once, "POZİTİF KONTROL başarısız: dondurma hiçbir şey dondurmadı"
    kalan = [ad for ad, p in m.named_parameters()
             if p.requires_grad and "lora_A" not in ad and "lora_B" not in ad]
    assert not kalan, f"taban parametresi eğitilebilir kaldı: {kalan[:3]}"

    # İKİNCİ DAL: modül takılmadan dondurma çağrılırsa DURMALI (sessiz "taban donuk" sanısı).
    bos = _taban_yukle(vocab, taban)
    with pytest.raises(RuntimeError):
        tabani_dondur(bos)


def test_kaydet_yukle_turu_ve_taban_sizintisi(vocab: Vocabulary, tmp_path: Any) -> None:
    """Modül dosyası YALNIZ lora tensörlerini taşır; tur sonunda çıktı özdeş kalır."""
    taban = str(tmp_path / "taban.pt")
    _taban_yaz(vocab, taban)
    m = _taban_yukle(vocab, taban)
    modul_ekle(m, SPEC)
    with torch.no_grad():  # sıfırdan farklı bir delta üret
        for _, kat in modul_katmanlari(m):
            kat.lora_B.add_(0.01)
    beklenen = _logit(m, vocab)

    yol = str(tmp_path / "deneme.mod.pt")
    meta = modul_kaydet(m, SPEC, yol, taban, ek={"adim": 3})
    paket = torch.load(yol, map_location="cpu", weights_only=False)

    assert meta["katman_sayisi"] == len(modul_katmanlari(m))
    assert set(paket["modul"].keys()) == set(modul_durumu(m).keys())
    sizan = [k for k in paket["modul"] if "lora_A" not in k and "lora_B" not in k]
    assert not sizan, f"modül dosyası taban ağırlığı taşıyor: {sizan[:3]}"

    m2 = _taban_yukle(vocab, taban)
    modul_yukle(m2, yol, taban)
    m2.eval()
    assert torch.equal(beklenen, _logit(m2, vocab)), "kaydet/yükle turu çıktıyı bozdu"


def test_yanlis_taban_reddedilir(vocab: Vocabulary, tmp_path: Any) -> None:
    """Modül BAŞKA bir tabana yüklenirse DURMALI; doğru tabana yüklenince geçmeli."""
    taban = str(tmp_path / "taban.pt")
    diger = str(tmp_path / "diger_taban.pt")
    _taban_yaz(vocab, taban, tohum=1)
    _taban_yaz(vocab, diger, tohum=2)  # farklı ağırlıklar => farklı digest

    m = _taban_yukle(vocab, taban)
    modul_ekle(m, SPEC)
    yol = str(tmp_path / "deneme.mod.pt")
    modul_kaydet(m, SPEC, yol, taban)

    hedef = _taban_yukle(vocab, diger)
    with pytest.raises(RuntimeError):
        modul_yukle(hedef, yol, diger)
    assert not modul_katmanlari(hedef), "reddedilen yükleme modeli YİNE DE değiştirdi"

    dogru = _taban_yukle(vocab, taban)
    modul_yukle(dogru, yol, taban)  # temiz dal: geçmeli
    assert modul_katmanlari(dogru), "doğru tabana yükleme modülü takmadı"


def test_hedef_deseni_bos_ise_durur(vocab: Vocabulary, tmp_path: Any) -> None:
    """Hiçbir katmana uymayan hedef deseni SESSİZCE 0 katman takmamalı."""
    taban = str(tmp_path / "taban.pt")
    _taban_yaz(vocab, taban)
    m = _taban_yukle(vocab, taban)

    with pytest.raises(ValueError):
        modul_ekle(m, ModulSpec(ad="bos", hedefler=("yok.boyle.katman",)))
    modul_ekle(m, SPEC)  # temiz dal: gerçek desen takılmalı
    assert len(modul_katmanlari(m)) == 2 * len(SPEC.hedefler)


def test_takma_cihaz_ve_tip_devralir() -> None:
    """Takılan katman taban katmanın CİHAZ ve TİP'ini devralmalı.

    ÖLÇÜLEN KUSUR (21 Eyl 2026, MPS): `nn.Linear` parametrelerini varsayılan CPU/fp32
    doğurur. Model MPS'te iken `modul_ekle` çağrılınca takılan katman CPU'da kalıyordu ve
    ilk forward `Tensor for argument weight is on cpu but expected on mps` ile çöküyordu.
    Eğitim koşumunda sıra (önce tak, sonra `.to(device)`) kusuru GİZLİYORDU; gerçek
    değerlendirme yolu ise tabanı cihaza taşıyıp SONRA modülü yükler ⇒ orada zorunlu ateşler.

    VEKİL: sandbox MPS'i gizlediği için cihaz burada sınanamaz; **tip** aynı sınıfın CPU'da
    sınanabilir vekilidir. Kanıtın sınırı budur — asıl cihaz kanıtı sandbox DIŞINDA,
    `scripts/modul_olcum.py --device mps` koşumudur.
    """
    import torch.nn as nn

    temel = nn.Linear(8, 8, dtype=torch.float64)
    # POZİTİF KONTROL: sınama VAKUM olmasın — varsayılan tip fp32'dir, yani "devralma"
    # kendiliğinden doğru değil.
    assert nn.Linear(8, 8).weight.dtype == torch.float32

    kat = LoRAKatmani(temel, r=4, alpha=8)
    assert kat.weight.dtype == torch.float64, "taban ağırlık tipi devralınmadı"
    assert kat.bias is not None and kat.bias.dtype == torch.float64
    assert kat.lora_A.dtype == torch.float64 and kat.lora_B.dtype == torch.float64

    x = torch.randn(2, 8, dtype=torch.float64)
    with torch.no_grad():
        cikti = kat(x)
    assert cikti.dtype == torch.float64, "forward tipi bozdu"
    assert torch.equal(cikti, nn.functional.linear(x, temel.weight, temel.bias)), \
        "takma anında tabanla özdeşlik cihaz/tip devralmayla birlikte bozuldu"


def test_egitim_tabani_degistirmez(vocab: Vocabulary, tmp_path: Any) -> None:
    """ASIL İDDİA: birkaç optimizer adımı taban tensörlerine DOKUNMAMALI, modülü değiştirmeli."""
    taban = str(tmp_path / "taban.pt")
    _taban_yaz(vocab, taban)
    m = _taban_yukle(vocab, taban)
    modul_ekle(m, SPEC)
    tabani_dondur(m)

    once = {k: v.clone() for k, v in m.state_dict().items()
            if "lora_A" not in k and "lora_B" not in k}
    lora_once = [t.clone() for _, k in modul_katmanlari(m) for t in (k.lora_A, k.lora_B)]

    opt = optim.AdamW(egitilebilir_parametreler(m), lr=1e-2)
    torch.manual_seed(5)
    m.train()
    for _ in range(3):
        x = torch.randint(0, len(vocab.stoi), (2, 16))
        y = torch.randint(0, len(vocab.stoi), (2, 16))
        opt.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        opt.step()

    simdi = m.state_dict()
    degisen = [k for k, v in once.items() if not torch.equal(v, simdi[k])]
    assert not degisen, f"taban tensörü eğitimde değişti: {degisen[:3]}"

    lora_simdi = [t for _, k in modul_katmanlari(m) for t in (k.lora_A, k.lora_B)]
    degisen_modul = [i for i, (a, b) in enumerate(zip(lora_once, lora_simdi))
                     if not torch.equal(a, b)]
    assert degisen_modul, "POZİTİF KONTROL başarısız: eğitim modülü hiç değiştirmedi"
