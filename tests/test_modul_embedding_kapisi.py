"""GÖMME KATMANI LoRA YOLU ve KISMİ-UYUM KAPISI — her test İKİ DALLI (T-0075).

NEDEN VAR (ölçüldü, 22 Eyl 2026): `modul_ekle` yalnız `nn.Linear` sarıyordu. `embedding`
hedefi verildiğinde **hiçbir katman takılmıyordu** ama hata da çıkmıyordu — başka hedefler
eşleştiği için "hiçbir katman uymadı" kapısı ateşlenmiyordu. Yani `--hedefler …,embedding`
ile koşan bir deney, ölçülmek istenen DEĞİŞKENİ HİÇ İÇERMEDEN tamamlanır ve sonuç
"değişken etkisiz" diye **sahte bir null** olarak yazılırdı.

İkinci iddia: `KristalEmbedding`'in kendisi DEĞİL, içteki `embedding.embedding` sarılır ⇒
imza maskesi (kristal yapı) delta'nın ÜSTÜNDE kalır ve baypas edilmez.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

import pytest
import torch

REPO_KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_KOK not in sys.path:
    sys.path.insert(0, REPO_KOK)

from scripts.train_step_demo import KristalLM  # noqa: E402
from src.llm.modules import (  # noqa: E402
    VARSAYILAN_HEDEFLER,
    LoRAEmbedding,
    ModulSpec,
    beklenen_parametre_sayisi,
    modul_ekle,
    modul_katmanlari,
    tabani_dondur,
)
from src.llm.tokenizer import Vocabulary  # noqa: E402

SOZLUK = "data/rebuild/vocab_anka_r1_33114.json"
GECERLI = tuple(VARSAYILAN_HEDEFLER)
GECERLI_G = GECERLI + ("embedding",)


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    v = Vocabulary()
    v.load(SOZLUK)
    return v


def _kucuk(vocab: Vocabulary) -> KristalLM:
    """Kasıtlı küçük: 2 blok, 64 boyut ⇒ hızlı ama gerçek sınıf."""
    return KristalLM(vocab_size=len(vocab.stoi), n_embd=64, vocab=vocab,
                     n_layer=2, n_head=2, dropout=0.0)


def _cift(vocab: Vocabulary) -> tuple:
    """AYNI tabandan iki kopya — rastgele iki model karşılaştırması SAHTE fark üretir."""
    import copy
    a = _kucuk(vocab)
    a.eval()
    return a, copy.deepcopy(a)


def _logit(m: KristalLM, vocab: Vocabulary) -> torch.Tensor:
    torch.manual_seed(99)
    x = torch.randint(0, len(vocab.stoi), (2, 16))
    with torch.no_grad():
        logits, _ = m(x)
    return logits


# ------------------------------------------------------- takma
def test_gomme_hedefi_takilir(vocab: Vocabulary) -> None:
    """POZİTİF: 'embedding' hedefi içteki nn.Embedding'i sarar. NEGATİF: varsayılan sarmaz."""
    _, m = _cift(vocab)
    tak = modul_ekle(m, ModulSpec(ad="g", r=4, alpha=8, hedefler=GECERLI_G))
    assert "embedding.embedding" in tak, "gömme hedefi takılmadı"
    assert isinstance(m.get_submodule("embedding.embedding"), LoRAEmbedding)

    _, m2 = _cift(vocab)
    modul_ekle(m2, ModulSpec(ad="v", r=4, alpha=8, hedefler=GECERLI))
    assert not isinstance(m2.get_submodule("embedding.embedding"), LoRAEmbedding), \
        "varsayılan hedefler gömmeye DOKUNMAMALI"
    assert len(modul_katmanlari(m)) == len(modul_katmanlari(m2)) + 1


def test_gomme_parametre_formulu(vocab: Vocabulary) -> None:
    """Formül iki sınıfta AYNI olmalı: r*(in_features + out_features)."""
    _, m = _cift(vocab)
    modul_ekle(m, ModulSpec(ad="g", r=4, alpha=8, hedefler=GECERLI_G))
    eg = tabani_dondur(m)
    assert eg == beklenen_parametre_sayisi(m), "gömme dâhil formül tutmuyor"
    k = m.get_submodule("embedding.embedding")
    assert (k.in_features, k.out_features) == (k.num_embeddings, k.embedding_dim)


def test_gomme_takma_ani_bit_ozdes(vocab: Vocabulary) -> None:
    """SÖZLEŞME 2 gömmе için de geçerli: B sıfır ⇒ çıktı ÖZDEŞ. Pozitif kontrol dâhil."""
    a, b = _cift(vocab)
    once = _logit(a, vocab)
    modul_ekle(b, ModulSpec(ad="g", r=4, alpha=8, hedefler=GECERLI_G))
    b.eval()
    assert torch.equal(once, _logit(b, vocab)), "gömme modülü takılırken çıktıyı DEĞİŞTİRDİ"

    # POZİTİF KONTROL: eşitlik testi duyarsız olmasın.
    with torch.no_grad():
        k = b.get_submodule("embedding.embedding")
        k.lora_B[0, 0] = 1e-2
    assert not torch.equal(once, _logit(b, vocab)), "bozulan delta görülmedi ⇒ test duyarsız"


# ------------------------------------------------------- kristal yapı
def test_kristal_yapi_deltanin_ustunde_kalir(vocab: Vocabulary) -> None:
    """İmza maskesi (kristal yapı) sarma sonrası da uygulanmalı — baypas EDİLMEMELİ.

    İddia: `KristalEmbedding(x) = mask * (W[x] + delta[x])`. Maske delta'dan SONRA çarpıyor.
    """
    _, m = _cift(vocab)
    modul_ekle(m, ModulSpec(ad="g", r=4, alpha=8, hedefler=GECERLI_G))
    m.eval()
    k = m.get_submodule("embedding.embedding")
    with torch.no_grad():
        k.lora_B[0, 0] = 1e-2          # delta ARTIK sıfır değil ⇒ test boş olmasın

    bnd = sorted(m.embedding.boundary_ids)[:2]
    neg = sorted(m.embedding.neg_ids)[0]
    # [kök, NEG, kök] ⇒ olumsuzlanan sözcüğün aralığı -1 ile çarpılmalı
    x = torch.tensor([[bnd[0], neg, bnd[1]]])
    mask = m.embedding.compute_sign_mask(x)
    assert (mask < 0).any(), "kurulan dizide maske ATEŞLEMEDİ ⇒ test vakum olurdu"

    with torch.no_grad():
        cikti = m.embedding(x)
        beklenen = (k.weight[x] + k.delta(x)) * mask
    assert torch.allclose(cikti, beklenen, atol=1e-6), "maske delta'nın ÜSTÜNDE değil"

    # NEGATİF DAL: olumsuzlama yoksa maske tümüyle +1 olmalı (yapı boşuna ateşlemiyor)
    x2 = torch.tensor([[bnd[0], bnd[1]]])
    assert not (m.embedding.compute_sign_mask(x2) < 0).any()


# ------------------------------------------------------- kısmi uyum kapısı
def test_kismi_uyum_kapisi_ATESLER(vocab: Vocabulary) -> None:
    """POZİTİF: geçerli + uymayan hedef karışımı DURMALI (sessiz düşme yasak).

    NEGATİF: tümü geçerliyse durmamalı — 'her şeye dur' diyen sabit kapı değildir.
    """
    with pytest.raises(ValueError) as e:
        _, m = _cift(vocab)
        modul_ekle(m, ModulSpec(ad="x", r=4, alpha=8,
                                hedefler=GECERLI + ("boyle.katman.yok",)))
    assert "KISMEN" in str(e.value)

    # NEGATİF dal: tümü geçerliyse durmaz. Katman sayısı DESEN sayısı değildir:
    # küçük model 2 blok × 6 hedef = 12 katman, +1 gömme = 13.
    _, m2 = _cift(vocab)
    tak: List[str] = modul_ekle(m2, ModulSpec(ad="y", r=4, alpha=8, hedefler=GECERLI_G))
    assert len(tak) == 2 * 6 + 1, f"beklenen 13 katman, gelen {len(tak)}"
    assert "embedding.embedding" in tak


def test_tam_uymsuzluk_kapisi_hala_ATESLER(vocab: Vocabulary) -> None:
    """Eski kapı korunmalı: hiçbir hedef uymazsa DUR (iki dal birlikte)."""
    with pytest.raises(ValueError):
        _, m = _cift(vocab)
        modul_ekle(m, ModulSpec(ad="z", hedefler=("yok.boyle.katman",)))
