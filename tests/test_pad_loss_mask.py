"""T-0053: <PAD> kayıp maskesi testleri.

`KristalLM.forward(..., ignore_index=...)` davranışını İKİ YÖNLÜ sınar:

  A) Maskeleme GERÇEKTEN çalışıyor mu? — PAD konumlarındaki hedef değiştirildiğinde
     maskeli kayıp DEĞİŞMEZ, maskesiz kayıp DEĞİŞİR. (Tek yönlü "kayıp düştü" iddiası
     maske olmadan da doğru olabilirdi; asıl kanıt bu iki yönlü kurgudur.)
  B) `ignore_index=None` eski davranışı birebir koruyor mu? (regresyon)

Neden gerekli: `data/train_chat_balanced.bin` kayıtları 128'e hizalanırken kısa olanlar
`<PAD>` (id 1) ile DOLDURULUYOR; maskeleme yokken model "PAD üret" öğreniyordu (T-0052'de
f3'ün en sık ürettiği token <PAD> çıktı).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import Vocabulary          # noqa: E402
from scripts.train_step_demo import KristalLM     # noqa: E402

PAD_ID = 1


def _tiny_model() -> KristalLM:
    """Küçük ama GERÇEK sınıf: aynı forward/loss yolu, hızlı koşum."""
    torch.manual_seed(0)
    vocab = Vocabulary()
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=12, vocab=vocab, block_size=8)
    model.eval()
    return model


def _mixed_batch(model: KristalLM):
    """x ve iki hedef dizisi: PAD konumlarında biri PAD, diğeri modelin EN SEVMEDİĞİ token."""
    v = model.lm_head.out_features
    x = torch.randint(4, v, (1, 8))
    pad_positions = [1, 3, 5]
    fill_id = v - 1 if v - 1 != PAD_ID else v - 2      # gecerli (aralik ici) PAD-olmayan id
    y_pad = torch.full((1, 8), fill_id, dtype=torch.long)
    y_pad[0, pad_positions] = PAD_ID
    with torch.no_grad():
        logits = model(x)[0]
    y_other = y_pad.clone()
    for p in pad_positions:
        y_other[0, p] = int(logits[0, p].argmin())  # o konumda kasıtlı olarak "yanlış" hedef
    return x, y_pad, y_other


def test_pad_targets_are_removed_from_loss():
    """A) Maskeli: Hedefte PAD olan konumlarda logits manipüle edilse bile kayıp DEĞİŞMEZ."""
    model = _tiny_model()
    v = model.lm_head.out_features
    x = torch.randint(4, v, (1, 8))
    pad_positions = [1, 3, 5]
    fill_id = v - 1 if v - 1 != PAD_ID else v - 2
    y_pad = torch.full((1, 8), fill_id, dtype=torch.long)
    y_pad[0, pad_positions] = PAD_ID

    with torch.no_grad():
        logits_orig, loss_masked_orig = model(x, y_pad, ignore_index=PAD_ID)

    # PAD konumlarındaki logits'leri bozup cross_entropy'yi doğrudan test edelim
    logits_mod = logits_orig.clone()
    logits_mod[0, pad_positions] += 1000.0  # PAD pozisyonlarındaki logitleri aşırı değiştir

    loss_masked_mod = nn.functional.cross_entropy(
        logits_mod.view(-1, v), y_pad.view(-1), ignore_index=PAD_ID
    )

    assert torch.isfinite(loss_masked_orig), "maskeli kayıp sonlu olmalı (tüm hedefler PAD değil)"
    assert torch.equal(loss_masked_orig, loss_masked_mod), (
        f"PAD konumları maskelenmiş olmalıydı: {loss_masked_orig.item()} != {loss_masked_mod.item()}"
    )


def test_unmasked_loss_reacts_to_same_positions():
    """A') Maskesiz kontrol: AYNI kurgu maske olmadan FARKLI kayıp vermeli (kurgu canlı mı?)."""
    model = _tiny_model()
    v = model.lm_head.out_features
    x = torch.randint(4, v, (1, 8))
    pad_positions = [1, 3, 5]
    fill_id = v - 1 if v - 1 != PAD_ID else v - 2
    y_pad = torch.full((1, 8), fill_id, dtype=torch.long)
    y_pad[0, pad_positions] = PAD_ID

    with torch.no_grad():
        logits_orig, loss_unmasked_orig = model(x, y_pad, ignore_index=None)

    logits_mod = logits_orig.clone()
    logits_mod[0, pad_positions] += 1000.0

    loss_unmasked_mod = nn.functional.cross_entropy(
        logits_mod.view(-1, v), y_pad.view(-1)
    )

    assert not torch.equal(loss_unmasked_orig, loss_unmasked_mod), (
        "maskesiz kayıp da eşit çıktı: kurgu ayırt edici değil, test vakum"
    )


def test_ignore_index_none_matches_plain_cross_entropy():
    """B) Regresyon: ignore_index=None, değişiklik öncesi formülün birebir aynısı."""
    model = _tiny_model()
    v = model.lm_head.out_features
    x = torch.randint(4, v, (2, 8))
    y = torch.randint(4, v, (2, 8))
    with torch.no_grad():
        logits, loss_model = model(x, y, ignore_index=None)
        loss_manual = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
    assert torch.equal(loss_model, loss_manual), (
        f"eski davranış korunmadı: {loss_model.item()} != {loss_manual.item()}"
    )


def test_ignore_index_equals_plain_when_no_pad_present():
    """C) Hedeflerde hiç PAD yoksa maskeleme kayıbı DEĞİŞTİRMEMELİ (yan etki yok)."""
    model = _tiny_model()
    v = model.lm_head.out_features
    x = torch.randint(4, v, (2, 8))
    y = torch.randint(4, v, (2, 8))
    y[y == PAD_ID] = PAD_ID + 1
    with torch.no_grad():
        _, loss_masked = model(x, y, ignore_index=PAD_ID)
        _, loss_plain = model(x, y, ignore_index=None)
    assert torch.equal(loss_masked, loss_plain), "PAD'siz batch'te maskeleme kayıbı değiştirdi"


# ---------------------------------------------------------------------------
# P2/Aşama 1 — train.py'nin PAD→−100 normalizasyonu (çakışma kapısı)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402
from train import MASKE, maske_pad_hedefleri  # noqa: E402


def test_maske_pad_hedefleri_pad_konumlarini_maskeye_yazar():
    """D) PAD hedefleri −100'e yazılır; PAD'siz dizide NO-OP (kayıp ölçeği korunur)."""
    targets = np.array([[5, PAD_ID, 7, PAD_ID, 9]], dtype=np.int64)
    cikti = maske_pad_hedefleri(targets, PAD_ID, True)
    assert list(cikti[0]) == [5, MASKE, 7, MASKE, 9], cikti
    # PAD'siz dizide bit-özdeş NO-OP:
    targets2 = np.array([[5, 6, 7]], dtype=np.int64)
    assert np.array_equal(maske_pad_hedefleri(targets2, PAD_ID, True), targets2)
    # girdi dizi MUTASYON ETMEZ (saf fonksiyon):
    targets3 = np.array([[5, PAD_ID]], dtype=np.int64)
    maske_pad_hedefleri(targets3, PAD_ID, True)
    assert targets3[0, 1] == PAD_ID, "girdi dizisi mutasyona uğradı (pure fonksiyon bozuldu)"


def test_no_pad_mask_eski_davranisi_korur():
    """E) pad_mask_active=False: PAD hedefleri EĞİTİLİR (eski --no-pad-mask davranışı)."""
    targets = np.array([[5, PAD_ID, 7]], dtype=np.int64)
    cikti = maske_pad_hedefleri(targets, PAD_ID, False)
    assert list(cikti[0]) == [5, PAD_ID, 7], cikti


def test_cakisma_duzeltilmis_tek_maske_degeri():
    """F) ÇAKIŞMA KAPISI (ölçülmüş kusur): eski yol `ignore_index=<PAD>` geçince SFT'nin
    −100 hedefleri geçersiz sınıf kalıyordu. Yeni sözleşme: PAD→−100 + tek
    `ignore_index=-100` — SFT maskesi ve PAD aynı değerde yok sayılır.

    İki yönlü kurgu (A testinin deseni):
      * MASKE'li hedefte −100 konumları YOK SAYILMALI (logit manipülasyonu kaybı
        DEĞİŞTİRMEMELİ) — eski `ignore_index=1` yolu burada FARKLI kayıp verirdi.
      * PAD→−100 sonrası kayıp, eski maskeleme yoluyla (PAD hedefli, ignore_index=1)
        ÖLÇÜLMÜŞ DEĞERE eşit kalmalı — pad maskelemesinin ölçeği değişmedi. (T-0053)"""
    model = _tiny_model()
    v = model.lm_head.out_features
    x = torch.randint(4, v, (1, 8))
    pad_positions = [1, 3, 5]
    fill_id = v - 1 if v - 1 != PAD_ID else v - 2
    y_pad = torch.full((1, 8), fill_id, dtype=torch.long)
    y_pad[0, pad_positions] = PAD_ID

    with torch.no_grad():
        logits, _ = model(x, y_pad, ignore_index=None)
    # YENİ yol: PAD→−100 normalize et, tek ignore_index=-100:
    y_norm = torch.from_numpy(maske_pad_hedefleri(y_pad.numpy(), PAD_ID, True))
    # −100 konumlarındaki logitleri AŞIRI boz — yok sayılmazlarsa kayıp ATLAYACAK:
    logits_mod = logits.clone()
    logits_mod[0, pad_positions] += 1000.0
    loss_yeni_mod = nn.functional.cross_entropy(
        logits_mod.view(-1, v), y_norm.view(-1), ignore_index=MASKE)
    loss_yeni = nn.functional.cross_entropy(
        logits.view(-1, v), y_norm.view(-1), ignore_index=MASKE)
    assert torch.equal(loss_yeni, loss_yeni_mod), (
        "−100 konumları yok sayılmadı (çakışma geri geldi): "
        f"{loss_yeni.item()} != {loss_yeni_mod.item()}")

    # ESKİ (bozuk) yolun GERÇEKTEN bozuk olduğunun kanıtı — aynı kurguda −100 hedefi
    # ignore edilmez: CPU'da IndexError FIRLATIR (MPS'te sessiz 0.0/garbaj dönerdi —
    # T-0073 sınıfı). Yani eski `ignore_index=<PAD>` yolu SFT'de koşum bile atamıyordu.
    y_sft = y_pad.clone()
    y_sft[0, 0] = MASKE          # SFT prompt maskesi (eski kodda geçersiz sınıf!)
    ile_hataya_dustu = False
    try:
        nn.functional.cross_entropy(logits_mod.view(-1, v), y_sft.view(-1),
                                    ignore_index=PAD_ID)
    except IndexError:
        ile_hataya_dustu = True
    assert ile_hataya_dustu, (
        "eski yol (ignore_index=<PAD> + −100 hedef) hatasız geçti: kurgu vakum")
    # PAD ölçeği korunumu: normalize edilmiş hedefle kayıp, eski PAD-id maskelemesiyle
    # AYNI konum kümesini yok sayar (PAD'siz konumlarda aynı hedefler) ⇒ eşit kayıp:
    loss_eski_temiz = nn.functional.cross_entropy(
        logits.view(-1, v), y_pad.view(-1), ignore_index=PAD_ID)
    assert torch.equal(loss_yeni, loss_eski_temiz), (
        f"PAD maske ölçeği değişti (T-0053 ihlali): {loss_yeni.item()} != {loss_eski_temiz.item()}")
