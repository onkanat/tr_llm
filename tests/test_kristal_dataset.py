"""T-0056: KristalDataset.get_batch testleri.

KristalDataset sınıfının temel veri sağlama sözleşmesini sınar:
  (i)   ŞEKİL: x ve y tensörleri [batch_size, block_size] boyutunda olmalı.
  (ii)  KAYDIRMA: y, x'in tam 1 token ileri kaydırılmış halidir.
        y[b, k] == data[offset_b + k + 1] olmalı (hedefler maskesiz ham veriden gelir).
  (iii) DTYPE: x ve y torch.int64 (torch.long) türünde olmalı (embedding için zorunlu).
  (iv)  KÜÇÜK VERİ YOLU: len(data) <= block_size durumunda çökme olmaz, şekiller tutarlıdır.
  (v)   PAD YOĞUNLUĞU: data/train_chat_balanced.bin üzerinde pencerelerde <PAD> (id 1)
        oranı ~%47-%50 bandındadır (> 0.30 olarak belgelenip pinlenir).
"""
import os
import numpy as np
import pytest
import torch
from scripts.train_step_demo import KristalDataset


def test_dataset_batch_shape(tmp_path):
    """(i) ŞEKİL: x ve y tensörleri [batch_size, block_size] boyutunda olmalı."""
    bin_file = tmp_path / "test_data.bin"
    raw_arr = np.arange(1000, dtype=np.uint16)
    raw_arr.tofile(bin_file)

    batch_size = 8
    block_size = 32
    ds = KristalDataset(str(bin_file), block_size=block_size)
    x, y = ds.get_batch(batch_size)

    assert x.shape == (batch_size, block_size), f"Beklenen {(batch_size, block_size)}, gelen {x.shape}"
    assert y.shape == (batch_size, block_size), f"Beklenen {(batch_size, block_size)}, gelen {y.shape}"


def test_dataset_shift_property(tmp_path):
    """(ii) KAYDIRMA: y, x'in tam 1 token ileri kaydırılmış halidir.
    
    y[b, k] == data[offset_b + k + 1] ve x[b, k] == data[offset_b + k].
    """
    bin_file = tmp_path / "test_shift.bin"
    # Tekil ve tahmin edilebilir değerler
    raw_arr = np.arange(500, dtype=np.uint16) + 100
    raw_arr.tofile(bin_file)

    block_size = 16
    batch_size = 4
    ds = KristalDataset(str(bin_file), block_size=block_size)
    x, y = ds.get_batch(batch_size)

    raw_mem = np.memmap(str(bin_file), dtype=np.uint16, mode='r')

    for b in range(batch_size):
        first_token = x[b, 0].item()
        # raw_arr içinde bu ilk tokenın konumunu bul
        matches = np.where(raw_mem == first_token)[0]
        assert len(matches) > 0, f"Token {first_token} veride bulunamadı"
        offset = matches[0]

        # x doğrulaması
        expected_x = raw_mem[offset:offset + block_size].astype(np.int64)
        np.testing.assert_array_equal(x[b].numpy(), expected_x)

        # y doğrulaması (1 token ileri kaydırılmış)
        expected_y = raw_mem[offset + 1:offset + 1 + block_size].astype(np.int64)
        np.testing.assert_array_equal(y[b].numpy(), expected_y)

        # Mutanta duyarlı kritik iddia: y asla x ile özdeş olamaz
        assert not np.array_equal(x[b].numpy(), y[b].numpy()), "y kaydırılmamış (x ile özdeş)"


def test_dataset_dtypes(tmp_path):
    """(iii) DTYPE: x ve y torch.int64 türünde olmalı (embedding katmanı gereksinimi)."""
    bin_file = tmp_path / "test_dtype.bin"
    raw_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint16)
    raw_arr.tofile(bin_file)

    ds = KristalDataset(str(bin_file), block_size=4)
    x, y = ds.get_batch(2)

    assert x.dtype == torch.int64, f"x dtype torch.int64 olmalı, gelen: {x.dtype}"
    assert y.dtype == torch.int64, f"y dtype torch.int64 olmalı, gelen: {y.dtype}"


def test_dataset_small_data_fallback(tmp_path):
    """(iv) KÜÇÜK VERİ YOLU: len(data) <= block_size (max_idx <= 0) durumunda çökme olmaz."""
    bin_file = tmp_path / "test_small.bin"
    # Sadece 8 tokenlık çok küçük bir veri
    raw_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    raw_arr.tofile(bin_file)

    # block_size = 16 > len(data)
    block_size = 16
    batch_size = 3
    ds = KristalDataset(str(bin_file), block_size=block_size)

    # get_batch çökmemeli ve len(data) - 1 boyutunda tensör dönmeli
    x, y = ds.get_batch(batch_size)

    expected_len = len(raw_arr) - 1  # 7
    assert x.shape == (batch_size, expected_len)
    assert y.shape == (batch_size, expected_len)

    # Kaydırma küçük veri yolunda da geçerli olmalı
    for b in range(batch_size):
        np.testing.assert_array_equal(x[b].numpy(), raw_arr[:expected_len].astype(np.int64))
        np.testing.assert_array_equal(y[b].numpy(), raw_arr[1:expected_len + 1].astype(np.int64))


def test_dataset_pad_density_measurement():
    """(v) PAD YOĞUNLUĞU: data/train_chat_balanced.bin pencerelerinde <PAD> oranı belgelenir."""
    chat_bin = "data/train_chat_balanced.bin"
    if not os.path.exists(chat_bin):
        pytest.skip(f"{chat_bin} bulunamadı, atlanıyor.")

    block_size = 128
    batch_size = 20
    ds = KristalDataset(chat_bin, block_size=block_size)

    total_targets = 0
    total_pad = 0

    # 10 batch x 20 pencere = 200 pencere (25,600 token)
    for _ in range(10):
        _, y = ds.get_batch(batch_size)
        y_np = y.numpy()
        total_targets += y_np.size
        total_pad += (y_np == 1).sum()

    pad_ratio = total_pad / total_targets

    # Ölçüm belgesi: Danışman ~%47.8, test ~%49.5 ölçtü.
    # Gevşek sınır: PAD yoğunluğu > %30 olmalıdır (kaybolursa yakalanır)
    assert pad_ratio > 0.30, f"Beklenen PAD yoğunluğu > 0.30, ölçülen: {pad_ratio:.4f}"
