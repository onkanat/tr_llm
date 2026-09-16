"""T-0057: Prompt maskeleme tekilleştirme ve compute_val_loss testleri.

Testler:
  (i)   AST/grep ile kod tabanında 'is_output' kopyası kalmadığı (yalnızca
        scripts/train_step_demo.py içinde geçtiği) doğrulanır.
  (ii)  compute_val_loss: dosya mevcut değilse -1.0 döner.
  (iii) compute_val_loss: dosya blok boyutundan çok kısaysa -1.0 döner.
  (iv)  compute_val_loss: DETERMINIZM - aynı girdiyle art arda iki çağrı aynı değeri üretir.
  (v)   compute_val_loss: küçük sentetik .bin ve küçük model ile sonlu bir kayıp döner.
  (vi)  KANIT: compute_val_loss hedef tensörü ile eski satır içi mantık karşılaştırılır;
        farkın YALNIZCA x[k] == eos_id konumlarında olduğu doğrulanır.
"""
import os
import subprocess
import numpy as np
import pytest
import torch
from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalLM, mask_prompt_targets
from scripts.evaluate_sft_benchmarks import compute_val_loss


def test_no_inline_prompt_masking_copies_left():
    """(i) AST/grep denetimi: 'is_output' değişkeni SADECE scripts/train_step_demo.py içinde olmalı."""
    res = subprocess.run(
        ["grep", "-rln", "is_output", "--include=*.py", "scripts/", "train.py"],
        capture_output=True,
        text=True,
        check=True
    )
    files = [f.strip() for f in res.stdout.strip().splitlines() if f.strip()]
    expected = ["scripts/train_step_demo.py"]
    assert files == expected, f"Beklenmeyen dosyalarda is_output kopyası bulundu: {files}"


def _create_mock_vocab():
    vocab = Vocabulary()
    # Temel belirteçlerin stoi'de olmasını sağla
    vocab.stoi["<PAD>"] = 1
    vocab.stoi["<EOS>"] = 3
    vocab.stoi["<OUTPUT>"] = 11
    vocab.itos[1] = "<PAD>"
    vocab.itos[3] = "<EOS>"
    vocab.itos[11] = "<OUTPUT>"
    return vocab


def _create_mock_model(vocab):
    torch.manual_seed(42)
    from scripts.evaluate_sft_benchmarks import DEVICE
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=12, vocab=vocab, block_size=16)
    model.to(DEVICE)
    model.eval()
    return model


def test_compute_val_loss_file_not_found():
    """(ii) compute_val_loss: Dosya yoksa -1.0 döner."""
    vocab = _create_mock_vocab()
    model = _create_mock_model(vocab)
    loss = compute_val_loss(model, "non_existent_path.bin", vocab, block_size=16, n_batches=2)
    assert loss == -1.0


def test_compute_val_loss_short_file(tmp_path):
    """(iii) compute_val_loss: Veri çok kısaysa (val_tokens <= block_size + 1) -1.0 döner."""
    vocab = _create_mock_vocab()
    model = _create_mock_model(vocab)
    
    bin_file = tmp_path / "too_short.bin"
    raw_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    raw_arr.tofile(bin_file)

    loss = compute_val_loss(model, str(bin_file), vocab, block_size=16, n_batches=2)
    assert loss == -1.0


def test_compute_val_loss_determinism(tmp_path):
    """(iv) DETERMINIZM: Aynı model ve veriyle art arda iki compute_val_loss çağrısı eşit değer üretir."""
    vocab = _create_mock_vocab()
    model = _create_mock_model(vocab)

    bin_file = tmp_path / "det_data.bin"
    # 200 tokenlık veri: val_split_idx = 180, val_tokens = 20 > 16 + 1
    # İçinde OUTPUT ve EOS olsun
    np.random.seed(0)
    arr = np.random.randint(4, len(vocab.stoi), size=300).astype(np.uint16)
    for i in range(180, 280, 25):
        arr[i] = 11  # <OUTPUT>
        arr[i+10] = 3 # <EOS>
    arr.tofile(bin_file)

    loss1 = compute_val_loss(model, str(bin_file), vocab, block_size=16, n_batches=3)
    loss2 = compute_val_loss(model, str(bin_file), vocab, block_size=16, n_batches=3)

    assert loss1 != -1.0
    assert np.isclose(loss1, loss2, atol=1e-7), f"Determinizm ihlali: {loss1} != {loss2}"


def test_compute_val_loss_finite_loss(tmp_path):
    """(v) compute_val_loss: Küçük sentetik veri ve modelle sonlu geçerli kayıp döner."""
    vocab = _create_mock_vocab()
    model = _create_mock_model(vocab)

    bin_file = tmp_path / "valid_data.bin"
    arr = np.random.randint(4, len(vocab.stoi), size=400).astype(np.uint16)
    # Val dilimine bol miktarda OUTPUT ve EOS ekle
    for i in range(360, 390, 10):
        arr[i] = 11  # OUTPUT
        arr[i+5] = 3 # EOS
    arr.tofile(bin_file)

    loss = compute_val_loss(model, str(bin_file), vocab, block_size=16, n_batches=3)
    assert np.isfinite(loss), f"Kayıp sonlu değil: {loss}"
    assert loss > 0.0


def test_compute_val_loss_target_difference_only_at_eos():
    """(vi) KANIT: compute_val_loss'taki hedef üretimi, eski satır içi döngüyle YALNIZCA x[k]==EOS'ta farklıdır."""
    output_start_id = 11
    eos_id = 3
    
    # 1D sentetik sekans
    x_np = np.array([101, 102, output_start_id, 201, 202, eos_id, 1, 1], dtype=np.int64)
    y_np = np.array([102, output_start_id, 201, 202, eos_id, 1, 1, 1], dtype=np.int64)

    # Eski satır içi döngü simülasyonu
    targets_old = y_np.copy()
    seq = x_np.tolist()
    if output_start_id in seq:
        is_output = False
        for i in range(len(seq)):
            tok = seq[i]
            if tok == output_start_id:
                is_output = True
            if not is_output:
                targets_old[i] = -100
            if tok == eos_id:
                is_output = False
    else:
        targets_old[:] = -100

    # Yeni paylaşılan fonksiyon çağrısı (evaluate_sft_benchmarks.py içindeki gibi)
    targets_new = mask_prompt_targets(x_np[None, :], y_np[None, :], output_start_id, eos_id)[0]

    eos_pos = np.where(x_np == eos_id)[0]
    non_eos_pos = np.where(x_np != eos_id)[0]

    # EOS konumunda eski mantık y_np[5] == 1 bırakıyordu, yeni mantık -100 yaptı
    for p in eos_pos:
        assert targets_old[p] == 1, f"Eski mantıkta sızıntı vardı (1 bekleniyordu, {targets_old[p]})"
        assert targets_new[p] == -100, f"Yeni mantıkta sızıntı kapandı (-100 bekleniyordu, {targets_new[p]})"

    # EOS dışındaki TÜM konumlarda eski ve yeni birebir aynı olmalı
    np.testing.assert_array_equal(targets_new[non_eos_pos], targets_old[non_eos_pos])
