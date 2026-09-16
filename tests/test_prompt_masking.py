"""T-0055: Prompt maskeleme ve EOS sızıntısı testleri.

`mask_prompt_targets` saf fonksiyonunun davranışını sınar:
  (i) <OUTPUT> öncesi hedefler -100 olmalı.
  (ii) <OUTPUT> bölgesindeki hedefler korunmalı (içerik ve <EOS> üretimi).
  (iii) <EOS> sonrası hedefler -100 olmalı.
  (iv) x[k] == eos_id konumundaki hedef de -100 olmalı (EOS->sonraki sızıntısı kapalı).
  (v) <OUTPUT> içermeyen pencerelerde tüm hedefler -100 olmalı.
  (vi) Regresyon: (iv) sızıntı konumu dışındaki tüm konumlarda yeni fonksiyonun
       çıktısı eski satır içi döngü ile birebir aynı olmalı.
"""
import numpy as np
import pytest
from scripts.train_step_demo import mask_prompt_targets

OUTPUT_START_ID = 11
EOS_ID = 3
PAD_ID = 1


def _legacy_inline_mask(x: np.ndarray, targets: np.ndarray, output_start_id: int, eos_id: int) -> np.ndarray:
    """train.py'nin eski satır içi döngüsü (karşılaştırma oracle'ı)."""
    targets_np = np.array(targets, copy=True)
    batch_size_curr, seq_len = x.shape
    x_list = x.tolist()
    
    for b in range(batch_size_curr):
        seq = x_list[b]
        if output_start_id in seq:
            is_output = False
            for i in range(seq_len):
                token_id = seq[i]
                if token_id == output_start_id:
                    is_output = True
                
                if not is_output:
                    targets_np[b, i] = -100
                    
                if token_id == eos_id:
                    is_output = False
        else:
            targets_np[b, :] = -100
            
    return targets_np


def test_mask_before_output_start():
    """(i) <OUTPUT> görülmeden önceki prompt hedefleri -100 olmalı."""
    # seq: [PROMPT_1, PROMPT_2, <OUTPUT>, CONTENT_1, <EOS>, PAD]
    x = np.array([[101, 102, OUTPUT_START_ID, 201, EOS_ID, PAD_ID]])
    # y: [PROMPT_2, <OUTPUT>, CONTENT_1, <EOS>, PAD, PAD]
    y = np.array([[102, OUTPUT_START_ID, 201, EOS_ID, PAD_ID, PAD_ID]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    # 0 ve 1. konumlar <OUTPUT>'tan öncedir, hedefler -100 olmalı
    assert res[0, 0] == -100, "PROMPT_1 hedefi -100 olmalı"
    assert res[0, 1] == -100, "PROMPT_2 hedefi -100 olmalı"


def test_preserve_output_and_eos_production():
    """(ii) <OUTPUT> bölgesindeki hedefler korunmalı, <EOS> üretimi öğrenilmeli."""
    x = np.array([[101, 102, OUTPUT_START_ID, 201, 202, EOS_ID, PAD_ID]])
    y = np.array([[102, OUTPUT_START_ID, 201, 202, EOS_ID, PAD_ID, PAD_ID]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    # OUTPUT_START_ID konumunda hedef 201'dir (korunmalı)
    assert res[0, 2] == 201
    # 201 konumunda hedef 202'dir (korunmalı)
    assert res[0, 3] == 202
    # 202 konumunda hedef EOS_ID'dir (korunmalı: model EOS üretmeyi öğrenmeli!)
    assert res[0, 4] == EOS_ID, "<EOS> hedefinin kendisi korunmalı"


def test_mask_after_eos():
    """(iii) <EOS> sonrasındaki hedefler -100 olmalı."""
    x = np.array([[OUTPUT_START_ID, 201, EOS_ID, PAD_ID, PAD_ID]])
    y = np.array([[201, EOS_ID, PAD_ID, PAD_ID, PAD_ID]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    # EOS_ID pos 2, PAD pos 3 ve 4
    assert res[0, 3] == -100, "EOS sonrasındaki PAD konumu -100 olmalı"
    assert res[0, 4] == -100, "EOS sonrasındaki ikinci PAD konumu -100 olmalı"


def test_mask_at_eos_position_leakage_closed():
    """(iv) x[k] == eos_id konumundaki hedef -100 olmalı (sızıntı kapalı)."""
    x = np.array([[OUTPUT_START_ID, 201, EOS_ID, PAD_ID]])
    # x[2] == EOS_ID, y[2] == PAD_ID
    y = np.array([[201, EOS_ID, PAD_ID, PAD_ID]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    # Eski satır içi döngüde res[0, 2] == PAD_ID kalıyordu (sızıntı)
    # Yeni fonksiyonda -100 olmalı
    assert res[0, 2] == -100, "x[k]==EOS konumundaki hedef -100 olmalı (sızıntı kapalı)"


def test_mask_entire_window_without_output():
    """(v) <OUTPUT> içermeyen pencerelerde tüm hedefler -100 olmalı."""
    x = np.array([[101, 102, 103, 104, 105]])
    y = np.array([[102, 103, 104, 105, 106]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    assert np.all(res == -100), "<OUTPUT> olmayan pencerede tüm hedefler -100 olmalı"


def test_regression_against_legacy_except_leak():
    """(vi) Regresyon: Sızıntı konumu dışında yeni fonksiyon eski döngü ile birebir aynıdır."""
    np.random.seed(42)
    # 5 batch, 32 uzunluk
    B, T = 5, 32
    x = np.random.randint(100, 500, size=(B, T))
    y = np.random.randint(100, 500, size=(B, T))
    
    # Rastgele OUTPUT ve EOS enjekte et
    for b in range(B):
        if b < 4:  # Son batch OUTPUT içermesin
            out_pos = np.random.randint(2, 10)
            eos_pos = np.random.randint(15, 25)
            x[b, out_pos] = OUTPUT_START_ID
            x[b, eos_pos] = EOS_ID
            y[b, out_pos-1] = OUTPUT_START_ID
            y[b, eos_pos-1] = EOS_ID
            
    res_new = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    res_old = _legacy_inline_mask(x, y, OUTPUT_START_ID, EOS_ID)
    
    # Sızıntı konumu: x == EOS_ID olan yerler
    eos_mask = (x == EOS_ID)
    
    # 1. Sızıntı konumlarında yeni fonksiyon -100 vermeli, eski döngü ise y değerini korumuştu
    assert np.all(res_new[eos_mask] == -100)
    
    # 2. Sızıntı OLMAYAN tüm konumlarda yeni ve eski BİREBİR eşit olmalı
    non_eos_mask = ~eos_mask
    np.testing.assert_array_equal(res_new[non_eos_mask], res_old[non_eos_mask])
