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


# =========================================================================
# G5 (T-0131): Çoklu-Kayıt Pencereleri, Kırpılmış Prompt ve Mutant Sızıntı Testleri
# =========================================================================

def test_multi_record_window_masking():
    """G5 (vii): Aynı pencerede ardışık birden fazla soru-cevap (çoklu kayıt) bulunması."""
    # Kayıt 1: PROMPT_1 -> <OUTPUT> -> ANS_1 -> <EOS>
    # Kayıt 2: PROMPT_2 -> <OUTPUT> -> ANS_2 -> <EOS>
    # Son: PAD
    x = np.array([[101, OUTPUT_START_ID, 201, EOS_ID, 102, OUTPUT_START_ID, 202, EOS_ID, PAD_ID]])
    y = np.array([[OUTPUT_START_ID, 201, EOS_ID, 102, OUTPUT_START_ID, 202, EOS_ID, PAD_ID, PAD_ID]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    
    # Beklenen:
    # Pos 0: 101 (Prompt 1) -> -100
    # Pos 1: OUTPUT_START_ID -> Hedef 201 korunmalı
    # Pos 2: 201 -> Hedef EOS_ID korunmalı
    # Pos 3: EOS_ID -> -100 (EOS sızıntısı kapalı)
    # Pos 4: 102 (Prompt 2) -> -100 (yeni prompt maskeli)
    # Pos 5: OUTPUT_START_ID -> Hedef 202 korunmalı
    # Pos 6: 202 -> Hedef EOS_ID korunmalı
    # Pos 7: EOS_ID -> -100 (EOS sızıntısı kapalı)
    # Pos 8: PAD_ID -> -100
    expected = np.array([[-100, 201, EOS_ID, -100, -100, 202, EOS_ID, -100, -100]])
    np.testing.assert_array_equal(res, expected)


def test_clipped_prompt_without_output():
    """G5 (viii): Kırpılmış pencerede yalnızca prompt kuyruğu var, <OUTPUT> pencereye girmemiş."""
    x = np.array([[101, 102, 103, 104]])
    y = np.array([[102, 103, 104, 105]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    # <OUTPUT> olmadığı için tüm pencere -100 olmalı
    assert np.all(res == -100), "Kırpılmış prompt parçasında tüm hedefler -100 olmalı"


def test_clipped_output_without_eos():
    """G5 (ix): Pencere <OUTPUT> ile başlamış ama pencere sonuna kadar <EOS> gelmemiş (uzun cevap kesilmiş)."""
    x = np.array([[OUTPUT_START_ID, 201, 202, 203]])
    y = np.array([[201, 202, 203, 204]])
    
    res = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    # Tüm hedefler cevap içinde kaldığı için korunmalı
    expected = np.array([[201, 202, 203, 204]])
    np.testing.assert_array_equal(res, expected)


def _mutant_leak_prompt_targets(x: np.ndarray, targets: np.ndarray, output_start_id: int, eos_id: int) -> np.ndarray:
    """Mutant maskeleme fonksiyonu: Prompt veya EOS bölgesini kasıtlı olarak sızdırır."""
    res = mask_prompt_targets(x, targets, output_start_id, eos_id)
    # Kasıtlı hata (mutant): İlk prompt token'ının maskesini kaldır ve hedefini sızdır!
    # Eğer x'te <OUTPUT> öncesi prompt varsa oraya gerçek hedefi geri koy
    batch_size, seq_len = x.shape
    for b in range(batch_size):
        for i in range(seq_len):
            if x[b, i] != output_start_id and res[b, i] == -100:
                # Kasıtlı sızıntı mutantı
                res[b, i] = targets[b, i]
                return res
    return res


def _mutant_leak_eos_targets(x: np.ndarray, targets: np.ndarray, output_start_id: int, eos_id: int) -> np.ndarray:
    """Mutant maskeleme fonksiyonu: EOS sonrası token'ı kasıtlı olarak açık bırakır (legacy sızıntısı)."""
    return _legacy_inline_mask(x, targets, output_start_id, eos_id)


def _verify_no_prompt_leakage(x: np.ndarray, targets_masked: np.ndarray, output_start_id: int, eos_id: int):
    """Denetçi: SFT maskelenmiş hedeflerde prompt ve EOS sızıntısı olmadığını doğrular.
    
    Herhangi bir sızıntı tespit edilirse AssertionError fırlatır.
    """
    batch_size, seq_len = x.shape
    for b in range(batch_size):
        seq = x[b]
        is_output = False
        for i in range(seq_len):
            tok = seq[i]
            if tok == output_start_id:
                is_output = True
            elif tok == eos_id:
                # EOS token'ının kendisinden sonraki hedefe geçişi -100 olmalı
                if targets_masked[b, i] != -100:
                    raise AssertionError(f"Batch {b}, Pos {i}: EOS konumundaki hedef maskelenmemiş (sızıntı!)")
                is_output = False
            elif not is_output:
                # Prompt bölgesinde hedef mutlaka -100 olmalı
                if targets_masked[b, i] != -100:
                    raise AssertionError(f"Batch {b}, Pos {i}: Prompt hedefi maskelenmemiş (sızıntı! tok={tok})")


def test_mutant_leakage_detection():
    """G5 (x): Mutant test — Sentetik prompt veya EOS sızıntısı olduğunda denetçi KIRMIZI,
    doğru mask_prompt_targets çalıştığında YEŞİL olmalıdır."""
    x = np.array([[101, 102, OUTPUT_START_ID, 201, EOS_ID, PAD_ID]])
    y = np.array([[102, OUTPUT_START_ID, 201, EOS_ID, PAD_ID, PAD_ID]])
    
    # 1. Doğru fonksiyon YEŞİL olmalı
    clean_masked = mask_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    _verify_no_prompt_leakage(x, clean_masked, OUTPUT_START_ID, EOS_ID)
    
    # 2. Prompt sızıntısı mutantı KIRMIZI olmalı (AssertionError)
    prompt_mutant = _mutant_leak_prompt_targets(x, y, OUTPUT_START_ID, EOS_ID)
    with pytest.raises(AssertionError, match="Prompt hedefi maskelenmemiş"):
        _verify_no_prompt_leakage(x, prompt_mutant, OUTPUT_START_ID, EOS_ID)
        
    # 3. EOS sızıntısı mutantı KIRMIZI olmalı (AssertionError)
    eos_mutant = _mutant_leak_eos_targets(x, y, OUTPUT_START_ID, EOS_ID)
    with pytest.raises(AssertionError, match="EOS konumundaki hedef maskelenmemiş"):
        _verify_no_prompt_leakage(x, eos_mutant, OUTPUT_START_ID, EOS_ID)

