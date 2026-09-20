# T-0073 TANI — `train.py` ön-eğitimde **öğrenme sinyali üretmiyor**

Damga: `2026-09-18T14:11:13Z` · Bu bir **TANI** raporudur; sayılar keşif koşumlarında elde edildi ve
tek, yeniden üretilebilir ölçüme toplandı — **önceden ilan edilmiş kapı değildir**.

## 0. Nasıl ortaya çıktı

Sondanın 3 adımlık doğrulama koşumu `rc=0` ile bitti ama **`Kayıp (Loss): 0.0000`** bastı.
Rastgele başlatılmış bir model 1.024 jeton üzerinde 0 kayıp veremez.

## 1. Ölçüm

| kontrol | sonuç |
|---|---|
| **D1** `<OUTPUT>` (id 1706) külliyatta | **2** kez / 100,000,000 jeton ⇒ **yok hükmünde** |
| **D2** maskeden sonra hayatta kalan hedef | blok 64: **0/19200** · blok 128: **0/38400** · blok 256: **0/76800** — üçünde de **%0,0000**, hayatta kalan pencere **0/300** |
| **D3** tümü `-100` hedef, `ignore_index=1` | **CPU → `IndexError: Target -100 is out of bounds.`** (gürültülü) · **MPS → `0.0`** (sessiz) |
| **D4** depodaki ön-eğitim yolları | `train_step_a_ablation.py`: düz-LM çağrısı ×2, maske çağrısı ×0 · `train_step_b1_canonical.py`: ×2/×0 · **`train.py`: düz-LM ×0, maske ×2** |

## 2. Mekanizma

`train.py` bir **SFT** eğiticisidir. `mask_prompt_targets` **kural 5**:
*"`<OUTPUT>` içermeyen pencerelerde **tüm** hedefler -100 yapılır."* A1 külliyatı düz metin
olduğu için bu kural **her pencerede** ateşler. Kalan hedef: **0** ⇒ gradyan **sıfır**.

Üstüne bir sessizlik katmanı biner: kayıp `ignore_index=1` ile çağrılır ve maskeden çıkan
değer `-100`'dür. **CPU bu geçersiz hedefte `IndexError` verir; MPS vermez, `0.0` döndürür.** Yani
koşum MPS'te "başarıyla tamamlandı" diye biter — hata yok, uyarı yok, kayıp 0.

> **Hüküm: YOK.**

## 3. Bunun plana etkisi

T-0072'de ben `336 saat` yazdım (batch=1 varsayımı), Antigravity `2,91 saat` diye düzeltti
(batch ölçeği karışıklığı). **İkisi de yanlış bir soruyu tartışıyordu:** ölçülen, *hiçbir şey
öğrenmeyen* bir koşumun maliyetiydi.

Ek olarak: sonda (verim) **no-op koşum üzerinde ölçülürse iyimser olabilir** — tümü yok sayılan
hedefte PyTorch kısa yola girebilir. Bu yüzden sonda, **hedef düzeltildikten sonra** koşmalıdır.

## 4. Çözüm yönü

Ön-eğitim **düz sonraki-jeton** hedefi kullanmalıdır. Depoda bu yol zaten var — eski taban model
böyle eğitilmiş: `model(x, y, sign_mask)`, prompt maskesi **yok**. `train.py`'ye ön-eğitim modu
eklenmeli (prompt maskesi kapanır, `<PAD>` maskesi kalır) **veya** ayrı bir ön-eğitim döngüsü
kullanılmalıdır.

## 5. Bu raporun İDDİA ETMEDİĞİ şeyler

1. **Eski Kristal zinciri bozuk değildi** — ön-eğitimi düz-LM döngüsüyle yapılmıştı (D4).
2. **Model kalitesi hakkında hiçbir şey söylenmiyor** — kalite ölçülmedi.
3. Sondanın ölçüm düzeneği geçersiz değil; yalnız no-op üzerinde koşarsa **iyimser** olur.

Üreticiler ve tam digest'ler (JSON): `data/eval/t0073_pretrain_objective_finding_2026-09-18.json`.
İlgili: [[sessiz-soyagaci-geri-dusmesi]], [[vakum-kanarya-satir-numarasi]],
[[tavan-artefakti-kapi-gecmez-kanitsizlik]], [[sure-hesabinda-boyut-kontrolu]].
