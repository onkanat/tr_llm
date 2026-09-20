# T-0069 — TÜM MODEL CHECKPOINT'LERİ SİLİNDİ (geri dönüşsüz)

**Sonuç (tek cümle):** 51 model checkpoint'i (**20.41 GiB)
kalıcı olarak silindi; veri kümesi olan 6 `.pt` dosyasının **6/6**
tanesi bit düzeyinde değişmedi.

| | |
|---|---|
| görev | T-0069 (Onkanat talimatı: *"Tüm Modelleri silin"*) |
| silme biçimi | **kalıcı sil (rm)** — kullanıcı onayı, AskUserQuestion 18 Eyl 2026 |
| kapsam | **hepsi, temel model dâhil** (kullanıcı onayı) |
| manifesto (silmeden önce) | `data/eval/t0069_pre_delete_manifest.json` · sha256 `4e1c4d51a896936909b1750b52088e8939e03e966b42fd8ea84934e0993e6e61` |
| silme sonucu | `data/eval/t0069_deletion_result_2026-09-18.json` · sha256 `bd4a1618e047cb84a227fe647cae126b82dd1fd0ad1624e888b116790dd0ef48` |
| damga | 2026-09-18T12:12:35Z |

## 0. Yeni modelin adı: **ANKA**

> *"Yeni modelin adı Anka çünkü Kristalin küllerinden doğuyor."* — Onkanat, 18 Eyl 2026

Yeni checkpoint'ler `data/anka_*.pt` olarak yazılacak; `kristal_*` adlandırması **bitti**.
Mimari korunur (kullanıcı: *"mimari temel yapısını koruyun"*) → `scripts/train_step_demo.py`'deki
`KristalLM` sınıf adları **değişmez**; değişen yalnız checkpoint adıdır.

## 1. Ne silindi

Ölçüm (bkz. `data/eval/t0069_pre_delete_manifest.json`, sınıflandırıcıdan geçti: hedeflerin **51/51**'i torch zip
arşivinde ≥50 tensor deposu taşıyan **model checkpoint'i**):

| kitle | yol deseni | adet |
|---|---|---|
| üretim zinciri | `data/*.pt` | 33 |
| arşivlenmiş RAG pilotu | `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_*.pt` | 9 |
| yedek + probe | `scratch/**/*.pt` | 9 |
| **toplam** | | **51** |

Bu kitle şunları içerir: **kabul edilen üretim noktası** `data/kristal_model_f4_r05.pt` (T-0064
kararı), taban `data/kristal_model.pt`, ve F3/F4 zincirinin tamamı (f3a…f4_r10). Karar **kayıtları**
duruyor; **artefaktları** yok.

## 2. Ne KORUNDU — ve neden

`6` dosya hedef kumeye **hiç girmedi**: bunlar model checkpoint'i değil, **pickle edilmiş
veri kümesi nesnesidir** (`SampleAlignedDataset` / `FastSampleAlignedDataset` / `FastRagPilotDataset`).
`data/b1_5_splits/*` resmî B1.5 eğitim bölmesinin kendisidir.

| yol | bayt | silme öncesi sha256 |
|---|---|---|
| `data/b1_5_splits/train_ds.pt` | 69857903 | `688931afacf15cc8b3b855d3f45634f452789089f5ada1b9f43d2d4b6b139bd5` |
| `data/b1_5_splits/train_fast_ds.pt` | 44027101 | `80408c67b7a3684c3229512f28864e893a3aaaba1d8dbd9276dee6cc0b27d085` |
| `data/b1_5_splits/val_ds.pt` | 16681209 | `59b9a64c02ca00b9128da3252cd4bc507ac8376545496199af02ee82654fcb26` |
| `data/b1_5_splits/val_fast_ds.pt` | 5471221 | `e15cbd6f0985851fdb896f1601653225d0ce17e6a6a9bac9e9dfd43782db93ff` |
| `data/_archive/rag_pilot_invalidated/train_fast_ds.pt` | 4755355 | `f538f2cc95325cc6082d8f3d172500aebc8f5e2b8a21708b0249b60fe01ace47` |
| `data/_archive/rag_pilot_invalidated/val_fast_ds.pt` | 593967 | `fa38009db4272bb6963be8c7c4c76714bbb3e8e894a9b570ed729e878809e23f` |

| yol | silme sonrası durum |
|---|---|
| `data/b1_5_splits/train_ds.pt` | ayni |
| `data/b1_5_splits/train_fast_ds.pt` | ayni |
| `data/b1_5_splits/val_ds.pt` | ayni |
| `data/b1_5_splits/val_fast_ds.pt` | ayni |
| `data/_archive/rag_pilot_invalidated/train_fast_ds.pt` | ayni |
| `data/_archive/rag_pilot_invalidated/val_fast_ds.pt` | ayni |

**Yorum farkı açıkça kayda geçiyor:** "tüm modeller" ifadesi *veri kümesi* `.pt`'lerini kapsamaz.
Kullanıcı bu yorumu düzeltmek isterse bu dosyalar hâlâ yerindedir (ve gerekiyorsa ayrı bir onayla silinir).

## 3. Silme nasıl güvenli yapıldı (geri dönüşsüzlüğe karşı)

`.pt` dosyaları `.gitignore`'da (`*.pt`) ve **repo dışında yedek yok** (tarandı) ⇒ git'ten geri
getirilemez. Bu yüzden sıra:

1. **Manifesto** (`data/eval/t0069_pre_delete_manifest.json`): 51 hedefin yol/bayt/mtime/**tam sha256**'sı + 6 korunanın digest'i. Manifest'in kendi sha256'sı `4e1c4d51a8969369…`.
2. **Betik içi bütünlük kapısı:** silme betiği manifestin sha256'sını beklenen değerle karşılaştırır; tutmazsa `DURDURULDU`.
3. **Silmeden önce yeniden doğrulama:** her hedefin boyutu + sha256'sı manifestoyla kıyaslanır; sapan dosya **silinmez** (`uyusmazlik`).
4. **Artımlı kayıt:** her silmeden sonra sonuç JSON'u yazılır.
5. **Sonra kanıt:** korunan 6 dosyanın sha256'sı yeniden ölçülür.

Ölçülen sonuç: `atlanan = 0` · `uyusmazlik = 0` ·
`kalan_hedef = 0` · disk **26.1 GiB → 46.2 GiB**
(kazanç **20.10 GiB**).

## 4. ⚠️ Kırılan bağımlılıklar (bilerek kabul edildi, Anka yazılana kadar geçerli)

`data/kristal_model.pt` şu giriş noktalarının **varsayılan yoluydu**; artık YOK:

- `chat_prompt.py:309` (üretim çıkarımı) · `train.py:89` · `train_dpo.py:89`
- `scripts/retrain_clean_models.py` (5 aşamalı zincir) · `scripts/run_goal_pipeline.py`

Yani **şu anda hiçbir model yok**: çıkarım, sohbet ve tüm türev eğitim komutları çalışmaz.
Bu, "sil ve yeniden eğit" talimatının doğrudan sonucudur; geçici kırılma olarak kayda geçer.

## 5. Bu görevin İDDİA ETMEDİĞİ şeyler

1. **"Yeni model daha iyi olacak" DEĞİL.** Sıfırdan eğitim devralınmış soyağacını
   ([[model-lineage-continued-not-scratch]]) yeniden üretmez; **yeni** bir model üretir. Kıyas
   cümleleri ancak Anka'nın kendi aşama kapılarıyla kurulabilir.
2. **Silmenin kendisi bir kazanç değil**, bir sıfırlamadır. Ceketin ölçülmüş kusuru (T-0067/T-0068:
   genel alanda belgeden bağımsız −0,52…−0,63 CE) silinerek **giderilmiş sayılmaz** — ancak Anka
   o eksende ölçülürse konuşulabilir.
3. **Veri elden geçirme ve eğitim bu görevin kapsamı DIŞINDA.** Ayrı görevlerde yürütülür; oradaki
   kararlar (ölçek = ~100 M jeton, tek yürütücü = claude-code) kullanıcı tarafından 18 Eyl'de verildi.

## 6. Artefaktlar (tam 64 karakter sha256; raporun kendisi hariç)

| yol | bayt | sha256 |
|---|---|---|
| `data/eval/t0069_pre_delete_manifest.json` | 15840 | `4e1c4d51a896936909b1750b52088e8939e03e966b42fd8ea84934e0993e6e61` |
| `data/eval/t0069_deletion_result_2026-09-18.json` | 10594 | `bd4a1618e047cb84a227fe647cae126b82dd1fd0ad1624e888b116790dd0ef48` |
| `scratch/t0069_manifest.py` | 5464 | `2f4e1b3d55b2ff7815c291f51d2df267f680dce8385939cc6c52e79b22b50433` |
| `scratch/t0069_delete.py` | 4355 | `81990e3cbf194437dcc47952f8ab008adb2dab6827d42eca0b1db2ea3fde5d95` |
| `scratch/t0069_make_report.py` | 10595 | `7137572d1e392b1b7e9acc7ec3b05a7ed881d70d7255d31ee0df95a084fddfb0` |

Raporun kendi sha256'sı tabloya **konmadı**: kendine referanslı bir satır, değeri yazıldığı anda
değiştirir. Nihai değer, üretici betiğin koşum çıktısında basılır (`[sha256]`) ve kapanış kanıtına
o değer yazılır.

## EK — SİLİNEN 51 CHECKPOINT'İN ANIT TABLOSU

Bu tablo, yok olan tek kopyanın **tek kaydıdır**. Hiçbir satırı başka bir yerde tutulmuyor.

| yol | bayt | mtime (UTC) | sha256 |
|---|---|---|---|
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_best.pt` | 472572950 | 2026-09-13T20:27:55Z | `19a23a11396997cc64ed883b37e4473af2a5b05ce67364d4e741190bee8d1c73` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep1.pt` | 472572767 | 2026-09-13T19:44:54Z | `ca7233e11ef4bcd79c8d5bcf39ed46a277ca88901e297c6c3fd887db7ffdfd31` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep2.pt` | 472572767 | 2026-09-13T19:57:25Z | `81325720631ca747560a0b890860a32110154455d1747a075e2dd549dc01a38f` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep3.pt` | 472572767 | 2026-09-13T20:06:43Z | `aded4b7a037c2c52091d4ca8760f09b07b63efd5663923f860678b004081eaca` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep4.pt` | 472572767 | 2026-09-13T20:15:46Z | `730d5242bda577e46cccfa447434481def2e1851e705330bb8a09c2b98bfb840` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep5.pt` | 472572767 | 2026-09-13T20:18:35Z | `e7e8117e290aa1147d922c93e0209ce44b0a9e8be28485c565b42eb0bac693bd` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep6.pt` | 472572767 | 2026-09-13T20:21:28Z | `2d47f840a57ca9272f9f9e6fe706f11c85a6824fbf1c520db6e3cb77519f94ed` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep7.pt` | 472572767 | 2026-09-13T20:25:06Z | `2cfae17f71e3cdbc8f0b20f8d77080f25ed0d991779c321e757529e35699c6c9` |
| `data/_archive/rag_pilot_invalidated/kristal_rag_pilot_ep8.pt` | 472572767 | 2026-09-13T20:27:54Z | `dae643011c5095c1b1b93fa93f4759267edb755694e5fccac429c5844c0f07a9` |
| `data/kristal_b1_5_best.pt` | 472572529 | 2026-09-13T18:14:30Z | `02e15b8198f8f3334609eb52691d699e80f88972245377c1a79772cb42ae1679` |
| `data/kristal_b1_5_entity_ready.pt` | 472794619 | 2026-09-14T02:06:54Z | `b684dd4899ac0be5035635db6a9477c1ae7ea3aced50b6097b2229d63d1d4330` |
| `data/kristal_b1_5_epoch1.pt` | 472572529 | 2026-09-13T16:40:15Z | `7e5f07fe08c0ac076addab8d7b5eb0dd4e4fe03ddc8d0263e2f653f822d92e85` |
| `data/kristal_b1_5_epoch2.pt` | 472572529 | 2026-09-13T17:21:23Z | `25c43aa8cf0251656f1a795a3559b6d9cbf9b4371091533205c80dda96b5714b` |
| `data/kristal_b1_5_epoch3.pt` | 472572529 | 2026-09-13T18:14:29Z | `02e15b8198f8f3334609eb52691d699e80f88972245377c1a79772cb42ae1679` |
| `data/kristal_carpenter_model.pt` | 374026701 | 2026-09-13T08:40:29Z | `5f0b8c71727d15e338e4301634b0d05652c3230cebc532b1124aff3761e93b4d` |
| `data/kristal_model.pt` | 472572529 | 2026-09-13T18:14:31Z | `02e15b8198f8f3334609eb52691d699e80f88972245377c1a79772cb42ae1679` |
| `data/kristal_model_backup.pt` | 343943271 | 2026-09-09T07:54:57Z | `eb0d5350262196d121ffc002244b8163bc922850f3f631ce10ee24fd57faa131` |
| `data/kristal_model_backup_v160.pt` | 463601831 | 2026-09-12T18:17:10Z | `bf2805a0bfa228e2f4234c35d8a4826f32d111a0b5d254e0394528108d478ca3` |
| `data/kristal_model_base.pt` | 281717382 | 2026-06-15T06:17:19Z | `c77849d272e31fc52495ff27aad854ca47859b55ebe3360f8eeb6e707312c4da` |
| `data/kristal_model_carpenter_v2.pt` | 372126054 | 2026-09-16T23:45:20Z | `1c81f3059ab18779cab3e55898de3b1f33059b62fea34e3d2ef059a2e1342f3f` |
| `data/kristal_model_f3.pt` | 372124920 | 2026-09-16T17:33:43Z | `eefc3ab85dcfb454dd80cc1b4519ec9c923df2587ad1c6ecaa4a2a535daebbfd` |
| `data/kristal_model_f3_clean.pt` | 372125626 | 2026-09-16T20:05:25Z | `fc964af53534b818329c0da6e86aabfb729144615b85bfb6adc6e3c40e767aee` |
| `data/kristal_model_f3_clean_sft.pt` | 372126054 | 2026-09-16T19:40:29Z | `03db1630ff95d3ec0bbaa1bc9b9ebedcbecbc6ed75676b57f266f3991b8142bf` |
| `data/kristal_model_f3_sft.pt` | 372125348 | 2026-09-16T16:35:31Z | `5a0a03c22d4efdff331deb826024bc6f7cce416da13aaaf85e4b24d2cf117108` |
| `data/kristal_model_f3a.pt` | 372125027 | 2026-09-16T16:01:42Z | `dbf79bc8149bc8fa4df6e7e5b24f7377d7a1f25113bccef0feecaf59e6ff205d` |
| `data/kristal_model_f3b.pt` | 372125027 | 2026-09-16T16:04:46Z | `027a75e71a7d0872d8f7fc356767652115418e40703a4a737db2d619815bd710` |
| `data/kristal_model_f4_r05.pt` | 372125348 | 2026-09-17T01:39:23Z | `bc35352af126179e21b803159de13250fe54c5a4d940a110afe94ad1f76ba2fe` |
| `data/kristal_model_f4_r10.pt` | 372125348 | 2026-09-17T01:50:27Z | `724bbdbf91b8fa37723488e3753aaf69e50e395e5558b701164c5b6d38178b77` |
| `data/kristal_model_f4_replay.pt` | 372125733 | 2026-09-17T00:47:13Z | `6c81409244811dd8976bb078644a4ee4374501c471920b4dbd9fe38bb8c5d407` |
| `data/kristal_model_pre_clean.pt` | 463601831 | 2026-09-13T08:06:47Z | `d1f04433d81b5691f310bca691baf50f90be0527701c4b757460e5c1b368dcac` |
| `data/kristal_model_sft.pt` | 374025447 | 2026-09-13T08:38:35Z | `b6764c7062ce627a0ba9437cf052a28d0ac4e2e4cf3b3df3e3110d05a1ba2957` |
| `data/kristal_model_step_a_ep1.pt` | 468454020 | 2026-09-13T10:55:04Z | `2c29e7f7179d0c929ae89c2ec13cef6a64326ea9ab580a5d71d9c6d6cf3d7812` |
| `data/kristal_model_step_a_ep2.pt` | 468454020 | 2026-09-13T11:25:57Z | `ae36f36d25ed88b6db27638e7af0b3d4cfd8718fb66fe835a5fe47ad987fa301` |
| `data/kristal_model_step_a_ep3.pt` | 468454020 | 2026-09-13T12:06:12Z | `f6d8ee88a05de407860a441b3da4f1b3354d10b45cfd3d01436be3d31757e698` |
| `data/kristal_model_step_a_final.pt` | 468454258 | 2026-09-13T12:06:13Z | `809bba6d851b16ffd5382e0bd3548b34fd35430345415f217576d06b05740861` |
| `data/kristal_model_step_b1_ep1.pt` | 468515643 | 2026-09-13T13:14:52Z | `4d2bcfa66a3b3c304f0d4a478b72609ce4dc37cad0db06af74f237eed8c49720` |
| `data/kristal_model_step_b1_ep2.pt` | 468515643 | 2026-09-13T14:09:10Z | `9a9f46a002bc3f713faa9395e6af7d0d088b30c5bc4d380f63626b528d2604aa` |
| `data/kristal_model_step_b1_ep3.pt` | 468515643 | 2026-09-13T14:48:01Z | `99497f5cf476c2338659dabeea2af8f3752c11d71c1059368f6365900e980d4c` |
| `data/kristal_model_step_b1_final.pt` | 468515881 | 2026-09-13T14:48:03Z | `01d7918e4e5fa8e4fdbf0c4628b41a734639d9736d54f3307b44a0b5fd77174e` |
| `data/kristal_rag_arm_a_best.pt` | 472572950 | 2026-09-14T06:36:44Z | `e276081526db9af76afad35cb63c4de84ff4b763594dbb08efa66dddf42be81d` |
| `data/kristal_rag_arm_b_best.pt` | 472794262 | 2026-09-14T07:59:36Z | `27802a2d837f6e3efe4ecd6961c15366fa87f8e829e23af459be384db9685d3b` |
| `data/kristal_rag_arm_c_best.pt` | 472794262 | 2026-09-14T11:14:05Z | `1bd6e444f71b822057a8f2d25b14e2917cafe2ef39e3e325a89d5270d686affa` |
| `scratch/f3_backup_2026-09-16/kristal_carpenter_model.pt` | 374026701 | 2026-09-16T15:29:58Z | `5f0b8c71727d15e338e4301634b0d05652c3230cebc532b1124aff3761e93b4d` |
| `scratch/f3_backup_2026-09-16/kristal_model.pt` | 472572529 | 2026-09-16T15:29:55Z | `02e15b8198f8f3334609eb52691d699e80f88972245377c1a79772cb42ae1679` |
| `scratch/f3_backup_2026-09-16/kristal_model_pre_clean.pt` | 463601831 | 2026-09-16T15:29:56Z | `d1f04433d81b5691f310bca691baf50f90be0527701c4b757460e5c1b368dcac` |
| `scratch/f3_backup_2026-09-16/kristal_model_sft.pt` | 374025447 | 2026-09-16T15:29:57Z | `b6764c7062ce627a0ba9437cf052a28d0ac4e2e4cf3b3df3e3110d05a1ba2957` |
| `scratch/f3_backup_2026-09-16/kristal_model_step_b1_final.pt` | 468515881 | 2026-09-16T15:29:55Z | `01d7918e4e5fa8e4fdbf0c4628b41a734639d9736d54f3307b44a0b5fd77174e` |
| `scratch/t0046_smoke.pt` | 372124385 | 2026-09-16T12:48:28Z | `cc92dd8b82af164e092ec3aa9382e3c70e86b8f693d9f1c07ff1d9f25ab65282` |
| `scratch/t0046_verify_smoke.pt` | 372125134 | 2026-09-16T13:09:53Z | `1e9ddeccbdbdec8153c8b831ffba4b15d1858c6c3b7ae5dd53e97602063f958b` |
| `scratch/t0053_padfix_probe.pt` | 372125134 | 2026-09-16T18:54:48Z | `27780dbf7c57057c2079ff50b248e7e0695558fc26970c88182cbb54ea21003c` |
| `scratch/t0053_unmasked_probe.pt` | 372125348 | 2026-09-16T19:15:16Z | `8e7f15006a7208e9ad91fdb3691d0de7e2c6b008a66508255a5e1d35cf100065` |
