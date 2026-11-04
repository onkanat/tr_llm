# P2.5 KALEM-3 — LEGAL KART-KİMLİĞİ VERİSİ (v3): İLAN→HÜKÜM→RAPOR

**Tarih:** 2026-11-04 · **Repo:** ~/Git/tr_llm · **Deliverable:**
`data/pedagogy/legal_canonical_sft_v3.jsonl` (5.346 kayıt) + `.meta.json` +
üretici `scripts/prepare_legal_dataset_v3.py` (hepsi YENİ dosya) + bu hüküm.
**Sert kurallar (uyuldu):** commit YOK · Qdrant YOK · mevcut dosyalara yazma
YOK (yalnız okundu; `legal_canonical_sft_v2.jsonl` sha değişmedi) · eğitim YOK
(kalem 5 ayrıca koşulacak).

---

## 1. İLAN (ölçütler koşumdan önce donduruldu)

### 1.1 Görev ve donmuş ölçütler

| Kapı | Donmuş ölçüt |
|---|---|
| Kayıt bandı | +300-600 YENİ kayıt (üç hedefli desen; korpus maddelerinden, uydurma YOK) |
| Birebirlik | her «» alıntısı korpus madde satırından izlenebilir (bağımsız doğrulayıcı; v2 denetim kalıbı) |
| Şema | `{instruction, input, output}` birebir; FAZ-1 `validate_record` kapıları aynen |
| Dedupe | yeni-yeni VE yeni-v2 (sha256(instruction‖output); v2 kalıbı aynen) |
| VAL sözleşmesi | v3 son-50 == v2 son-50 BAYT-BİREBİR (bağımsız `cmp`); yeni kayıtlar tohum-42 ile ORTADA |
| UNK | her yeni kayıt tokenizer ölçümü (anka 33.114, mode=True); talimat şablonlarında 0 UNK hedefi |

### 1.2 GÖREV-1 bulgusu — K2 rig'inin tam şekli

Kaynak: `scripts/train_legal_ceket.py::kanonik_dogruluk` (2A/2C/p25c rig'i,
ölçüm kanıtı `p25c_legal_v3_olcum_rig_2026-11-04.json`).

- **Örnekleme:** `random.Random(42).sample(range(len(recs)), 20)` — TÜM kayıt
  listesinden; val/train ayrımı yok (`val_mi` yalnız raporlanır).
- **Karşılaştırma:** talimat `render_prompt` sonrası argmax üretim (≤48 jeton)
  vs altın gövde jeton-id'leri (`tok.encode(output)`, BOS/EOS soyulmuş).
- **kart4** = `gen[:4]==gold[:4]` → kart-TİP iskeleti. **etiket16** =
  `gen[:16]==gold[:16]` → kart-tip + kanun/kavram KİMLİĞİ. **K2 barı: etiket16
  ≥ 5/20** (p25c İLAN §3).
- **Kart-tip ilk-jeton şekilleri (ölçüldü, tam jeton dizileri):**
  - `Mevzuat Hükmü —` → `[mevzuat, <ENT>, <CAP>, h, ü, k, m, ü, </ENT>]` — **9
    jetonluk SABİT iskelet** (kart4'ü ilk 4'ü oluşturur);
  - `Mevzuat Yanıtı —` → `[mevzuat, yanıt, POSS_3SG]` + kanun adı (İş Kanunu:
    ilk-16 **TAM kanun numarasını 4 8 5 7** dahil eder);
  - `Kanun Tanımı:` → `[kanun, tanım, POSS_3SG, :]` + kavram jetonları;
  - `Kavram Ayrımı:` → `[kavram, ayrım, POSS_3SG, :, (, 1, )]` + kavram;
  - `Prosedür` → `[prosedür, …]` (2 jetonda ayrışır).
- **etiket16 kapsamı (b-kartları):** sabit iskelet + kanun ADI jetonları +
  `( kanun No` öneki. Yani K2'nin istediği şekil = *talimat→kart-tip eşlemesi*
  + *talimattaki kanun kimliğini çıktı iskeletine doğru aktarma* (uzun adlarda
  numara rakamları ilk-16 dışındadır; kısa adlı Yanıtı kartında numara dahil).
- **v2 ölçüm zemini kompozisyonu:** kanonik-20 örneklemi 18 Mevzuat Hükmü + 1
  Kanun Tanımı + 1 Kavram Ayrımı; val-50 ise 30 Mevzuat Yanıtı + 11 Kavram
  Ayrımı + 9 Prosedür (2A val avlusu).
- **Modelin edinemediği ŞEKİL (p25c §3.3 karakterizasyonu + jeton ölçümü):**
  1. **Kart iskeletinin ilk-4'ü kurulamadı** — 18/20 b-kartında ortak-önek
     yalnız 1 jeton (`mevzuat`); sabit `h ü k m ü` varlık bloğu üretilemiyor
     (argmax `<PROPER_NOUN>` dalıyor). CE 4,26'ya inerken bile iskelet oturmadı.
  2. **Talimat→kart-tip ayrımı yok** — 'Kanun Tanımı:'/'Kavram Ayrımı:'
     sorularına da %84'lük b1∪b3 'Mevzuat' çerçevesi üretiliyor.
  3. **Kanun adı↔numara bağı kurulamadı** — 'Kanun No. 477' halüsinasyonu
     (eğitim verisindeki numaralar 657-6698; talimatta Çevre/2872 yazsa da
     yanlış kanun adı+numara üretildi).
  4. **Eşleme denetimi öğretilmemişti** — v2 desen envanterinde KONU→MADDE-NO
     ve MADDE-NO→METİN yönlü hiçbir desen yok (yalnız konu→alıntı var);
     koşulsuz alan-istatistiği numara üretimini halüsinasyona itti.

## 2. HÜKÜM — ÜRETİM TAMAMLANDI, TÜM KAPILAR GEÇTİ

| Kapı | Ölçüt | Ölçüm | Hüküm |
|---|---|---|---|
| Kayıt bandı | 300-600 yeni | **533** (540 üretildi, 7 v2-örtüşmesi dedupe) | **GEÇTİ** |
| Desen kapsamı | a/b/c üçlü hedef | a=150 · b=150 · c=233 (c1 60/c2 50/c3 50/c4 50/c5 30-7) | **GEÇTİ** |
| Birebirlik | her alıntı izlenebilir | **629/629** (bağımsız varyant-doğrulayıcı) | **GEÇTİ** |
| Şema | 5.346/5.346 alan-birebir | **5346/5346** `{instruction,input,output}` | **GEÇTİ** |
| VAL-50 | bayt-özdeş | iç kanıt `True` + **bağımsız `cmp` EXIT=0**; kuyruk sha `67da2b94…cef99fee` her iki dosyada aynı | **GEÇTİ** |
| Montaj | v2 satırı kaybı 0 | bağımsız çokluk denetimi: kayıp **0**, yeni **533**; orta blok tohum-42 | **GEÇTİ** |
| UNK | her kayıt ölçülü | yeni blok **%1,235** (v2 %1,828'in altında); şablon UNK **0** | **GEÇTİ** |

## 3. RAPOR

### 3.1 Üretilen desenler (K2'nin eksik şekline birebir hedefli)

| Desen | Kayıt | Talimat → Çıktı kartı | Denetlenen kimlik |
|---|---|---|---|
| (a) madde-bulma | 150 | "'konu' hangi maddede düzenlenmiştir?" (talimatta numara YOK) → **Mevzuat Yanıtı** + Madde NO + birebir alıntı | konu→madde numarası |
| (b) ters-kimlik | 150 | "{k}. maddesi ne düzenler?" → **Mevzuat Hükmü** + birebir alıntı + kısa öz (başlıklıda yapısal başlık özü, başlıksızda alıntının ilk cümlesi — birebir) | numara→metin |
| (c1) kart Hükmü | 60 | YENİ şablon, talimatta '(Kanun No. N)' + etiket → çıktı `f1.rec_b` BİREBİR | ad↔numara bağı |
| (c2) kart Yanıtı | 50 | kaynak-madde sorusu → `rec_b_hedefli` çerçevesi BİREBİR | tip+etiket |
| (c3) kart Tanımı | 50 | YENİ tanım şablonları → `f1.rec_a` çıktısı BİREBİR | tip ayrımı |
| (c4) kart Ayrımı | 50 | YENİ bent çiftleri **(i, i+2)** (v2'de i,i+1 idi) + YENİ şablon → `f1.rec_c` biçimi BİREBİR | tip ayrımı |
| (c5) kart Dizisi | 23 | YENİ pencereler (ofset-1; v2'de ofset-0) → `f2.rec_d_dizi` BİREBİR | madde dizisi |

- **konu kapısı (a):** %60 resmi madde başlığı, %40 gövdeden birebir öbek
  (öbeğin `m.text` içinde geçiği doğrulanır; 'madde'/dipnot/kurulum/giriş
  öbekleri elenir; talimata numara sızması ayrıca kapılanır).
- **Uydurma YOK:** tüm alıntılar korpus madde satırından; başlıksız/Geçici-Ek
  maddeler (b) havuzuna ~%25 karıştı; kısa özler ya yapısal ('hüküm X
  konusundadır') ya alıntıdan birebir.
- **Kapsam:** 17/17 kanun temsil ediliyor (yeni kayıtlarda kanun başına 14-36).

### 3.2 Dosya-düzeyi kart dağılımı (K2 tip-dengesi dozu)

| Kart | v2 (4.813) | v3 (5.346) | Değişim |
|---|---|---|---|
| Mevzuat Hükmü | 4.275 (%88,8) | 4.485 (%83,9) | pay ↓ 4,9 puan |
| Mevzuat Yanıtı | 35 (%0,7) | **235** (%4,4) | **6,7×** |
| Kanun Tanımı | 181 (%3,8) | 231 (%4,3) | ↑ |
| Kavram Ayrımı | 156 (%3,2) | 206 (%3,9) | ↑ |
| Mevzuat Dizisi | 157 (%3,3) | 180 (%3,4) | ↑ |
| Prosedür | 9 | 9 | sabit |

Yeni blok tip-mix'i **tip-dengeli**dir (Hükmü %39,4 · Yanıtı %37,5 · Tanımı
%9,4 · Ayrımı %9,4 · Dizisi %4,3) — v2 yeni-bloğunun %93 b-çerçevesine karşı
bilinçli kırılma (p25c §3.6.1(a) önerisinin uygulanması).

### 3.3 Token/UNK ölçümü (anka 33.114, literal_entity_mode=True; v2 rig'i aynen)

- **Yeni blok:** 105.842 token · UNK 1.307 (**%1,235**) · UNK'lu kayıt 419/533
  (kaynak: korpus alıntıları/başlıklar; maks/kayıt 15) · talimatında UNK olan
  kayıt 88 (korpus başlık/öbek/kavram kelimeleri; şablon metinlerinde UNK 0 —
  CB şablonundaki 'hükmü' UNK'lu çıktı, 'Kanun metni sorusu' ile düzeltildi).
- **Tam v3:** 983.563 token · UNK 17.356 (**%1,765**) · ort. 184,0/kayıt
  (v2: 877.721 · %1,828 · 182,4) — UNK oranı v2'nin altında.

### 3.4 Montaj kanıtları

- v3 = tohum-42 karışık `[v2-train 4.763 + yeni 533]` (orta 5.296) + v2 son-50
  (kuyruk, yeniden serileştirilmeden).
- **Bağımsız doğrulama (üreteciden ayrı süreç):** satır 5.346; son-50 bayt-özdeş
  (`cmp` EXIT=0; iki kuyruk sha `67da2b94…cef99fee`); v2 satır kaybı 0; yeni
  satır 533; 5.346/5.346 şema-ok.
- **v2 dokunulmamış:** `legal_canonical_sft_v2.jsonl` sha
  `1c822c9f…bf6524f` == p25c hükmündeki donmuş değer.

### 3.5 Kalem-5 (yeniden koşum) için hazırlık hükmü

1. **Hazır.** v3 dosyası `train_p26_sft_replay.py --sft-path …v3.jsonl` ile
   doğrudan koşulabilir; val-50 bayt-özdeş olduğundan K1 ölçümü p25c çapası
   (4,782289 taban, final 4,259029) ile apples-to-apples kalır.
2. **Kanonik-20 yeni nüfus önizlemesi (tohum-42, nüfus 5.346):** idx
   [204, 217, 244, 260, 712, **767**, 839, 912, 1143, 1791, 1828, 1905, 2006,
   2253, 3456, 4139, 4467, 4837, 4931, 5238] → 15 Mevzuat Hükmü + 1 Mevzuat
   Yanıtı + 4 Kanun Tanımı; val'deki 0; **örneklemde yeni-kayıt 1/20**
   (beklenti ~2/20; %9,97'lik yeni pay). K2 örneklem kompozisyonu v2'ye yakın
   kalır — dozun K2'ye etkisi ÖĞRENME yoluyladır, örneklem yoluyla değil.
3. **Kalem-5 İLAN'ının dondurması gereken parametre:** train bloğu
   4.663 → **5.296** oldu. Pencere karışımı %20,003'te tutmak için pencere-n
   1.166 → **~1.324** olmalı; pencere-n 1.166 kalırsa karışım %18,0'a düşer
   (K3 ankrajı zayıflar). Karar İLAN'da açıkça dondurulmalı.
4. **Dürüst risk beyanı:** K2 0/20'nin kaynaklandığı dört eksik şeklin hepsi
   (§1.2/1-4) dozda hedefli; ancak yeni pay dosyada %10,0 ve K2 barı 5/20
   katı. Bu dozun K2'yi barın üstüne taşıyacağı GARANTİ DEĞİLDİR — kalem 5
   ölçer; K2 yine kalırsa teşhis 'doz yetersiz' olur (sıradaki eksen: tip-dengeli
   yeniden-ağırlıklama / b-çerçeve alt-örnekleme, yeni İLAN gerektirir).
5. K1 riski düşük (val sabit, yeni kayıtlar alan-içi); K3 riski v3'e özel
   değişmedi (yeni kayıtların tümü legal-alan; ölçüm kalem 5'te).

### 3.6 Çıktılar ve sha256 (programatik `hashlib`/`shasum`; elle dökülmedi)

| Dosya | sha256 |
|---|---|
| `data/pedagogy/legal_canonical_sft_v3.jsonl` (5.346 kayıt; son-50 v2 ile bayt-özdeş) | `3b9ca7d92ea77f47dcd60527543355cbdea5a9190a1ed399da5d69039e6a0325` |
| `data/pedagogy/legal_canonical_sft_v3.jsonl.meta.json` | meta içinde `sha256_v3`/`sha256_uretici` beyanlı |
| `scripts/prepare_legal_dataset_v3.py` (üretici; FAZ-1/v2 kapıları import) | `095cfb3dee0f8a599f79fa14ef7997d5b24f81974856d29cab5819fca83ee1f9` |
| `data/pedagogy/legal_canonical_sft_v2.jsonl` (girdi — DOKUNULMADI) | `1c822c9f41bd0896f65ed85ff24f59d883346b46f26f8ba6c58922ad2bf6524f` (p25c ile aynı) |
| `data/eval/p25d_kart_kimligi_2026-11-04.md` (BU dosya) | — (yazım sonrası üst ajana bildirilir) |

Düşenler: 7 kayıt (c5 dizilerinin v2 ofset-0 pencereleriyle birebir örtüşmesi —
dedupe kapısı), 0 şema dışı, 0 birebir-düşüş. Eğitim koşulMADI; commit YAPILMADI
(controller commit eder).