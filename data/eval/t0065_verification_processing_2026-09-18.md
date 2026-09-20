# Bağımsız Doğrulama Bulgularının İşlenmesi — T-0065

- **Damga (UTC):** 2026-09-18T07:33Z · **Yazan:** claude · **Görev:** T-0065
- **Tetikleyici:** `data/eval/t0063_verification.json` — hüküm **PASS_WITH_FINDINGS**, bulgular **B1 / B2 / B3**
- **Doğrulayıcı dosyası sha256:** `cb41734f911268fdf623b5aab16f14cd0fd82b9d0f88f45ba7b57b178a90319e` · **bu turda DEĞİŞMEDİ** (bkz §7)
- **Bu kayıt güncel hash zincirinin TEK YETKİLİSİDİR.**

---

## 1. Tek cümlede

Doğrulayıcının **B1** bulgusu, T-0063 raporumda ve notlarımda **%0 kolunun B′ eksenindeki yönünü ters okuduğumu** söylüyordu. **İddia ölçülmeden kabul edilmedi** — kendi ham verimden yeniden üretildi, **doğrulandı**, düzeltildi. **Hiçbir eşik, hiçbir ölçüm sayısı, hiçbir kol hükmü değişmedi.**

## 2. Bulgu B1 — yön tersliği: **yeniden üretildi**

Yöntem: `scratch/t0065_b1_verify.py`, T-0063 ölçümünün **sakladığı hitvec'lerden** McNemar'ı yeniden hesaplar (taşmasız: `fractions.Fraction` + `math.comb`). Çıktı: `scratch/t0065_b1_verify.json`.

| korpus | n | taban doğru | %0 doğru | n_kayıp | n_kazanç | düşüş (puan) | p | **yön** |
|---|---|---|---|---|---|---|---|---|
| `raw_full` | 11865 | 5143 | 5023 | 981 | 861 | +1.011 | 5.545e-03 | **KAYIP** |
| `gts_dict` | 6207 | 2834 | 2594 | 468 | 228 | +3.867 | 5.631e-20 | **KAYIP** |

> **İşaret kuralı (bu görevin konusu olduğu için açıkça yazılır):** `düşüş` sütunu **pozitifken KAYIP**tır ve aracın `drop_points` alanıyla **aynı işareti** taşır. Örn. `+1.011` = %0, tabana göre **1,011 puan kaybetti**. `n_kayıp` = taban doğru & kol yanlış; `n_kazanç` = taban yanlış & kol doğru. **Her iki korpusta da `n_kayıp > n_kazanç`** → yön **KAYIP**.

**Doğrulayıcı ile birebir uyum:** yeniden hesaplanan p değerleri (`0.00554507979819711` · `5.63146210386241e-20`) doğrulayıcının `p_exact` alanlarıyla **aynı** çıktı.

- **Özdeşlik denetimi geçti:** her kolda `n_kazanç − n_kayıp = kol_doğru − taban_doğru`.
- **Hitvec hizası doğrulandı:** uzunluk eşitliği sağlanmazsa betik `SystemExit` ile durur.
- Saklanan etiket çifti denetimi: `n01_loss==n_loss, n10_gain==n_gain` / `n01_loss==n_loss, n10_gain==n_gain`

## 3. Kök neden — **araçta değil, yürütücüde**

Aracın sakladığı alan `n01_loss: 981` — yani **981 zaten KAYIP sayısıydı** ve yeniden hesapla birebir uyuştu. **Araç kusurlu değildi.** Hata, metinde `981/861` çiftini **ters okumamdı**. Aracın kendi alanları düşüşü zaten söylüyordu: `drop_points` **+1.011**, `retention_pct` **97,67** (< 100).

> **Ders (yeniden kullanılabilir):** bir çiftin hangi teriminin ne olduğunu **varsayma** — alan adını ve **işaretini** oku.

**Doğru okuma:** %0 **A′'da da B′'de de tabanın altında**; üstün olduğu **tek** eksen **C** (+0,0371). *"%0 dış-alan yetkinliğinden verip iç-alan kazancı satın alıyor."*

## 4. Bulgular B2 / B3 — sınır kayıtları (**eşik değiştirilmedi**)

**B2 (medium) — raw_full K3 Farkı (10.55 Puan) İstatistiki Olarak 10.0 Eşiğinden Ayırt Edilemiyor**

raw_full üzerinde K3 farkı nokta tahmini olarak 10.552 puandır (ilan edilen eşik: >= 10.0). Ancak eşleştirilmiş %95 Güven Aralığı [9.328, 11.776] (bootstrap: [9.313, 11.850]) olup alt sınır 10.0 eşiğinin altındadır. H0: fark <= 10.0 hipotezinin tek yönlü p değeri p=0.1884tür. Dolayısıyla raw_fullun K3 kontrolünü geçmesi istatistiksel bir kesinlik değil, nokta tahmini başarısıdır. Claudenun rapora yazdığı F7 sınır uyarısı yerindedir ve bu analizle desteklenmektedir.

**B3 (low) — carpenter_v2 / gts_dict Koruma Oranının %95 Güven Aralığı %90 Eşiğini Kesmektedir**

carpenter_v2nin gts_dict üzerindeki göreli koruma oranı nokta tahmini %91.53tür (ilan edilen sınır bandı %88-92). Eşleştirilmiş bootstrap %95 GA [%89.81, %93.25] olup alt sınır %90 eşiğinin altına inmektedir. Bu kol zaten A ekseninden elendiği için nihai birleşik kararı etkilememektedir.

Her iki bulgu da **sınır kaydıdır**: ilan edilen eşikler (`raw_full` K3 ≥ 10 puan · göreli koruma ≥ %90) **değiştirilmedi**. Eksenin geçerliliği B2'nin marjına dayanmaz — asıl kanıt **K1/K3'ün şans düzeyine oturmasıdır** (32,6–33,4).

## 5. Karar üzerindeki etki: **YOK**

| soru | cevap |
|---|---|
| F4 kararı değişti mi? | **HAYIR** — hâlâ `data/kristal_model_f4_r05.pt` (%5) |
| %0'ın elenme gerekçesi değişti mi? | **HAYIR** — %0 **A′'dan** elendi (+%15,96/+%14,84 > ilan edilen %+10) |
| Eşik değişti mi? | **HAYIR** — hiçbiri |
| Ölçüm sayısı değişti mi? | **HAYIR** — yalnız bir **yön okuması** ve iki **sınır kaydı** |
| Ölçüm betiği değişti mi? | **HAYIR** — `scratch/t0063_b_probe.py` (§7) |
| Doğrulayıcının dosyası değişti mi? | **HAYIR** — `data/eval/t0063_verification.json` (§7) |

## 6. Değişen dosyalar — **tam digest, ALAN ALAN**

"Önce" değerleri `scratch/t0065_orig_hashes.json`'dan (bu turun başında betikle yakalandı), "sonra" değerleri canlı `sha256`. **Hiçbir digest elle yazılmadı.**

| dosya | sha256 (ÖNCE) | sha256 (SONRA) |
|---|---|---|
| `.agent-bus/notes/T-0063.md` | `671bf1e5ecb317c38f368c7a1cae3264324fd0bdcaa31ef50e854bbe71cfad23` | `69d9fb1baaf7a6cdf18cce41dd7b6672a47d103a3670b207bb2c2e596a743077` |
| `.agent-bus/notes/T-0064.md` | `35b9cb2af022bddaa88a14e73347df5712507e5de68b76ed809740c494549b0c` | `804ea4c3b8d02f17d73f8b1f90dae9f85f1cc57e57f63f936f5b9be3d1282d40` |
| `data/eval/b_prime_axis_design_2026-09-18.md` | `08ce2b59b2f4ce8162e9e808f156e6deb001754cc6a9c869a54bb5acf7c16778` | `1777bbf73db92716fffddefef6bef48476b95109741ab5023efa411eac6e16b2` |
| `data/eval/f4_decision_2026-09-18.md` | `2522b2311d44bcd52cd42289ec150624208d6753c08f4c2132c6226c93cc465b` | `048886ffff1cfe7b331f05d6a8dc0f1240249ff1cb492df5465c6498b320c5f3` |
| `data/eval/forgetting_measurement_design_2026-09-18.md` | `6b201f343ca4a57c95e50279eb6dae709c238f7c49530e065ab5a13a48de54fe` | `6d8e29f78b1ef6b845c28a54d9df69fc7580f01d0787d3e3008e606b054306f7` |
| `data/eval/t0063_b_prime_axis_2026-09-18.json` | `7cd4bd7057ee98fa317b059ed358501c20a0003896ce7440d046ba67944d260f` | `1f771ae0a20c1c2be498509ea365c9d8d48d6f72bfeb9ce3979a9a9f400be758` |
| `scratch/t0063_make_report.py` | `96af8ae62b594bb18b2a4c9642a1b92469d143cae0ecf000dfaaa77e4f2bfe81` | `554d3e3e04fb0496d67eb4ab8107fc45a1f747cbc9824f3d2113f03c25def1a0` |
| `scratch/t0064_make_decision.py` | `1d0688e44952097b9c936ac0276c127077d734b4c5d4ad695648f994d264aefd` | `1cc388cd38bb47e871b4bf024494c7e12f28670786499750831791b8f12a64ec` |

## 7. **Değişmeyenler** — kanıt: digest aynı

Beklenen ve **ölçülen**: bu iki dosyaya hiç dokunulmadı. Yani düzeltme **ölçümün kendisine** ve **doğrulayıcının dosyasına** dokunmadı.

| dosya | sha256 (turun başında = şimdi) |
|---|---|
| `data/eval/t0063_verification.json` | `cb41734f911268fdf623b5aab16f14cd0fd82b9d0f88f45ba7b57b178a90319e` |
| `scratch/t0063_b_probe.py` | `1b9f6fd241e828ecec8b81be83901609256dad09523774c9157f43dddd62ef4c` |

Ayrıca hiçbir `.pt` / `.bin` / `vocab*.json` dosyasına dokunulmadı.

## 8. Yeniden üretilen kayıtlar ve **pozitif kontrollü** denetim

**T-0063 raporu** ve **T-0064 karar kaydı** BETİKLE yeniden üretildi (`scratch/t0063_make_report.py`, `scratch/t0064_make_decision.py`); hash'leri **betik hesaplayıp yeniden denetledi** (8/8 ve 8/8 uyumlu), elle digest yazılmadı.

**T-0064 karar kaydının §5 sapma tablosu artık CANLI ÖLÇÜMDEN BESLENMİYOR:** T-0065 aynı iki dosyayı **ikinci kez** düzenlediği için canlı `sha()` hesabı "bu kayıttan sonra" derken **T-0065 sonrası** değeri gösterir ve iki turu sessizce birleştirirdi. Tablo artık arşiv kayıtlarından okunuyor ve **T-0064 anının değerlerini birebir koruduğu doğrulandı** (orijinal kayıtla satır satır aynı).

### Denetim sonucu (yalnız-yokluk değil, **pozitif kontrol de** arandı)

| ne arandı | sonuç |
|---|---|
| `**ezberin kaybıdır**` — iddia konumundan **silinmiş** | 6 dosyada **YOK** ✔ |
| `B′'de ise tabanı **geçiyor**` — iddia konumundan **silinmiş** | 6 dosyada **YOK** ✔ |
| `anlamlı KAZANÇ` — kalan geçişler **düzeltme bağlamında** mı | düzeltme dışı geçiş: **0** ✔ |
| **Pozitif kontrol:** iddiayı taşıyan her dosyada **düzeltme izi var** mı | 6/6 **VAR** ✔ |
| **Yapısal:** raporun `findings[F5]` maddesi artık "kazanç" iddia ediyor mu | **ETMİYOR** ✔ |

> **Neden pozitif kontrol şart:** yalnız "yanlış kalıp yok" aramak, **düzeltme hiç yazılmamış olsa da geçerdi**. Denetim bu yüzden iki yönlü: kalıp gitti **ve** düzeltme var.

**Denetim hükmü:** **TEMİZ** — düşen kontrol yok

## 9. Artefakt hash tablosu (tam sha256)

| dosya | boyut | sha256 |
|---|---|---|
| `data/eval/t0065_verification_processing_2026-09-18.md` | — | *(bu dosya; kendini hash'lemez)* |
| `data/eval/t0063_b_prime_axis_2026-09-18.json` | 15951 | `1f771ae0a20c1c2be498509ea365c9d8d48d6f72bfeb9ce3979a9a9f400be758` |
| `data/eval/f4_decision_2026-09-18.md` | 9398 | `048886ffff1cfe7b331f05d6a8dc0f1240249ff1cb492df5465c6498b320c5f3` |
| `data/eval/b_prime_axis_design_2026-09-18.md` | 10459 | `1777bbf73db92716fffddefef6bef48476b95109741ab5023efa411eac6e16b2` |
| `data/eval/forgetting_measurement_design_2026-09-18.md` | 18928 | `6d8e29f78b1ef6b845c28a54d9df69fc7580f01d0787d3e3008e606b054306f7` |
| `data/eval/t0063_verification.json` | 12243 | `cb41734f911268fdf623b5aab16f14cd0fd82b9d0f88f45ba7b57b178a90319e` |
| `.agent-bus/notes/T-0063.md` | 16538 | `69d9fb1baaf7a6cdf18cce41dd7b6672a47d103a3670b207bb2c2e596a743077` |
| `.agent-bus/notes/T-0064.md` | 5103 | `804ea4c3b8d02f17d73f8b1f90dae9f85f1cc57e57f63f936f5b9be3d1282d40` |
| `.agent-bus/notes/T-0065.md` | 5910 | `8ac5b30f846bfd45687fb8380b1225d33144e7237b544acabe61c2f478542010` |
| `scratch/t0063_make_report.py` | 13067 | `554d3e3e04fb0496d67eb4ab8107fc45a1f747cbc9824f3d2113f03c25def1a0` |
| `scratch/t0064_make_decision.py` | 12907 | `1cc388cd38bb47e871b4bf024494c7e12f28670786499750831791b8f12a64ec` |
| `scratch/t0065_b1_verify.py` | 7165 | `02dbc36797cd67ea5d8cfb147cad77825634c1813050d9a6618a5416ae855663` |
| `scratch/t0065_b1_verify.json` | 5764 | `597aa2d8e15af5b9e637ba987a9d1b7e5782f0cf325e2d82c7f79197ad255566` |
| `scratch/t0065_orig_hashes.json` | 1229 | `5811d0c0275b868af18714e17f494e366c743312ce1a13028ba8b32bcd20e44c` |
| `scratch/archive/t0063_report_original.json` | 13083 | `7cd4bd7057ee98fa317b059ed358501c20a0003896ce7440d046ba67944d260f` |
| `scratch/archive/t0063_notes_original.md` | 11430 | `671bf1e5ecb317c38f368c7a1cae3264324fd0bdcaa31ef50e854bbe71cfad23` |
| `scratch/archive/t0064_decision_original.md` | 6343 | `2522b2311d44bcd52cd42289ec150624208d6753c08f4c2132c6226c93cc465b` |

---

*Kaynaklar: `data/eval/t0063_verification.json` (doğrulayıcı hükmü) · `scratch/t0065_b1_verify.json` (B1'in bağımsız yeniden üretimi) · `scratch/t0065_orig_hashes.json` (tur başı digestleri) · `scratch/archive/` (orijinal kayıtlar). Anlatı: `.agent-bus/notes/T-0065.md`. Karar kaydı: `data/eval/f4_decision_2026-09-18.md`. Ölçüm raporu: `data/eval/t0063_b_prime_axis_2026-09-18.json`.*
