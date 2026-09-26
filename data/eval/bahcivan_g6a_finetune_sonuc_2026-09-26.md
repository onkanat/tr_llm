# G6a (T-0137) RAPOR — FİNE-TUNE KOŞUMU + ÖLÇÜM (İLAN-3b) — 2026-09-26 (claude)

ILAN ≠ RAPOR: ilan `bahcivan_g6a_finetune_ilan_2026-09-26.md` (koşum ÖNCESİ
damga + koşum sırası canlılık kanıtları); hüküm BETİKTEN
(`scratch/t0137/bahcivan_ft_hukum_betigi.py`, elle sayı YOK).

## Hüküm

**KABUL_YOK** — `scratch/t0137/bahcivan_ft_hukum.json` (`d8802979…`;
betik rc=2). Kabul birleşimi A9 VE A10 VE A11 VE A12 VE A13 (İLAN-3b §5).

| Ayrac | Ölçüm (betikten) | Eşik | Dal |
|---|---|---|---|
| A9 ARTİS (Bahçıvan) | **+0,0129** (0,0365 − çıpa 0,0236) | ≥ +0,0230 | **BAND_ALTINDA** |
| A10 LM-bedeli \|f\| (marangoz) | **f = −0,0018** (0,1374 − çıpa 0,1392) | \|f\| ≤ 0,0230 | **KORUNDU** ✓ |
| A11 A-ekseni CE artış | **+%0,446** (3,3265 vs taban 3,3117) | ≤ +%10 | **A_GECTİ** ✓ |
| A12 B top-1 düşüş | **0,3965** | ≤ 5,0 | **B_GECTİ** ✓ |
| A13 tutarsızlık < %5 · ezber < %10 | tutarsızlık O-1 **%2,0** ✓ / O-2 **%43,0** ✗ · ezber 0,00/0,00 | — | **IHLAL** |

- Unutma kapısı (kap): `unutma_gec=True` — LM-bedeli yalnızlık değil, A+B
  eksenleri de korundu.
- Sonda ÖLÇÜM-1 DECOMP 0,2254; ÖLÇÜM-2 DECOMP 0,0568; kesişim (sızıntı)
  O-2 %0,00 (carve ayracı işliyor); DECOMP ROUGE hüküme bağlanmaz (aday eşik).

## Koşum kanıtları (sonuc.json + segments.jsonl — sürücü kapıları 3/3)

- `durum=BITTI` (rc=0), 3.117 sn ≈ 52 dk; adım süresi 0,41-0,50 sn/adım
  (beyanlı 0,506 — süre beyanı sağlandı). L1 ilk kayıp **3,3254** (beyanlı
  bant 2,0-3,5 ✓; L1_ESİK 4,736 — maske-dışı-oran %55,11 ölçümünden, betikten).
- Segmentler: seg_1 CE 3,3301/ROUGE 0,1508 (+0,0116) · seg_2 3,3281/0,1419
  (−0,0089) · seg_3 3,3265/**0,1374** (−0,0045) — **TEPE hiç düşmedi** (eşik
  −0,05), CE kapısı 3/3, ezber 0,00/0,00/0,00, tutarsızlık %1,0/%4,0/%2,0
  (marangoz heldout sondaları).
- Kaynak dağılımı bitiş: wiki 9.617 · ceket 19.234 · sft 19.234 (PATERN=5
  dengeli); `kisa_pencere_atlanan=0` (T-0113b); heldout baş=son çıpa
  `c397eb08…` **birebir**.
- VOCAB-UYUM 3 bin `f9940a8d…` ✓ · H1 tek eğitici (pgrep `EGITICI_YOK`
  koşum öncesi; lsof üç bin + MPS metallib, PID 22209) · CARRY sidecar YOK
  → optimizer sıfırdan (beyanlı).
- Nihai checkpoint: `scratch/t0137_g6a_kos/seg_3.pt` `bd68c450…` + `.opt.pt`
  sidecar; `seg_1` `8ab9e025…`, `seg_2` `9d923913…`.

## A9 BAND_ALTINDA + A13 İHLAL — bulgu ölçeği (tahmin değil, ölçümler)

1. **Yetenek rampası kırık:** taban 0,0236 → ft 0,0365 (**+0,0129**, bant
   +0,0230 altı). Kardeş rampa kanıtı marangozda 1,6 epoch → 0,1014 idi
   (P3); Bahçıvan'da 6,8 epoch → yalnız +0,0129. Marangoz ROUGE aynı koşumda
   0,1392→0,1374 (−0,0018) — genel LM yeteneği SAGLAM, yetenek transferi
   gerçekleşmemiş.
2. **Tutarsızlık %43 (ÖLÇÜM-2):** taban-çıpa ÖN-3'te de %61'di; marangoz
   heldout'ta aynı model %2,0. Bahçıvan cevap-uzayında üretim kilitlenmemiş
   (T-0112 koşum-rejim kilit sınıfı Bahçıvan tarafında).
3. **Kök neden ölçümü ayrı görevdir** — adaylar (ölçülmeden ilan edilmez):
   cevap-uzunluk etkisi (Bahçıvan cevaplar kısa; ROUGE-L kısa referanslarda
   düşük tavan), 6,8 epoch dozajı, ceket çekicilik, üretim sıcaklığı/kararlılık.
   Bu rapor tahmin YAZMAZ; ölçüm ayracaçları yukarıdakilerdir.

## Kanıt digest tablosu (tam sha256)

| Dosya | sha256 |
|---|---|
| `scratch/t0137/bahcivan_ft_marangoz_sonda.json` (ÖLÇÜM-1) | `64aba0f5c0362c34e2b5e6b872fb1deeaa1d8132b088b9f854c26dafe4712f5e` |
| `scratch/t0137/bahcivan_ft_bahcivan_sonda.json` (ÖLÇÜM-2) | `be84f93eae4539f36db4db1279fff59e061c466dae67c558d51cd858239073f6` |
| `scratch/t0137/bahcivan_ft_hukum.json` (hüküm, betikten) | `d8802979f706e8df73283b3adae6f6c55216be9ee63ec498142bc8960b7aec60` |
| `scratch/t0137/bahcivan_ft_hukum_betigi.py` (betik) | `53b370b608a4816cf2ae8f0fb5d9e290cfd882aa1631e54e1e0ab3e2909c6776` |
| `scratch/t0137_g6a_kos/seg_1.pt` | `8ab9e02578e345e9b9d02843d5fc5ac65a000ef0bdbd7cef4c0164376bc24e2f` |
| `scratch/t0137_g6a_kos/seg_2.pt` | `9d92391334ec66fff99a08c4602401ca01a2bab8a90b9aae07f7a98c870f635f` |
| `scratch/t0137_g6a_kos/seg_3.pt` (nihai) | `bd68c4505995064add6cb8cc67f055e121a678e6ae1f4947e67557907c482770` |
| `scratch/t0137_g6a_kos/segments.jsonl` (artımlı kayıtlar) | `5f112daac986948f1a1ab364c2a89f17c455c83b01140cb336019172d0cfff47` |
| `scratch/t0137_g6a_kos/sonda_seg_3.json` | `fb3924bb8fb12ebe67c28f1afb66f762999c1606551e5055143d8043642ca289` |
| `scratch/t0137/t0137_kos.log` (sürücü logu) | `bfd9cfda2e428e856573ccb950956732f2b74af43e9db0f211b513ea33243200` |
| `scratch/t0137/olcum1_marangoz.log` | `55a864f8611de3b4d1d4458ba245b489983d65d53eae12a400603baa2b840219` |
| `scratch/t0137/olcum2_bahcivan.log` | `6dc5fc3e17a50c8b8b375cadca9549cf490f9360b20dcfbe1f691d1b43934edc` |

## Dürüst kayıtlar

1. **Heldout şema düzeltmesi:** İLK carve koşumu heldout'u ham arena satırı
   yazdı; kanonik kap `r["output"]` ZORUNLU (`evaluate_carpenter_anka.py:463`)
   ⇒ ÖN-3 `KeyError: 'output'` ile DURDU. Düzeltme: heldout D3 şemasına
   çevrildi (kardeş marangoz heldout şeması `{"instruction","input","output"}`).
   **Eğitim adapter'ı bit-özdeş korundu** (`7c2b8578…`) ⇒ koşum kanıtları
   geçerli. İLAN §2'ye işlendi; `--train-source` adapter'a bağlandı (arena ham
   satırları `output` alanı taşımadığından kesişim ölçümü yapay 0 üretirdi).
2. **Çıpa onayı:** ÖN-3 BAHÇIVAN_CIPA_DUSTU (taban 0,0236 — marangoz/ceket
   kalıbı üretimi 99/100, içerik kesişimi ~0) ⇒ çıpa OPERATÖR onayına gitti;
   onay "çıpa 0,0236 onaylıyorum, koşumu başlat" (2026-09-26T17:01Z); İLAN'a
   damgalı.
3. **Süre beyanı güncellendi:** İLAN'daki ~2-3 saat beyanı koşum gerçek
   ölçümüyle 52 dk eğitim + 3 sonda (~17 dk/segment dahil) olarak ölçüldü;
   sonda başına ~3,5 dk (ÖN-3 eval'in ~2,4 saati tek başına eval koşumu —
   A+B eksenleri + decompile dahil; sürücü sondaları daha hafif).
4. **İlk adım-kayıp izlemesi iki kez tetiklendi** (grep deseni geniş — TEPE
   referans satırıyla) — izleme deseni daraltılıp tekrar kuruldu; ölçüm
   değeri (3,3254) her iki koşumda aynıdır.
5. **Hüküm betiği iki kez koşuldu** (heredoc ilk koşum + kalıcı dosya
   `53b370b6…` ikinci koşum) — hüküm sha birebir `d8802979…` (determinizm
   kanıtı); ikinci koşum rc=2 (KABUL_YOK kanıtı).

## Sonuç dalları

- A10+A11+A12 **KORUNDU** — taban mühürlü modelde LM-bedeli kapısı geçildi;
  unutma yok.
- **A9 BAND_ALTINDA + A13 IHLAL → KABUL_YOK**: Bahçıvan yeteneği sinyal
  eşiğine ulaşmadı; kök neden ölçümü AYRI GÖREVDİR (İLAN'da beyanlı dallar:
  "BAND_ALTINDA (kabul yok)").
- Checkpoint silme OPERATÖR kapısı (`scratch/t0137_g6a_kos/` doğrulama
  sonrası); commit OPERATÖR kapısı (`git add -A` YASAK — satır-satır).
- G6b (T-0138 Berber) antigravity'ye açık.

## Sınırlar

ESIK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509 DOKUNULMAZ (Bahçıvan'a
BAĞLANMAZ; hukum.rouge_gec kapın kendi 0,3221 bayrağıdır, bu raporun kabul
birleşimine girmez) · kanonik kod IMPORT · elle sayı YOK · data/** salt-okunur
· checkpoint'ler scratch'ta (kiralanmış) · commit + checkpoint silme operatör
kapısıdır.