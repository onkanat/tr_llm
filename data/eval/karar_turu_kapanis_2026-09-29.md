# Karar-turu K1..K6 kapanış kaydı (T-0180)

**Damga:** 2026-09-29T03:30:24Z (BETİKTEN, `date -u`) · **Yürütücü:** claude (T-0180 sahibi)
**Kaynak:** operatör kararları, AskUserQuestion iki-tur (29 Eyl 2026, 4+2 soru)
**Keşif:** Explore-ajanı altı-madde taraması, tam-disk-tarama raporu (33 araç-koşumu)

Altı açık-karar maddesinin operatör kararları ve BETİKTEN çıparları. KOD DOKUNULMAZ;
data/** yazımı yalnız data/eval/; canlı sunucular (192.168.1.9:6333 anka_bellek,
192.168.1.5:6333, 127.0.0.1:8080) 0 istek.

---

## K1 — F2 hedef-kümesi → **VAZGEÇ**

- **Karar:** üç hedef `.bin`'in yeniden derlemesi (chat_balanced, balanced_sft, carpenter)
  **yapılmaz**; karar kapanır.
- **Gerekçe (ölçülmüş):** Anka zinciri bu külliyatı kullanmıyor — A1-r/A2 pretraining
  `data/anka_a1r_pretrain.bin` (191 MB, 19 Eyl) / `anka_a2_pretrain.bin` (23 Eyl);
  carpenter dikeyi kendi `train_carpenter_specialization_anka_r18_v4.bin`'ini okuyor
  (24 Eyl). Külliyat-kendisinde/epoch-arz-doyumu kanıtı: P4 faz-kapanışları
  (T-0144/T-0145; [[p3-ceket-carve-sonuc]]).
- **Koruma-beyanı:** üç `.bin` + `scratch/f2_backup_2026-09-16/` (6 dosya, 53 MB)
  durduğu gibi korunur. T-0175 dokunulmazluk koşulu artık "karar AÇIK"tan
  "VAZGEÇİL"e döndü; f2_backup'un silinmesi ayrı bir operatör-emri içeriğinde değerlendirilir.

## K2 — C2 taban-derleme → **ERTELEME-devam**

- Güncel tanım yeri `data/eval/t0164_turc_ilan_20260928.md` (28 Eyl — taze).
- Yeni eylem YOK. Taban büyütme koşulları önceki beyanlarda durmaya devam eder:
  kaynak-ÇEŞİTLİLİK ([[operatör-mimari-kararı-2026-09-27]]), canlılık-imza
  ([[canlilik-imzasi-moda-bagli]]), `--vocab` külliyat eşleşmesi.

## K3 — SPEC.md açık-kusur #3 → **KAPATILDI (T-0164 C5)**

- Ölçüm çıpası (BETİKTEN, 03:30:24Z): `.agent-bus/tasks/` **0 dosya**;
  `get_task_file_path` tek-kaynak `state/tasks/` (T-0164 C5, commit `7c680a7`).
- SPEC.md "Bilinen açık kusurlar" bölümünde madde 3 bu çıpa ile KAPATILDI olarak işaretlendi
  (bu turda kusur-unvanı: onarım gerçekte T-0164'te yapıldı; metin 29 Eyl'de güncellendi).

## K4 — G6a/G6b bekleyen karar → **YOK (kuyruk boş)**

- Çıpa: `state/results/T-0137.json` "KABUL_YOK tüm zincir boyunca değişmez";
  `state/results/T-0138.json` "KABUL_YOK, rc=2" (BETİKTEN grep, 03:30:24Z).
- Bellek-kuyruğundaki "sonraki-ilk-plan ONARIM-GENİŞLETME" beyanı **kaynak-sız çıktı**
  (data/eval g6a/g6b 19 dosyada arama; tek eşleşme ilgisiz ileri-çalışma cümlesi).
- Sonuç: karar-turu paketine madde olarak girmez. T-0137 raporundaki izleyen eksenler
  (cevap-sonlandırma/EOS eğitimi, T=0,7 tekrarlanabilirliği) ölçülmeden ilan edilmez —
  ayrı İLAN'lı görev gerektirir; bu turda açılmadı (operatör: "kapat — kuyruk yok").

## K5 — RAG arm C → **KAPATILIR**

- Çıpa: pilot `data/_archive/rag_pilot_invalidated/` (INVALIDATED.md — %97-98
  4-gram sızıntıyla geçersiz-kılınmış; 14 Eyl); arm C eğitimi 14 Eyl'de kullanıcı
  kararıyla durduruldu, son resmî ölçüm val 5,7079 (`state/results/T-0009.json`).
- "rev2 arm C" etiketi kaynak-sız bulundu (keşif: diskte/İLAN'larda karşılığı yok).
- Üretim-RAG yeteneği bundan bağımsız T-0142 P2 hattında canlı-yoldadır
  (vector_memory + RRF; eşik 0,33 — commit `deac1cd`). Pilot-derse katkı sınıfı
  (küçük-külliyat sızıntısı) bu kayıttan okunabilir; yeni pilot İLAN'lı ayrı görev ister.

## K6 — hüküm-dosyası iz-kuralı → **KURAL YOK**

- Ölçüm (keşif, BETİKTEN): data/eval **513 dosya / 8,4 MB**; damga dağılımı
  513/513 = 2026-09; 2026-08 bandı YOK; git 510 tracked + 1 untracked.
- Zorla kurulan bir iz/temizlik kuralı kanıt desteklemiyor — kurulmaz.
- Tek kanıt-borcu: untracked `data/eval/t0174_scratch_temizlik_2026-09-28.md`
  bu karar-turu kayıt-commit'ine girer.

---

## Kayıt-disiplini

- Bu dosya BETİKTEN damga ile yazıldı (koşumsuz kayıt-turu; İLAN/hüküm ikilisi gerekmedi
  çünkü ölçüm-koşumu YOK).
- SPEC.md düzenlemesi kural-uyumlu: kiralama `.agent-bus` (T-0180), yazım-öncesi-okuma,
  yazım-sonrası hizalama teyidi.
- Commit: AYRI operatör onayı; satır-satır `git add` (`git add -A` YASAK); push YOK.