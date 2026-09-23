# P2 · AŞAMA 3.1 — YETENEK AŞAMASI İNCE AYAR İLANI (ön-kayıtlı)

**Damga:** 22 Eyl 2026 21:05 (+03) · **Yazım:** koşumdan ÖNCE · Plan:
`~/.claude/plans/enchanted-wiggling-moon.md` (Aşama 3). Temel 2.0 koşum kapandı
(`anka_p2_temel2_sonuc_2026-09-22.md`, 4/4 CE geçti).

## 1. Aşama 3'ün sıfır noktası (ÖLÇÜLDÜ, koşumdan önce)

`data/eval/anka_p2_temel2_zemin_2026-09-22.json` (sha256 `7cc909f62220898b…`,
sandbox dışı MPS, kanonik kap `--ceket-ekseni`):

| eksen | Temel 2.0 zemin | taban zemin (0e) | kalibre eşik |
|---|---|---|---|
| tutarsızlık | **%96,00** | %100,00 | < 5,0 |
| ROUGE-L | **0,0056** | 0,0000 | ≥ 0,35 |
| LCS-kesişim | **%0,00** (F1 0,0018) | %0,00 | ≥ 10,92 |
| ezber | %0,00 | %0,00 | < 10,0 |

Hüküm: 4 eksen de eşik ALTINDA (beklendiği gibi — Aşama 2'nin işi LM+zarftı;
yetenek Aşama 3'ün işidir). ROUGE 0,0000→0,0056 ve tutarsızlık %100→%96 küçük
hareket: kesişim eşiği 10,92 kalibre olduğundan bu zemin ARTIK ölçülebilir bir
başlangıç noktasıdır (Ç4 vakumu kapandı).

## 2. Tarif (sürücü: `scratch/anka_p2_temel2_surucu.py` — parametrik çalışma)

* Başlangıç: `scratch/anka_p2_temel2_kos/seg_4.pt` + **carry VAR**
  (`seg_4.pt.opt.pt`, `f8e844f6…`; digest uyuşmazsa DUR — G5 kapısı).
* Veri: AŞAMA 2A MIX BIN'İ (`data/rebuild/anka_p2_sft_mix.bin`, ceketsiz + held-out'suz
  kanıtlı temiz — sızıntı kapısı 414 kayıt düşürdü) + wiki %25 aynı patern.
  **Karar ve gerekçe:** carpenter/ceket verisi eğitimde OLMAZ (planın açık yasası +
  sızıntı); yetenek eksenleri bu aşamada modülle taşınacak (3.2 — ayrı ilan).
* lr peak **1e-4** · warmup 50 + cosine · toplam **2.000 adım** (2 × 1.000) ·
  min_lr 1e-6 · clip 1,0 · weight_decay 0,01 · b8 blok 128 · seed 42.
* CE kapısı her segmentte ≤ 3,8887 (kodda kanonik sabitten assert'li) · L1 · H1 ·
  CARRY digest kapısı · artımlı yazım (`scratch/anka_p2_ince_ayr_kos/`).
* Süre: ~15-20 dk (0,42-0,46 sn/adım ölçülmüş band) · sandbox DIŞI MPS.

## 3. Kabul ve ölçüm

1. 2/2 segment CE ≤ 3,8887 + sentinel (koşum kabulü).
2. Koşum sonrası kanonik kap zemin tekrarı → zemin (§1) ile KIYAS tablosu;
   yetenek eksenlerinde hareket (varsa) raporlanır — eşik hükmü 3.2 sonrası.
3. **3.2 (onarıcı modül) ayrı ilanla** başlar; bu ilan onu tarif etmez.

## 4. Dallar (önceden ilanlı)

* CE DÜŞERSE (kapı DÜŞTÜ): koşum DURUR; wiki payı %25→%50 + carry ile tek tekrar
  (Aşama 2 DAL-1 deseni); hâlâ yüksekse sonuç yazılır, 3.2 modül yolu korunur.
* ROUGE/kesişim bu aşamada eşik hükmine tabi DEĞİL (ilanda beyan: ince ayarın
  işi zarf+LM korunumu; yetenek ölçümü modül sonrası anlamlıdır).

## 5. Beyan

Sürücü Aşama 2b koşumunun bitmiş sürücüsünün parametrik çalışmasıdır
(`--taban/--kos-adi/--segmentler/--lr/--wiki-patern/--ilan`); kanonik kod parçaları
aynı import'larla, kapılar aynen. Kapanmış koşum dizinleri (`anka_p2_temel2_kos`)
SALT OKUNUR — ince ayar yeni dizine (`anka_p2_ince_ayr_kos`) yazar.