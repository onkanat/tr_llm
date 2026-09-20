# T-0066 — SONUÇ-SONRASI ÇERÇEVE EK'İ

> **Bu ek ölçümden SONRA yazıldı (as-of 2026-09-18T09:04:43Z).** Hiçbir sayıyı, eşiği, kapı hükmünü
> veya ölçüm dosyasını değiştirmez. T-0066'nın *çerçevesine* ilişkin bir düzeltmedir.
> Ölçümün kendisi geçerliliğini korur.

## 1. Neden yazıldı

Kullanıcı, T-0066 kapandıktan sonra rol atamasını netleştirdi (18 Eyl 2026):

> *"temel model belge üretecek, ceket sadece yeni bağlam ekleyecek, temel model chat ya da
> diğer konularda sft ve dpo da eğitilmedi."*

ve soru üzerine **"temel model" = `data/kristal_model.pt`** olduğunu bildirdi.

## 2. Düzeltilen iki şey

### 2.1 Referans: "temel model" T-0066'da kapı DIŞI bırakılmıştı

T-0066, `f3_clean`'i "taban" sayıp kapı içine aldı; `kristal_model.pt`'yi ise **betimleyici,
kapı dışı** ilan etti (gerekçe: 32816 başlık, 36 satır doldurma, farklı servis sözleşmesi).

Kullanıcı kararı bu atamayı tersine çevirir: **üretici olan model `kristal_model.pt`'dir.**
Yani T-0066, üreticiyi kapı dışı bırakmış; kapı içindeki "taban" ise kullanıcının kastettiği
model değildi. Bu bir *referans* hatasıdır, ölçüm hatası değil — sayılar doğru, adres yanlış.

### 2.2 Kapı sorusu: ceket ÜRETİCİ sayılarak yargılandı

T-0066'nın **"Ceket RAG'ı bozmadı"** kapısı `P3 ∧ G1` olarak ilan edilmişti; ikisi de ceketin
**üretim kalitesini** ölçer (CE oranı ve ROUGE oranı). Rol atamasına göre ceketin işi üretim
değil, **yeni bağlam eklemektir** → bu kapı, ceketin kendi rolüne göre değil, bir başka role
göre sorulmuş bir sorudur.

Ölçülen değerler değişmez: P3 oranı **1.1201** (eşik ≤ 1.1) · G1 oranı **0.2374** (eşik ≥ 0.9).

## 3. Değişmeyenler (açıkça)

| öğe | durum |
|---|---|
| T-0066'nın ilan edilmiş eşikleri | **değişmedi** (P3 ≤ 1.1 · G1 ≥ 0.9) |
| T-0066'nın kapı hükümleri | **değişmedi** (GEÇTİ / GEÇMEDİ olarak kalır) |
| T-0066'nın ölçüm dosyaları | **değişmedi** (elle düzenlenmedi) |
| tasarım sha256 bütünlüğü | **SAĞLAM — kayıtlı hash canlı hash ile aynı** |
| "Belge kullanımı KANITLANDI" hükmü | **geçerli** — P1/P2 eksenlerine dayanır, rol atamasından etkilenmez |

## 4. Kapsam daralması (hükmü çevirmez, sınırlar)

T-0066'nın RAG külliyatı (`data/realistic_rag/test_natural_150.jsonl`) **arkeoloji, mimarlık,
astronomi, botanik, jeoloji** alanlarındandır. Ceketin eğitim aldığı "yeni bağlam" ise
**marangozluk** içeriğidir (ceket verisi: 867.650 token, T-0061). Yani T-0066, ceketi
**kendi alanı dışında** bir belgeyle ölçtü.

Bu, ceket hakkındaki hükmü **tersine çevirmez**; kapsamını daraltır: hüküm "ceket, alan dışı
belgeleri üretimde kullanamıyor" olarak okunmalıdır. Ceketin kendi alanındaki davranışı
T-0066'da ölçülmedi.

## 5. Sonraki adım

Rol atamasına göre yeniden ölçüm: **T-0067** (`data/eval/t0067_role_design_2026-09-18.md`).
İki eksen ayrı ölçülür: üretici (`kristal_model.pt`) için belge kullanımı; ceket (`f4_r05`) için
kendi alanında **içerik kazancı**.

---

*Bu ek: `data/eval/t0066_frame_addendum_2026-09-18.md` · T-0066 raporu: `data/eval/t0066_text_vs_rag_2026-09-18.md` · T-0066 tasarımı: `data/eval/t0066_text_vs_rag_design_2026-09-18.md`*
