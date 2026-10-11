# T-0209 İLAN — Saf Taban Model ve Müstakil Marangozluk Ceketi Ayrışımı (MoE)

Damga (BETİMTEN): 2026-10-02T00:10:00Z
Operatör Kararı: Marangozluk bilgisinin taban modelden arındırılması, tabanın saf tutulması ve Marangozluğun bağımsız bir MoE Ceketi (`anka_marangoz.pt`) olarak sisteme bağlanması.

## 1. Problem & Gerekçe
- `t0197_uretim_saf.jsonl` külliyatı içerisinde 993 adet marangozluk/ahşap sorusu bulunuyordu. Bu veriler taban modele (`anka_t0203_polish.pt`) sızdığı için taban model dikey uzmanlık bilgisiyle kirlenmişti.
- `TriModalRouter` içinde `carpenter` uzman sınıfı tanımlı olmasına rağmen, canlı sistemde marangozluk ceketi bulunmadığı için istekler taban modele yönlendiriliyordu.
- **Hedef:** 
  1. `pure_base_sft.jsonl` (4.498 kayıt) + 1.124 pretrain penceresi ile **Saf Akıcı Taban Model (`data/anka_taban_pure.pt`)** eğitmek.
  2. `marangoz_canonical_sft.jsonl` (6.123 kayıt) + 1.530 pretrain penceresi ile **Marangozluk Ceketi (`data/anka_marangoz.pt`)** eğitmek.
  3. Gateway MoE sözlüğüne hem Bahçıvan (`gardener`) hem Marangoz (`carpenter`) ceketlerini bağlayarak canlıda doğrulamak.

## 2. Çıpa-Tablo (BETİMTEN)
| Çıpa | Değer / Dosya | Açıklama |
|---|---|---|
| Başlangıç Akıcı Taban | `data/anka_t0201_b05.pt` | Akıcılığı kanıtlanmış ortak atar model |
| KL-Öğretmen | `data/anka_base_v2.pt` | KL-çıpa ($\beta=0.2$) |
| Sözlük | `data/rebuild/vocab_anka_r1_33114.json` | 33.114 kanonik sözlük |
| Saf Taban SFT | `data/pedagogy/pure_base_sft.jsonl` | 4.498 kayıt (marangozluktan arındırılmış) |
| Marangoz Ceketi SFT | `data/pedagogy/marangoz_canonical_sft.jsonl` | 6.123 kayıt (kapsamlı marangozluk dikey seti) |
| Pretrain Bin | `data/anka_a1r_pretrain.bin` | 4:1 oranında pretrain akıcılık desteği |
