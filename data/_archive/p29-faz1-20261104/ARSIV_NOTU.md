# p29-faz1-20261104 — P2.9 Faz-1 Arşivi

P2.9 arşiv politikası FAZ-1 uygulamasıyla taşındı (onay: 2026-11-04, onaylayan: operatör/Kullanıcı).
Kaynak politika: `data/eval/p29_checkpoint_envanteri_2026-11-04.md` (Bölüm 7 FAZ-1 listesi).
Kapsam: **C2 (b1.5 ailesi, 13) + C4 (eski 3-sınıflı router, 1) + C3 (T-02xx merdiveni, 6) = 20 dosya / 7.103.354.683 bayt = 6,616 GiB.**

Bunlar superseded checkpoint'lerdir; hüküm ve ölçüm kayıtları `data/eval/` içinde kalıcıdır.
C1 (a1r/a2/base_v2), B2 (v3r/v3r2) ve B3 (v3.opt) bu fazın kapsamı DIŞINDA — `data/` kökünde kalır.

## Bütünlük

- `SHA256_MANIFEST.txt` — arşiv-dizin-göreceli, `shasum -a 256 -c` ile doğrulanır (20/20 OK).
- `SHA256_ONCE.txt` / `SHA256_SONRA.txt` — taşıma öncesi/sonrası ölçümler (data/ ve arşiv yollarıyla); birebir eşleşir.
- `BEKLENEN_SHA256_envanter.txt` — envanter raporundaki çıpalar (taşıma öncesi 20/20 birebir doğrulandı).
- `MV_LISTESI.txt` — taşınan 20 dosya adı.
- `TASINMA_ONCE_boyut.txt` — taşıma öncesi bayt boyutları.

## Geri alma

```bash
cd <repo-kökü>
while read f; do mv "data/_archive/p29-faz1-20261104/$f" "data/$f"; done < data/_archive/p29-faz1-20261104/MV_LISTESI.txt
shasum -a 256 -c data/_archive/p29-faz1-20261104/BEKLENEN_SHA256_envanter.txt  # (repo kökünden; data/ yollarıyla)
```

## Notlar

- Bu taşınma `mv`'dir (aynı birim), silme değildir; diskte yer açılmaz.
- Tarihsel kanıt betikleri (`scripts/t0199_*`, `t0200_mutasyon_kanit.py`, `t0196/0197/0198_faz3_kabul.py`, `dogrulama_t015x/016x/0172*` vb.) bu dosyaları `data/` sabit yollarıyla arar — eski halleriyle koşamazlar (politika §8.2: kabul edilen kırılma; gerekirse dosya geri taşınarak koşulur).
- Canlı soyağacı (`anka_taban_pure/marangoz/bahcivan/router_v2`) ve bugünün ailesi (v3, v3r3) DOKUNULMADI (sha birebir aynı).
- Uygulama raporu: `data/eval/p29_faz1_uygulama_2026-11-04.md`