# T-0194 SONUÇ — Giyim+system_message şartlanma testi (3-kol × 2-yüzey)

- **Hüküm:** **T0194_GIYIM_DUR** (rc=2) · BETİKTEN hüküm-JSON:
  `data/eval/t0194_giyim_hukum_2026-09-29.json` (damga 11:45:46Z BETİKTEN)
- **Soru (İLAN):** ceket-giyim yüzeyi + system_message şartlanması eklendiğinde
  B1.5 modeli taban-çıplak eşiği (decomp-mean ≥ 0,35) geçer mi? → **Geçmedi.**

## Koşullama-matris (BETİKTEN, 300 üretim; eşli-100 örneklem; bağımsız-disk-teyit)

| Kol | Şartlanma | ROUGE raw | ROUGE **decomp** (giyim) | Δ decomp | Tutarsızlık | kesisim-F1 |
|---|---|---|---|---|---|---|
| **A** | — (T-0192 birebir) | 0,1058 | **0,1325** | — | %22,0 | 0,0202 |
| **B1** | kod-canonical ROL_ZARFI (carpenter-metni, salt-eklenti) | 0,1086 | **0,1402** | **+0,0076** | %25,0 | 0,0209 |
| **B2** | alan-canonical SYS_MESAJ (salt-eklenti) | 0,1027 | **0,1292** | **−0,0034** | %21,0 | 0,0209 |

- **K1 pozitif-kontrol PASS:** Kol-A raw sayım T-0192 hüküm-sayım birebir
  (0,0 / %22,0 / 0,105808… / 6,0) — üçüncü-bağımsız tekrar-kanıt.
- **K2 PASS:** decomp-istisna 3 kolda 0 · boş-decompile 0 · 300/300 kayıt-disk.
- **SYS_MESAJ-canonical PASS:** train verbatim 4.849 (`[SYS_MESAJ_CANONICAL]` BETİKTEN);
  B2'nin kendinde birebir-yineleme 54/100 (üretimi değiştirmedi → gürültü-bant).

## Hüküm-okuma

1. **System_message şartlanması ROUGE'u taşımadı:** her iki şartlanma-kolu
   decomp-mean'de |Δ| < 0,01 — eşik 0,35'ten uzak (0,129–0,140). Şartlanma
   metni değiştirmekle soru-ilişkili içerik problemi çözülmüyor.
2. **Rol-zarf yönü:** domain-mismatch'li B1 hafif +0,0076 (gürültü-bant),
   tutarsızlığı %22→%25 KÖTÜLEŞTİRDİ — kod-canonical zarf-metni marangoz-
   alanlı olduğundan bu davranış-dönüşü beklentide de değildi; KAPI DEĞİL,
   gözlem-beyanı.
3. **Ceket-giyim yüzeyi (decomp) kol-A'da T-0193 birebir 0,1325** — giyim
   yüzeyi tekrar-bilirlik-kanıtı da oldu.
4. T-0192'nin DUS hükmü üç-yüzeyde artık pekiştirildi: raw, giyim-decomp,
   giyim+şartlanan — hepsi eşikten uzak (0,10–0,14). Kalite-açığı
   içerik-yakını (T-0193 hükmü) bu koşumda da değişmedi.

## Koşum-disiplin ve beyanlar

1. Koşum 11:42→11:45Z ~4 dk; ARM süreleri 64,6/77,1/74,7 sn BETİKTEN.
2. `src/compiler/**` + `prompt_contract.py` yalnız-okuma (sha'lar İLAN'da,
   dokunuş yok); tarihsel betikler dokunulmAZ; canlı 8080'e 0 istek.
3. Betik-ilk-turda iki açık-kusur BETİKTEN yakalandı ve onarıldı
   (tutarsızlık-ifade kalıntısı + kayıt-collection) — koşum öncesi; koşum
   tek-geçişte tamamlandı.
4. Elle-parça-yüzey tüm kollarda; render_prompt delegasyonu İLAN'da
   beyanlı (Kol-A birebirliğini korumak için).

## Bus-disiplin

T-0194 claim ok · kiralama `data`+`scripts`+`data/eval`+`.agent-bus/notes`
ok (conflict 0) · hüküm/İLAN/sonuç/kayıtlar BETİKTEN — ayrı konumlar ·
commit ayrı operatör-onayı.