# İLAN-2 — T-0151 revizyonu: K1 ölçüm-kabı istisna-beyanı (tek-madde)

**Damga (koşum-öncesi, betikten `date -u`):** 2026-09-28T06:32:35Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Hüküm BETİKTEN; elle sayı/hüküm YOK.

## Meşruiyet

Operatör onayı (28 Eyl 2026, AskUserQuestion: "İLAN-2 onayla → koşum-2").
**İLAN-1** (`anka_bellek_yeniden_adlandirma_ilan1_2026-09-28.md`, damga
06:16:09Z, sha koşum-1 hükmünde kayıtlı) ve **koşum-1 hüküm-artefaktları**
(`..._hukum_2026-09-28.json`, DUR rc=2, damga 06:28:14Z) DOKUNULMAZ.

## Koşum-1 bulgusu (hüküm dokunulmaz; kayıt)

K1 DÜŞTÜ — sebep ölçüm-kabı kusuru: K1 taraması **kendi hüküm-betiğini**
(`scripts/dogrulama_t0151_anka_bellek.py`) eski-ad kalıntısı saydı. Betik
K6 canlı-dokunulmazlık ölçümü için URL'sinde `kristal_bellek` taşımak
ZORUNDADIR (ölçüm-aracı, hedef-değil). K2-K6 GEÇTİ (36/36 envanter;
statik-ön 0; davranış :memory:; P2-P5 çıpa-shaları; canlı 36 SABİT green).

## Tek-madde revizyon (İLAN-1'den devralınır; değerler değişmez)

**K1 ESKİ-AD SIFIR:** grep `kristal_bellek` src/+scripts/+tests/ == 0.
İSTİSNA beyanı: (a) `dogrulama_p[2345]_*.py` geçmiş-hüküm çıpa-betikleri
(İLAN-1'den), (b) **`scripts/dogrulama_t0151_anka_bellek.py` — ölçüm-kabı
betiğinin kendisi** (K6 URL'si + İLAN beyan-metni ölçüm-aracıdır;
yeniden-adlandırma hedef-yüzeyi DEĞİL). Diğer tüm kapılar (K2 36-geçiş
kırılımı, K3 statik-ön, K4 davranış, K5 çıpa-shalar, K6 canlı-36) İLAN-1'den
birebir geçerli.

## Kapılar (koşum öncesi sabit — İLAN-1 + bu revizyon)

6/6 → **T0151_GECTI rc=0**; aksi her dal → **DUR rc=2**.
Koşum-2 çıktı-adları: `data/eval/anka_bellek_yeniden_adlandirma_hukum2_2026-09-28.json`
+ `..._rapor2_2026-09-28.md` (ilan==rapor-yolu ön-kayıt-ezme dersi).

## DOKUNULMAZLAR

İLAN-1; koşum-1 hüküm-artefaktları; P2-P5 betik-shaları (K5 değerleri İLAN-1
revizyon-2'de); canlı sunucu 192.168.1.9:6333 salt-okuma; `data/**` istisna
`data/eval/`.