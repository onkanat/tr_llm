# İLAN-2 — T-0155: koşum-1 ölçüm-kabı istisna-beyanı (kapılar DEĞİŞMEZ)

**Damga (koşum-2 ÖNCESİ, betikten `date -u`):** 2026-09-28T07:15:23Z
Ana İLAN: `data/eval/t0155_router_entegrasyon_ilan_2026-09-28.md`
(sha256 koşum-2 betiğinde ölçülür; damga 07:12:22Z; kapılar bu belgede
DEĞİŞMEZ — yalnız ölçüm-kabı onarımı beyan edilir; T-0151/T-0149 kalıbı).

## Koşum-1 DUR (istisna; hüküm-JSON YAZILMADI)

- Koşum-1 (log: `t0155_router_entegrasyon_kosum_2026-09-28.log`):
  **K1 GEÇTİ · K5 GEÇTİ · K2 GEÇTİ · K4 GEÇTİ**; K3 adımında
  `RuntimeError: Tensor for argument input is on cpu but expected on mps`
  — **ölçüm-kabı cihaz-istisnası**: `EpistemicCuriosityAgent` kurucusu
  `self.router.to(self.device)` yapar (MPS); betik özellikleri CPU'da
  tutuyordu (T-0154 ölçüm-yüzeyi CPU). K3/K6 ölçülemeden çöktü; hüküm
  yazılmadı. Üretim-kod hatası DEĞİL — betik hatası.
- Koşum-1 artefaktları DOKUNULMAZ (yalnız koşum-logu; hüküm/rapor yok).

## Betik-onarımı (yalnız ölçüm-kabı; kapılar DEĞİŞMEZ)

- K3 ölçümü **CPU-router kopyası** ile: `agent.router.state_dict()` CPU'ya
  taşınıp yeni `TriModalRouter`'a `strict=True` yüklenir; özellikler de
  CPU'da ⇒ T-0154 ölçüm-cihazıyla BİREBİR (K3'ün birebir-üretim taahhüdü
  korunur). Kopya state_dict özeti == `9b85f951…` yeniden teyit (K2-b).
- Koşum-2 çıktıları AYRI adlarla (İLAN≠RAPOR; ön-kayıt-ezme dersi):
  `t0155_router_entegrasyon_hukum2_2026-09-28.json` /
  `t0155_router_entegrasyon_rapor2_2026-09-28.md` /
  `t0155_router_entegrasyon_kosum2_2026-09-28.log`.
- Kapı tanımları, eşikler, kaynak-çıpası değerleri ANA İLAN'dan AYNEN.

**İLAN SABİT — yumuşatma YOK.** Hüküm BETİKTEN; elle sayı/hüküm YOK.