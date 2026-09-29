# T-0185 — T=0,7 tekrarlanabilirliği ölçümü: KAPANIŞ-KAĞIDI (fail-closed, DÜŞEN-KAYNAK)

- **Damga:** BETİKTEN 2026-09-29T05:15:10Z (`date -u`; yok-teyit koşum) · **Yürütücü:** claude
- **Claim/kiralama:** T-0185 `ok:true` + `data/eval` + `.agent-bus/notes` `ok:true`
- **Koşum:** BU GÖREVDE YOK — K1 (kaynak-keşfi) fail-closed DURTU.

## K1 keşif-kalıt (BETİKTEN, keşif-ajandan)

- T=0,7 orijinal ölçüm-yüzeyi: `scratch/t0137/bahcivan_uretim_eksenleri.py` (FAZ-4
  E1_sicaklik_07; softmax(logits/T)+multinomial, satır 103-114) + `bahcivan_heldout.jsonl`
  (220 örnek) + `scripts/evaluate_carpenter_anka.py` (kanonik, sıcaklık-YOK greedy).
- **Orijinal ölçümün model-checkpoint'i `scratch/t0137_g6a_kos/seg_3.pt` DİSKTE YOK:**
  `find <repo> -name "seg_3*.pt"` → 0 satır (BETİKTEN 05:15:10Z; ikinci doğrulama
  `find -iname "*bah*ivan*" -name "*.pt"` → 0 satır). T-0176 silme-kaydı t0113'in
  seg_3.pt'sini dokümente eder; `t0186 t0175 envanterinde T-0137_g6a_kos dir hiç
  listede yok → daha-önce silinmiş; silme-kaydı hükümde BELIRSIZ-BAND'dır.
- İLAN-4 kabul-eşiği (koşum-öncesi sabit): TEKRARLANIR = ROUGE(0,7) > ROUGE(0,0)
  koşumların ≥ 2/3'ünde VE ort(delta)>0. (Ölçek notu: orijinal tek-koşum
  +0,0106 — mutlak-eşik +0,03 koymak orijinal-etkinin ÜSTÜNDE çubuk olurdu.
  Yine koşum yapılamadı.)

## Hüküm

**T0185_TEKRARLANIRLIK_ÖLÇULEMEZ (rc=2 — kaynak-yok).** Orijinal hüküm
(`bahcivan_g6a_uretim_eksenleri_2026-09-26.md`, E1 ROUGE 0,0421 +0,0106)
artefakt-kanıt olarak KORUNUR; tekrarlanabilirlik iddiası bu pencerede
kaynak-dışıdır ve İSTİŞARAT-yoksayı DEĞİL — ölçüm-yüzeyi bayatlar.

## Kalan-yol (ayrı İLAN'lı görev, T-0189 olarak bus'ta açık)

Orijinal-model regen (bahçıvan-FT re-koşumu; pahalı MPS) VEYA mevcut-çapa
(anka_base_v2/a2) üzerinde YENİ-sorgu: "T=0,7, BUGÜNKÜ zincirde ROUGE-kilit
etkisi?" — model değişikliği İLAN'da birebir beyan edilmeli. Ayrı emirsiz
koşum YOK.