# İLAN — T-0153: canlı-migrate `kristal_bellek` → `anka_bellek` (192.168.1.9:6333)

**Damga (koşum-ÖNCESİ, betikten `date -u`):** 2026-09-28T06:43:18Z
**İLAN SABİT — koşum-sonrası yumuşatma YOK.** Hüküm BETİKTEN
(`scripts/dogrulama_t0153_canli_migrate.py`); elle sayı/hüküm YOK. rc ∈ {0, 2}.

## Meşruiyet

Operatör onayı (28 Eyl 2026): T-0151 RAPOR2 §7 canlı-migrate maddesi —
"2. ile devam et sonra 1." Bus-görev T-0153; kiralamalar `data/eval/`,
`scripts/`, `.agent-bus/notes/` (ÜST-DİZİN). **Canlı sunucuya YAZIM** bu
turda operatör-onaylıdır (önceki turların aksine).

## Kaynak-düzeltmesi (RAPOR2 §7 beyan-hatası; koşum-ÖNCESİ düzeltme)

RAPOR2 §7 "kaynak: `data/pedagogy_canonical/**` re-index" yazmıştı — **bu
yanlıştır**. P2 çıpa-betiği (`dogrulama_p2_rag_gezgini.py`, K5 sha-sabit)
koleksiyonu `data/realistic_rag/test_natural_150.jsonl` (DONMUŞ) içindeki
benzersiz `<BELGE>…</BELGE>` metinlerinden kurdu (domain `realistic_rag_p2`;
36 benzersiz belge **bu tur koşum-öncesi yeniden ölçüldü**). Ayrıca
`src/compiler/core.py` T-0150'da onarıldı — yeniden-derleme tokenizasyon
sapması riski taşır. Bu nedenle yöntem **NOKTA-KOPYASI** (verbatim migrate):
koleksiyondan okunan 36 nokta (id + dense + sparse + payload) yeni
koleksiyona birebir kopyalanır; birebirlik K3'te ÖLÇÜLÜR.

## Koşum-öncesi ölçülen çıpalar (betik-yokken ölçüldü; İLAN değeri sabit)

- Sunucu 192.168.1.9:6333 ayakta; **10 koleksiyon** (kristal_bellek + 9
  foreign: elektor_articles, octave_articles, rapberry_pi_pico_all_articles,
  rendergit_01/02/03_articles, rp2040_articles, sdr_articles, türk_articles).
- `kristal_bellek`: **green, points 36, indexed 36**; şema: dense **768
  Cosine** (ad "dense") + sparse adı **"sparse"** (modifier idf) +
  **on_disk_payload=true**.
- Payload anahtarları: `crystal_tags, domain, text, token_ids`
  (scroll örnekleme id=1).
- `anka_bellek` sunucuda **YOK** (B0 koşulu koşum-öncesi sağlandı).

## Kapılar (koşum öncesi sabit)

- **K1 B0 + ÇIPA-ŞEMA:** `anka_bellek` YOK; `kristal_bellek` 36-point green;
  şema yukarıdaki çıpalarla birebir. Aksi → DUR (yazım başlamadan).
- **K2 ARŞİV:** silme-ÖNCESİ tam nokta-arşivi
  (`data/eval/anka_bellek_canli_migrate_eski_nokta_arshivi_2026-09-28.jsonl`)
  — 36 nokta: id + payload + dense (768 float) + sparse (indices+values);
  dosya SHA-256 hüküm-JSON'a yazılır (silme geri-dönüşü kanıtı).
- **K3 KOPYA-BİREBİR:** `anka_bellek` şema-birebir kurulur (dense 768 Cosine
  + sparse "sparse"/idf + on_disk_payload=true — ham client, VM-kurucu
  on_disk_payload ayarlamaz) + 36 nokta kopyalanır; doğrulama: yeni
  koleksiyon scroll'unda **id kümesi ==** + **payload JSON ==** + **dense
  liste ==** + **sparse (indices, values) ==** nokta-başına 36/36.
- **K4 SELF-RETRIEVAL:** nokta id=1'in dense vektörüyle anka_bellek'te
  arama → top-1 kendisi (koleksiyon aranabilir; 36/36 index).
- **K5 DOKUNULMAZLIK:** 9 foreign koleksiyon ad+nokta-sayısı ÖNCE==SONRA
  (kristal_bellek/anka_bellek dışındaki tüm küme).
- **K6 ESKİ-SİLME (fail-closed sıra):** K1-K5 TAMAM olmadan silme
  ÇALIŞMAZ. Silme `VectorMemory.delete_collection` wrapper'ı (T-0148 4D)
  ile; VM-instance `anka_bellek`'e bağlanır (kristal_bellek'e VM dokunmaz).
  Silme-sonrası: `kristal_bellek` YOK + `anka_bellek` 36-point green.
- **6/6 → T0153_MIGRATE_GECTI rc=0**; herhangi kapı düşerse → **DUR rc=2**
  (eski koleksiyon DOKUNULMAZ kalır — yarıda kalan migrate geri-alınır:
  anka_bellek silinir, kristal_bellek arşivden geri yüklenir, hüküm-JSON
  raporlar).

## DOKUNULMAZLAR

- 9 foreign koleksiyon (salt-okuma; K5 kanıtı).
- `data/**` salt-okunur (istisna `data/eval/`); `data/realistic_rag/**`
  frozen (yalnız okunur — provenans).
- P2-P5 çıpa-betik shaları SABİT (bu tur betik-yüzeyine dokunmaz).
- `data/eval/` mevcut 447 dosya.

## Çıktılar (koşum-öncesi sabit; ilan==rapor-yolu YOK)

- Hüküm: `data/eval/anka_bellek_canli_migrate_hukum_2026-09-28.json`
- Rapor: `data/eval/anka_bellek_canli_migrate_rapor_2026-09-28.md`
- Arşiv: `data/eval/anka_bellek_canli_migrate_eski_nokta_arshivi_2026-09-28.jsonl`

## Beyan: damga-yöntemi

Damga bu İLAN'a betik koşulmadan `date -u` işlendi (mtime-çıpası);
hüküm-JSON: ilan_sha + damga + kapılar. Hüküm BETİKTEN; elle sayı YOK.