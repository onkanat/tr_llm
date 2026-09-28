# İLAN — T-0157 Gateway İnsan-Okunur Üretim Onarımı

**Damga (BETİKTEN, koşum-ÖNCESİ):** `2026-09-28T07:48:56Z`
**Meşruiyet:** Kullanıcı talebi "insan okunabilir chat arayüzünü düzenle";
**operatör kapsam-seçimi (AskUserQuestion, 28 Eyl 2026): GATEWAY ONARIMI**
(CLI `chat_prompt.py` KORUNUR; web HTML/JS yüzeyi kapsam-DIŞI).

## Kök-neden (keşif-kanıtlı)

`src/rag/epistemic_agent.py:222-272` `generate_tokens` greedy-only ve
**yapısal-jeton bastırması YOK** → `/api/query` metni `<BOS>`,
`<INSTRUCTION>`, `<INPUT>` gibi yapısal jetonlar içerebilir (referans
uygulama: `chat_prompt.py:174-182`). Temizlik `:351-354` yalnız
EOS/`</OUTPUT>` filtreler (PAD sızıntı-riski); decode `:356-361`
`decompile_sentence` **capitalize'siz**.

## Dokunma-yüzeyi (TEK kod dosyası; imza DEĞİŞİKLİĞİ YOK)

Yalnız `src/rag/epistemic_agent.py`:

1. **Modül-sabitleri** (chat_prompt.py:174-182 kalıbı):
   - `YAPISAL_BASTIRMA_JETONLARI = ("<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>", "<INPUT>", "</INPUT>", "<OUTPUT>")`
   - `TEMIZLIK_JETONLARI = YAPISAL_BASTIRMA_JETONLARI + ("<EOS>", "</OUTPUT>")`
2. **`generate_tokens`**: eos/output_end çözümünden sonra çağrı-başına bir
   kez bastırma-id listesi (`self.vocab.stoi` üye-testi; vocab None/eksik
   jeton → atla — mock-güvenli); **vocab dolu AMA bastırma-listesi BOŞ →
   RuntimeError** (fail-closed). Döngüde: mevcut tekrar-cezasından SONRA,
   argmax'tan ÖNCE `logits_last[sid] = -float("inf")`.
3. **`process_query` temizlik + decode**: temizlik seti =
   `TEMIZLIK_JETONLARI` id'leri ∪ {eos_id, output_end_id} (id-düzeyi ikinci
   savunma hattı); `decompile_sentence(morpheme_output, capitalize=True)`.

## Bilinçli-sabit kararlar (koşum-sonrası değişmez)

- **GREEDY KALIR.** temperature/top_k/frekans-ölçekli ceza/4-gram bloklama
  kapsam-DIŞI (minimal-onarım; ayrı madde). Mevcut
  `repetition_penalty=1.4/window=10` aynen.
- Bastırma = 7 yapısal jeton; `EOS`/`</OUTPUT>` **bastırılmaz** (meşru
  dur-jetonları); `UNK`/`PROPER_NOUN`/`NUMBER` bastırılmaz.
- **Temizlikte `UNK` FİLTRELENMEZ** — `morpheme_output.count("<UNK>") >= 2`
  epistemik-sinyali korunur; `PROPER_NOUN`/`NUMBER` decompiler'da
  `[Özel İsim]`/`[sayı]` placeholder olur (`src/compiler/**` DOKUNULMAZ).
- **`capitalize=True` → DAVRANIŞ-DEĞİŞİMİ:** `ask().response_text` ilk
  karakteri büyük metin olur; `morphemes`/future_train kayıtları
  yapısal-jeton taşımaz. Mock-yığında üretim ilk adımda EOS'te durursa
  `response_text` `""` olabilir (testler anahtar-assert — geçer; K4
  belgeler). **Bu turda test DEĞİŞİKLİĞİ YOK** (`MockDecompiler` tek-arg
  sorunu yalnız DOLU morpheme_output ile tetiklenir — mevcut 307'de
  tetiklenmez).
- `max_new_tokens` 40 (default) / 45 (process_query) tutarsızlığı
  DOKUNULMAZ (görünür-not; ayrı madde).

## Kaynak-çıpa shaları (K1; TAM-DİGEST)

- `data/anka_base_v2.pt` = `d0f415f3d882beb4a3dace87fc4a6024bf3c667f033790fc1e472cb60a664a50`
- `data/rebuild/vocab_anka_r1_33114.json` = `f9940a8d8e1f7cd9428d389f12ff4c5ee448e5a7bfcdcc8ecc9c616fce950984`
- `data/lexicon/roots.tsv` = `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598`
- `data/anka_router.pt` = **KOŞULLU çıpa:** ya
  `64527c725f319db6b3a4d11df1ae98964fa92534e7e4011ca4857376a0ef4720`
  (T-0156-öncesi) **ya** T-0156 hüküm-JSON `cikti_sha256` (T-0156-sonrası);
  ikisi de değilse **DUR** (T-0156 sıralamasına bağışık fail-closed).

## Ölçüm-cihazı (T-0155 dersi — açık beyan)

K2 canlı-üretim **MPS** üzerinde ölçülür (yoksa **DUR** — sessiz-fallback
YOK). `EpistemicCuriosityAgent` kurucusu `router.to(device)` yapar; model
ve tüm çıkarım aynı cihazda. K3/K4 stub'ları CPU'da. Koşum **sandbox
DIŞI** (sandbox MPS'i gizler) ve **sarmal-sız** (`nohup …; echo rc=$?`
rc'yi YUTAR — T-0155 dersi).

## K2 canlı-üretim sorguları (SABİT; İLAN-değer)

1. `Kırlangıç kuyruğu nedir?`
2. `Osmanlı Devleti ne zaman kuruldu?`
3. `Meşe ağacı nedir?`

K2 kapıları: (i) üretilen jeton-id'leri ∩ bastırma-id-seti = ∅;
(ii) `decompiled_text` içinde `<[A-Z_/]+>` regex eşleşmesi YOK;
(iii) metin boş-DEĞİLSE ilk karakter büyük. **Boş çıktı raporlanır, kapı
düşürmez** (küçük model meşru boş üretebilir — bilinçli-beyan).

## Kapılar (hüküm BETİKTEN; elle sayı/hüküm YOK)

- **K1 KAYNAK-ÇIPASI** — yukarıdaki 4 dosya tam-digest (koşullu router-çıpası).
- **K2 CANLI-ÜRETİM (MPS)** — 3 sabit sorgu; sızıntı-∅ + regex-YOK +
  capitalize; kurulumda `load_trained_router_state` (T-0155 entegrasyonu
  yeniden kanıtlanır); VectorMemory `:memory:` (storage_path=None,
  host=None).
- **K3 MUTASYON-KANITI (stub, CPU)** — (i) bastırma sabiti boşaltılırsa →
  RuntimeError; (ii) PAD işaret eden stub + bastırma-kapalı yol →
  çıktıda PAD VAR (pozitif-kontrol: ölçüm-yüzeyi sızıntıyı yakalar);
  (iii) tam sabit → çıktıda yapısal jeton YOK.
- **K4 DAVRANIŞ-KORUMA** — mock-yığınla `ask()` 17-anahtar birebir;
  `response_text` isinstance str; `process_query` anahtar-kümesi sabit;
  UNK-sinyali korunur (stub'da UNK 2-kez + skor ≥ 0,85 → `epistemic_failure`
  tetiklenir); `load_trained_router_state` çağrı-yolu AST'te mevcut.
- **K5 STATİK-ÖN** — py_compile + AST (iki sabit tanımlı;
  `generate_tokens`'ta bastırma + `-inf`; `process_query`'de
  `capitalize=True` + temizlik sabiti) + bağımsız pyflakes TEMİZ.
- **K6 CANLI-DOKUNULMAZLIK** — 192.168.1.9:6333 → 0 istek; localhost →
  0 istek; `create_default` çağrılmaz; VectorMemory `:memory:`.

6/6 → `T0157_OKUNABILIRLIK_GECTI` rc=0; aksi **DUR** rc=2.

## DOKUNULMAZLAR

`src/gateway/agent_gateway.py` (HTTP `:432-487` + `ask()` `:210-249`;
T-0155 kapanışı DEĞİŞMEZ) · `chat_prompt.py` (CLI referans; ortak-refactor
YOK — ayrı uygulamalar beyan edilir) · `src/llm/tokenizer.py` (DONMUŞ) ·
`src/compiler/**` (DONMUŞ) · `tests/` (bu turda değişiklik YOK) · canlı
192.168.1.9:6333 (0 istek; `anka_bellek` DOKUNULMAZ).

## pytest (AYRI koşum — hüküm-dışı; kod DEĞİŞTİ)

Baseline **307/307**. Yeni düşen beklenmez (mock analizi yukarıda);
düşerse İLAN'da beyan — **yumuşatma YOK**.

## Kiralamalar

`src/` (üst-dizin), `scripts/`, `data/eval/`, `.agent-bus/notes/` —
claim-ile-birlikte (T-0153 dersi). Donmuş-desen yazımı YOK.

## Çıktılar

`data/eval/t0157_gateway_okunabilirlik_hukum_2026-09-28.json` ·
`data/eval/t0157_gateway_okunabilirlik_rapor_2026-09-28.md` ·
`data/eval/t0157_gateway_okunabilirlik_kosum_2026-09-28.log` ·
notes `T-0157.md`. Hüküm BETİKTEN; İLAN ≠ RAPOR.