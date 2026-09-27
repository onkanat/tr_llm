# MİMARİ DOĞRULAMA PAKET-1 İLANI — Lexicon → Compile → Decompile Gidiş-Dönüş (T-0141)

**Damga (koşum ÖNCESİ, betikten):** 2026-09-27T10:55:59Z (UTC, `date -u`)
**Operatör onayı:** plan modu onayı 27 Eyl 2026 (.claude/plans/enchanted-wiggling-moon.md)
**Yürütücü:** claude (operatör kararı: "Ben doğrudan")
**Kira:** T-0141 — `scripts/dogrulama_p1_roundtrip.py`, `data/eval/` (dir), `.agent-bus/state/` (dir), `.agent-bus/notes/T-0141.md` (file) — acquire ok:true
**Girdi (salt-okunur):** `data/lexicon/roots.tsv` — sha256 `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` (52.374 satır = 1 başlık + 52.373 veri)

---

## 1. Kapsam ve kural

Kanonik zincir **IMPORT** edilir (kopya YASAK; `src/compiler/**` DONMUŞ):
`LexiconManager().load_from_tsv("data/lexicon/roots.tsv")` → `build_default_graph()` →
`CrystalCompiler(lexicon, graph)` (vocab=None) → `MorphemeDecompiler(compiler, vocab=None)`.

Saf CPU, tek süreç, seed 42, Qdrant GEREKMEZ. Kanonik kodda YAZIM YOKTUR.

**Hüküm (operatör kararı, koşum ÖNCESİ sabit):** FAZ-B birincil kapı + FAZ-C envanter birebirliği → `P1_GECTİ` (rc=0); aksi her dal → `DUR` (rc=2, stderr+rc kayıtlı — duran-dal dersi). Hüküm BETİK İÇİNDEDİR; elle sayı/hüküm YOK.

## 2. FAZ-B örneklem planı (birincil hüküm)

- **Hedef:** ~10.000 gidiş-dönüş örneği; havuz = TSV'deki **çekimlenebilir** lemma'lar.
- **Kapsam dışı (ÖNCE ilan edilir, adetleri raporlanır):** lemma'lar ki
  (a) `lemma != lemma.lower()` (büyük harfli/özel ad — compile girdiyi küçültür; KSUR-2/3 kapsamı),
  (b) boşluk içerir (çok-sözcüklü lemma — find_stems tek-sözcük arar),
  (c) apostrof içerir (kesme sınıfı — KSUR-2 kapsamı).
- **Örnekleme:** deterministik seed 42; POS katmanları hedef paylarla
  NOUN 5.000 · VERB 2.500 · ADJ 1.500 · ADV 800 · NUM/PRON/POSTP/INTJ/CONJ toplam 200.
- **Ek zinciri:** graph'tan **geçerli geçiş yürüyüşü** (`get_valid_transitions`) — kök state'ten
  derinlik 1..3 rasgele geçiş, terminal/çıkmazda dur; kök state = compile'ın `pos_to_state`
  eşlemesiyle AYNI (VERB/NOUN/ADJ/ADV kökleri; NUM/PRON/POSTP/INTJ/CONJ → NOUN_ROOT).
- **Örnek protokolü:** `tags = [lemma] + ek_id_dizisi` → `yüzey = decompile_tags(tags)`
  → `pack = compile(yüzey)` → `geri = decompile_tags(pack["token_vector"])`
  (token_vector = best-path morfem id dizisi, core.py:101).
- **Birincil ölçüt (hükme bağlı):** `geri == yüzey` — çekimlenebilir sınıfta **%100**.
- **İkincil kayıt (hükme bağlanmaz — "id karşılaştırması sahte-sıfır" dersi):**
  `pack["best_surface"] == yüzey` oranı; `pack["token_vector"] == tags` (tam-morfem eş) oranı;
  `needs_disambiguation` payı.
- **Yeni kusur sınıfları (ÖNCEDEN İLANLI = DUR tetikleyicileri):** 0-yol örneği
  (`analyses == []`), boş yüzey üretimi, beklenmedik istisna. Bunlar FAZ-B kapsamında
  **bilinmeyen** sınıflardır; görülmeleri fail-closed DUR üretir.

## 3. FAZ-A tam tarama (envanter — hükme bağlanmaz)

52.373 lemma (benzersiz) tam `compile(lemma)` taraması: POS-bazlı yol-bulma oranı,
`analyses == []` sessiz-boş sınıfı kırılımı (KSUR-4'ün lemma-öz hâli), istisna FIRLATMAMA
davranış teyidi. FAZ-A çıktısı rapor verisidir; hüküm FAZ-B+FAZ-C'dedir.

## 4. FAZ-C bilinen-ksur envanteri — İLANLI SAYILAR (birebir beklenir)

| Sınıf | Ölçüm | İLANLI değer | Kaynak |
|---|---|---|---|
| **KSUR-1** ADJ-çekim engeli | NOUN satırı olmayan ADJ lemma sayısı (TSV parse) | **3.627 / 6.352 ADJ** | morfotactics.py:141,157; T-0093 sonrası ölçüm |
| **KSUR-1 probe** | 50 yalnız-ADJ lemma + PLURAL → `compile` 0-yol | **50/50 sessiz-boş** | ADJ_ROOT'ta isim-ek geçişi yok |
| **KSUR-2** kesme-apostrof | isupper-lemma probe (20 örnek) + placeholder probe → yüzeyde apostrof | **%100 kesmeli** (davranış envanteri) | decompiler.py:140-141 (koşulsuz); :239 kümede kesme yok |
| **KSUR-3** İ/I düz-lower | `lemma.lower() != turkish_lower(lemma)` olan lemma sayısı | **79 (72 İ + 7 I)** | T-0080 ölçümü; düz-lower anahtarı KORUNUR (lexicon.py:42) |
| **KSUR-4** sessiz-boş OOV | 200 sahte OOV probe (trie'de kök YOK teyitli) → `analyses == []`, istisna YOK | **200/200 sessiz-boş** | core.py compile istisna fırlatmaz (CLAUDE.md OOV standardına aykırılık — envanter kalemi) |

Sapma (her sınıfta) = fail-closed **DUR**. KSUR sınıfları FAZ-B birincil örneklemine
girmez (yukarıdaki kapsam-dışı kuralı); böylece %100 bit-özdeşlik ölçütü bu sınıflarla
kontamine edilmez.

## 5. Koşum komutu (damgalı)

```
venv/bin/python scripts/dogrulama_p1_roundtrip.py \
  --lexicon data/lexicon/roots.tsv \
  --ilan data/eval/mimari_dogrulama_p1_ilan_2026-09-27.md \
  --rapor data/eval/mimari_dogrulama_p1_sonuc_2026-09-27.md
```

- Saf CPU; sandbox İÇİNDE koşabilir (MPS/soket YOK). Süre pilotu: ilk 1.000 satır
  sn/hız ölçülür, toplam ekstrapole edilir (boyut-kontrol dersi).
- Çıktı: rapor + hüküm JSON (`data/eval/mimari_dogrulama_p1_hukum_2026-09-27.json`)
  betikten; rc ∈ {0, 2}.
- Bu İLAN dosyasının sha256'sı RAPORA işlenecek (İLAN ≠ RAPOR; koşum-öncesi mtime).

## 6. KOŞUM-ÖNCESİ REVİZYON (probe bulguları — damga 2026-09-27T11:01Z'ye kadar, koşum BAŞLAMADI)

Koşum-öncesi keşif probe'ları ($TMPDIR, repo-yazımsız) İLAN'ı iki noktada netleştirdi:

1. **KSUR-3 ölçüm tanımı ÇİFT İLANLI:** satır-bazlı `lemma.lower() != turkish_lower(lemma)`
   sayımı = **79 satır (72 İ + 7 I)** — birebir doğrulandı; benzersiz-lemma bazlı = **70
   (63 İ + 7 I)**. Betik İKİSİNİ de ölçer; hükümde ikisi de İLANLI değerdir (birebir
   eşleşme şartı her iki tanım için geçerlidir).
2. **KSUR-1 probe havuzu tanımı:** probe lemma havuzu = küçük-harf, boşluksuz,
   apostrofsuz **yalnız-ADJ** (POS kümesi == {ADJ}) lemma'lar — ölçülen havuz **3.249**;
   İLAN'lı 3.627 ise "NOUN satırı olmayan ADJ" (yalnız-ADJ'yi kapsamayan daha geniş
   küme) tanımıdır — her ikisi ayrı satırda raporlanır.
3. **Probe teyitleri:** yalnız-ADJ + PLURAL → 0-yol **8/8**; NOUN/VERB yürüyüş
   gidiş-dönüşleri EŞİT (örn. `['git','TENSE_OPTATIVE','PERSON_2PL'] → 'gidesiniz'`);
   `token_vector == tags` eşliği bazı çok-yollu örneklerde SAĞLANMAYABİLİR (surface
   eşliği korunarak) — ikincil kayıt hükmü doğrulamaz (İLAN §2 ile tutarlı).

## 7. DOKUNULMAZ / YAPILMAYANLAR

`src/compiler/**` · `src/llm/tokenizer.py` · `evaluate_carpenter_anka.py` DOKUNULMAZ ·
`data/**` yazımı yalnız `data/eval/`'e · ESİK_ROUGE 0,3221 / TAVAN_ROUGE_DECOMP 0,9509
ilgilendirmez (bu paket ROUGE ölçmez) · commit ayrı operatör onayıyla · `git add -A` YASAK.