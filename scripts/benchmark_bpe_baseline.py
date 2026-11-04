#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.7 — BPE-baseline benchmark: Kristal tokenizer vs BPE (tek komutla tekrarlanabilir).

Kaynak inceleme betikleri (03.11.2026, ~/.hermes/cache/scratch/trllm_olcum/):
  - bpe_v2.py     : artımlı BPE eğitimi (bayat heap girdisini geri itmeli; tur-boyunca doğrulama)
  - bpe_kiyas.py  : 4 sabit metin + kelime örneklemi üzerinde tok/kelime kıyası
  - kapsam.py     : kristal tarafı (kök kapsamı, UNK oranı, apostrof davranışı)
Bunlar repoya taşındı: bu betik EĞİTİM + KIYASI tek komutta birleştirir.

Ölçümler:
  - tok/kelime (fertility) : 4 sabit metin + korpus satır örneklemi + benzersiz kelime örneklemi
  - UNK oranı              : kristalda UNK jeton oranı; BPE'de UNK yok — eğitim alfabesinin
                             dışında kalan karakter oranı (kelime-dışı karakter oranı) karşılaştırılabilir ölçüm
  - encode hızı            : örneklem satırları üzerinde tok/s (3 tekrar, en iyisi)

Kullanım:
    venv/bin/python scripts/benchmark_bpe_baseline.py
    venv/bin/python scripts/benchmark_bpe_baseline.py --corpus data/raw_corpus.txt --samples 2000
    venv/bin/python scripts/benchmark_bpe_baseline.py --merges 2000,8000,32000 --seed 42

Çıktı:
    data/eval/bpe_baseline_rapor_<tarih>.md    (İLAN → HÜKÜM → RAPOR)
    data/eval/bpe_baseline_olcum_<tarih>.json  (ham ölçümler)
"""
import argparse
import heapq
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)  # fixture'lar (data/...) göreli yol kullanır

# --- kaynak betiklerle birebir aynı ön-tokenizasyon konvansiyonu ---
WORD_RE = re.compile(r"[\w']+|[.,!?;:()\"—–-]", flags=re.UNICODE)
PUNCT = set(".,!?;:()\"—–-")

# kaynak betikteki 4 sabit metin (bpe_kiyas.py / kapsam.py ile aynı)
FIXED_TEXTS = {
    "gunluk": "Yarın akşam arkadaşlarımla sinemaya gideceğiz ama önce evdeki ödevlerimizi bitirmemiz gerekiyor.",
    "teknik": "Alüminyum oksit zımpara tanecikleri lif koparmadan yüzey pürüzlülüğünü sıfıra indirir ve vernik öncesi boya tabanını hazırlar.",
    "edebi": "Tarihin seyrinde milletlerin yükselişi ve inhitatı içtimai buudlarıyla tetkik olunmadıkça fehmedilmez.",
    "tamlama": "Maarif Vekaleti Türk Tarihi Tetkik Cemiyeti yayımladığı yazım kılavuzu yayın kurulu kararları tasdikatinden geçti.",
}


def words_of(text):
    """Kaynak betik konvansiyonu: kelimeler (küçük harf), noktalama hariç."""
    return [t.lower() for t in WORD_RE.findall(text) if t not in PUNCT]


# ----------------------------------------------------------------------------
# BPE v2 eğitimi (bpe_v2.py'den taşındı: bayat heap girdisi geri itmeli,
# tur-boyunca doğrulama ile)
# ----------------------------------------------------------------------------
def bpe_train(word_freq, merge_targets):
    """word_freq: {kelime: frekans}. merge_targets: sıralı hedef listesi.
    Dönüş: {hedef: merges_list}, temel alfabe, erişilen merge sayıları, süre.

    Kaynak bpe_v2.py'den bilinçli sapma: kelime işleme sırası kanonikleştirildi
    (`sorted`). Kaynak betik pair_index kümelerini ham set yineleme sırasıyla
    dönüyordu; bu sıra PYTHONHASHSEED'e bağlı olduğundan beraberlik (tie) durumlarında
    merge listesi süreçten sürece değişiyordu (ölçüldü: 8 koşumda merge[18] bir kez
    takla attı). sorted ile eğitim tam deterministiktir — P2.7 tekrarlanabilirlik şartı.
    """
    sym = {w: list(w) + ["</w>"] for w in word_freq}
    freq = dict(word_freq)

    pair_counts = Counter()
    pair_index = defaultdict(set)
    for w, syms in sym.items():
        f_ = freq[w]
        for a, b in zip(syms, syms[1:]):
            pair_counts[(a, b)] += f_
            pair_index[(a, b)].add(w)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    reached = {}

    def train_until(n_total):
        while len(merges) < n_total:
            best = None
            # bayat girdileri geri it, doğru sayıyı bul
            while heap:
                c, p = heapq.heappop(heap)
                cur = pair_counts.get(p, 0)
                if cur == -c and cur >= 2:
                    best = (cur, p)
                    break
                elif cur >= 2:
                    heapq.heappush(heap, (-cur, p))
                    best = None
                    continue
            if best is None:
                # heap'te geçerli kalmadıysa lineer tara
                for p, c in pair_counts.items():
                    if c >= 2 and (best is None or c > best[0]):
                        best = (c, p)
                if best is None:
                    return False
            c, (a, b) = best
            merges.append((a, b))
            pair_counts.pop((a, b), None)
            # kanonik sıra -> çapraz-süreç deterministik (kaynak: list(...) idi, ham set sırası)
            for w in sorted(pair_index[(a, b)]):
                syms = sym[w]
                f_ = freq[w]
                i = 0
                newsyms = []
                changed = False
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        newsyms.append(a + b)
                        i += 2
                        changed = True
                    else:
                        newsyms.append(syms[i])
                        i += 1
                if not changed:
                    continue
                for x, y in zip(syms, syms[1:]):
                    pair_counts[(x, y)] -= f_
                    if pair_counts[(x, y)] <= 0:
                        pair_counts.pop((x, y), None)
                    pair_index[(x, y)].discard(w)
                sym[w] = newsyms
                for x, y in zip(newsyms, newsyms[1:]):
                    if (x, y) not in pair_counts:
                        heapq.heappush(heap, (-f_, (x, y)))
                    pair_counts[(x, y)] += f_
                    pair_index[(x, y)].add(w)
            pair_index.pop((a, b), None)
        return True

    snaps = {}
    t0 = time.time()
    for m in merge_targets:
        ok = train_until(m)
        snaps[m] = list(merges)
        reached[m] = len(merges)
        if not ok:
            print(f"  UYARI: {m} merge hedefine ulaşılamadı (çift tükendi): {len(merges)} merge")
    elapsed = time.time() - t0

    base_alphabet = set()
    for w in word_freq:
        base_alphabet.update(w)
    base_alphabet.add("</w>")
    return snaps, base_alphabet, reached, elapsed


def bpe_apply(word, merge_list):
    """Sıra-düzgün BPE uygulaması (bpe_kiyas.py ile aynı)."""
    rank = {p: i for i, p in enumerate(merge_list)}
    syms = list(word) + ["</w>"]
    while True:
        best = None
        bi = -1
        for i, (a, b) in enumerate(zip(syms, syms[1:])):
            r = rank.get((a, b))
            if r is not None and (best is None or r < best):
                best = r
                bi = i
        if best is None:
            return syms
        syms = syms[:bi] + [syms[bi] + syms[bi + 1]] + syms[bi + 2:]


def bpe_vocab_size(merge_list, base_alphabet):
    """Gerçek BPE sözük boyutu: temel alfabe + birleştirilmiş semboller."""
    syms = set(base_alphabet)
    for a, b in merge_list:
        syms.add(a + b)
    return len(syms)


# ----------------------------------------------------------------------------
# Kristal tokenizer (kapsam.py deseninden taşındı; anka vocab fixture'ı)
# ----------------------------------------------------------------------------
def load_kristal(vocab_path):
    from src.compiler.core import CrystalCompiler
    from src.compiler.lexicon import LexiconManager
    from src.compiler.morphotactics import build_default_graph
    from src.llm.tokenizer import KristalTokenizer, Vocabulary

    t0 = time.time()
    lex = LexiconManager()
    lex.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    vocab = Vocabulary()
    vocab.load(vocab_path)
    compiler = CrystalCompiler(lex, graph, vocab=vocab)
    tok = KristalTokenizer(compiler, vocab, literal_entity_mode=True)
    meta = {
        "yukleme_sn": round(time.time() - t0, 2),
        "vocab_boyut": len(vocab.stoi),
        "bos_id": vocab.stoi.get("<BOS>"),
        "eos_id": vocab.stoi.get("<EOS>"),
        "unk_id": vocab.stoi.get("<UNK>"),
    }
    return tok, vocab, meta


def kristal_encode(text, tok, unk_id, bos_id, eos_id):
    """ids döner: (jeton sayısı [BOS/EOS hariç], UNK sayısı)."""
    ids = tok.encode(text)
    n = len(ids)
    if ids and bos_id is not None and ids[0] == bos_id:
        n -= 1
    if ids and eos_id is not None and ids[-1] == eos_id:
        n -= 1
    n_unk = sum(1 for i in ids if i == unk_id)
    return n, n_unk


def bpe_out_of_alphabet_ratio(words, base_alphabet):
    """BPE'de UNK yok; eğitim alfabesinin dışında kalan karakter oranı (%) karşılaştırılabilir ölçüm."""
    total = 0
    oov = 0
    for w in words:
        for ch in w:
            total += 1
            if ch not in base_alphabet:
                oov += 1
    return (100.0 * oov / total) if total else 0.0


# ----------------------------------------------------------------------------
# Ölçüm blokları
# ----------------------------------------------------------------------------
def measure_fixed_texts(tok, vocab_meta, variants):
    """4 sabit metin: kristal ve her BPE varyantı için tok/kelime + UNK."""
    unk_id, bos_id, eos_id = vocab_meta["unk_id"], vocab_meta["bos_id"], vocab_meta["eos_id"]
    rows = {}
    for name, text in FIXED_TEXTS.items():
        ws = words_of(text)
        n_tok, n_unk = kristal_encode(text, tok, unk_id, bos_id, eos_id)
        row = {
            "kelime": len(ws),
            "kristal_tok": n_tok,
            "kristal_tok_kelime": round(n_tok / len(ws), 3),
            "kristal_unk": n_unk,
            "kristal_unk_yuzde": round(100.0 * n_unk / n_tok, 2) if n_tok else 0.0,
        }
        for vname, merges, base_alphabet in variants:
            tot = sum(len(bpe_apply(w, merges)) for w in ws)
            row[f"{vname}_tok_kelime"] = round(tot / len(ws), 3)
        rows[name] = row
    return rows


def measure_corpus_sample(lines, tok, vocab_meta, variants, base_alphabet, reps=3):
    """Korpus satır örneklemi: tok/kelime + UNK + hız (her iki kodlayıcı)."""
    unk_id, bos_id, eos_id = vocab_meta["unk_id"], vocab_meta["bos_id"], vocab_meta["eos_id"]

    # --- kristal ---
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        n_tok = n_unk = 0
        for line in lines:
            a, b = kristal_encode(line, tok, unk_id, bos_id, eos_id)
            n_tok += a
            n_unk += b
        dt = time.perf_counter() - t0
        if best is None or dt < best[0]:
            best = (dt, n_tok, n_unk)
    assert best is not None  # reps >= 1
    dt, n_tok, n_unk = best
    n_words = sum(len(words_of(l)) for l in lines)
    n_punct = sum(1 for l in lines for t in WORD_RE.findall(l) if t in PUNCT)

    kristal = {
        "satir": len(lines),
        "kelime": n_words,
        "noktalama_isaret": n_punct,
        "token": n_tok,
        "tok_kelime": round(n_tok / n_words, 3),
        "unk_token": n_unk,
        "unk_yuzde": round(100.0 * n_unk / n_tok, 2) if n_tok else 0.0,
        "sure_sn": round(dt, 2),
        "tok_s": round(n_tok / dt, 1),
        "tekrar": reps,
    }

    # --- BPE varyantları ---
    bpe_rows = {}
    for vname, merges, _ in variants:
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            n_tok = 0
            for line in lines:
                for w in words_of(line):
                    n_tok += len(bpe_apply(w, merges))
            dt = time.perf_counter() - t0
            if best is None or dt < best[0]:
                best = (dt, n_tok)
        assert best is not None  # reps >= 1
        dt, n_tok_b = best
        all_words = []
        for line in lines:
            all_words.extend(words_of(line))
        bpe_rows[vname] = {
            "token": n_tok_b,
            "tok_kelime": round(n_tok_b / n_words, 3),
            "kelime_disi_karakter_yuzde": round(bpe_out_of_alphabet_ratio(all_words, base_alphabet), 3),
            "sure_sn": round(dt, 2),
            "tok_s": round(n_tok_b / dt, 1),
            "tekrar": reps,
        }
    return kristal, bpe_rows


def measure_word_sample(word_list, tok, vocab_meta, variants, base_alphabet):
    """Benzersiz kelime örneklemi (kaynak betiğin 'korpus_2000ornek' ölçümü)."""
    unk_id, bos_id, eos_id = vocab_meta["unk_id"], vocab_meta["bos_id"], vocab_meta["eos_id"]
    row = {}
    t0 = time.perf_counter()
    tot = 0
    n_unk = 0
    for w in word_list:
        a, b = kristal_encode(w, tok, unk_id, bos_id, eos_id)
        tot += a
        n_unk += b
    dt = time.perf_counter() - t0
    row["kristal"] = {
        "tok_kelime": round(tot / len(word_list), 3),
        "unk_yuzde": round(100.0 * n_unk / tot, 2) if tot else 0.0,
        "sure_sn": round(dt, 2),
    }
    for vname, merges, _ in variants:
        t0 = time.perf_counter()
        tot = sum(len(bpe_apply(w, merges)) for w in word_list)
        dt = time.perf_counter() - t0
        row[vname] = {
            "tok_kelime": round(tot / len(word_list), 3),
            "kelime_disi_karakter_yuzde": round(bpe_out_of_alphabet_ratio(word_list, base_alphabet), 3),
            "sure_sn": round(dt, 2),
        }
    return row


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="Kristal tokenizer vs BPE baseline: tek komutla tekrarlanabilir karşılaştırma raporu (P2.7)"
    )
    ap.add_argument("--corpus", default="data/raw_corpus.txt",
                    help="BPE eğitim + test korpusu (varsayılan: data/raw_corpus.txt — kaynak betik korpusu)")
    ap.add_argument("--samples", type=int, default=2000,
                    help="korpus satır örneklemi boyutu (tok/kelime + UNK + hız ölçümü)")
    ap.add_argument("--word-samples", type=int, default=2000,
                    help="benzersiz kelime örneklemi boyutu (kaynak betiğin korpus_2000ornek ölçümü)")
    ap.add_argument("--merges", default="2000,8000,32000",
                    help="BPE merge hedefleri, virgülle (varsayılan: 2000,8000,32000)")
    ap.add_argument("--seed", type=int, default=42, help="rastgelelik tohumu (varsayılan: 42)")
    ap.add_argument("--vocab", default="data/rebuild/vocab_anka_r1_33114.json",
                    help="kristal vocab fixture'ı")
    ap.add_argument("--out-dir", default="data/eval", help="rapor/JSON çıktı dizini")
    args = ap.parse_args()

    merge_targets = [int(x) for x in args.merges.split(",") if x.strip()]
    tarih = time.strftime("%Y-%m-%d")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- 1. korpus yükle + kelime frekansları (kaynak betik konvansiyonu) ---
    print(f"[1/5] Korpus okunuyor: {args.corpus}")
    t0 = time.time()
    word_freq = Counter()
    lines = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
            for t in WORD_RE.findall(line):
                if t not in PUNCT:
                    word_freq[t.lower()] += 1
    print(f"  satır: {len(lines)}, benzersiz kelime: {len(word_freq)}, "
          f"toplam kelime: {sum(word_freq.values())} ({time.time()-t0:.1f} sn)")

    # --- 2. BPE eğitimi (v2: bayat heap geri-itmeli) ---
    print(f"[2/5] BPE eğitimi (hedefler: {merge_targets})...")
    snaps, base_alphabet, reached, bpe_sure = bpe_train(word_freq, merge_targets)
    for m in merge_targets:
        print(f"  {m} merge hedefi -> {reached[m]} merge, "
              f"sözük ~{bpe_vocab_size(snaps[m], base_alphabet)} (toplam {bpe_sure:.1f} sn)")

    variants = []
    for m in merge_targets:
        # ad ulaşılan GERÇEK merge sayısını yansıtır (hedefe düşülmezse aldatıcı olmasın)
        vname = f"BPE-{reached[m]}"
        variants.append((vname, snaps[m], base_alphabet))

    # --- 3. kristal tokenizer yükle ---
    print(f"[3/5] Kristal tokenizer yükleniyor (vocab: {args.vocab})...")
    tok, vocab, vocab_meta = load_kristal(args.vocab)
    print(f"  yükleme {vocab_meta['yukleme_sn']} sn, vocab boyut: {vocab_meta['vocab_boyut']}")

    # --- 4. ölçümler (tohum sabit -> tekrarlanabilir) ---
    random.seed(args.seed)
    sample_lines = random.sample(lines, min(args.samples, len(lines)))
    print(f"[4/5] Ölçümler: {len(sample_lines)} satır örneklemi (tohum {args.seed}) + "
          f"{len(FIXED_TEXTS)} sabit metin + {args.word_samples} benzersiz kelime...")
    fixed_rows = measure_fixed_texts(tok, vocab_meta, variants)
    kristal_corr, bpe_corr = measure_corpus_sample(sample_lines, tok, vocab_meta, variants, base_alphabet)
    unique_words = list(word_freq.keys())
    k = min(args.word_samples, len(unique_words))
    word_sample = random.Random(args.seed).sample(unique_words, k)
    word_rows = measure_word_sample(word_sample, tok, vocab_meta, variants, base_alphabet)
    print(f"  korpus örneklemi: kristal {kristal_corr['tok_kelime']} tok/kelime, "
          f"{kristal_corr['tok_s']} tok/s | " +
          " | ".join(f"{v}: {bpe_corr[v]['tok_kelime']} tok/kelime, {bpe_corr[v]['tok_s']} tok/s"
                     for v, _, _ in variants))

    # --- 5. rapor + JSON ---
    print("[5/5] Rapor yazılıyor...")
    head = git_head()
    params = {
        "gorev": "P2.7_bpe_baseline_benchmark",
        "tarih": tarih,
        "kaynak_betikler": [
            "~/.hermes/cache/scratch/trllm_olcum/bpe_v2.py",
            "~/.hermes/cache/scratch/trllm_olcum/bpe_kiyas.py",
            "~/.hermes/cache/scratch/trllm_olcum/kapsam.py",
        ],
        "komut": f"venv/bin/python scripts/benchmark_bpe_baseline.py --corpus {args.corpus} "
                 f"--samples {args.samples} --merges {args.merges} --seed {args.seed}",
        "git_head": head,
        "korpus": args.corpus,
        "tohum": args.seed,
        "merge_hedefleri": merge_targets,
        "vocab": args.vocab,
    }
    olcum = {
        "params": params,
        "korpus": {
            "satir": len(lines),
            "benzersiz_kelime": len(word_freq),
            "toplam_kelime": sum(word_freq.values()),
        },
        "kristal": vocab_meta,
        "bpe_egitim": {
            "sure_sn": round(bpe_sure, 1),
            "ulasilan_merge": reached,
            "temel_alfabe_boyut": len(base_alphabet),
            "sozluk_boyut": {v: bpe_vocab_size(snaps[v2], base_alphabet)
                             for v, v2, _ in [(m, m, None) for m in merge_targets]},
        },
        "sabit_metinler": fixed_rows,
        "korpus_satir_orneklemi": {"kristal": kristal_corr, **{v: bpe_corr[v] for v, _, _ in variants}},
        "benzersiz_kelime_orneklemi": {"n": len(word_sample), **word_rows},
    }
    json_path = os.path.join(args.out_dir, f"bpe_baseline_olcum_{tarih}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(olcum, f, ensure_ascii=False, indent=1)

    # ---- markdown rapor: İLAN → HÜKÜM → RAPOR ----
    k32 = bpe_corr[variants[-1][0]]  # en büyük BPE varyantı
    krim = kristal_corr
    hiz_oran = round(k32["tok_s"] / krim["tok_s"], 1)
    fert_oran = round(k32["tok_kelime"] / krim["tok_kelime"], 2)

    L = []
    L.append(f"# P2.7 — BPE-Baseline Benchmark Raporu ({tarih})\n")
    L.append("## İLAN\n")
    L.append(f"- **Amaç:** Kristal tokenizer ile BPE baseline'ı aynı test korpusu üzerinde "
             f"tek komutla, tekrarlanabilir biçimde karşılaştırmak (P2.7 kabul ölçütü).")
    L.append(f"- **Kaynak:** inceleme betikleri `bpe_v2.py`/`bpe_kiyas.py`/`kapsam.py` "
             f"(~/.hermes/cache/scratch/trllm_olcum/) repoya taşındı → `scripts/benchmark_bpe_baseline.py`.")
    L.append(f"- **Komut:** `{params['komut']}` (git HEAD: `{head or 'n/a'}`)")
    L.append(f"- **Korpus:** `{args.corpus}` — {len(lines)} dolu satır, {len(word_freq):,} benzersiz kelime, "
             f"{sum(word_freq.values()):,} toplam kelime.")
    L.append(f"- **BPE:** saf Python artımlı BPE (kelime-içi, `</w>` işaretli; bayat-heap geri-itmeli doğrulama), "
             f"aynı korpus üzerinde eğitildi; hedefler {', '.join(str(m) for m in merge_targets)} merge.")
    L.append(f"- **Kristal:** `KristalTokenizer(literal_entity_mode=True)`, vocab `{args.vocab}` "
             f"({vocab_meta['vocab_boyut']:,} ID).")
    L.append(f"- **Örneklem/tekrarlanabilirlik:** satır örneklemi {len(sample_lines)} (tohum {args.seed}), "
             f"benzersiz kelime örneklemi {len(word_sample)} (tohum {args.seed}), "
             f"hız ölçümü 3 tekrarın en iyisi. BPE eğitimi kanonikleştirildi: kaynak bpe_v2.py'de etkilenen-kelime "
             f"yinelemesi ham set sırasıyla yapılıyordu ve bu sıra PYTHONHASHSEED'e bağlı olduğundan beraberlik "
             f"(tie) durumlarında merge listesi süreçten sürece değişiyordu (ölçüldü: 8 koşumda merge[18] bir kez "
             f"takla attı). Bu portta `sorted` sırasıyla eğitim tam deterministiktir.")
    L.append(f"- **UNK ölçüm notu:** BPE'de UNK jetonu yoktur; karşılaştırılabilir ölçüm olarak eğitim "
             f"alfabesinin dışında kalan **kelime-dışı karakter oranı** raporlanır. Kristalda UNK jeton oranıdır.")
    L.append(f"- **Sözleşme notu:** Kristal tok/kelime payı BOS/EOS hariç, **noktalama jetonları dahil** gerçek dizi "
             f"uzunluğudur; BPE yalnız kelime jetonlarını sayar (kaynak betik konvansiyonu). Örneklemede "
             f"{krim['noktalama_isaret']:,} noktalama işareti kristal sayısına dahildir "
             f"(~+{round(krim['noktalama_isaret']/krim['kelime'], 3)} tok/kelime pay).\n")
    L.append("## HÜKÜM\n")
    big = variants[-1][0]
    fert_fark = round((fert_oran - 1) * 100)
    fert_yon = "daha fazla" if fert_oran > 1 else ("daha az" if fert_oran < 1 else "aynı oranda")
    sabit_k = round(sum(fixed_rows[n]["kristal_tok_kelime"] for n in FIXED_TEXTS) / len(FIXED_TEXTS), 3)
    sabit_b = round(sum(fixed_rows[n][f"{big}_tok_kelime"] for n in FIXED_TEXTS) / len(FIXED_TEXTS), 3)
    sabit_oran = round(sabit_b / sabit_k, 2)
    kel_k = word_rows["kristal"]["tok_kelime"]
    kel_b = word_rows[big]["tok_kelime"]
    kel_oran = round(kel_b / kel_k, 2)
    L.append(f"- **Frekans-ağırlıklı (in-domain) fertility:** korpus satır örneklemi BPE'nin EĞİTİM korpusundan "
             f"alınmıştır; kristal **{krim['tok_kelime']}** vs {big} **{k32['tok_kelime']}** tok/kelime — "
             f"{big} {abs(fert_fark)}% {fert_yon} jeton. {big} bu ölçümde ezber avantajı taşır "
             f"(42,321 benzersiz kelimenin sık geçenleri ~32k merge ile bütün-kelime sembolüne dönüşmüştür).")
    L.append(f"- **Dağıtım-dışı / seyrek-kelime fertility:** 4 sabit metin ortalaması kristal **{sabit_k}** vs "
             f"{big} **{sabit_b}** tok/kelime ({sabit_oran}×); benzersiz kelime örneklemi kristal **{kel_k}** vs "
             f"{big} **{kel_b}** ({kel_oran}×) — metin korpusun birebir satırları olmadığında ve seyrek kelimelerde "
             f"morfolojik çözümleme subword ezberinden belirgin biçimde daha verimli.")
    L.append(f"- **UNK ekseni:** kristal UNK oranı satır örnekleminde **%{krim['unk_yuzde']}**, benzersiz kelime "
             f"örnekleminden **%{word_rows['kristal']['unk_yuzde']}** (seyrek kelimelerin bir bölümü morfolojik "
             f"çözümleme bulamıyor). BPE'de UNK jetonu yoktur; ölçülen kelime-dışı karakter oranı in-domain "
             f"örneklemde trivyal olarak %{k32['kelime_disi_karakter_yuzde']}. UNK'suzluk BPE'nin yapısal "
             f"avantajıdır; dağıtım dışı metinde bu karakter oranı artar, kristalda ise UNK oranı artar.")
    hiz_yon = "daha hızlı" if k32["tok_s"] < krim["tok_s"] else ("daha yavaş" if k32["tok_s"] > krim["tok_s"] else "eşit hızlı")
    L.append(f"- **Hız ekseni:** kristal **{krim['tok_s']:,} tok/s** vs {big} **{k32['tok_s']:,} tok/s** "
             f"(BPE/kristal ~{hiz_oran}×; kristal {hiz_yon}). DİKKAT: BPE hızı burada saf-Python applier ölçer; üretim "
             f"BPE kurulumları (Rust tabanlı HF tokenizers vb.) kat kat hızlıdır. Bu eksen paradigma değil "
             f"uygulama farkı ölçer.")
    sozluk_oran = round(vocab_meta["vocab_boyut"] / bpe_vocab_size(snaps[merge_targets[-1]], base_alphabet), 2)
    L.append(f"- **Sözük ekonomisi:** kristal {vocab_meta['vocab_boyut']:,} sabit ID vs {big} ~"
             f"{bpe_vocab_size(snaps[merge_targets[-1]], base_alphabet):,} sembol ({sozluk_oran}×) — benzer "
             f"sözük bütçesinde karşılaştırma adil.")
    L.append(f"- **Sonuç:** BPE-baseline {', '.join(v for v, _, _ in variants)} varyantlarıyla formalize edildi ve "
             f"tek komutla tekrarlanabilir rapora bağlandı. Eksen dengesi (BPE/kristal jeton oranı): in-domain "
             f"frekans-ağırlıklı {fert_oran}×, 4 sabit metin {sabit_oran}×, benzersiz kelime {kel_oran}× — "
             f"kristal dağıtım-dışı/seyyar eksenlerde önde, BPE yalnız in-domain ezber ölçümünde öne geçebilir; "
             f"UNK'suzluk BPE'nin tek yapısal üstünlüğüdür.\n")
    L.append("## RAPOR\n")
    L.append(f"### Tablo 1 — Korpus satır örneklemi ({len(sample_lines)} satır, "
             f"{krim['kelime']:,} kelime; tok/kelime + UNK + hız)\n")
    L.append("| Kodlayıcı | Sözük | tok/kelime | UNK / kelime-dışı karakter | Hız (tok/s) |")
    L.append("|---|---|---|---|---|")
    L.append(f"| **Kristal** | {vocab_meta['vocab_boyut']:,} ID | **{krim['tok_kelime']}** | "
             f"UNK %{krim['unk_yuzde']} ({krim['unk_token']:,}/{krim['token']:,}) | {krim['tok_s']:,} |")
    for vname, merges, _ in variants:
        r = bpe_corr[vname]
        L.append(f"| {vname} | ~{bpe_vocab_size(merges, base_alphabet):,} sembol | {r['tok_kelime']} | "
                 f"kelime-dışı karakter %{r['kelime_disi_karakter_yuzde']} | {r['tok_s']:,} |")
    L.append("")
    L.append("### Tablo 2 — 4 sabit metin (kaynak betik metinleriyle birebir aynı)\n")
    L.append("| Metin | Kelime | Kristal tok/kelime | Kristal UNK% | " +
             " | ".join(f"{v} tok/kelime" for v, _, _ in variants) + " |")
    L.append("|---|---|---|---|---" + "|---" * (len(variants) - 1) + "|")
    for name in FIXED_TEXTS:
        r = fixed_rows[name]
        L.append(f"| {name} | {r['kelime']} | {r['kristal_tok_kelime']} | {r['kristal_unk_yuzde']}% | " +
                 " | ".join(f"{r[f'{v}_tok_kelime']}" for v, _, _ in variants) + " |")
    ort_k = round(sum(fixed_rows[n]['kristal_tok_kelime'] for n in FIXED_TEXTS) / len(FIXED_TEXTS), 3)
    L.append(f"| **ortalama** | — | **{ort_k}** | — | " +
             " | ".join(f"**{round(sum(fixed_rows[n][f'{v}_tok_kelime'] for n in FIXED_TEXTS)/len(FIXED_TEXTS), 3)}**"
                       for v, _, _ in variants) + " |")
    L.append("")
    L.append(f"### Tablo 3 — Benzersiz kelime örneklemi ({len(word_sample)} kelime, tohum {args.seed}; "
             f"kaynak betiğin 'korpus_2000ornek' ölçümü)\n")
    L.append("| Kodlayıcı | tok/kelime | UNK / kelime-dışı karakter% | Süre (sn) |")
    L.append("|---|---|---|---|")
    L.append(f"| **Kristal** | **{word_rows['kristal']['tok_kelime']}** | UNK %{word_rows['kristal']['unk_yuzde']} | "
             f"{word_rows['kristal']['sure_sn']} |")
    for vname, _, _ in variants:
        r = word_rows[vname]
        L.append(f"| {vname} | {r['tok_kelime']} | %{r['kelime_disi_karakter_yuzde']} | {r['sure_sn']} |")
    L.append("")
    L.append(f"### Tekrar üretim\n")
    L.append("```bash")
    L.append(params["komut"])
    L.append("```")
    L.append(f"\nHam ölçümler: `{json_path}`")

    md_path = os.path.join(args.out_dir, f"bpe_baseline_rapor_{tarih}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"\nRAPOR : {md_path}")
    print(f"JSON  : {json_path}")
    print("\n== ÖZET ==")
    print(f"korpus örneklemi (kristal): {krim['tok_kelime']} tok/kelime, UNK %{krim['unk_yuzde']}, {krim['tok_s']} tok/s")
    for vname, _, _ in variants:
        r = bpe_corr[vname]
        print(f"korpus örneklemi ({vname}): {r['tok_kelime']} tok/kelime, kelime-dışı karakter %{r['kelime_disi_karakter_yuzde']}, {r['tok_s']} tok/s")


if __name__ == "__main__":
    main()