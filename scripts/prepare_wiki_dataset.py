"""
Wikipedia (open-wikipedia-markdown, TR) parçalarını çok-parça okuyup
Kristal tokenizasyonuyla data/train_wiki.bin üretir.

Dalga-3 değişikliği (T3.2):
  - Eski sürüm tek dosyayı (tr-00000.parquet) pd.read_parquet ile tümüyle
    belleğe yükleyip ilk 5.000 makaleyi örnekliyordu.
  - Yeni sürüm dizindeki TÜM tr-*.parquet parçalarını sıralı okur
    (pd.concat'e denk düşen birleşik akış) ancak 8 GB RAM güvenliği için
    hiçbir parçayı bütün olarak belleğe almaz: pyarrow iter_batches ile
    satır-öbekli akıtır ve token'ları artımlı olarak bin dosyasına yazar.
  - Kalabilir/uzatılabilir koşum için flush sınırında atomik ilerleme
    dosyası yazar; kesilen koşum kaldığı makaleden devam eder.

Koşum: venv/bin/python scripts/prepare_wiki_dataset.py [--budget N] [--max-articles N]
  --budget       toplam token bütçesi (varsayılan: sınırsız — tüm külliyat!)
  --max-articles işlenecek toplam makale üst sınırı (örneklem modu)
  --fresh        önceki ilerlemeyi yok sayıp sıfırdan başla
Not: cwd repo kökü olmalı (göreli data/... yolları).
"""

import os
import sys
import json
import re
import glob
import time
import argparse

import numpy as np
import pyarrow.parquet as pq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

# --- Sabitler ---
DEFAULT_WIKI_DIR = '/Users/hakankilicaslan/Prompts/Outputs/datasets/open-wikipedia-markdown/tr'
DEFAULT_OUTPUT_BIN = 'data/train_wiki.bin'
VOCAB_PATH = 'data/rebuild/vocab_anka_r1_33114.json'  # dalga-1 sabiti — dokunma
FLUSH_TOKENS = 1_000_000     # bin dosyasına yazma aralığı (RAM tamponu üst sınırı)
BATCH_ROWS = 200             # parquet satır-öbek boyutu
LOG_EVERY_ARTICLES = 500
UINT16_MAX = 65535
GATE_TOKENS = 15_000_000     # plan eşiği: wiki payı >= 15M token


def clean_markdown(text: str) -> str:
    """Removes basic markdown links and headers for cleaner tokenization."""
    if not text:
        return ""
    # Remove markdown links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove bold/italic markup: **text** or *text* -> text
    text = re.sub(r'\*\*([^*]+)\*\*|\*([^*]+)\*', r'\1\2', text)
    # Remove headers: # Header -> Header
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def find_shards(wiki_dir: str):
    return sorted(glob.glob(os.path.join(wiki_dir, 'tr-*.parquet')))


def atomic_write_json(path: str, obj: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def fmt_n(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def main():
    ap = argparse.ArgumentParser(description='TR Wikipedia -> Kristal morfem akışı (çok-parça, artımlı)')
    ap.add_argument('--dir', default=DEFAULT_WIKI_DIR, help='tr-*.parquet dizini')
    ap.add_argument('--out', default=DEFAULT_OUTPUT_BIN, help='çıktı bin dosyası (uint16 token akışı)')
    ap.add_argument('--budget', type=int, default=None, help="toplam token bütçesi (bin'deki toplam token)")
    ap.add_argument('--max-articles', type=int, default=None, help='toplam makale üst sınırı')
    ap.add_argument('--fresh', action='store_true', help='ilerleme dosyasını yok say, sıfırdan başla')
    args = ap.parse_args()

    t_start = time.time()
    print("=" * 60)
    print(" WIKIPEDIA MARKDOWN VERİ SETİ OKUMA VE KRİSTAL TOKENİZASYONU")
    print(" (çok-parça tr-*.parquet · artımlı yazım · kalabilir koşum)")
    print("=" * 60)

    # 1. Derleyici & tokenizer (eski sürümle birebir aynı)
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)

    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)  # freeze=True -> encode yeni token eklemez (uint16 güvenli)
    print(f"Sözlük yüklendi: {len(vocab.stoi)} token.")

    tokenizer = KristalTokenizer(compiler, vocab)

    # 2. Parça listesi
    shards = find_shards(args.dir)
    if not shards:
        print(f"Hata: '{args.dir}' altında tr-*.parquet bulunamadı!")
        sys.exit(1)

    shard_names = [os.path.basename(s) for s in shards]
    shard_rows = {}
    print(f"\nParça dizini: {args.dir}")
    total_rows_all = 0
    total_mb = 0.0
    for s, name in zip(shards, shard_names):
        pf = pq.ParquetFile(s)
        n = pf.metadata.num_rows
        mb = os.path.getsize(s) / 1e6
        shard_rows[name] = n
        total_rows_all += n
        total_mb += mb
        print(f"  {name}: {n:,} satır, {mb:.1f} MB")
    print(f"TOPLAM: {len(shards)} parça, {total_rows_all:,} makale, {total_mb:.1f} MB")

    # şema teyidi
    schema_names = pq.ParquetFile(shards[0]).schema_arrow.names
    for col in ('title', 'markdown'):
        if col not in schema_names:
            print(f"Hata: parquet şemasında '{col}' sütunu yok! Şema: {schema_names}")
            sys.exit(1)
    print(f"Şema: {schema_names}")

    # 3. Çıktı & ilerleme (kalabilir koşum)
    out_path = args.out
    progress_path = out_path + '.progress.json'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    tokens_flushed = 0
    rows_done = {}   # parça adı -> o parçada tamamlanan makale sayısı
    resume = False
    if (not args.fresh) and os.path.exists(progress_path) and os.path.exists(out_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                prog = json.load(f)
            if prog.get('version') == 1 and prog.get('shards') == shard_names:
                tokens_flushed = int(prog.get('tokens_flushed', 0))
                rows_done = {k: int(v) for k, v in prog.get('rows_done', {}).items()}
                resume = True
                print(f"\nDEVAM: '{out_path}' içinde {tokens_flushed:,} token zaten yazılı; "
                      f"kaldığı makaleden devam ediliyor.")
        except (ValueError, KeyError) as e:
            print(f"İlerleme dosyası okunamadı ({e}); sıfırdan başlanıyor.")

    if resume:
        bin_file = open(out_path, 'r+b')
        bin_file.truncate(tokens_flushed * 2)  # son flush sınırına geri al
        bin_file.seek(0, os.SEEK_END)          # yeni token'lar SONA eklensin
    else:
        bin_file = open(out_path, 'wb')
        if os.path.exists(progress_path):
            os.remove(progress_path)

    buf = []  # henüz diske yazılmayan token tamponu

    state = {
        'version': 1,
        'shards': shard_names,
        'tokens_flushed': tokens_flushed,
        'rows_done': rows_done,
        'out': out_path,
    }

    def flush():
        """Tamponu uint16 olarak diske yaz + atomik ilerleme güncelle."""
        nonlocal tokens_flushed
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.int64)
        if arr.size and int(arr.max()) > UINT16_MAX:
            raise RuntimeError(f"Token ID uint16 sınırını aşıyor: {int(arr.max())} "
                               f"(sözlük {len(vocab.stoi)} — frozen olması gerekir)")
        arr.astype(np.uint16).tofile(bin_file)
        tokens_flushed += len(buf)
        state['tokens_flushed'] = tokens_flushed
        state['rows_done'] = rows_done
        atomic_write_json(progress_path, state)
        buf.clear()

    # 4. Sıralı çok-parça işleme (pd.concat'e denk akış, artımlı)
    budget = args.budget
    max_articles = args.max_articles
    articles_done = sum(rows_done.values())  # önceki oturumlar dahil kümülatif
    stop_reason = None
    t_enc = time.time()
    tokens_at_start = tokens_flushed

    print(f"\nTokenizasyon başlıyor... (bütçe: {'sınırsız' if budget is None else f'{budget:,} token'})")
    for shard_path, shard_name in zip(shards, shard_names):
        if stop_reason:
            break
        skip = rows_done.get(shard_name, 0)
        n_rows = shard_rows[shard_name]
        if skip >= n_rows:
            print(f"\n--- {shard_name}: {skip:,}/{n_rows:,} makale zaten işlenmiş, atlanıyor.")
            continue
        print(f"\n--- Parça: {shard_name} ({n_rows:,} makale, {'devam: ' + format(skip, ',') if skip else 'baştan'})")
        pf = pq.ParquetFile(shard_path)
        shard_articles = skip
        skip_remaining = skip
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=['title', 'markdown']):
            rows = batch.to_pylist()
            if skip_remaining:
                if skip_remaining >= len(rows):
                    skip_remaining -= len(rows)
                    continue
                rows = rows[skip_remaining:]
                skip_remaining = 0
            for row in rows:
                title = (row.get('title') or '').strip()
                markdown_text = row.get('markdown') or ''
                article_text = f"{title}\n{clean_markdown(markdown_text)}"

                segments = [seg.strip() for seg in re.split(r'\n\s*\n+', article_text) if seg.strip()]
                for segment in segments:
                    if len(segment) < 20:
                        continue
                    token_ids = tokenizer.encode(segment)
                    if len(token_ids) > 2:
                        buf.extend(token_ids)

                shard_articles += 1
                articles_done += 1
                rows_done[shard_name] = shard_articles

                if len(buf) >= FLUSH_TOKENS:
                    flush()

                if articles_done % LOG_EVERY_ARTICLES == 0:
                    total_tokens = tokens_flushed + len(buf)
                    dt = time.time() - t_enc
                    rate = (total_tokens - tokens_at_start) / dt if dt > 0 else 0.0
                    line = (f"  [{time.strftime('%H:%M:%S')}] {shard_name} makale {shard_articles:,}/{n_rows:,}"
                            f" | kümülatif {articles_done:,} | token {fmt_n(total_tokens)}")
                    if rate > 0:
                        line += f" | {fmt_n(rate)} tok/s"
                        if budget is not None:
                            rem = max(budget - total_tokens, 0)
                            line += f" | kalan {fmt_n(rem)} (~{rem/rate/60:.0f} dk)"
                    print(line, flush=True)

                if budget is not None and (tokens_flushed + len(buf)) >= budget:
                    stop_reason = f"token bütçesine ulaşıldı ({fmt_n(budget)})"
                    break
                if max_articles is not None and articles_done >= max_articles:
                    stop_reason = f"makale üst sınırına ulaşıldı ({max_articles:,})"
                    break
            if stop_reason:
                break
        if not stop_reason:
            rows_done[shard_name] = n_rows
            print(f"  -> {shard_name} tamamlandı ({n_rows:,} makale).")
        flush()

    bin_file.close()

    total_tokens = tokens_flushed
    elapsed = time.time() - t_start
    enc_elapsed = time.time() - t_enc
    bin_bytes = os.path.getsize(out_path)
    rate = (total_tokens - tokens_at_start) / enc_elapsed if enc_elapsed > 0 else 0.0

    # 5. Sözlük kaydı (eski davranışın birebir korunması — içerik değişmez)
    vocab.save(VOCAB_PATH)

    # 6. Özet
    print("\n" + "=" * 60)
    print(" WİKIPEDİA VERİ HAZIRLIĞI TAMAMLANDI — ÖZET")
    print("=" * 60)
    if stop_reason:
        print(f" Duraklama nedeni   : {stop_reason}")
    print(f" Parça sayısı (disk): {len(shards)} ({total_mb:.1f} MB)")
    print(f" İşlenen makale     : {articles_done:,} / {total_rows_all:,}")
    for name in shard_names:
        print(f"   {name}: {rows_done.get(name, 0):,}/{shard_rows[name]:,} makale")
    print(f" Toplam token       : {total_tokens:,}")
    print(f" Bin dosyası        : {out_path} ({bin_bytes:,} bayt = {bin_bytes/1e6:.1f} MB)")
    assert bin_bytes == total_tokens * 2, "bin boyutu token sayısıyla uyuşmuyor!"
    print(f" Bütçe              : {'sınırsız' if budget is None else f'{budget:,}'}")
    print(f" Hız                : {fmt_n(rate)} tok/s (koşum {elapsed/60:.1f} dk)")
    if articles_done > 0:
        print(f" Ortalama token/makale: {total_tokens/articles_done:.0f}")
    gate_ok = total_tokens >= GATE_TOKENS
    print(f" KAPI (>= {GATE_TOKENS:,} token) : {'GEÇTİ ✓' if gate_ok else 'GEÇİLMEDİ ✗'}")

    # 7. Örnek tokenizasyon çıktısı: bin başından ilk 3 <EOS> sınırlı bölüm
    eos_id = vocab.stoi.get('<EOS>')
    if eos_id is not None and total_tokens > 0:
        with open(out_path, 'rb') as f:
            head = np.fromfile(f, dtype=np.uint16, count=min(total_tokens, 200_000))
        eos_pos = np.flatnonzero(head == eos_id)
        shown = 0
        start = 0
        print("\n ÖRNEK TOKENİZASYON ÇIKTISI (bin başından ilk 3 bölüm):")
        for p in eos_pos:
            seg = head[start:p + 1]
            start = p + 1
            if len(seg) <= 3:
                continue
            morphs = [vocab.decode(int(i)) for i in seg[:60]]
            print(f"\n  [ÖRNEK {shown + 1}] token sayısı: {len(seg)}")
            print(f"    id'ler (ilk 20) : {seg[:20].tolist()}")
            print(f"    morfem dizisi  : {' '.join(morphs[:50])}{' …' if len(morphs) > 50 else ''}")
            shown += 1
            if shown >= 3:
                break

    print("\n" + "=" * 60)
    print(" WIKIPEDIA VERİ HAZIRLIĞI TAMAMLANDI")
    print("=" * 60)


if __name__ == '__main__':
    main()