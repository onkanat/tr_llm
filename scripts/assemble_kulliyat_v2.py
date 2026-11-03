"""
Külliyat v2 birleşmesi + kapsam ölçümü (Faz-3 T3.3).

ADIM 1 — birleşme:
  Metin kaynakları:
    - data/raw_corpus.txt                 -> her satır (boş satırlar hariç)
    - data/kulliyat_v2/gts_definitions.txt -> yalnız ANLAM sütunu (ilk sekme sonrası, olduğu gibi)
  Satır-bazında tam yinelenenler tek set ile atılır (kaynaklar arası dahil).
  Tokenizasyon: KristalTokenizer.encode (frozen anka vocab; encode satır başına
  <BOS> ... <EOS> sınırları koyar) -> data/kulliyat_v2/kulliyat_v2_text.bin (uint16).
  Birleşme: kulliyat_v2_text.bin ++ data/train_wiki.bin (uint16, parça-parça kopya)
    -> data/kulliyat_v2/kulliyat_v2.bin  (dosya boyutu = toplam token x 2)
  İstatistik -> data/kulliyat_v2/kulliyat_v2_stats.json (toplam + kaynak başına token).

ADIM 2 — kapsam ölçümü (--coverage-only):
  data/kulliyat_v2/kulliyat_v2.bin üzerinde tek geçiş: her vocab ID'nin görülme
  sayısı (np.bincount). 33.114 ID'nin yüzde kaçı en az 1 kez görünüyor,
  0 kez görülen ID listesi -> data/kulliyat_v2/zero_gradient_ids.json
  özet -> data/kulliyat_v2/kulliyat_v2_kapsam.json

Koşum (cwd = repo kökü; göreli data/... yolları):
  venv/bin/python scripts/assemble_kulliyat_v2.py                # birleşme
  venv/bin/python scripts/assemble_kulliyat_v2.py --coverage-only # kapsam

Not: RESUME desteklidir (text bin ilerleme dosyası); koşum kesilirse aynı
komutla kaldığı satırdan devam eder. HİÇBİR commit yapılmaz.
"""

import os
import sys
import json
import time
import shutil
import argparse

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

# --- Sabitler ---
RAW_PATH = 'data/raw_corpus.txt'
GTS_PATH = 'data/kulliyat_v2/gts_definitions.txt'
WIKI_BIN = 'data/train_wiki.bin'
OUT_DIR = 'data/kulliyat_v2'
TEXT_BIN = os.path.join(OUT_DIR, 'kulliyat_v2_text.bin')
FINAL_BIN = os.path.join(OUT_DIR, 'kulliyat_v2.bin')
STATS_JSON = os.path.join(OUT_DIR, 'kulliyat_v2_stats.json')
ZERO_IDS_JSON = os.path.join(OUT_DIR, 'zero_gradient_ids.json')
KAPSAM_JSON = os.path.join(OUT_DIR, 'kulliyat_v2_kapsam.json')
VOCAB_PATH = 'data/rebuild/vocab_anka_r1_33114.json'  # dalga-1 sabiti — dokunma
FLUSH_TOKENS = 1_000_000
UINT16_MAX = 65535
COPY_CHUNK = 8 * 1024 * 1024


def atomic_write_json(path: str, obj: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_tokenizer():
    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)  # freeze=True -> encode yeni token eklemez (uint16 güvenli)
    tokenizer = KristalTokenizer(compiler, vocab)
    return tokenizer, vocab


def load_text_sources():
    """(kaynak, satır) akışı: raw her satır + gts yalnız ANLAM sütunu."""
    stream = []

    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        raw_lines = f.read().splitlines()
    for line in raw_lines:
        stream.append(('raw', line))

    gts_read = 0
    gts_no_tab = 0
    with open(GTS_PATH, 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            gts_read += 1
            if '\t' not in line:
                gts_no_tab += 1
                continue
            stream.append(('gts', line.split('\t', 1)[1]))  # anlam sütunu, olduğu gibi

    return stream, len(raw_lines), gts_read, gts_no_tab


def assemble(args):
    t_start = time.time()
    print("=" * 60)
    print(" KÜLLİYAT V2 BİRLEŞMESİ (raw + GTS -> text bin ++ wiki bin)")
    print("=" * 60)

    tokenizer, vocab = build_tokenizer()
    vocab_size = len(vocab.stoi)
    print(f"Sözlük yüklendi: {vocab_size} token (frozen).")

    stream, raw_read, gts_read, gts_no_tab = load_text_sources()
    print(f"Kaynak satırları: raw={raw_read:,} · gts={gts_read:,} "
          f"(anlam sütunu çıkarılan: {gts_read - gts_no_tab:,}; sekmesiz atlanan: {gts_no_tab})")

    os.makedirs(OUT_DIR, exist_ok=True)
    progress_path = TEXT_BIN + '.progress.json'

    # --- RESUME durumu ---
    lines_done = 0
    tokens_flushed = 0
    resume = False
    if os.path.exists(progress_path) and os.path.exists(TEXT_BIN):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                prog = json.load(f)
            if prog.get('version') == 1:
                lines_done = int(prog.get('lines_done', 0))
                tokens_flushed = int(prog.get('tokens_flushed', 0))
                resume = True
                print(f"DEVAM: text bin içinde {tokens_flushed:,} token zaten yazılı; "
                      f"satır {lines_done:,}'den devam.")
        except (ValueError, KeyError) as e:
            print(f"İlerleme dosyası okunamadı ({e}); sıfırdan başlanıyor.")

    if resume:
        bin_file = open(TEXT_BIN, 'r+b')
        bin_file.truncate(tokens_flushed * 2)
        bin_file.seek(0, os.SEEK_END)
    else:
        bin_file = open(TEXT_BIN, 'wb')
        if os.path.exists(progress_path):
            os.remove(progress_path)

    buf = []

    def flush():
        nonlocal tokens_flushed
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.int64)
        if arr.size and int(arr.max()) > UINT16_MAX:
            raise RuntimeError(f"Token ID uint16 sınırını aşıyor: {int(arr.max())} "
                                f"(sözlük {vocab_size} — frozen olması gerekir)")
        arr.astype(np.uint16).tofile(bin_file)
        tokens_flushed += len(buf)
        atomic_write_json(progress_path, {
            'version': 1,
            'lines_done': lines_done,
            'tokens_flushed': tokens_flushed,
            'out': TEXT_BIN,
        })
        buf.clear()

    # --- Sayaçlar (resume'da replay ile tutarlı kurulur) ---
    stats = {
        'raw': {'read': raw_read, 'kept': 0, 'dup': 0, 'blank': 0, 'tokens': 0},
        'gts': {'read': gts_read, 'no_tab': gts_no_tab, 'kept': 0, 'dup': 0,
                'blank': 0, 'tokens': 0},
    }
    seen = set()
    t_enc = time.time()
    enc_tokens = 0

    for i, (src, line) in enumerate(stream):
        if i < lines_done:
            # resume replay: yalnız seen-set ve sayaçlar (kodlama yok)
            if not line.strip():
                stats[src]['blank'] += 1
            elif line in seen:
                stats[src]['dup'] += 1
            else:
                seen.add(line)
                stats[src]['kept'] += 1
            continue

        lines_done = i + 1
        if not line.strip():
            stats[src]['blank'] += 1
            continue
        if line in seen:
            stats[src]['dup'] += 1
            continue
        seen.add(line)
        ids = tokenizer.encode(line)  # <BOS> ... <EOS> sınırları dahil
        buf.extend(ids)
        stats[src]['kept'] += 1
        stats[src]['tokens'] += len(ids)
        enc_tokens += len(ids)
        if len(buf) >= FLUSH_TOKENS:
            flush()
            dt = time.time() - t_enc
            rate = enc_tokens / dt if dt > 0 else 0.0
            print(f"  satır {i + 1:,}/{len(stream):,} · text token {tokens_flushed + len(buf):,} "
                  f"· {rate / 1000:.1f}K tok/s", flush=True)

    flush()
    bin_file.close()
    dt_enc = time.time() - t_enc
    print(f"\nMetin fazı bitti: {enc_tokens:,} yeni token kodlandı "
          f"({dt_enc / 60:.1f} dk; {len(stream):,} satır taralandı).")
    print(f"  raw: kept={stats['raw']['kept']:,} dup={stats['raw']['dup']:,} "
          f"blank={stats['raw']['blank']:,} tokens={stats['raw']['tokens']:,}")
    print(f"  gts: kept={stats['gts']['kept']:,} dup={stats['gts']['dup']:,} "
          f"blank={stats['gts']['blank']:,} tokens={stats['gts']['tokens']:,}")

    os.remove(progress_path)

    # --- Wiki ile birleşme (text ++ wiki -> final) ---
    if not os.path.exists(WIKI_BIN):
        print(f"Hata: {WIKI_BIN} yok — önce prepare_wiki_dataset.py çalıştır.")
        sys.exit(1)
    text_tokens = os.path.getsize(TEXT_BIN) // 2
    wiki_tokens = os.path.getsize(WIKI_BIN) // 2
    total_tokens = text_tokens + wiki_tokens
    print(f"\nBirleşme: text={text_tokens:,} + wiki={wiki_tokens:,} "
          f"= {total_tokens:,} token")

    tmp_final = FINAL_BIN + '.tmp'
    with open(tmp_final, 'wb') as out:
        for path in (TEXT_BIN, WIKI_BIN):
            with open(path, 'rb') as src:
                shutil.copyfileobj(src, out, COPY_CHUNK)
    os.replace(tmp_final, FINAL_BIN)

    final_size = os.path.getsize(FINAL_BIN)
    if final_size != total_tokens * 2:
        raise RuntimeError(f"Final boyut uyumsuz: {final_size} bayt != {total_tokens * 2}")
    print(f"Yazıldı: {FINAL_BIN} ({final_size:,} bayt = {total_tokens:,} token)")

    stats_json = {
        'gorev': 'T3.3_kulliyat_v2_birlesme',
        'kaynaklar': {
            'raw_corpus_txt': {'yol': RAW_PATH,
                               'bayt': os.path.getsize(RAW_PATH),
                               **stats['raw']},
            'gts_definitions_txt': {'yol': GTS_PATH,
                                    'bayt': os.path.getsize(GTS_PATH),
                                    **stats['gts']},
            'train_wiki_bin': {'yol': WIKI_BIN,
                               'bayt': os.path.getsize(WIKI_BIN),
                               'token': wiki_tokens},
        },
        'text_token': text_tokens,
        'wiki_token': wiki_tokens,
        'toplam_token': total_tokens,
        'cikti': {'yol': FINAL_BIN, 'bayt': final_size,
                  'duzen': 'text ++ wiki (uint16, boyut = token x 2)'},
        'vocab': VOCAB_PATH,
        'vocab_boyut': vocab_size,
        'sure_sn': round(time.time() - t_start, 1),
    }
    atomic_write_json(STATS_JSON, stats_json)
    print(f"İstatistik: {STATS_JSON}")
    print("=" * 60)
    print(f" TOPLAM: {total_tokens:,} token ({total_tokens / 1e6:.2f}M)")
    print("=" * 60)


def coverage(args):
    t_start = time.time()
    print("=" * 60)
    print(" KÜLLİYAT V2 KAPSAM ÖLÇÜMÜ (tek geçiş, np.bincount)")
    print("=" * 60)

    if not os.path.exists(FINAL_BIN):
        print(f"Hata: {FINAL_BIN} yok — önce birleşme adımını çalıştır.")
        sys.exit(1)

    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)
    vocab_size = len(vocab.stoi)

    total_tokens = os.path.getsize(FINAL_BIN) // 2
    arr = np.fromfile(FINAL_BIN, dtype=np.uint16)
    assert arr.size == total_tokens, "bayt/token uyumsuzluğu"
    print(f"Bin: {FINAL_BIN} · {total_tokens:,} token")

    counts = np.bincount(arr, minlength=vocab_size)
    if len(counts) > vocab_size:
        raise RuntimeError(f"Vocab dışı ID görüldü (max ID {len(counts) - 1} >= {vocab_size})")

    seen_ids = np.nonzero(counts)[0]
    zero_ids = np.nonzero(counts == 0)[0]
    coverage_pct = 100.0 * len(seen_ids) / vocab_size

    atomic_write_json(ZERO_IDS_JSON, {
        'aciklama': 'kulliyat_v2.bin içinde hiç (0 kez) görünmeyen vocab ID listesi',
        'vocab_boyut': vocab_size,
        'toplam_token': int(total_tokens),
        'sifir_id_sayisi': int(len(zero_ids)),
        'sifir_idler': zero_ids.tolist(),
    })

    kapsam = {
        'gorev': 'T3.3_kulliyat_v2_kapsam',
        'bin': FINAL_BIN,
        'toplam_token': int(total_tokens),
        'vocab_boyut': vocab_size,
        'gorulen_id': int(len(seen_ids)),
        'sifir_id': int(len(zero_ids)),
        'kapsam_yuzde': round(coverage_pct, 4),
        'esik_yuzde': 50.0,
        'esik_saglandi': bool(coverage_pct >= 50.0),
        'zero_ids_json': ZERO_IDS_JSON,
        'sure_sn': round(time.time() - t_start, 1),
    }
    atomic_write_json(KAPSAM_JSON, kapsam)

    print(f"Vocab: {vocab_size} ID · görülen: {len(seen_ids):,} · 0 kez: {len(zero_ids):,}")
    print(f"KAPSAM: {coverage_pct:.2f}% (eşik ≥50% -> "
          f"{'GEÇTİ' if coverage_pct >= 50.0 else 'GEÇİLEMEDİ'})")
    print(f"Yazıldı: {ZERO_IDS_JSON} · {KAPSAM_JSON}")


def main():
    ap = argparse.ArgumentParser(description='Külliyat v2 birleşmesi + kapsam ölçümü (T3.3)')
    ap.add_argument('--coverage-only', action='store_true',
                    help='birleşme atla; mevcut kulliyat_v2.bin üzerinde kapsam ölç')
    args = ap.parse_args()
    if args.coverage_only:
        coverage(args)
    else:
        assemble(args)


if __name__ == '__main__':
    main()