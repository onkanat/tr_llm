#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import os
import json
import pytest
from src.llm.tokenizer import Vocabulary, KristalTokenizer
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

# --- Provenance çıpası (T-0080, 19 Eyl 2026 ölçümü) -------------------------
# BİLİNÇLİ OLARAK sabit-kodlu: roots.tsv sessizce değişemesin. Faz 1 yeni bir
# dosya üretecek (data/lexicon/roots_anka_r1.tsv) ve roots.tsv YERİNDE
# DEĞİŞMEYECEK; sözleşme değişirse bu sabitler BEYAN EDİLEREK güncellenir.
LEXICON_SHA256 = "fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598"
LEXICON_SATIR = 52373
LEXICON_BENZERSIZ_LEMMA = 48200
LEXICON_KUYRUK_BASI = 50475   # sırasız `NOUN\t-` kuyruk bloğunun ilk satırı

# Yer-gerçek oracle: ROOT görevi (25 öğe). 19 Eyl 2026'da 24/25 doğru.
# BİLİNEN başarısızlık: `tatlısı` -> beklenen `tat`, gelen `Tat`. Büyük-harf
# homografı (`Tat`, bir etnonim) küçük-harf kökü (`tat`) gölgeliyor; trie
# düğümündeki girdi SIRASI belirliyor. Yeni başarısızlık = REGRESYON.
ROOT_ORACLE_BILINEN_BASARISIZ = {"tatlısı": ("tat", "Tat")}

ROOT_GOREVI = 'Kelimedeki kök morfemini bul.'


def _lexicon_satirlari():
    with open('data/lexicon/roots.tsv', 'r', encoding='utf-8') as f:
        return [ln for ln in f.read().split('\n')[1:] if ln.strip()]


def _fixture():
    json_path = os.path.join(os.path.dirname(__file__), 'morphology_regression_100.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _derleyici():
    lex = LexiconManager()
    lex.load_from_tsv('data/lexicon/roots.tsv')
    return CrystalCompiler(lex, build_default_graph())


def test_lexicon_provenance():
    """roots.tsv sessizce değişemez: digest + satır/lemma sayısı + şema + kuyruk."""
    with open('data/lexicon/roots.tsv', 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == LEXICON_SHA256, (
        f'roots.tsv digesti değişti: {sha}. Sözleşme bilinçli değiştiyse '
        f'LEXICON_SHA256 BEYAN EDİLEREK güncellenir (sessizce gevşetilmez).')

    satirlar = _lexicon_satirlari()
    assert len(satirlar) == LEXICON_SATIR, f'satır {len(satirlar)} != {LEXICON_SATIR}'
    lemma = [ln.split('\t')[0] for ln in satirlar]
    assert len(set(lemma)) == LEXICON_BENZERSIZ_LEMMA
    for i, ln in enumerate(satirlar, start=2):
        assert len(ln.split('\t')) == 3, f'satır {i} şemaya aykırı: {ln[:60]!r}'

    # Sırasız kuyruk bloğu BURADA başlar: sessiz bir "sıralama/temizlik" görünür olsun.
    kirilma = next(i for i in range(1, len(lemma)) if lemma[i] < lemma[i - 1]) + 2
    assert kirilma == LEXICON_KUYRUK_BASI, (
        f'sırasız kuyruk {kirilma}. satırda başlıyor, beklenen {LEXICON_KUYRUK_BASI}')


def test_root_oracle_no_regression():
    """ROOT görevi gerçek oracle: yeni başarısızlık = REGRESYON."""
    comp = _derleyici()
    root_items = [x for x in _fixture() if x['instruction'] == ROOT_GOREVI]
    assert len(root_items) == 25

    basarisiz = {}
    for x in root_items:
        bek = x['output'].split('ROOT:')[1].strip()
        an = comp.compile(x['input']).get('analyses') or []
        gel = an[0]['morphemes'][0]['id'] if an else None
        if gel != bek:
            basarisiz[x['input']] = (bek, gel)

    yeni = {k: v for k, v in basarisiz.items() if k not in ROOT_ORACLE_BILINEN_BASARISIZ}
    assert not yeni, f'ROOT oracle REGRESYONU: {yeni}'
    assert len(root_items) - len(basarisiz) >= 24, (
        f'ROOT oracle {len(root_items) - len(basarisiz)}/25 — 24 tabanının altında')


def test_morphology_regression_integrity():
    data = _fixture()
    assert len(data) == 100, f'Beklenen 100 örnek, bulunan: {len(data)}'

    tasks = set()
    for item in data:
        assert 'instruction' in item and item['instruction']
        assert 'input' in item and item['input']
        assert 'output' in item and item['output']
        tasks.add(item['instruction'])

    assert len(tasks) == 4, f'4 temel görev bekleniyor, bulunan: {tasks}'


def test_morphology_regression_tokenization():
    """ESKİ assert `len(enc) >= 2` ASLA DÜŞEMEZDİ: <BOS>+<EOS> zaten 2 eder,
    yani her token <UNK> olsa da geçerdi.

    Yerine SADAKAT özelliği sınanır: derleyicinin seçtiği kök, sözlükte VARSA
    kodlamada da bulunmak zorundadır — karar sessizce kaybolamaz.

    DİKKAT: `enc_out` üzerinde "hiç <UNK> olmasın" İDDİA EDİLMEZ. `data/vocab.json`
    (31.357) noktalama bloğunu (32137-32145) taşımıyor; fixture çıktısındaki `:`
    bu yüzden <UNK> olur. Bu bilinen bir bayatlıktır, bu testin konusu değildir."""
    vocab = Vocabulary()
    vocab.load('data/vocab.json')
    comp = _derleyici()
    tok = KristalTokenizer(comp, vocab)
    unk = vocab.stoi['<UNK>']
    bos, eos = vocab.stoi['<BOS>'], vocab.stoi['<EOS>']

    root_items = [x for x in _fixture() if x['instruction'] == ROOT_GOREVI]
    assert len(root_items) == 25

    sadakatsiz = []
    for item in root_items:
        an = comp.compile(item['input']).get('analyses') or []
        assert an, f"{item['input']!r} hiç çözülemedi"
        kok = an[0]['morphemes'][0]['id']

        enc_in = tok.encode(item['input'])
        enc_out = tok.encode(item['output'])
        assert len(enc_in) >= 3 and len(enc_out) >= 3
        assert enc_in[0] == bos and enc_in[-1] == eos
        assert enc_out[0] == bos and enc_out[-1] == eos

        kid = vocab.stoi.get(kok)
        if kid is not None and kid != unk and kid not in enc_in:
            sadakatsiz.append((item['input'], kok, kid, enc_in))
        # ROOT girdileri çözülebilir kelimelerdir: kök <UNK>'a düşerse görünür olsun
        if kid is None and (unk not in enc_in
                            and vocab.stoi['<PROPER_NOUN>'] not in enc_in):
            sadakatsiz.append((item['input'], kok, None, enc_in))

    assert not sadakatsiz, f'kök kodlamaya taşınmadı: {sadakatsiz}'

