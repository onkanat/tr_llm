#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A EKSENİYLE ÇAKIŞMAYAN WIKIPEDIA REPLAY DİLİMİ — ve karışımın sızıntı denetimi.

NEDEN VAR (ölçüldü, 21 Eyl 2026): `mix_r4.bin`'in replay kaynağı `data/train_chat_balanced.bin`
— içinde **Wikipedia yok**. Unutma ekseni A ise tam olarak Wikipedia'da ölçülüyor
(`evaluate_carpenter_anka.py:59 WIKI_BIN`). Yani dört modül konfigürasyonunun hiçbirinde A
ekseninin kendi alanından replay yapılmadı. `anka_r27_wikireplay_ilani_2026-09-21.md` bu tek
değişkeni sınar: replay'in ALANI.

TEK GERÇEK TEHLİKE — SIZINTI. Eğitime giren bir Wikipedia bloğu, A ekseninin 256 penceresinden
biriyle aynı olursa A CE'nin düşüşü **koruma değil ezber** olur ve H3 sahte biçimde doğrulanır.
Bu betik onu iki katmanda kapatır:

  1. ÜRETİM (`--uret`): A ekseninin 256 penceresiyle **tek jeton bile** paylaşmayan bloklar
     seçilir. Seçim ARALIK birleştirme ile yapılır (blok blok değil, koşu koşu) ⇒ hem hızlı
     hem de dosyanın tamamına yayılmış kalır.
  2. DENETİM (`--denetle <karisim.bin>`): üretilen nihai karışımın **128 hizalı her bloğu**
     hash'lenir ve A ekseninin 256 penceresinin hash kümesiyle kesişim aranır. Kesinlikle 0
     olmalı; değilse `rc=2` ile DURULUR. Bu, "seçtim" iddiasının ölçülmüş kanıtıdır.

Kullanım:
  venv/bin/python scripts/wiki_replay_disjoint.py --uret scratch/t0099_wiki_replay.bin
  venv/bin/python scripts/wiki_replay_disjoint.py --denetle scratch/t0099_wiki_mix_r4.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

WIKI_BIN = "data/anka_a1r_pretrain.bin"
B_PENCERE = 128          # `evaluate_carpenter_anka.py:81` ile AYNI olmalı
N_PENCERE = 256          # `evaluate_carpenter_anka.py:80` ile AYNI olmalı
SEED_B = 7               # `evaluate_carpenter_anka.py:84` kanonik seed
BOS_ID = 2               # kanonik `stoi["<BOS>"]` — makale sınırı (ölçüldü)

# İKİ DIŞLAMA KURALI. Hangisinin kullanıldığı dilim meta'sına YAZILIR ⇒ denetim kuralı
# yeniden türetebilir (kuralı dosyada gizli bırakmak, denetimi imkânsız kılardı).
KURAL_BLOK = "eval_blok_haric"        # yalnız 256 pencerenin BİREBİR bloğu dışlanır (r27)
KURAL_MAKALE = "eval_makale_haric"    # pencereyi İÇEREN MAKALENİN TÜM blokları dışlanır (r28)


def durdur(mesaj: str) -> None:
    """Sessiz durma yasağı: rc=0 ile çıkan bir 'durma' çağıran için 'başarı'dır (T-0089)."""
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def eval_pencereleri(wiki_bin: str) -> Tuple[List[int], int]:
    """A ekseninin kullanacağı pencere başlangıçlarını KANONİK seed ile birebir türetir."""
    mm = np.memmap(wiki_bin, dtype=np.uint16, mode="r")
    n_win = int(mm.size // B_PENCERE)
    if n_win < N_PENCERE:
        durdur(f"A ekseni: yalnız {n_win} pencere var, {N_PENCERE} gerekli")
    rng = random.Random(SEED_B)
    baslar = sorted(rng.sample(range(n_win), N_PENCERE))
    return baslar, n_win


def makale_sinirlari(wiki_bin: str) -> np.ndarray:
    """Makale sınırları: `<BOS>` konumları + dosya sonu (kesme noktaları)."""
    mm = np.memmap(wiki_bin, dtype=np.uint16, mode="r")
    bos = np.flatnonzero(mm == BOS_ID)
    if bos.size == 0:
        durdur(f"'{wiki_bin}' içinde <BOS> (id={BOS_ID}) yok — makale sınırı çıkarılamaz")
    if bos[0] != 0:
        bos = np.concatenate(([0], bos))     # ilk makale <BOS> ile başlamıyorsa kapat
    del mm
    return np.append(bos, np.memmap(wiki_bin, dtype=np.uint16, mode="r").size)


def makale_numaralari(baslar: List[int], sinir: np.ndarray) -> set:
    """Verilen pencere başlangıçlarını İÇEREN makale numaraları (pencere sınırı kesebilir)."""
    kume: set = set()
    for b in baslar:
        a, son = b * B_PENCERE, (b + 1) * B_PENCERE
        i = int(np.searchsorted(sinir, a, side="right") - 1)
        j = int(np.searchsorted(sinir, son - 1, side="right") - 1)
        kume |= set(range(max(i, 0), j + 1))
    return kume


def izinli_blok_maskesi(baslar: List[int], n_win: int, kural: str, sinir: np.ndarray) -> np.ndarray:
    """Kurala göre blok maskesi: True = replay olarak KULLANILABİLİR."""
    izinli = np.ones(n_win, dtype=bool)
    if kural == KURAL_BLOK:
        for b in baslar:
            # pencere [b*128, (b+1)*128) tek bir bloğun TAM içindedir (hizalı) ⇒ tek indeks.
            izinli[b] = False
        return izinli
    if kural == KURAL_MAKALE:
        engel = np.zeros(n_win * B_PENCERE, dtype=bool)
        for m in makale_numaralari(baslar, sinir):
            engel[sinir[m]:min(sinir[m + 1], engel.size)] = True
        return ~engel.reshape(n_win, B_PENCERE).any(axis=1)
    durdur(f"bilinmeyen kural: '{kural}' (beklenen: {KURAL_BLOK} | {KURAL_MAKALE})")


def kosular(maske: np.ndarray) -> List[Tuple[int, int]]:
    """True koşularını [bas, son) aralıkları olarak döner (deterministik, artan)."""
    if not maske.any():
        return []
    d = np.diff(maske.astype(np.int8))
    baslar = list(np.flatnonzero(d == 1) + 1)
    sonlar = list(np.flatnonzero(d == -1) + 1)
    if maske[0]:
        baslar.insert(0, 0)
    if maske[-1]:
        sonlar.append(int(maske.size))
    return list(zip(baslar, sonlar))


def uret(hedef: str, kural: str) -> int:
    if not os.path.exists(WIKI_BIN):
        durdur(f"Wikipedia dilimi YOK: {WIKI_BIN}")
    baslar, n_win = eval_pencereleri(WIKI_BIN)
    sinir = makale_sinirlari(WIKI_BIN)
    izinli = izinli_blok_maskesi(baslar, n_win, kural, sinir)
    kosu = kosular(izinli)
    n_blok = int(izinli.sum())
    if n_blok < 2044:
        durdur(f"Yeterli çakışmasız blok yok: {n_blok} < 2044 (mix_r4'ün replay ihtiyacı)")

    mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    os.makedirs(os.path.dirname(os.path.abspath(hedef)) or ".", exist_ok=True)
    out = np.memmap(hedef, dtype=np.uint16, mode="w+", shape=(n_blok * B_PENCERE,))
    yazilan = 0
    for a, b in kosu:                       # koşu koşu kopyala (blok blok değil)
        n = (b - a) * B_PENCERE
        out[yazilan:yazilan + n] = mm[a * B_PENCERE:b * B_PENCERE]
        yazilan += n
    out.flush()
    del out

    h = hashlib.sha256()
    with open(hedef, "rb") as f:
        for par in iter(lambda: f.read(1 << 22), b""):
            h.update(par)
    yan = {
        "kaynak": WIKI_BIN, "cikti": hedef,
        "kural": kural,                      # denetimin kuralı YENİDEN TÜRETMESİ için şart
        "bos_id": BOS_ID, "makale": int(sinir.size - 1),
        "eval_pencere": N_PENCERE, "blok": B_PENCERE, "seed": SEED_B,
        "kaynak_pencere": n_win, "cakismasiz_blok": n_blok,
        "atilan_blok": n_win - n_blok, "kosu_sayisi": len(kosu),
        "jeton": int(n_blok * B_PENCERE), "sha256": h.hexdigest(),
        "not": "A ekseninin 256 penceresiyle TEK JETON paylaşmayan bloklar, orijinal sırada.",
    }
    with open(hedef + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(yan, f, ensure_ascii=False, indent=2)
    print(f"[üret] {hedef}: {n_blok:,} blok ({yan['jeton']:,} jeton) · "
          f"atılan {n_win - n_blok} blok · {len(kosu)} koşu · KURAL={kural}")
    print(f"[üret] sha256={h.hexdigest()[:16]}… · yan dosya: {hedef}.meta.json", flush=True)
    return 0


def dilim_bloklari(dilim_meta: Dict[str, Any]) -> np.ndarray:
    """Dilimdeki i. bloğun WIKIPEDIA blok indeksi — kuraldan YENİDEN türetilir."""
    kural = dilim_meta.get("kural")
    if kural is None:
        durdur("dilim meta'sında `kural` yok: hangi dışlama kuralıyla üretildiği belirsiz ⇒ "
               "denetim kuralı yeniden türetemez. Dilimi geçerli betikle yeniden üretin.")
    baslar, n_win = eval_pencereleri(WIKI_BIN)
    izinli = izinli_blok_maskesi(baslar, n_win, str(kural), makale_sinirlari(WIKI_BIN))
    return np.flatnonzero(izinli)


def secim_indeksleri(ceket_jeton: int, replay_every: Any, mevcut_blok: int) -> np.ndarray:
    """`build_replay_mix.py:67,74`'ün seçim indeksleri — SAF fonksiyon (I/O yok, test edilebilir).

    `replay_every` meta'dan HAM gelir: yokluğu ya da bozukluğu SESSİZ varsayıma düşmemeli
    (sabit yazmak yanlış denetim üretir — bu tam olarak bu projenin cezalandırdığı sınıf).
    """
    if replay_every is None:
        durdur("karışım meta'sında `replay_every` yok ⇒ seçim türetilemez "
               "(sabit varsaymak yanlış denetim üretir)")
    re = int(replay_every)
    if re < 2:
        durdur(f"replay_every={re} < 2: karışım meta'sı tutarsız")
    n_j_blocks = int(ceket_jeton) // B_PENCERE
    n_r_blocks = (n_j_blocks + (re - 1) - 1) // (re - 1)
    if mevcut_blok < n_r_blocks:
        durdur(f"dilim yetersiz: gereken {n_r_blocks} blok, mevcut {mevcut_blok}")
    return np.round(np.linspace(0, mevcut_blok - 1, n_r_blocks)).astype(np.int64)


def kural_oku(karisim: str) -> str:
    """Karışımın kullandığı dilimin DIŞLAMA KURALI — metadan, varsaymadan."""
    with open(karisim + ".meta.json", encoding="utf-8") as f:
        dilim = json.load(f)["replay_source"]["path"]
    with open(dilim + ".meta.json", encoding="utf-8") as f:
        kural = json.load(f).get("kural")
    if kural not in (KURAL_BLOK, KURAL_MAKALE):
        durdur(f"dilim meta'sında geçerli `kural` yok: {kural!r}")
    return str(kural)


def secilen_bloklar(karisim: str) -> np.ndarray:
    """`build_replay_mix.py:67,74` seçimini BİREBİR yeniden üretir (kopya değil, türetme)."""
    meta_yolu = karisim + ".meta.json"
    if not os.path.exists(meta_yolu):
        durdur(f"karışım meta'sı yok: {meta_yolu}")
    with open(meta_yolu, encoding="utf-8") as f:
        km = json.load(f)
    dilim = km["replay_source"]["path"]
    if not os.path.exists(dilim + ".meta.json"):
        durdur(f"dilim meta'sı yok: {dilim}.meta.json")
    with open(dilim + ".meta.json", encoding="utf-8") as f:
        dm = json.load(f)
    bloklar = dilim_bloklari(dm)
    # `replay_every` KARIŞIM META'SINDAN okunur — sabit yazmak sessiz yanlış cevap sınıfıdır.
    idx = secim_indeksleri(int(km["jacket_source"]["total_tokens"]),
                           km.get("mix_parameters", {}).get("replay_every"),
                           len(bloklar))
    return bloklar[idx]


def blok_hashleri(yol: str, sinir: int = 0) -> set:
    """128 hizalı blokların sha1 kümesi (ilk `sinir` bayt ile sınırlanabilir)."""
    mm = np.memmap(yol, dtype=np.uint16, mode="r")
    n = int(mm.size // B_PENCERE)
    kume = set()
    for i in range(n):
        blok = np.asarray(mm[i * B_PENCERE:(i + 1) * B_PENCERE], dtype=np.uint16)
        kume.add(hashlib.sha1(blok.tobytes()).hexdigest())
    del mm
    return kume


def denetle(karisim: str) -> int:
    """Nihai karışımın hiçbir bloğu A ekseni penceresiyle AYNI olmamalı (içerik düzeyi kanıt)."""
    if not os.path.exists(karisim):
        durdur(f"karışım YOK: {karisim}")
    baslar, _ = eval_pencereleri(WIKI_BIN)
    mm = np.memmap(WIKI_BIN, dtype=np.uint16, mode="r")
    eval_h = set()
    for b in baslar:
        blok = np.asarray(mm[b * B_PENCERE:(b + 1) * B_PENCERE], dtype=np.uint16)
        eval_h.add(hashlib.sha1(blok.tobytes()).hexdigest())
    del mm
    mix_h = blok_hashleri(karisim)
    kesisim = eval_h & mix_h
    print(f"[denetim] A ekseni penceresi: {len(eval_h)} · karışım bloğu: {len(mix_h):,} · "
          f"BİREBİR BLOK KESİŞİMİ: {len(kesisim)}", flush=True)
    if kesisim:
        durdur(f"SIZINTI: {len(kesisim)} karışım bloğu A ekseni penceresiyle AYNI — "
               f"A CE düşüşü KORUMA değil EZBER olurdu. İlk 3: {list(kesisim)[:3]}")

    # --- MAKALE DÜZEYİ: replay bloğu, pencereyi İÇEREN bir makaleden mi geliyor? ---
    secilen = secilen_bloklar(karisim)
    sinir = makale_sinirlari(WIKI_BIN)
    etkilenen = makale_numaralari(baslar, sinir)
    mk = np.array([int(np.searchsorted(sinir, b * B_PENCERE, side="right") - 1) for b in secilen])
    temas = int(np.isin(mk, list(etkilenen)).sum())
    print(f"[denetim] replay bloğu: {len(secilen)} · AYNI MAKALEDEN gelen: {temas} "
          f"(%{100.0 * temas / len(secilen):.2f}) · farklı makale: {len(secilen) - temas}", flush=True)

    kural = kural_oku(karisim)
    if kural == KURAL_MAKALE and temas:
        durdur(f"MAKALE KURALI İHLALİ: kural '{KURAL_MAKALE}' ama {temas} replay bloğu "
               f"eval'i içeren makaleden geliyor")
    if kural == KURAL_BLOK:
        print(f"[denetim] kural '{KURAL_BLOK}': birebir blok sızıntısı YOK, makale teması "
              f"{temas} blok — bu ÖLÇÜLEN TEMAS'tır, ihlal değil.")
    print("[denetim] GEÇTİ.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="A ekseniyle çakışmayan Wikipedia replay dilimi")
    ap.add_argument("--uret", type=str, default=None, help="çakışmasız dilimi yaz")
    ap.add_argument("--denetle", type=str, default=None, help="nihai karışımda sızıntı denetle")
    ap.add_argument("--kural", type=str, default=KURAL_BLOK,
                    choices=[KURAL_BLOK, KURAL_MAKALE],
                    help=f"dışlama kuralı (varsayılan: {KURAL_BLOK})")
    a = ap.parse_args()
    if not a.uret and not a.denetle:
        durdur("--uret ya da --denetle verilmeli")
    if a.uret:
        uret(a.uret, a.kural)
    if a.denetle:
        denetle(a.denetle)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
