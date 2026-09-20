#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T-0090: `chat_prompt.py` yapılandırma kapıları (fail-closed).

ÖLÇÜLMÜŞ TABAN (20 Eyl 2026, değişiklikten ÖNCE): üç ayrı dal — sözlük yolu VERİLİP
dosya yoksa, sözlük (lexicon) dosyası yoksa, model yolu VERİLİP dosya yoksa — `print` +
çıplak `return` ile bitiyordu ⇒ `main()` normal dönüyor ve betik **rc=0** ile çıkıyordu.
Çağıran, "dosya yok" ile "başarıyla koştu"yu AYIRT EDEMİYORDU. Üç dalın üçü de rc=0
olarak ölçüldü; aynı sınıf `train_dpo.py`'de T-0089'da kapatılmıştı.

Ölü artefakt ADLARI burada anılmaz: kanıt (tam komutlar, stdout/stderr ayrımı ve rc
tablosu) `data/eval/anka_r13_onaysiz_kucuk_aciklar_2026-09-20.md`'de KAYIT kapsamında
durur. Canlı bir test yüzeyi, silinmiş bir yolun adını yaşatmamalıdır.

Bu dosya kapının YALNIZ DURMADIĞINI değil **AYIRT ETTİĞİNİ** de gösterir:
  * DURAN dallar — üçü de `rc=2`, durma mesajı **stderr**'de, stdout'ta DEĞİL.
  * GEÇEN dal (pozitif kontrol) — yollar VARSA kapı ateşlenmez ve betik kapıların
    ALTINDAKİ satıra ulaşır. Yalnız "duruyor mu" bakan bir test, her şeyi durduran bir
    kapıyı da geçirirdi (VAKUM KAPI).

İDDİA SINIRI (beyan — testteki pozitif kap ile ilan edilen kapı arasındaki fark):
otomatik testteki pozitif kontrol UCUZ kaptır: `--model` VAR OLAN ama checkpoint
OLMAYAN bir dosyaya işaret eder ⇒ üç kapı da geçilir, ardından `torch.load` sesli hata
verir ve süreç KENDİ KENDİNE biter (asılmaz). Gerçek checkpoint'li koşum (356 MB) da
ölçüldü ve interaktif istem satırına ulaştı; otomatik teste ALINMADI çünkü stdin EOF'ta
REPL sonsuz döngüye giriyor — bu, bu görevin kapsamı dışında **ayrı bir açık maddedir**
ve rapora yazılmıştır.
"""
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
BETIK = KOK / "chat_prompt.py"
GUNCEL_VOCAB = "data/rebuild/vocab_anka_r1_33114.json"
LEXICON = "data/lexicon/roots.tsv"


def _kosum(*args: str, cwd: Path = KOK) -> "subprocess.CompletedProcess[str]":
    """Betiği alt süreçte koşar.

    Kapılar AĞIR YÜKLEMEDEN ÖNCE durur; yalnız K2 dalı sözlüğü ve lexicon'u yükler
    (yine de model YÜKLENMEZ). `stdin=DEVNULL`: interaktif döngüye girilirse EOF alır.
    """
    return subprocess.run(
        [sys.executable, str(BETIK), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=300,
    )


def _kapı_durdu(res: "subprocess.CompletedProcess[str]") -> None:
    """Duran dalın ORTAK sözleşmesi: rc=2 · mesaj stderr'de · stdout'ta DEĞİL."""
    assert res.returncode == 2, (
        f"kapı rc=2 ile durmalıydı, rc={res.returncode}\n"
        f"--- stdout ---\n{res.stdout[-800:]}\n--- stderr ---\n{res.stderr[-800:]}"
    )
    assert "DURDURULDU" in res.stderr, f"durma mesajı stderr'de yok:\n{res.stderr[-800:]}"
    assert "DURDURULDU" not in res.stdout, (
        "durma mesajı stdout'a da basılmış: bu bir UYARI değil DURMA'dır ve boru "
        f"hattında stdout sohbet akışıdır.\n{res.stdout[-800:]}"
    )


# ---------------------------------------------------------------------------
# DURAN dallar (3/3) — hepsi eskiden rc=0 dönüyordu
# ---------------------------------------------------------------------------

def test_kapi1_sozluk_yolu_verilip_dosya_yoksa_rc2():
    """`--vocab` VERİLİR ama dosya yok — eski davranış rc=0 (ölçüldü)."""
    res = _kosum("--vocab", "YOK_boyle_bir_sozluk.json")
    _kapı_durdu(res)
    assert "YOK_boyle_bir_sozluk.json" in res.stderr, "mesaj verilen yolu ADIYLA anmalı"


def test_kapi2_lexicon_yoksa_rc2(tmp_path):
    """Lexicon yolu GÖRELİdir ⇒ başka bir cwd'de çözülmez ve kapı ateşlenir."""
    res = _kosum("--vocab", str(KOK / GUNCEL_VOCAB), cwd=tmp_path)
    _kapı_durdu(res)
    assert LEXICON in res.stderr, f"mesaj eksik yolu anmalı: {res.stderr[-400:]}"


def test_kapi3_model_yolu_verilip_dosya_yoksa_rc2():
    """`--model` VERİLİR ama dosya yok — eski davranış rc=0 (ölçüldü).

    Bu dal sözlüğü ve lexicon'u GERÇEKTEN yükler (kapı onlardan SONRA gelir) ⇒
    kapıların yükleme yapmadan durduğu iddia EDİLMEZ.
    """
    res = _kosum("--vocab", GUNCEL_VOCAB, "--model", "YOK_boyle_bir_model.pt")
    _kapı_durdu(res)
    assert "YOK_boyle_bir_model.pt" in res.stderr, "mesaj verilen yolu ADIYLA anmalı"


# ---------------------------------------------------------------------------
# GEÇEN dal (pozitif kontrol) — kapı AYIRT ediyor mu, yoksa her şeyi durduruyor mu?
# ---------------------------------------------------------------------------

def test_pozitif_dal_kapilar_ateslenmez():
    """Yollar VARSA kapı ateşlenmez: betik kapıların ALTINDAKİ satıra ulaşır.

    `--model` var olan ama checkpoint OLMAYAN bir dosyaya işaret eder ⇒ kapı geçilir,
    sonra `torch.load` sesli hata verir (rc=1) ve süreç kendi kendine biter.

    Ölçüt "rc=0" DEĞİLDİR (o, tüm programın başarısı olurdu); ölçüt, durma kapısının
    bu dalda ATEŞLENMEMESİ ve betiğin model yükleme satırına ULAŞMASIDIR.
    """
    res = _kosum("--vocab", GUNCEL_VOCAB, "--model", LEXICON)
    assert "DURDURULDU" not in res.stderr, (
        f"pozitif dalda kapı ateşlendi ⇒ kapı AŞIRI GENİŞ (her şeyi durduruyor):\n"
        f"{res.stderr[-800:]}"
    )
    assert res.returncode != 2, "pozitif dalda rc=2 döndü ⇒ kapı ateşlenmiş"
    assert "Model Ağırlıkları Yükleniyor" in res.stdout, (
        "betik model yükleme satırına ULAŞMADI ⇒ kapılar aşılamıyor:\n"
        f"--- stdout ---\n{res.stdout[-800:]}\n--- stderr ---\n{res.stderr[-800:]}"
    )
