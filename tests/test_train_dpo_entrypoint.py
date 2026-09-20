"""T-0089: `train_dpo.py` yapılandırma kapıları (fail-closed).

ÖLÇÜLMÜŞ TABAN (20 Eyl 2026, değişiklikten ÖNCE): betik argümansız koşumda
"SFT referans model dosyası ... bulunamadı" deyip **rc=0** dönüyordu ⇒ çağıran boru
hattı DPO hiç yapılmamış gibi başarı sayıyordu. Ayrıca sözlük BAYAT bir dosyaya
(31.357 satır) sabit-kodluydu ve `--vocab` kabul edilmiyordu.

Ölü artefakt ADLARI burada anılmaz: kanıt (tam komut + stderr + rc tablosu)
`data/eval/anka_r12_train_dpo_kapisi_2026-09-20.md`'de KAYIT kapsamında durur.
Canlı bir test yüzeyi, silinmiş bir yolun adını yaşatmamalıdır (T-0086 aracının
yakaladığı sınıf; kendi kusurum olarak rapora yazıldı).

Bu dosya kapının YALNIZ DURMADIĞINI değil **AYIRT ETTİĞİNİ** de gösterir:
  * saf fonksiyonlar — uyuşan çift **GEÇER** (pozitif kontrol) / uyuşmayan **DURUR**
  * CLI — argümansız rc=2 · kısmi rc=2 (eksikleri **ADIYLA** sayar, verileni saymaz)
    · yolu olmayan rc=2

DİKKAT: bu testler yalnız kapıları sınar; hiçbiri eğitim koşusu başlatmaz ve
hiçbir `.pt`/`.bin` üretmez (kapı, modeller yüklenmeden ÖNCE durur).
"""
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from train_dpo import (
    KapsamHatasi,
    argv_deger,
    sozluk_checkpoint_uyumu,
    yol_dogrula,
    zorunlu_eksikler,
)

KOK = Path(__file__).resolve().parent.parent
BETIK = KOK / "train_dpo.py"


def _kosum(*args: str) -> "subprocess.CompletedProcess[str]":
    """Betiği alt süreçte koşar; kapı, ağır yükleme YAPILMADAN durduğu için hızlıdır."""
    return subprocess.run(
        [sys.executable, str(BETIK), *args],
        cwd=str(KOK),
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# Saf fonksiyonlar (pozitif + negatif kontrol aynı testte)
# ---------------------------------------------------------------------------

def test_zorunlu_eksikler_verileni_saymaz_verilmeyeni_adlandirir():
    """Eksik listesi: verilen parametre listeye GİRMEZ, verilmeyen ADIYLA girer.

    İki yönlü: yalnız "eksik var mı" değil, "DOĞRU olanı mı eksik saydı" da ölçülür.
    Yoksa her şeyi eksik sayan bir kapı da bu testi geçerdi.
    """
    eksik = zorunlu_eksikler({
        "--vocab": "data/rebuild/vocab_anka_r1_33114.json",
        "--ref-model": None,
        "--active-model": "",
    })
    assert eksik == ["--ref-model", "--active-model"], (
        f"beklenen ['--ref-model', '--active-model'], bulunan: {eksik}"
    )
    # Pozitif kontrol: hepsi verilirse HİÇBİR şey eksik sayılmamalı
    assert zorunlu_eksikler({
        "--vocab": "a", "--ref-model": "b", "--active-model": "c",
    }) == []


def test_yol_dogrula_var_olan_yolda_gecer_yoksa_durur(tmp_path):
    """Pozitif kontrol (var olan yol) + negatif kontrol (olmayan yol) birlikte."""
    var_olan = tmp_path / "var.pt"
    var_olan.write_bytes(b"x")
    yol_dogrula("--deneme", str(var_olan))  # durmamalı

    with pytest.raises(KapsamHatasi) as hata:
        yol_dogrula("--deneme", str(tmp_path / "yok.pt"))
    assert "VERILEN YOL YOK" in str(hata.value)
    assert "--deneme" in str(hata.value), "mesaj bayrağı ADIYLA anmalı"


def test_sozluk_checkpoint_uyumu_uyusani_gecirir_uyusmayani_durdurur():
    """K4'ün ayırt ediciliği: eşit çift GEÇER, farklı çift DURUR.

    Çürütme maddesi Ç1: ikisinde de dursaydı kapı VAKUM olurdu. Bu test tam olarak
    "geçen" dalı da ölçtüğü için vakumluğu dışlar.
    """
    # Pozitif kontrol: güncel külliyat çifti (33.114 ↔ 33.114)
    sozluk_checkpoint_uyumu(33114, 33114, "v.json", "m.pt")  # durmamalı

    # Negatif kontrol: ölçülmüş gerçek uyuşmazlık (31.357 ↔ 33.114)
    with pytest.raises(KapsamHatasi) as hata:
        sozluk_checkpoint_uyumu(31357, 33114, "data/vocab.json", "data/anka_a1r.pt")
    mesaj = str(hata.value)
    assert "UYUSMAZLIGI" in mesaj
    assert "1757" in mesaj, f"fark (1757) mesajda görünmeli; bulunan: {mesaj}"


def test_argv_deger_ilk_esleseni_alir_ve_varsayilana_duser(monkeypatch):
    """`argv_deger` eş-alias'ları ve varsayılanı doğru ele almalı (saf fonksiyon)."""
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--sft-model", "a.pt", "--beta", "0.1"])
    assert argv_deger(("--ref-model", "--sft-model")) == "a.pt"
    assert argv_deger(("--vocab",)) is None
    assert argv_deger(("--vocab",), "varsayilan.json") == "varsayilan.json"


# ---------------------------------------------------------------------------
# CLI kapıları (alt süreç) — rc=2 ölçülür
# ---------------------------------------------------------------------------

def test_cli_argumansiz_rc2_ve_uc_bayragi_adlandirir():
    """TABAN ÖLÇÜMÜN TERSİ: eskiden rc=0 idi; kapı artık rc=2 ile durur."""
    sonuc = _kosum()
    assert sonuc.returncode == 2, (
        f"argümansız koşum rc=2 olmalı, bulunan: {sonuc.returncode}\n"
        f"stdout={sonuc.stdout[-500:]}\nstderr={sonuc.stderr[-500:]}"
    )
    for bayrak in ("--vocab", "--ref-model", "--active-model"):
        assert bayrak in sonuc.stderr, f"{bayrak} durma mesajında adıyla geçmeli"
    # ve hiçbir model yolu yüklenmeye çalışılmamalı
    assert "Kelime dağarcığı boyutu" not in sonuc.stdout, "kapı yüklemeden ÖNCE durmalı"


def test_cli_kismi_veri_eksik_olanlari_sayar_verileni_saymaz():
    """Ayırt edicilik: yalnız `--vocab` verilince mesaj İKİ eksiği anmalı, `--vocab`'ı değil."""
    sonuc = _kosum("--vocab", "data/rebuild/vocab_anka_r1_33114.json")
    assert sonuc.returncode == 2, f"kısmi koşum rc=2 olmalı, bulunan: {sonuc.returncode}"
    satir = [s for s in sonuc.stderr.splitlines() if "DURDURULDU" in s]
    assert satir, f"stderr'de DURDURULDU yok: {sonuc.stderr[-400:]}"
    assert "--ref-model" in satir[0] and "--active-model" in satir[0]
    assert "--vocab" not in satir[0].split("ZORUNLUDUR")[0], (
        "verilmiş bir parametre 'eksik' diye sayılmamalı"
    )


def test_cli_yolu_olmayan_girdi_rc2():
    """Bayraklar verilip yollar YOKSA da durur (kapı çağrı anında)."""
    sonuc = _kosum(
        "--vocab", "data/rebuild/yok_boyle_bir_sozluk.json",
        "--ref-model", "data/yok_boyle_bir_model.pt",
        "--active-model", "data/yok_boyle_bir_model.pt",
    )
    assert sonuc.returncode == 2, f"olmayan yolda rc=2 beklenirdi, bulunan: {sonuc.returncode}"
    assert "VERILEN YOL YOK" in sonuc.stderr, f"stderr={sonuc.stderr[-400:]}"
