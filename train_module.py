"""YETENEK MODÜLÜ EĞİTİMİ — taban model DONUK, yalnız modül öğrenir.

NEDEN AYRI BİR EĞİTİCİ: `train.py` optimizer'ı her zaman `model.parameters()` üzerine kurar
(ölçüldü: `train.py:287`); orada dondurma yolu YOKTUR. Bu betik onun panzehiri değil,
tamamlayıcısıdır: tabanı hiç değiştirmez, yalnız `src/llm/modules.py` modülünü eğitir.

ÖLÇÜLEN GEREKÇE (T-0094/T-0096, `data/eval/anka_r17_ceket_giydirme_2026-09-20.md`): bugüne
kadar her yetenek eğitimi TAM AĞIRLIK ince ayardı ve dört kolun (R0/R4/R10/R20) HEPSİ
unutma kapısını düşürdü — Wikipedia CE **+%24,52…+%35,80**, ilan edilen eşik ≤+%10. R18
ilanı (`data/eval/anka_r18_ceket_ilani_2026-09-21.md:256`) denenmemiş tek kaldıracı adıyla
yazar: **katman dondurma**. Bu betik onu uygular: tabanın 93,4 M parametresi donuk, eğitilen
modül **1.327.104** (%1,40).

DÖRT SESSİZ ARIZA KAPISI (hepsi fail-closed, hepsi ölçülmüş sınıftan):
  1. `--base` yok/eksik            -> DUR (modül hangi tabana ait olacak belirsiz)
  2. hedef deseni hiçbir katmana uymaz -> DUR (`modul_ekle`; yoksa 0 katman "eğitilir")
  3. eğitilebilir sayı != formül   -> DUR (mekanizma ile niyet ayrışmış)
  4. `--module` == `--base`        -> DUR (modül kaydı tabanı EZERDİ)
Ayrıca koşum sonunda taban dosyasının digest'i BAŞTA alınanla karşılaştırılır ve eşit
değilse DURULUR: "tabana dokunmadım" iddiası ölçülmeyen bir iddia olamaz.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.train_step_demo import KristalDataset, KristalLM, mask_prompt_targets
from src.llm.modules import (
    VARSAYILAN_HEDEFLER,
    ModulSpec,
    beklenen_parametre_sayisi,
    egitilebilir_parametreler,
    modul_ekle,
    modul_katmanlari,
    modul_kaydet,
    modul_yukle,
    sha256_dosya,
    tabani_dondur,
)
from src.llm.prompt_contract import resize_state_dict
from src.llm.tokenizer import Vocabulary


def _durdur(mesaj: str) -> None:
    """Sessiz durma yasağı: rc=0 ile çıkan bir 'durma', çağıran için 'başarı'dır (T-0089)."""
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def _arg(adlar: Tuple[str, ...], varsayilan: Optional[str] = None) -> Optional[str]:
    for i, a in enumerate(sys.argv):
        if a in adlar and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return varsayilan


def _bayrak(ad: str) -> bool:
    return ad in sys.argv


def _device_sec() -> torch.device:
    istenen = _arg(("--device",), "cpu")
    if istenen == "mps" and torch.backends.mps.is_available():
        print("Metal GPU (MPS) seçildi.", flush=True)
        return torch.device("mps")
    if istenen == "cuda" and torch.cuda.is_available():
        print("CUDA GPU seçildi.", flush=True)
        return torch.device("cuda")
    if istenen not in ("cpu", "", None):
        # Açıkça istenen cihaz yoksa SESSİZCE CPU'ya düşmek, 12 saatlik koşuyu "cihaz"
        # ekseninde yanlış yapar (T-0075/T-0080). DUR.
        _durdur(
            f"--device {istenen} istendi ama kullanılamıyor "
            f"(mps={torch.backends.mps.is_available()}, cuda={torch.cuda.is_available()}). "
            f"CPU isteniyorsa --device cpu verin."
        )
    n = os.cpu_count() or 8
    torch.set_num_threads(n)
    print(f"CPU modu seçildi ({n} iş parçacığı).", flush=True)
    return torch.device("cpu")


def main() -> None:
    print("=" * 64)
    print(" ANKA · YETENEK MODÜLÜ EĞİTİMİ (taban donuk, modül öğrenir)")
    print("=" * 64)

    base_yol = _arg(("--base",))
    module_yol = _arg(("--module",))
    if not base_yol:
        _durdur("--base verilmedi: modülün hangi tabana ait olacağı belirsiz. "
                "Ör. --base data/anka_a1r.pt")
    if not module_yol:
        _durdur("--module verilmedi: modül nereye kaydedilecek? Ör. --module modules/marangoz.mod.pt")
    if not os.path.exists(base_yol):
        _durdur(f"--base dosyası yok: '{base_yol}'")
    if os.path.abspath(base_yol) == os.path.abspath(module_yol):
        _durdur(f"--module ile --base AYNI dosya: '{base_yol}'. Modül kaydı tabanı EZERDİ.")

    # Modül yolu donmuş bir desene düşerse (ör. yanlışlıkla data/*.pt) kapı yine sorar.
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(module_yol) and not _bayrak("--allow-frozen-write"):
        _durdur(f"'{module_yol}' donmuş desene uyuyor; modüller 'modules/' altına yazılmalıdır.")

    device = _device_sec()

    if _bayrak("--seed"):
        tohum = int(_arg(("--seed",), "0") or 0)
        torch.manual_seed(tohum)
        np.random.seed(tohum)
        print(f"Tohum kuruldu: {tohum}", flush=True)

    # --------------------------- sözlük ---------------------------
    vocab = Vocabulary()
    vocab_yol = _arg(("--vocab",), "data/rebuild/vocab_anka_r1_33114.json")
    vocab.load(str(vocab_yol))
    vocab_size = len(vocab.stoi)
    print(f"Sözlük: {vocab_yol} ({vocab_size} giriş)", flush=True)

    # KAYIP MASKESİ — CİHAZDAN BAĞIMSIZ SÖZLEŞME (ölçüldü, bu oturum).
    # `mask_prompt_targets` maskelenen yerlere **-100** yazar. `train.py` ise CE'ye
    # `ignore_index=<PAD id>` (=1) verir; -100 hedefi ile bu çakışır:
    #   · CPU  -> `IndexError: Target -100 is out of bounds`  (1. adımda çöker; ölçüldü)
    #   · MPS  -> HATA VERMEZ, koşum sürer (T-0073 sınıfı: geçersiz hedefte sessiz kalma)
    # T-0096/T-0097'nin SFT koşuları MPS'te bu yüzden "çalıştı" (seg_1.log: 3,1864 -> 0,9659).
    # Burada cihazdan bağımsız doğru yol kurulur: PAD hedefleri de -100'e çevrilir ve CE
    # TEK bir ignore_index (-100) alır => hem CPU hem MPS'te aynı, tanımlı davranış.
    # REJİM KIYASI AÇIK KALDI (ölçüldü, bu oturum): bu betik PAD+istem maskesi aktifken
    # ilk kaybı **6,6154** veriyor; T-0096'nın kendi logu (`scratch/t0096_kos/seg_1.log`)
    # ise **3,1864 -> 0,9659** (son60 1,0256 ± 0,2404) veriyor. İkisi de "PAD maskesi aktif"
    # olmakla birlikte KAYIP ÖLÇEĞİ AYNI DEĞİL => T-0096 ile kıyas **kurulmuş sayılmaz**.
    # Muhtemel neden MPS'in -100 hedefini sessizce kabul etmesi (yani T-0096'da maskenin
    # gerçekte nasıl davrandığı ölçülmemiştir). Ölçülmeden "aynı rejim" denemez.
    pad_id: Optional[int] = None if _bayrak("--no-pad-mask") else vocab.stoi.get("<PAD>", 1)
    MASKE = -100

    # --------------------------- veri ---------------------------
    bin_yol = _arg(("--data", "--dataset"), "data/train.bin")
    block_size = int(_arg(("--block-size",), "0") or 0)
    if block_size <= 0:
        # BLOK BOYUTU SESSİZCE VARSAYILANA DÜŞMEZ (ölçüldü, 21 Eyl 2026): her üretici
        # meta'yı aynı şemayla yazmıyor — `mix_r4.bin.meta.json` block_size'ı
        # `mix_parameters` İÇİNE koyuyor, üst düzeye değil. Eski okuma
        # `.get("block_size", 64)` ile sessizce **64**'e düşüyordu ⇒ 128'lik külliyat
        # 64'lük pencereyle eğiliyordu ve kayıt da 64 yazdığı için uyumsuzluk
        # görünmüyordu; yalnızca "iki koşum kıyaslanamaz" hâline geliyordu (sessiz).
        meta_yol = str(bin_yol) + ".meta.json"
        if not os.path.exists(meta_yol):
            block_size = 64  # meta YOK ⇒ eski davranış korunur (açık bayrak önerilir)
        else:
            import json as _json
            with open(meta_yol, "r", encoding="utf-8") as f:
                _m = _json.load(f)
            _bulunan = _m.get("block_size")
            if _bulunan is None:
                _bulunan = (_m.get("mix_parameters") or {}).get("block_size")
            if _bulunan is None:
                _durdur(f"'{meta_yol}' var ama `block_size` bilinen hiçbir yerde yok "
                        f"(üst düzey ya da mix_parameters). Sessiz varsayılana düşmek "
                        f"yerine DURUYORUM — `--block-size` ile açıkça verin.")
            block_size = int(_bulunan)
    if not os.path.exists(str(bin_yol)):
        _durdur(f"veri dosyası yok: '{bin_yol}'")

    dataset = KristalDataset(str(bin_yol), block_size=block_size)
    jeton_max = int(dataset.data.max()) if len(dataset.data) else 0
    if jeton_max >= vocab_size:
        _durdur(f"'{bin_yol}' en büyük jeton id={jeton_max} >= vocab_size={vocab_size}: "
                f"sözlük bu külliyat için KÜÇÜK (--vocab ile doğrusunu verin).")
    print(f"Veri: {bin_yol} ({len(dataset.data):,} jeton, blok {block_size}, "
          f"max id {jeton_max:,} < {vocab_size:,})", flush=True)

    # --------------------------- taban model ---------------------------
    model = KristalLM(vocab_size=vocab_size, n_embd=768, vocab=vocab)
    durum = torch.load(base_yol, map_location="cpu")
    for k in [k for k in list(durum.keys())
              if "cos_cached" in k or "sin_cached" in k or "mask" in k]:
        del durum[k]
    model.load_state_dict(resize_state_dict(model, durum), strict=False)
    print(f"Taban yüklendi: {base_yol} (sha256={sha256_dosya(base_yol)[:16]}…)", flush=True)

    # --------------------------- modül ---------------------------
    hedefler = tuple((_arg(("--hedefler",)) or ",".join(VARSAYILAN_HEDEFLER)).split(","))
    spec = ModulSpec(
        ad=str(_arg(("--ad",), os.path.basename(str(module_yol)).split(".")[0])),
        r=int(_arg(("--r",), "16") or 16),
        alpha=int(_arg(("--alpha",), "32") or 32),
        dropout=float(_arg(("--dropout",), "0.0") or 0.0),
        hedefler=hedefler,
    )
    devam_modulu = _arg(("--load-module",))
    if devam_modulu:
        meta = modul_yukle(model, str(devam_modulu), base_yol)
        print(f"Modül devam ettiriliyor: {devam_modulu} (adım={meta.get('adim')}, "
              f"spec={meta['spec']['ad']}/r{meta['spec']['r']})", flush=True)
        spec = ModulSpec.sozlukten(meta["spec"])
    else:
        takilan = modul_ekle(model, spec)
        print(f"Modül takıldı: '{spec.ad}' r={spec.r} alpha={spec.alpha} "
              f"({len(takilan)} katman: {takilan[0]} … {takilan[-1]})", flush=True)

    # --------------------------- dondurma + kapı ---------------------------
    egitilebilir = tabani_dondur(model)
    beklenen = beklenen_parametre_sayisi(model)
    toplam = sum(p.numel() for p in model.parameters())
    if egitilebilir != beklenen:
        _durdur(f"eğitilebilir parametre {egitilebilir:,} != formül {beklenen:,}: "
                f"mekanizma ile niyet ayrıştı.")
    katman_sayisi = len(modul_katmanlari(model))
    print(f"Taban DONUK: {toplam - egitilebilir:,} parametre sabit | "
          f"eğitilen: {egitilebilir:,} ({100.0 * egitilebilir / toplam:.3f}%) over "
          f"{katman_sayisi} katman", flush=True)

    model.to(device)
    lr = float(_arg(("--lr",), "1e-3") or 1e-3)
    # Ağırlık çürümesi YOK: LoRA deltası taban ağırlığı değildir; onu sıfıra çekmek
    # "yetenek eğitimi"ni sessizce kısar.
    optimizer = optim.AdamW(egitilebilir_parametreler(model), lr=lr, weight_decay=0.0)
    print(f"Optimizer: AdamW(lr={lr}, wd=0.0), yalnız modül parametreleri", flush=True)

    batch_size = int(_arg(("--batch-size",), "8") or 8)
    max_steps = int(_arg(("--steps",), "200") or 200)
    save_every = int(_arg(("--save-every",), "0") or 0)
    pretrain = _bayrak("--pretrain")
    print(f"Eğitim -> adım {max_steps}, batch {batch_size}, hedef: "
          f"{'ÖN-EĞİTİM (düz sonraki-jeton)' if pretrain else 'SFT (istem maskeli)'} | "
          f"maske: -100 (PAD id={pad_id} "
          f"{'-> -100 çevrilir' if pad_id is not None else 'maskelenmiyor'})", flush=True)

    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    kayip_gecmisi: List[float] = []
    taban_once = sha256_dosya(base_yol)
    # `.cpu()` İNŞA GEREĞİ: bu satır bir kez MPS koşumunda çöktü (ölçüldü, 21 Eyl 2026) —
    # özet MPS'te klonlanınca `torch.equal` "Cannot compare two tensors on different
    # devices: mps:0 and cpu" ile koşumun EN SONUNDA patladı, yani 500 adımlık koşumun
    # bütünlük denetimi HİÇ ÇALIŞMADI. CPU koşumu temiz geçmişti ⇒ kusur yalnız gerçek
    # cihazda görünür (sandbox MPS'i gizlediği için test yüzeyim de göremezdi).
    taban_ozet = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if ".lora_A" not in k and ".lora_B" not in k}

    model.train()
    t0 = time.time()
    for step in range(1, max_steps + 1):
        adim_t0 = time.time()
        x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
        if pretrain:
            targets_np = np.asarray(y_cpu.detach().cpu().numpy(), dtype=np.int64).copy()
        else:
            targets_np = mask_prompt_targets(x_cpu, y_cpu, output_start_id, eos_id)
            if int((targets_np != MASKE).sum()) == 0:
                _durdur("SFT maskelemesi bu partide HİÇBİR hedef bırakmadı; düz metin "
                        "külliyatında --pretrain kullanın (T-0073 sessiz sıfır kayıp).")
        if pad_id is not None:
            targets_np[targets_np == pad_id] = MASKE

        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = model.embedding.compute_sign_mask(x_cpu).to(device)

        optimizer.zero_grad()
        _, loss = model(x, targets, sign_mask, ignore_index=MASKE)
        loss.backward()
        optimizer.step()

        kayip_gecmisi.append(float(loss.item()))
        if step % 25 == 0 or step == 1:
            print(f"  adım {step:5d}/{max_steps} | kayıp {kayip_gecmisi[-1]:.4f} | "
                  f"{time.time() - adim_t0:.2f}s/adım | toplam {time.time() - t0:.0f}s", flush=True)
        if save_every and step % save_every == 0:
            modul_kaydet(model, spec, str(module_yol), base_yol,
                         ek={"adim": step, "kismi": True, "veri": os.path.basename(str(bin_yol))})

    # --------------------------- kayıt + doğrulama ---------------------------
    son60 = kayip_gecmisi[-60:]
    ort = sum(son60) / len(son60)
    std = (sum((v - ort) ** 2 for v in son60) / len(son60)) ** 0.5
    meta = modul_kaydet(model, spec, str(module_yol), base_yol, ek={
        "adim": max_steps,
        "veri": os.path.basename(str(bin_yol)),
        "vocab": os.path.basename(str(vocab_yol)),
        # SÖZLÜK PARMAK İZİ ZORUNLU BEYAN. Taban `.pt` kendi sözlüğünü TAŞIMAZ (train.py
        # çıplak state_dict kaydeder), bu yüzden "bu modül hangi sözlükle eğitildi"
        # sorusunun tek kanıtı budur. Aynı UZUNLUKTA ama farklı SIRADA bir sözlük sessiz
        # jeton kayması üretir; boyut kontrolü bunu yakalayamaz (T-0077 sınıfı).
        "vocab_sha256": sha256_dosya(str(vocab_yol)),
        "lr": lr,
        "batch_size": batch_size,
        "block_size": block_size,
        "kayip_ilk": kayip_gecmisi[0],
        "kayip_son60_ort": ort,
        "kayip_son60_std": std,
        "sure_sn": round(time.time() - t0, 1),
    })

    taban_sonra = sha256_dosya(base_yol)
    # İki taraf da AÇIKÇA cpu(): karşılaştırma cihazdan bağımsız olsun (yukarıdaki ölçüm).
    degisen = [k for k, v in taban_ozet.items()
               if not torch.equal(v.cpu(), model.state_dict()[k].detach().cpu())]
    print("\n" + "=" * 64)
    print(f"Modül kaydedildi: {module_yol}")
    print(f"  boyut            : {os.path.getsize(str(module_yol)) / 1e6:.1f} MB "
          f"({meta['modul_parametre']:,} parametre)")
    print(f"  kayıp            : ilk {kayip_gecmisi[0]:.4f} -> son60 {ort:.4f} ± {std:.4f}")
    print(f"  taban dosyası    : {taban_once[:16]}… -> {taban_sonra[:16]}… "
          f"({'DEĞİŞMEDİ' if taban_once == taban_sonra else 'DEĞİŞTİ'})")
    print(f"  taban tensörleri : {len(degisen)} tanesi değişti "
          f"({'BEKLENEN: 0' if not degisen else 'KUSUR: ' + str(degisen[:3])})")
    print("=" * 64)

    if taban_once != taban_sonra or degisen:
        _durdur("Taban model DEĞİŞTİ: modül eğitimi tabana dokunmamalıydı.")


if __name__ == "__main__":
    main()
