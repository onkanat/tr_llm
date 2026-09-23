import os
import sys
import time
import json
import math
import hashlib
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.tokenizer import Vocabulary
from scripts.train_step_demo import KristalDataset, KristalEmbedding, KristalLM, mask_prompt_targets
from src.llm.prompt_contract import resize_state_dict

def check_frozen_save_path(save_path: str, allow_frozen_write: bool = False) -> None:
    """Belirtilen kaydetme yolunun donmuş olup olmadığını denetler.

    Donmuş yola yazma izni (allow_frozen_write=True) açıkça verilmemişse RuntimeError fırlatır.
    """
    from src.llm.frozen_guard import is_frozen_path
    if is_frozen_path(save_path) and not allow_frozen_write:
        raise RuntimeError(f"Donmuş yola yazma engellendi: {save_path} (allow_frozen_write=False)")


def _durdur(mesaj: str) -> None:
    """SESSİZ DURMA YASAĞI (T-0092). Yanlış/eşleşmeyen optimizer durumunu yüklemek
    koşumu yarım saat sonra anlaşılmaz biçimde bozmak yerine BURADA durdurur.

    `print` + çıplak `return` sınıfı (T-0089/T-0090): süreç rc=0 ile çıkar ve çağıran
    "başarıyla koştu" ile "durdu"yu ayırt edemez. Bu yüzden mesaj stderr'e gider ve
    süreç rc=2 ile çıkar."""
    print(f"DURDURULDU: {mesaj}", file=sys.stderr, flush=True)
    sys.exit(2)


def optimizer_sidecar_path(save_path: str) -> str:
    """AdamW momentlerinin yan dosya yolu.

    YAN DOSYA seçildi (tek-sözlük formatı DEĞİL): `torch.save(model.state_dict())`
    biçimini değiştirmek mevcut TÜM yükleyicileri (chat_prompt.py, run_agent_arena.py,
    test_model.py, src/llm/prompt_contract) kırardı. Yan dosya model biçimini
    DEĞİŞTİRMEZ => sıfır yükleme kırılması."""
    return save_path + ".opt.pt"


def sha256_file(path: str) -> str:
    """Dosyanın tam sha256'sı (akışlı; 356 MiB'lik checkpoint belleğe sığdırılmaz)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


MASKE = -100
"""Kayıp maskesi — CİHAZDAN BAĞIMSIZ SÖZLEŞME (P2/Aşama 1; train_module.py:145 şablonu).

ÇAKIŞMA ÖLÇÜLDÜ: SFT maskelemesi hedefleri −100 ile işaretlerken eski kod
`ignore_index=<PAD>` (1) geçiyordu — bu, cross_entropy'nin varsayılan −100 yok
sayımını EZBERDİ ve −100 hedefler geçersiz sınıf olarak kaldı (MPS'te sessiz bozuk
kayıp). Tek çözüm TEK maske değeri: PAD hedefleri de −100'e yazılır, forward'a
yalnız `ignore_index=-100` geçer."""


def maske_pad_hedefleri(targets_np: np.ndarray, pad_id: int, pad_mask_active: bool) -> np.ndarray:
    """PAD→−100 maske deseni (train_module.py:264-265 şablonu; P2/Aşama 1).

    `pad_mask_active=True` (--no-pad-mask YOK): hedefteki PAD konumları MASKE'ye
    yazılır; forward'a `ignore_index=MASKE` geçilir. PAD'siz dizide NO-OP
    (kayıp ölçeği değişmez). `pad_mask_active=False`: PAD hedefleri eğitilir
    (eski maskeleme yok davranışı bit-bit korunur)."""
    if pad_mask_active and pad_id is not None:
        targets_np = targets_np.copy()
        targets_np[targets_np == pad_id] = MASKE
    return targets_np


def get_lr(step: int, peak_lr: float, warmup_steps: int, toplam_adim: int, min_lr: float) -> float:
    """B1 scheduler şablonu (scripts/train_step_b1_canonical.py:201-242): warmup + cosine.

    Şablonun İLERLEME TUZAĞI KAPANDI: cosine `toplam_adim` sonunda min_lr'de DURUR
    (progress 1.0'da kırpılır); aşan adımda şablonun kosinüsü geri YÜKSELİYORDU
    (progress 2,0'da peak_lr'ye döner) — kırpılmayan değer hata değildir ama
    "bitti sanılan eğride lr yeniden zirveye çıkması" sessiz sürprizdir.

    `toplam_adim <= 0`: cosine ilan edilmedi — warmup'tan sonra lr PEAK'te KALIR
    (sabit). `warmup_steps <= 0`: warmup yok, ilk adımdan peak.
    Scheduler'in kendisi yalnız `warmup_steps > 0 veya toplam_adim > 0` iken AKTİF;
    ikisi de 0 ise hiç çağrılmaz ve lr sabittir (mevcut davranış bit-bit)."""
    if toplam_adim <= 0:
        if warmup_steps <= 0 or step >= warmup_steps:
            return peak_lr
        return peak_lr * step / max(1, warmup_steps)
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, toplam_adim - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: MODEL EĞİTİM DÖNGÜSÜ")
    print("=" * 60)

    # 1. Device Selection
    device_arg = "cpu"
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--device" and arg_idx + 1 < len(sys.argv):
            device_arg = sys.argv[arg_idx + 1]
            
    if device_arg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal GPU (MPS) seçildi.", flush=True)
    elif device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU seçildi.", flush=True)
    elif device_arg not in ("cpu", ""):
        # FAIL-CLOSED CIHAZ KAPISI (T-0080). Gerekce (OLCULDU, T-0075): acikca
        # `--device mps` istenip MPS yoksa asagidaki dala DUSUYORDU ve kosum CPU'da
        # basliyordu; tek ayirt edici sinyal "CPU modu secildi" satiriydi. Sandbox
        # MPS'i GIZLEDIGI icin (T-0075) bu sessiz dusus 12 saatlik kosuyu bozar ve
        # sonuc "cihaz" ekseninde YANLIS olur. Acik istek yerine gelmiyorsa DUR.
        raise RuntimeError(
            f"DURDURULDU: --device {device_arg} istendi ama kullanilamiyor "
            f"(mps_available={torch.backends.mps.is_available()}, "
            f"cuda_available={torch.cuda.is_available()}). "
            f"Sessizce CPU'ya dusmek yerine duruldu; CPU isteniyorsa --device cpu verin.")
    else:
        device = torch.device("cpu")
        num_cores = os.cpu_count() or 8
        torch.set_num_threads(num_cores)
        print(f"CPU modu seçildi ({num_cores} iş parçacığı aktif).", flush=True)

    # TOHUM (--seed, T-0092). VARSAYILAN YOK => mevcut davranis bit-bit ayni kalir (tohum
    # verilmezse hicbir sey cagrilmaz). Gerekce (OLCULDU): bu dosyada ve veri kumesinde
    # HICBIR yerde tohum kurulmuyordu; `KristalDataset.get_batch` global `torch.randint`
    # kullaniyor => AYNI komut farkli agirliklar uretiyor ve "degisiklik davranisi
    # degistirdi mi" sorusu OLCULEMIYOR. Bu bayrak, degismezleri olculebilir kilar.
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--seed" and arg_idx + 1 < len(sys.argv):
            _tohum = int(sys.argv[arg_idx + 1])
            torch.manual_seed(_tohum)
            np.random.seed(_tohum)
            print(f"Tohum (seed) kuruldu: {_tohum}", flush=True)

    # 2. Vocabulary & Data Loading    # SOZLUK YOLU (T-0080). VARSAYILAN DEGISMEDI => mevcut davranis bit-bit ayni kalir.
    # Gerekce (OLCULDU): sozluk yolu sabit-kodlu iken kulliyat yeni kimlikler tasiyorsa
    # (Anka A1-r: id 33.113'e kadar) model embedding tablosunun DISINA indeksler. Bu
    # SESSIZ bir arizadir: (a) kosum ortasinda RuntimeError, (b) MPS gecersiz indekste
    # HATA VERMEYIP cop deger dondurur (T-0073'te olculdu). --vocab ile acikca verilir.
    vocab = Vocabulary()
    vocab_path = 'data/rebuild/vocab_base_32852.json'
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--vocab" and arg_idx + 1 < len(sys.argv):
            vocab_path = sys.argv[arg_idx + 1]
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi ({vocab_path}). Kelime dağarcığı boyutu: {vocab_size}", flush=True)

    pad_id = vocab.stoi.get("<PAD>", 1)
    pad_mask_active = "--no-pad-mask" not in sys.argv
    if pad_mask_active:
        # PAD maskeleme artık hedef düzeyinde: PAD konumları -100'e yazılır (maske_pad_hedefleri)
        # ve forward'a tek maske değeri (ignore_index=-100) geçer — PAD id'si ignore_index
        # olarak EZBERLENDİĞİNDE SFT'nin -100'leri geçersiz sınıf kalıyordu (yukarıdaki MASKE).
        print(f"<PAD> kayıp maskesi aktif (PAD hedefleri {MASKE}'e yazılır; pad_id={pad_id}).", flush=True)
    else:
        print("<PAD> kayıp maskesi devre dışı (--no-pad-mask).", flush=True)

    bin_filepath = 'data/train.bin'
    block_size = 64
    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--data", "--dataset") and arg_idx + 1 < len(sys.argv):
            bin_filepath = sys.argv[arg_idx + 1]
        if arg == "--block-size" and arg_idx + 1 < len(sys.argv):
            block_size = int(sys.argv[arg_idx + 1])

    # Auto-detect block_size from metadata if available and not explicitly provided
    meta_path = bin_filepath + '.meta.json'
    if os.path.exists(meta_path) and "--block-size" not in sys.argv:
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                if "block_size" in meta:
                    block_size = meta["block_size"]
        except Exception:
            pass

    if not os.path.exists(bin_filepath):
        print(f"Hata: {bin_filepath} bulunamadı! Lütfen önce derleme adımını çalıştırın.", flush=True)
        return

    dataset = KristalDataset(bin_filepath, block_size=block_size)
    print(f"Veri kümesi yüklendi: {bin_filepath} (Blok boyutu: {block_size}). Toplam morfem token sayısı: {len(dataset.data):,}", flush=True)

    # FAIL-CLOSED SINIR KONTROLU (T-0080). Gerekce (OLCULDU): KristalDataset.get_batch
    # dilimi dogrudan embedding'e verir ve HICBIR sinir kontrolu yapmaz. Sozluk ile
    # kulliyat ayrisirsa iki sessiz ariza mumkundur: (a) kosumun ortasinda RuntimeError,
    # (b) MPS'in gecersiz indekste HATA VERMEYIP cop deger dondurmesi (T-0073). Ikisi de
    # 12,6 saatlik kosuyu bosa cikarir. Sozlesme: max jeton id < vocab_size OLMALI.
    _jeton_max = int(dataset.data.max()) if len(dataset.data) else 0
    if _jeton_max >= vocab_size:
        raise RuntimeError(
            f"DURDURULDU: {bin_filepath} en buyuk jeton id={_jeton_max} >= vocab_size={vocab_size}. "
            f"Yuklenen sozluk ({vocab_path}) bu kulliyat icin KUCUK. --vocab ile dogru sozlugu verin.")
    print(f"[sinir] en buyuk jeton id={_jeton_max:,} < vocab_size={vocab_size:,} (guvenli)", flush=True)

    # META CROSS-CHECK. Kardes <bin>.meta.json varsa sozluk digest'i ve giris sayisi
    # YUKLENEN sozlukle birebir olmali; degilse sozluk ile kulliyat FARKLI kaynaklardan
    # gelmistir (bayat .bin / tokenizer kaymasi sinifi). Alan yoksa SESSIZCE gecilmez,
    # "dogrulanmadi" diye BASILIR; yanlis eslesme ise DURDURUR.
    _meta_dogrulandi = "yok"
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as mf:
            _meta = json.load(mf)
        _m_giris = _meta.get("sozluk_giris")
        _m_sha = _meta.get("sozluk_sha256")
        if _m_giris is None or _m_sha is None:
            _meta_dogrulandi = (f"ATLANDI: meta'da sozluk_giris/sozluk_sha256 alani yok "
                                f"(anahtarlar: {sorted(_meta.keys())})")
        else:
            if int(_m_giris) != vocab_size:
                raise RuntimeError(
                    f"DURDURULDU: meta sozluk_giris={_m_giris} != yuklenen sozluk={vocab_size}. "
                    f"Kulliyat '{_meta.get('sozluk')}' ile derlenmis; --vocab ile onu verin.")
            with open(vocab_path, 'rb') as vf:
                _v_sha = hashlib.sha256(vf.read()).hexdigest()
            if _v_sha != _m_sha:
                raise RuntimeError(
                    f"DURDURULDU: yuklenen sozluk sha256 {_v_sha} != meta sozluk_sha256 {_m_sha} "
                    f"({meta_path})")
            _meta_dogrulandi = f"BIREBIR (sozluk_sha256={_v_sha[:16]}… · giris={vocab_size})"
    print(f"[meta] sozluk/kulliyat provenance: {_meta_dogrulandi}", flush=True)

    # 3. Model & Optimizer Setup
    n_embd = 768
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab)
    
    # EMEKLI (T-0085) — emekliye ayrılan VARSAYILAN, ilan edilen kümeden AYRILMIŞTIR.
    # Burada eskiden `model_save_path = '<silinmiş zincir>.pt'` varsayılanı vardı. O
    # checkpoint zinciri 18 Eyl 2026'da operatör kararıyla SİLİNDİ; varsayılan, silinmiş
    # bir zincirin adını diriltiyordu ve `--save-path` verilmeden koşulduğunda koşum o
    # ada yazmaya çalışıyordu.
    # Varsayılanı BAŞKA BİR ADA TAŞIMAK REDDEDİLDİ: uydurma bir ad, hangi soyağacına ait
    # olduğu belirsiz bir hedef yaratırdı. Bunun yerine varsayılan KALDIRILDI; kayıt yolu
    # artık ZORUNLUDUR ve yokluğunda koşum SESLİ olarak durur (aşağıdaki kontrol).
    # Not: `model_load_path` hâlâ None ise `model_save_path`e eşitlenir (mevcut davranış).
    model_save_path = None
    model_load_path = None
    from_scratch = "--from-scratch" in sys.argv
    # Hedef modu: --pretrain ile prompt maskelemesi UYGULANMAZ, hedef duz sonraki-jetondur.
    # Gerekce (T-0073 tanisi, olculdu): mask_prompt_targets kural 5 ("<OUTPUT> icermeyen
    # pencerede TUM hedefler -100") duz-metin kulliyatta HER pencerede atesler; A1
    # kulliyatinda <OUTPUT> = 2/100.000.000 ve maske sonrasi hayatta kalan hedef 0/38.400.
    # Gradyan sifir olur; ustune MPS gecersiz hedefte hata VERMEZ, 0.0 dondurur => kosum
    # "basarili" gorunup kayip 0.0000 basar. On-egitim duz-LM hedefi kullanmalidir.
    pretrain = "--pretrain" in sys.argv

    for arg_idx, arg in enumerate(sys.argv):
        if arg in ("--save-path", "--output-model") and arg_idx + 1 < len(sys.argv):
            model_save_path = sys.argv[arg_idx + 1]
        if arg in ("--load-path", "--base-model") and arg_idx + 1 < len(sys.argv):
            model_load_path = sys.argv[arg_idx + 1]

    # FAIL-CLOSED ZORUNLU ARGUMAN KAPISI (T-0085). Gerekce: varsayilan kayit yolu
    # KALDIRILDI (yukaridaki EMEKLI notu). Sessiz bir varsayilana dusmek yerine kosum
    # BURADA durur. Bu kontrol her yazimdan ve egitim dongusunden ONCE calisir.
    if model_save_path is None:
        raise RuntimeError(
            "DURDURULDU: --save-path verilmedi. Varsayilan kayit yolu KALDIRILDI (T-0085): "
            "eski varsayilan, silinmis bir checkpoint zincirinin adini tasiyordu. "
            "Kayit yolunu ACIKCA verin, or. --save-path data/anka_a2.pt")

    if model_load_path is None:
        model_load_path = model_save_path

    allow_frozen_write = "--allow-frozen-write" in sys.argv
    check_frozen_save_path(model_save_path, allow_frozen_write=allow_frozen_write)

    # ADAMW MOMENTLERI (T-0092). OLCULEN KUSUR: `torch.save(model.state_dict())` yalniz
    # agirliklari yazar; devam kosumunda optimizer.state BOS kalir (olculdu: 0 kayit) ve
    # ilk guncelleme %73 DAHA BUYUK cikar (||delta|| 8,194721 vs 4,735192). Cozum yan dosya.
    # VARSAYILAN KAPALI: bayrak verilmezse davranis BIT-BIT ayni kalir (yan dosya olusmaz).
    save_optimizer = "--save-optimizer" in sys.argv
    load_optimizer = "--load-optimizer" in sys.argv
    opt_save_path = optimizer_sidecar_path(model_save_path)

    # DONMUS KAPI YAN DOSYA ICIN DE (T-0048 AST degismezi: kontrol torch.save'dan ONCE).
    # `<yol>.opt.pt` da `data/*.pt` donmus desenine girer; kontrol ATLANMAZ.
    if save_optimizer:
        check_frozen_save_path(opt_save_path, allow_frozen_write=allow_frozen_write)

    # PERIYODIK KAYIT (--save-every N). VARSAYILAN 0 = KAPALI => mevcut davranis bit-bit
    # ayni kalir. Gerekce: model su ana kadar YALNIZCA dongu bittikten sonra kaydediliyordu;
    # 10 saatlik bir on-egitimde bu TEK NOKTA ARIZASIDIR (kesinti/OOM tum ilerlemeyi siler).
    # Kayit ATOMIK yapilir: '<yol>.tmp' yazilir, sonra os.replace ile yerine konur => yarim
    # yazilmis bir checkpoint gecerli checkpoint'i BOZMAZ.
    save_every = 0
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--save-every" and arg_idx + 1 < len(sys.argv):
            save_every = int(sys.argv[arg_idx + 1])
    if save_every < 0:
        raise RuntimeError(f"--save-every negatif olamaz: {save_every}")

    # KAYIP RAPORU (--loss-report). VARSAYILAN KAPALI => mevcut log bit-bit ayni kalir
    # (T-0076 K3b degismezi korunur). Gerekce (T-0078/T-0079'da OLCULDU): asagidaki
    # "Bitis Kaybi" satiri loss_history[-1], yani SON ADIMIN KENDI ORNEKLEMIDIR —
    # egitimin son kayip DUZEYI degil. Tek orneklemden bant kuran bir kabul kriteri
    # (T-0078 K1) saglikli bir olcumu yanlislikla "OLU" ilan etmisti: ilan edilen bant
    # 3,2 ± 0,5 iken egitimin gercek son60 duzeyi 3,9122 ± 0,2147 cikti. Adim kaybi
    # ±0,3-0,5 nats oynar; bir kayip referansi SAYI degil DAGILIMDIR.
    loss_report = "--loss-report" in sys.argv

    lr = 1e-3
    devam_edildi = False
    if os.path.exists(model_load_path) and not from_scratch:
        print(f"Mevcut model ağırlıkları '{model_load_path}' tespit edildi, eğitim devam ettiriliyor (Resume)...", flush=True)
        state_dict = torch.load(model_load_path, map_location=device)
        keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
        for k in keys_to_skip:
            del state_dict[k]
        state_dict = resize_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        lr = 2e-4  # Lower learning rate when fine-tuning/resuming
        devam_edildi = True
    else:
        print("Sıfırdan eğitim (From Scratch) başlatılıyor...", flush=True)

    model.to(device)
    
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--lr" and arg_idx + 1 < len(sys.argv):
            lr = float(sys.argv[arg_idx + 1])

    # AŞAMA 1 (P2) — SCHEDULER/CLIP/WEIGHT-DECAY BAYRAKLARI. Optimizer'dan ÖNCE okunurlar,
    # cunku weight_decay kurulumda, scheduler ise yukleme dalindan da beslenmelidir.
    weight_decay = 0.01
    warmup_steps = 0
    toplam_adim = 0
    min_lr = 1e-6
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--weight-decay" and arg_idx + 1 < len(sys.argv):
            weight_decay = float(sys.argv[arg_idx + 1])
        if arg == "--warmup-steps" and arg_idx + 1 < len(sys.argv):
            warmup_steps = int(sys.argv[arg_idx + 1])
        if arg == "--toplam-adim" and arg_idx + 1 < len(sys.argv):
            toplam_adim = int(sys.argv[arg_idx + 1])
        if arg == "--min-lr" and arg_idx + 1 < len(sys.argv):
            min_lr = float(sys.argv[arg_idx + 1])
    if weight_decay < 0 or warmup_steps < 0 or toplam_adim < 0 or min_lr < 0:
        raise RuntimeError(
            f"--weight-decay/--warmup-steps/--toplam-adim/--min-lr negatif olamaz "
            f"(wd={weight_decay}, warmup={warmup_steps}, toplam={toplam_adim}, min_lr={min_lr})")
    scheduler_aktif = warmup_steps > 0 or toplam_adim > 0
    # VARSAYILAN KAPALI (--warmup-steps 0 ve --toplam-adim 0) => mevcut davranis bit-bit ayni
    # (lr sabit; scheduler hic uygulanmaz, logda LR alani basilmaz).

    # Using AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Model mimarisi kuruldu ve cihaza taşındı. (Öğrenme Oranı: {lr}, weight_decay: {weight_decay})", flush=True)

    # ADAMW MOMENTLERINI GERI YUKLE (T-0092). Sirasi onemli: optimizer yukarida KURULMALI,
    # cunku state_dict ancak kurulmus bir optimizer'a yuklenebilir.
    opt_load_path = optimizer_sidecar_path(model_load_path)
    if os.path.exists(opt_load_path):
        if not load_optimizer:
            # SESSIZ SURPRIZ YASAK: yan dosya VAR ama bayrak YOK. Yuklememek mesru bir
            # tercihtir, ama kullanicinin BUNDAN HABERI OLMALIDIR.
            print(
                f"UYARI: optimizer yan dosyasi BULUNDU ama --load-optimizer verilmedi: "
                f"'{opt_load_path}'. Momentler YUKLENMEDI, sifirdan basliyor.",
                file=sys.stderr, flush=True)
        else:
            payload = torch.load(opt_load_path, map_location="cpu")
            if not isinstance(payload, dict) or "optimizer" not in payload:
                _durdur(f"optimizer yan dosyasi bozuk: '{opt_load_path}' icinde 'optimizer' alani yok "
                        f"(bulunan anahtarlar: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__})")
            # ESLESME KAPISI: yan dosya, yaninda kaydedildigi MODEL dosyasinin digest'ini
            # tasir. Uyusmazsa momentler BASKA bir agirlik kumesine aittir => yapistirmak
            # sessiz bir soykutugu bozulmasidir. DUR.
            beklenen = payload.get("model_sha256")
            if not beklenen:
                _durdur(f"optimizer yan dosyasinda 'model_sha256' alani YOK: '{opt_load_path}'. "
                        f"Eslestirme dogrulanamaz => momentler yuklenmedi.")
            gercek = sha256_file(model_load_path)
            if gercek != beklenen:
                _durdur(f"optimizer/model ESLESMIYOR: yan dosya '{opt_load_path}' model_sha256="
                        f"{beklenen} bekliyor, ama '{model_load_path}' digest'i {gercek}. "
                        f"Momentler BASKA bir agirlik kumesine ait; yuklenmedi.")
            optimizer.load_state_dict(payload["optimizer"])
            print(f"AdamW momentleri geri yuklendi: '{opt_load_path}' "
                  f"(adim={payload.get('adim')}, model_sha256={gercek[:16]}…, "
                  f"{len(optimizer.state)} parametre)", flush=True)
            # SCHEDULER DURUMU (P2/Aşama 1): yan dosya, momentlerin kaydedildiği koşumun
            # scheduler ayarını da taşır. Devam koşumu bayrak vermediyse EĞRİYI SÜRDÜRÜR —
            # yoksa cosine yeniden tepeden başlar ve "carry var ama LR eğrisi sıfırlandı"
            # sessiz bozulması olur (ölçüldü: optimizer carry'siz ilk güncelleme 1,73×).
            sch = payload.get("scheduler")
            if isinstance(sch, dict):
                if "--warmup-steps" not in sys.argv:
                    warmup_steps = int(sch.get("warmup_steps", warmup_steps))
                if "--toplam-adim" not in sys.argv:
                    toplam_adim = int(sch.get("toplam_adim", toplam_adim))
                if "--min-lr" not in sys.argv:
                    min_lr = float(sch.get("min_lr", min_lr))
                scheduler_aktif = warmup_steps > 0 or toplam_adim > 0
                print(f"Scheduler durumu geri yuklendi: warmup={warmup_steps}, "
                      f"toplam_adim={toplam_adim}, min_lr={min_lr}", flush=True)
    elif load_optimizer:
        _durdur(f"--load-optimizer verildi ama yan dosya YOK: '{opt_load_path}'. "
                f"Momentler sifirdan baslardi; sessizce devam etmek yerine duruldu.")
    elif devam_edildi:
        # OLCULEN VARSAYILAN DAVRANIS: eski checkpoint'ler moment TASIMAZ (anka_a1.pt ve
        # anka_a1r.pt'de optimizer durumu YOK). Bu yuzden yokluk bir HATA degil, bir
        # EKSIKLIKTIR: kosum durdurulmaz, ama sessiz de kalinmaz.
        # Yalniz DEVAM kosumunda basilir: sifirdan kosumda moment zaten BEKLENMEZ ve
        # her yeni kosumda uyari basmak gurultu olurdu (yanlis pozitif).
        print(
            f"UYARI: AdamW momenti bulunamadi ('{opt_load_path}' yok) => optimizer SIFIRDAN "
            f"basliyor. Ilk guncelleme, momentli bir devam kosumuna gore DAHA BUYUK olur "
            f"(olculdu: 1,73x; T-0092).", file=sys.stderr, flush=True)

    # 4. Training Loop Configuration
    batch_size = 32
    max_steps = 100
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--steps" and arg_idx + 1 < len(sys.argv):
            max_steps = int(sys.argv[arg_idx + 1])
        if arg == "--batch-size" and arg_idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[arg_idx + 1])
            
    eval_interval = 10

    # AŞAMA 1 (P2) — GRAD CLIP. VARSAYILAN 1,0 (plan onaylı onarım); `--clip 0` ile KAPALI.
    clip_deger = 1.0
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--clip" and arg_idx + 1 < len(sys.argv):
            clip_deger = float(sys.argv[arg_idx + 1])
    if clip_deger < 0:
        raise RuntimeError(f"--clip negatif olamaz: {clip_deger}")

    if scheduler_aktif and toplam_adim > 0 and max_steps > toplam_adim:
        print(f"UYARI: --toplam-adim ({toplam_adim}) < --steps ({max_steps}); cosine "
              f"{toplam_adim}. adımda min_lr'de DURUR ve geri YÜKSELMEZ (progress 1.0'da "
              f"kırpılır).", file=sys.stderr, flush=True)

    print(f"\nEğitim Başlatılıyor -> Adım Sayısı: {max_steps}, Batch Boyutu: {batch_size}, Block Boyutu: {block_size}, Hedef: {'ON-EGITIM (duz sonraki-jeton)' if pretrain else 'SFT (prompt maskeli)'}", flush=True)
    print("Periyodik kayıt: " + (f"her {save_every} adımda -> '{model_save_path}' ('.tmp' üzerinden atomik)"
                                 if save_every else "KAPALI (yalnız koşum sonunda kaydedilir)"), flush=True)
    if scheduler_aktif:
        print(f"Scheduler AKTİF: warmup {warmup_steps} + cosine (toplam_adim={toplam_adim or '—'}), min_lr {min_lr}", flush=True)
    print(f"Grad clip: {clip_deger if clip_deger > 0 else 'KAPALI'}", flush=True)
    
    model.train()
    start_time = time.time()
    
    loss_history = []
    
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)

    for step in range(1, max_steps + 1):
        step_t0 = time.time()
        # Fetch a batch and move tensors to device
        x_cpu, y_cpu = dataset.get_batch(batch_size=batch_size)
        
        # Compute sign mask on CPU to avoid device roundtrips
        sign_mask_cpu = model.embedding.compute_sign_mask(x_cpu)
        
        # Hedef secimi: ON-EGITIM -> duz sonraki-jeton (maskeleme YOK); SFT -> istem maskesi.
        if pretrain:
            targets_np = np.asarray(y_cpu.detach().cpu().numpy(), dtype=np.int64).copy()
        else:
            # Apply Causal Prompt Masking for SFT sequences on CPU
            targets_np = mask_prompt_targets(x_cpu, y_cpu, output_start_id, eos_id)
            # SESSIZ NO-OP'A KARSI DUR (T-0073): hicbir hedef kalmadiysa devam etmek
            # gradyani sifirlar ve kayip 0.0000 basar. Ayni girdi CPU'da IndexError
            # verirken MPS 0.0 donduruyor; dogruluk cihaza bagli olamaz.
            if int((targets_np != -100).sum()) == 0:
                raise RuntimeError(
                    "SFT maskelemesi bu partide HICBIR hedef birakmadi (tum pencereler -100). "
                    "Duz-metin kulliyatinda on-egitim icin --pretrain kullanin. "
                    "Sessizce devam etmek gradyani sifirlar ve kaybi 0.0000 gosterir (T-0073).")

        # PAD→−100 maske deseni (P2/Aşama 1): SFT'nin -100'leriyle TEK maske değerinde
        # birleşir; ignore_index=pad_id ezberi (çakışma) bu noktada kapanır.
        targets_np = maske_pad_hedefleri(targets_np, pad_id, pad_mask_active)
        if pad_mask_active and int((targets_np != MASKE).sum()) == 0:
            raise RuntimeError(
                "PAD maskelemesi bu partide HICBIR hedef birakmadi (tum hedefler PAD/-100). "
                "Batch hizalamasi ve veri kumesini denetleyin; sessizce devam etmek "
                "gradyani sifirlar ve kaybi 0.0000 gosterir (T-0073 sinifi).")

        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = sign_mask_cpu.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets, sign_mask, ignore_index=MASKE)
        loss.backward()
        if scheduler_aktif:
            cur_lr = get_lr(step, lr, warmup_steps, toplam_adim, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr
        if clip_deger > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_deger)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        step_dur = time.time() - step_t0
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            lr_alan = f" | LR: {cur_lr:.6f}" if scheduler_aktif else ""
            print(f"Adım {step:4d}/{max_steps} | Kayıp (Loss): {loss_val:.4f} | Adım Süresi: {step_dur:.2f}s | Toplam Süre: {elapsed:.1f}s{lr_alan}", flush=True)

        # KAYIT — TEK NOKTA: egitim SONU (step == max_steps) veya periyodik esik.
        # Tek kayit noktasi bilincli: (a) T-0048 AST degismezi "check_frozen_save_path
        # torch.save'dan ONCE" aynen korunur, (b) kayit yolu TEK yerde atomik kalir.
        # ATOMIK: gecici dosyaya yaz, sonra os.replace => hedef dosya ya ESKI ya YENI tam
        # haliyle bulunur; yarim yazim gecerli checkpoint'i BOZMAZ.
        # Donmus yol korumasi bu hedef icin yukarida BIR KEZ dogrulandi; periyodik yol da
        # ayni hedefe yazar, bu yuzden kontrolu ATLAMAZ.
        if step == max_steps or (save_every and step % save_every == 0):
            gecici = model_save_path + ".tmp"
            torch.save(model.state_dict(), gecici)
            os.replace(gecici, model_save_path)
            # ADAMW MOMENTLERI (T-0092) — MODEL ONCE, YAN DOSYA SONRA. Sira BILINCLI:
            # aradaki kesinti, yan dosyayi BIR ONCEKI aralikta birakir; o zaman yan dosyanin
            # `model_sha256`'si diskteki YENI modelle UYUSMAZ => bir sonraki devam kosumu
            # ESLESME KAPISINDA DURUR. Ters sirada yazsaydik, kesinti "uyusan" gorunen ama
            # yanlis cift uretebilirdi; bu sirada her kesinti TESPIT EDILEBILIR bir uyusmazlik
            # birakir (sessiz soykutugu bozulmasi yok).
            if save_optimizer:
                opt_gecici = opt_save_path + ".tmp"
                torch.save({
                    "optimizer": optimizer.state_dict(),
                    # SCHEDULER DURUMU (P2/Aşama 1): devam koşumu aynı eğriyi sürdürsün.
                    "scheduler": {"warmup_steps": warmup_steps, "toplam_adim": toplam_adim,
                                  "min_lr": min_lr, "peak_lr": lr},
                    "adim": step,
                    "model_sha256": sha256_file(model_save_path),
                }, opt_gecici)
                os.replace(opt_gecici, opt_save_path)
            # YENI SATIR YALNIZ --save-every VERILINCE basilir => varsayilan kosumun
            # logu BIT-BIT ayni kalir (K3; eski surumle karsilastirilarak dogrulanir).
            if save_every:
                tur = "son kayıt" if step == max_steps else "periyodik"
                print(f"  [ckpt] adım {step}: '{model_save_path}' güncellendi ({tur}, atomik)", flush=True)
                if save_optimizer:
                    print(f"  [ckpt] adım {step}: '{opt_save_path}' güncellendi (AdamW momentleri, atomik)", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 50, flush=True)
    print(" EĞİTİM BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 50, flush=True)
    print(f"Toplam Süre:       {total_time:.2f} saniye", flush=True)
    print(f"Başlangıç Kaybı:   {loss_history[0]:.4f}", flush=True)
    print(f"Bitiş Kaybı:       {loss_history[-1]:.4f}", flush=True)

    # YALNIZ --loss-report VERILINCE basilir => varsayilan kosumun logu BIT-BIT ayni kalir
    # (T-0076 K3b). Yukaridaki "Bitis Kaybi" = loss_history[-1] = SON ADIMIN ORNEKLEMI;
    # dogru referans adim-kaybi dagilimidir, o da burada basilir (T-0079).
    if loss_report:
        son60 = loss_history[-60:]
        o60 = sum(son60) / len(son60)
        s60 = (sum((v - o60) ** 2 for v in son60) / len(son60)) ** 0.5
        son200 = loss_history[-200:]
        o200 = sum(son200) / len(son200)
        s200 = (sum((v - o200) ** 2 for v in son200) / len(son200)) ** 0.5
        sirali = sorted(loss_history)
        ort = len(sirali) // 2
        medyan = (sirali[ort] if len(sirali) % 2
                  else (sirali[ort - 1] + sirali[ort]) / 2)
        print(f"Kayıp Dağılımı:    son60 {o60:.4f} ± {s60:.4f} (n={len(son60)})", flush=True)
        print(f"                   son200 {o200:.4f} ± {s200:.4f} (n={len(son200)})", flush=True)
        print(f"                   medyan {medyan:.4f} (n={len(sirali)})", flush=True)
        print("                   NOT: 'Bitiş Kaybı' tek adımın örneklemidir; "
              "karşılaştırma için bu dağılımı kullanın.", flush=True)
    
    # Kayit yukarida, dongunun ICINDE tek noktadan yapildi (step == max_steps VEYA periyodik
    # esik). Burada yalnizca bitis mesaji basilir. VARSAYILAN (--save-every yok) davranis
    # aynidir: egitim sonunda model kaydedilir; tek fark artik '.tmp' uzerinden ATOMIK
    # yazilmasidir. (Kenar durum: --steps 0 => dongu hic kosmaz, model kaydedilmez; egitim
    # yapilmadigi icin bu bilincli kabul edildi.)
    print(f"Eğitilmiş model ağırlıkları '{model_save_path}' dosyasına kaydedildi.", flush=True)
    print("=" * 50, flush=True)

if __name__ == '__main__':
    main()
