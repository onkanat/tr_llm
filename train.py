import os
import sys
import time
import json
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

    # 2. Vocabulary & Data Loading
    # SOZLUK YOLU (T-0080). VARSAYILAN DEGISMEDI => mevcut davranis bit-bit ayni kalir.
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

    pad_ignore_index = None if "--no-pad-mask" in sys.argv else vocab.stoi.get("<PAD>", 1)
    if pad_ignore_index is not None:
        print(f"<PAD> kayıp maskesi aktif (ignore_index={pad_ignore_index}).", flush=True)
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
    
    model_save_path = 'data/kristal_model.pt'
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

    if model_load_path is None:
        model_load_path = model_save_path

    allow_frozen_write = "--allow-frozen-write" in sys.argv
    check_frozen_save_path(model_save_path, allow_frozen_write=allow_frozen_write)

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
    if os.path.exists(model_load_path) and not from_scratch:
        print(f"Mevcut model ağırlıkları '{model_load_path}' tespit edildi, eğitim devam ettiriliyor (Resume)...", flush=True)
        state_dict = torch.load(model_load_path, map_location=device)
        keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
        for k in keys_to_skip:
            del state_dict[k]
        state_dict = resize_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        lr = 2e-4  # Lower learning rate when fine-tuning/resuming
    else:
        print("Sıfırdan eğitim (From Scratch) başlatılıyor...", flush=True)

    model.to(device)
    
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--lr" and arg_idx + 1 < len(sys.argv):
            lr = float(sys.argv[arg_idx + 1])

    # Using AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"Model mimarisi kuruldu ve cihaza taşındı. (Öğrenme Oranı: {lr})", flush=True)

    # 4. Training Loop Configuration
    batch_size = 32
    max_steps = 100
    for arg_idx, arg in enumerate(sys.argv):
        if arg == "--steps" and arg_idx + 1 < len(sys.argv):
            max_steps = int(sys.argv[arg_idx + 1])
        if arg == "--batch-size" and arg_idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[arg_idx + 1])
            
    eval_interval = 10
    
    print(f"\nEğitim Başlatılıyor -> Adım Sayısı: {max_steps}, Batch Boyutu: {batch_size}, Block Boyutu: {block_size}, Hedef: {'ON-EGITIM (duz sonraki-jeton)' if pretrain else 'SFT (prompt maskeli)'}", flush=True)
    print("Periyodik kayıt: " + (f"her {save_every} adımda -> '{model_save_path}' ('.tmp' üzerinden atomik)"
                                 if save_every else "KAPALI (yalnız koşum sonunda kaydedilir)"), flush=True)
    
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
                        
        x = x_cpu.to(device)
        targets = torch.from_numpy(targets_np).to(device)
        sign_mask = sign_mask_cpu.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, targets, sign_mask, ignore_index=pad_ignore_index)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        step_dur = time.time() - step_t0
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"Adım {step:4d}/{max_steps} | Kayıp (Loss): {loss_val:.4f} | Adım Süresi: {step_dur:.2f}s | Toplam Süre: {elapsed:.1f}s", flush=True)

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
            # YENI SATIR YALNIZ --save-every VERILINCE basilir => varsayilan kosumun
            # logu BIT-BIT ayni kalir (K3; eski surumle karsilastirilarak dogrulanir).
            if save_every:
                tur = "son kayıt" if step == max_steps else "periyodik"
                print(f"  [ckpt] adım {step}: '{model_save_path}' güncellendi ({tur}, atomik)", flush=True)

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
