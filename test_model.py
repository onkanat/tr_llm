import os
import sys
import json
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM

def main():
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: MODEL TEST VE ÇIKARIM (INFERENCE)")
    print("=" * 60)

    # 1. Device Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Çıkarım yapılacak cihaz: {device}")

    # 2. Load Vocab & Tokenizer
    vocab = Vocabulary()
    vocab_path = 'data/vocab.json'
    vocab.load(vocab_path)
    vocab_size = len(vocab.stoi)
    print(f"Sözlük Yüklendi. Kelime dağarcığı boyutu: {vocab_size}")

    lexicon = LexiconManager()
    lexicon.load_from_tsv('data/lexicon/roots.tsv')
    compiler = CrystalCompiler(lexicon, build_default_graph())
    tokenizer = KristalTokenizer(compiler, vocab)

    # 3. Load Model
    n_embd = 768
    model = KristalLM(vocab_size=vocab_size, n_embd=n_embd, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    model_path = 'data/kristal_model.pt'
    
    if not os.path.exists(model_path):
        print(f"Hata: Eğitilmiş model dosyası '{model_path}' bulunamadı! Lütfen önce eğitimi çalıştırın.")
        return

    state_dict = torch.load(model_path, map_location=device)
    keys_to_skip = [k for k in state_dict.keys() if "cos_cached" in k or "sin_cached" in k or "mask" in k]
    for k in keys_to_skip:
        del state_dict[k]
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Eğitilmiş model '{model_path}' adresinden başarıyla yüklendi.")
    print("=" * 60)

    # 4. Test Prompts
    test_prompts = [
        "Akmayan su kımıldanmayan yer",
        "Demirkır güney tepelerinin duldalarına çektiği atları",
        "Yarın okula gideceğim",
        "Kitabı okudum ve temizledim"
    ]

    print("\n--- Test Cümleleri Çıkarım Sonuçları ---")
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n[Girdi Cümle]: '{prompt}'")
            
            # Tokenize & Encode
            token_ids = tokenizer.encode(prompt)
            decoded_morphemes = tokenizer.decode(token_ids)
            print(f"  -> Morfemler: {decoded_morphemes}")
            print(f"  -> Token IDs: {token_ids}")
            
            # 1. Cümle İçi Tahmin (Son morfemden önceki duruma göre tahmin)
            # <EOS> token'ını çıkararak son morfemden sonra ne geleceğini tahmin edelim
            if len(token_ids) > 2:
                token_ids_before_eos = token_ids[:-1]
                x_mid = torch.tensor([token_ids_before_eos], dtype=torch.long, device=device)
                logits_mid, _ = model(x_mid)
                last_logits_mid = logits_mid[0, -1, :]
                probs_mid = torch.softmax(last_logits_mid, dim=-1)
                top_k = 5
                top_probs_mid, top_indices_mid = torch.topk(probs_mid, top_k)
                
                last_morpheme = vocab.decode(token_ids_before_eos[-1])
                print(f"  -> '{last_morpheme}' morfeminden sonra gelen en olası morfemler (Top {top_k}):")
                for idx in range(top_k):
                    pred_id = top_indices_mid[idx].item()
                    pred_prob = top_probs_mid[idx].item()
                    pred_token = vocab.decode(pred_id)
                    print(f"     {idx+1}. {pred_token:<20} | Olasılık: {pred_prob:.4f} (ID: {pred_id})")
            
            # 2. Cümle Sonu / Belge Sınırı Tahmini (<EOS> sonrası)
            x_eos = torch.tensor([token_ids], dtype=torch.long, device=device)
            logits_eos, _ = model(x_eos)
            last_logits_eos = logits_eos[0, -1, :]
            probs_eos = torch.softmax(last_logits_eos, dim=-1)
            top_probs_eos, top_indices_eos = torch.topk(probs_eos, top_k)
            
            print(f"  -> <EOS> (Cümle Sonu) sonrası en olası sonraki morfemler (Top {top_k}):")
            for idx in range(top_k):
                pred_id = top_indices_eos[idx].item()
                pred_prob = top_probs_eos[idx].item()
                pred_token = vocab.decode(pred_id)
                print(f"     {idx+1}. {pred_token:<20} | Olasılık: {pred_prob:.4f} (ID: {pred_id})")

    print("\n--- SFT (Ebeveynlik Fazı) Görev Çıkarım Sonuçları ---")
    sft_test_cases = [
        {"instruction": "Kelimedeki kök morfemini bul.", "input": "kitaplarda", "output": ""},
        {"instruction": "Kelimedeki eylemin zamanını veya kipini tespit et.", "input": "gideceğim", "output": ""},
        {"instruction": "Kelimenin aldığı durum eklerini (hâl eklerini) tespit et.", "input": "okula", "output": ""},
        {"instruction": "Kelimede çoğul eki (PLURAL) olup olmadığını tespit et.", "input": "çocuklar", "output": ""}
    ]

    output_start_id = vocab.stoi.get("<OUTPUT>", -1)

    with torch.no_grad():
        for case in sft_test_cases:
            print(f"\n[SFT Görevi]: '{case['instruction']}' | Girdi: '{case['input']}'")
            
            # Encode structured JSON
            token_ids = tokenizer.encode(json.dumps(case))
            decoded_morphemes = tokenizer.decode(token_ids)
            print(f"  -> Tam Akış: {decoded_morphemes}")
            
            # Find the position of <OUTPUT> token
            if output_start_id in token_ids:
                output_idx = token_ids.index(output_start_id)
                # Slice up to <OUTPUT> (inclusive) to let the model generate the answer
                eval_tokens = token_ids[:output_idx + 1]
                eval_decoded = tokenizer.decode(eval_tokens)
                print(f"  -> Model Girdisi: {eval_decoded}")
                
                # Predict first token after <OUTPUT>
                x_eval = torch.tensor([eval_tokens], dtype=torch.long, device=device)
                logits_eval, _ = model(x_eval)
                last_logits = logits_eval[0, -1, :]
                probs = torch.softmax(last_logits, dim=-1)
                
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k)
                
                print(f"  -> Modelin <OUTPUT> sonrasındaki 1. Tahminleri (Top {top_k}):")
                for idx in range(top_k):
                    pred_id = top_indices[idx].item()
                    pred_prob = top_probs[idx].item()
                    pred_token = vocab.decode(pred_id)
                    print(f"     {idx+1}. {pred_token:<20} | Olasılık: {pred_prob:.4f} (ID: {pred_id})")
                
                # Autoregressively feed the top prediction (or <PROPER_NOUN>) to see the 2nd prediction
                top_1_id = top_indices[0].item()
                eval_tokens_2 = eval_tokens + [top_1_id]
                x_eval_2 = torch.tensor([eval_tokens_2], dtype=torch.long, device=device)
                logits_eval_2, _ = model(x_eval_2)
                last_logits_2 = logits_eval_2[0, -1, :]
                probs_2 = torch.softmax(last_logits_2, dim=-1)
                top_probs_2, top_indices_2 = torch.topk(probs_2, top_k)
                
                top_1_decoded = vocab.decode(top_1_id)
                print(f"  -> '{top_1_decoded}' sonrasındaki 2. Tahminleri (Top {top_k}):")
                for idx in range(top_k):
                    pred_id_2 = top_indices_2[idx].item()
                    pred_prob_2 = top_probs_2[idx].item()
                    pred_token_2 = vocab.decode(pred_id_2)
                    print(f"     {idx+1}. {pred_token_2:<20} | Olasılık: {pred_prob_2:.4f} (ID: {pred_id_2})")
            else:
                print("  -> Hata: <OUTPUT> etiketine ulaşılamadı!")

    print("\n" + "=" * 60)
    print(" MODEL TEST SÜRECİ TAMAMLANDI")
    print("=" * 60)

if __name__ == '__main__':
    main()
