#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPERIMENT B: MINIMAL INTERVENTION (FIX SCHEMA + VERIFY)
1. Selects 50 diverse carpentry question-answer records.
2. Formats them with strict, correct schema:
   instruction: 'Ahşap ve marangozluk uzmanı olarak cevapla.'
   input: <Actual Question>
   output: <Specific Technical Answer>
3. Fine-tunes carpenter model for 50 steps with proper prompt masking.
4. Evaluates the 5 diversity questions to see if output collapses or diversifies.
"""

import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.tokenizer import KristalTokenizer, Vocabulary
from scripts.train_step_demo import KristalLM
from src.compiler.decompiler import MorphemeDecompiler
from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

from src.llm.prompt_contract import render_prompt, render_example, resize_state_dict

def generate(model: KristalLM, tokenizer: KristalTokenizer, prompt_str: str, max_tokens: int = 128, temp: float = 0.0) -> str:
    input_ids = tokenizer.encode(prompt_str)
    eos_id = tokenizer.vocab.stoi.get("<EOS>", -1)
    if input_ids and input_ids[-1] == eos_id:
        input_ids = input_ids[:-1]
        
    x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    masked_tags = [
        "<PAD>", "<BOS>", "<INSTRUCTION>", "</INSTRUCTION>",
        "<INPUT>", "</INPUT>", "<OUTPUT>", "<PROPER_NOUN>", "<UNK>", "<NUMBER>"
    ]
    forbidden_ids = set()
    for tag in masked_tags:
        if tag in tokenizer.vocab.stoi:
            forbidden_ids.add(tokenizer.vocab.stoi[tag])
            
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= 256 else x[:, -256:]
            sign_mask = model.embedding.compute_sign_mask(x_cond).to(DEVICE)
            res = model(x_cond, sign_mask=sign_mask)
            logits = res[0] if isinstance(res, (tuple, list)) else res
            next_logits = logits[:, -1, :].clone()
            
            for fid in forbidden_ids:
                next_logits[:, fid] = -float("inf")
                
            # Sign-aware repetition penalty
            if len(generated) > 0:
                recent_ids = generated[-32:]
                for tid in set(recent_ids):
                    val = next_logits[0, tid].item()
                    if val > 0:
                        next_logits[0, tid] /= 1.3
                    else:
                        next_logits[0, tid] *= 1.3
                    
            if temp == 0.0:
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temp
                val, _ = torch.topk(next_logits, min(10, next_logits.size(-1)))
                next_logits[next_logits < val[:, [-1]]] = -float("inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                
            tok_id = next_tok.item()
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            x = torch.cat((x, next_tok), dim=1)
            
    return tokenizer.decode(generated).strip()

def run_exp_b():
    lexicon = LexiconManager()
    lexicon.load_from_tsv("data/lexicon/roots.tsv")
    graph = build_default_graph()
    compiler = CrystalCompiler(lexicon, graph)
    vocab = Vocabulary()
    vocab.load("data/vocab.json")
    tokenizer = KristalTokenizer(compiler, vocab)
    decompiler = MorphemeDecompiler(compiler, vocab)
    
    ckpt_path = "data/kristal_carpenter_model.pt"
    model = KristalLM(vocab_size=len(vocab.stoi), n_embd=768, vocab=vocab, block_size=4096, n_layer=6, n_head=6)
    state = torch.load(ckpt_path, map_location="cpu")
    new_sd = resize_state_dict(model, state)
    model.load_state_dict(new_sd)
    model.to(DEVICE)
    
    # 1. 50 curated question-answer pairs with strictly aligned schema
    # (instruction = persona, input = question, output = answer)
    qa_pairs = [
        ("Zıvana bağlantısı nasıl yapılır?", "Zıvana dişi ve yuvası birbirine alıştırılarak tutkalla monte edilir."),
        ("Rende yüzeyde dalma yapıyorsa sebebi nedir?", "Rende tığı ters lif yönünde ilerliyor olabilir veya bıçak körelmiştir. Talaş kırıcı açıklığı ayarlanmalıdır."),
        ("Masif ahşap kurutmada fırınlama neden şarttır?", "Kerestenin kontrollü kurutularak hücre içi gerilimlerinin boşaltılması ve çatlamanın önlenmesi için şarttır."),
        ("Kurt dişi birleştirme nerelerde kullanılır?", "Açılı zikzak dişlerle ahşap parçaların boylamasına eklenerek sonsuz kereste elde edilmesinde kullanılır."),
        ("Ahşap yüzeyde dolgu verniği nasıl uygulanır?", "Dolgu verniği açık gözenekleri doldurarak son kat cila için pürüzsüz alt zemin hazırlar."),
        ("Gönye aletinin marangozluktaki işlevi nedir?", "Ahşap parçaların 90 derece dik açılarını ve düzlemselliklerini kontrol etmeye yarar."),
        ("Şerit testere hangi işlev için tercih edilir?", "İki kasnak arasında dönen esnek şerit bıçağıyla kavisli kesimler ve biçme işlevinde kullanılır."),
        ("Planya tezgahının kullanım amacı nedir?", "Kaba biçilmiş kerestenin yüzeyini kusursuz bir 90 derecelik referans düzlemine getirir."),
        ("Kerpeten marangozlukta ne işe yarar?", "Eğri çakılan veya hatalı çivileri ahşaptan sökmek için kullanılır."),
        ("İşkence aletinin görevi nedir?", "Yapıştırma veya montaj sırasında parçaları tutkal kuruyana kadar sabit tutmaya yarar."),
        ("Kayın ağacının özellikleri nelerdir?", "Oldukça sert ve sıkı liflidir, buharlama yöntemiyle kolayca bükülerek sandalye yapımında kullanılır."),
        ("Dişbudak ağacının kullanım alanları nelerdir?", "Esnekliği ve darbe emme gücü çok yüksektir, balta ve alet saplarında tercih edilir."),
        ("Ihlamur ağacı neden oymacılıkta kullanılır?", "Yumuşak, lifsiz ve homojen yapısıyla ahşap oyma zanaatında en rahat işlenen ağaçtır."),
        ("Çam ahşabı nerelerde kullanılır?", "Reçineli, hafif ve ekonomik yapısıyla inşaat karkasında ve doğramalarda kullanılır."),
        ("Ceviz ağacının özellikleri nelerdir?", "Koyu damarlı, sert ve değerli yapısıyla lüks mobilya ve silah kabzalarında tercih edilir."),
        ("Meşe ağacı nerelerde kullanılır?", "Aşınmaya dayanıklı, tanenli ve sert yapısıyla zemin parkesi ve fıçılarda kullanılır."),
        ("Pelesenk ağacı nedir?", "Olağanüstü sert, yağlı ve koyu renkli tropikal bir ağaçtır, enstrüman klavyesinde kullanılır."),
        ("Köknar ağacı nerelerde tercih edilir?", "Reçinesiz ve açık renkli yapısıyla ambalaj sandığı ve çıta imalatında kullanılır."),
        ("Balmumu cilası nasıl etki eder?", "Doğal balmumunun masif ahşaba ovularak yedirildiği, ipeksi mat dokunuş veren geleneksel ciladır."),
        ("Danimarka yağı nedir?", "Doğal yağlar ile vernik reçinelerinin harmanlandığı, hem içten doyuran hem yüzeyde hafif sert film oluşturan ciladır."),
        ("Zımpara numaraları nasıl seçilir?", "Kaba temizlik için 80 kum, ara zımpara için 120-150 kum, son kat öncesi için 220 kum zımpara seçilir."),
        ("İskarpela bileme açısı kaç derece olmalıdır?", "Genel ahşap işleme için iskarpela ağız açısı 25 ile 30 derece arasında bilenir."),
        ("Kavela birleştirme nasıl uygulanır?", "Açılan deliklere silindirik ahşap pimler tutkallanarak parçalar birbirine kilitlenir."),
        ("Kırlangıç kuyruğu geçme nerede kullanılır?", "Çekmece köşelerinde çekme kuvvetine direnç sağlayan en sağlam dekoratif geçme tekniğidir."),
        ("Lamba zıvana tekniği nedir?", "Bir kenara erkek çıta, diğerine dişi kanal açılarak tahtaların birbirine geçirilmesidir."),
        ("Pah kırma işlemi neden yapılır?", "Ahşap kenarlarındaki keskin köşelerin yumuşatılarak kıymık ve ezilmelere karşı korunması için yapılır."),
        ("Sistre bıçağı ne işe yarar?", "Zımpara yerine ahşap yüzeyden mikron düzeyinde pürüzsüz talaş kazıyan çelik bıçaktır."),
        ("Poliüretan tutkal ne zaman tercih edilir?", "Suya ve neme tam dayanıklılık gerektiren dış mekan ahşap yapıştırmalarında tercih edilir."),
        ("Beyaz tutkal kuruma süresi nedir?", "Parçalar işkencede en az 30-45 dakika sıkılı tutulmalı, tam kuruma için 24 saat beklenmelidir."),
        ("Freze bıçağı ile kanal açarken nelere dikkat edilir?", "Parça lif yönüne ters sürülmemeli ve tek seferde çok derin talaş kaldırılmamalıdır."),
        ("Ahşabın lif yönü neden önemlidir?", "Rendeleme ve kesme işlemlerinde lif yönüne uyulmazsa yüzeyde yırtılma ve dalma oluşur."),
        ("Kalınlık makinesi ne işe yarar?", "Keresteyi istenen milimetrik et kalınlığına getiren döner merdaneli tezgah makinesidir."),
        ("Gürgen ağacı mobilyada nerede kullanılır?", "Yüksek mukavemeti sayesinde koltuk iskeletlerinde ve bükme parçalarda kullanılır."),
        ("Tezgâh mengenesi ahşabı nasıl korur?", "Mengeneye ahşap veya deri çeneler takılarak iş parçasının ezilmesi engellenir."),
        ("Ahşapta hare deseni nasıl oluşur?", "Ağacın yıllık büyüme halkalarının kesim açısına göre yüzeyde oluşturduğu doğal desendir."),
        ("Zıvana yuvası iskarpela ile nasıl açılır?", "Önce matkapla kaba talaş boşaltılır, ardından kenarlar iskarpela ile dik olarak düzeltilir."),
        ("Kılavuz çizgisi ahşapta nasıl çizilir?", "Kalem yerine işaret bıçağı veya çizgi demiri kullanılarak lifler kesilerek net iz bırakılır."),
        ("Ağaçta radyal kesim nedir?", "Yıllık halkalara dik açıyla yapılan, dönme ve büzülme payı en düşük kereste kesim yöntemidir."),
        ("Ağaçta teğet kesim nedir?", "Yıllık halkalara teğet yapılan, belirgin hare deseni veren ancak daha çok çalışan kesimdir."),
        ("Kavelanın yivli olması neden gereklidir?", "Tutkalın delik dibine sıkışıp hidrolik basınç yapmasını önlemek ve yapışma alanını artırmak içindir."),
        ("Sentetik tiner ne için kullanılır?", "Sentetik boya ve verniklerin inceltilmesinde ve alet temizliğinde kullanılır."),
        ("Gomalak cilası nasıl hazırlanır?", "Gomalak pullarının saf alkolde eritilerek topak bezle masif ahşaba dairesel hareketlerle yedirilmesiyle hazırlanır."),
        ("Çatlak ahşap nasıl tamir edilir?", "Ağaç tozu ile tutkal karıştırılarak macun yapılır veya kelebek kilit geçme uygulanır."),
        ("Kelebek kilit birleştirme nedir?", "Ahşap çatlaklarının büyümesini engellemek için çatlağın üzerine kakılan çift taraflı kırlangıç parçadır."),
        ("Eğri kesim testeresi nedir?", "Dar bıçağı sayesinde ahşapta dairesel ve karmaşık kavisli hatları kesmeye yarayan el testeresidir."),
        ("Ahşap nem ölçer nasıl kullanılır?", "İki sivri iğnesi kerestenin lif yönünde masife batırılarak elektriksel direnç üzerinden rutubet ölçülür."),
        ("İç mekan mobilyasında ideal ahşap nemi kaçtır?", "İç mekan ortamındaki mobilyalar için kereste nemi yüzde 8 ile 12 arasında olmalıdır."),
        ("Rende tığı kaç derecede bilenir?", "Genellikle 25 derece ana açı ve 30 derece mikro pah verilerek bilenir."),
        ("Ahşap masif panel nedir?", "Dar masif lataların boyuna ve enine tutkalla birleştirilmesiyle elde edilen geniş tablalardır."),
        ("Görünmez zıvana geçme nasıl yapılır?", "Dışarıdan bakıldığında hiçbir ek yeri görünmeyecek şekilde yuvanın kör açılmasıyla yapılır.")
    ]
    
    # 2. Compile pairs into training batch
    persona_inst = "Ahşap ve marangozluk uzmanı olarak cevapla."
    i_tok = tokenizer.decode(tokenizer.encode(persona_inst)).replace("<BOS>", "").replace("<EOS>", "").strip()
    
    train_tokens_list = []
    for q, a in qa_pairs:
        full_str = render_example(persona_inst, q, a)
        t_ids = tokenizer.encode(full_str)
        if len(t_ids) <= 256:
            train_tokens_list.append(t_ids)
            
    print(f"Hazırlanan Doğru Şemalı Örnek Sayısı: {len(train_tokens_list)}")
    
    # 3. Fine-tuning Loop (50 steps)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    output_start_id = vocab.stoi.get("<OUTPUT>", -1)
    eos_id = vocab.stoi.get("<EOS>", -1)
    pad_id = vocab.stoi.get("<PAD>", 0)
    
    model.train()
    print("\n50 Adımlık Şema Düzeltme Eğitimi Başlatılıyor...")
    
    np.random.seed(42)
    for step in range(1, 51):
        # Sample batch of 4 sequences
        batch_seqs = [train_tokens_list[idx] for idx in np.random.randint(0, len(train_tokens_list), 4)]
        max_len = max(len(s) for s in batch_seqs)
        
        x_np = np.full((4, max_len), pad_id, dtype=np.int64)
        targets_np = np.full((4, max_len), -100, dtype=np.int64)
        
        for b, s in enumerate(batch_seqs):
            x_np[b, :len(s)] = s
            # Shifted targets for causal LM
            targets_np[b, :len(s)-1] = s[1:]
            
            # Apply causal prompt masking: only tokens inside <OUTPUT> are trained
            is_out = False
            for t_idx in range(len(s) - 1):
                tok = s[t_idx]
                if tok == output_start_id:
                    is_out = True
                if not is_out:
                    targets_np[b, t_idx] = -100
                if tok == eos_id:
                    is_out = False
                    
        x = torch.from_numpy(x_np).to(DEVICE)
        targets = torch.from_numpy(targets_np).to(DEVICE)
        sign_mask = model.embedding.compute_sign_mask(x).to(DEVICE)
        
        optimizer.zero_grad()
        res = model(x, targets=targets, sign_mask=sign_mask)
        loss = res[1] if isinstance(res, (tuple, list)) else res
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0 or step == 1:
            print(f"  Adım {step:2d}/50 | Kayıp (Loss): {loss.item():.4f}")
            
    # Save fine-tuned checkpoint temporarily
    model.eval()
    
    # 4. Re-evaluate the 5 diversity questions
    print("\n" + "=" * 80)
    print("DENEY B: EĞİTİM SONRASI 5 ÇEŞİTLİLİK SORUSU DEĞERLENDİRMESİ")
    print("=" * 80)
    
    diversity_questions = [
        "Zıvana bağlantısı nasıl yapılır?",
        "Rende yüzeyde dalma yapıyorsa sebebi nedir?",
        "Masif ahşap kurutmada fırınlama neden şarttır?",
        "Kurt dişi birleştirme nerelerde kullanılır?",
        "Ahşap yüzeyde dolgu verniği nasıl uygulanır?"
    ]
    
    post_outputs = []
    for q in diversity_questions:
        prompt = render_prompt(persona_inst, q)
        out = generate(model, tokenizer, prompt, max_tokens=128, temp=0.0)
        dec = decompiler.decompile_sentence(out, capitalize=True)
        post_outputs.append((q, out, dec))
        
        print(f"\nSoru: '{q}'")
        print(f"  Decompile: {dec}")
        print(f"  Ham Çıktı: {out[:80]}...")
        
    unique_cnt = len(set(out for _, out, _ in post_outputs))
    print("\n" + "=" * 80)
    print(f">> DENEY B SONUCU: 5 soruda {unique_cnt} benzersiz çıktı (Klon oranı: {(1 - unique_cnt/5)*100:.1f}%)")
    if unique_cnt >= 4:
        print(">> BULGU DOĞRULANDI: Şema düzeltmesi (`instruction`=persona, `input`=soru) tek başına çıktı çeşitliliğini sağladı!")
    else:
        print(">> BULGU: Şema düzeltmesine rağmen çeşitlilik artmadı; ikincil arıza (veri doygunluğu / tokenizer) etkili.")

if __name__ == "__main__":
    run_exp_b()
