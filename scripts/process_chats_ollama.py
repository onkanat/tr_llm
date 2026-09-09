#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import time
import re

def escape_unescaped_quotes_in_json(json_str):
    pattern_inst = re.compile(r'("instruction"\s*:\s*")(.*?)("\s*,\s*"input")', re.DOTALL)
    pattern_inp = re.compile(r'("input"\s*:\s*")(.*?)("\s*,\s*"output")', re.DOTALL)
    pattern_out = re.compile(r'("output"\s*:\s*")(.*?)("\s*\})', re.DOTALL)
    
    def replacer(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        content_clean = content.replace('\\"', '"')
        content_escaped = content_clean.replace('"', '\\"')
        return f"{prefix}{content_escaped}{suffix}"
        
    json_str = pattern_inst.sub(replacer, json_str)
    json_str = pattern_inp.sub(replacer, json_str)
    json_str = pattern_out.sub(replacer, json_str)
    return json_str

def call_ollama(url, model, prompt):
    api_url = f"{url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url, 
        data=data, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            raw_response = res_json.get("response", "").strip()
            
            # Extract JSON object from raw response (handling thinking blocks, markdown fences, or extra text)
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = raw_response[start_idx:end_idx+1]
                json_str_fixed = escape_unescaped_quotes_in_json(json_str)
                return json.loads(json_str_fixed)
            else:
                raise ValueError("Response does not contain a JSON object.")
    except urllib.error.URLError as e:
        print(f"\n[Hata] Ollama sunucusuna bağlanılamadı: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"\n[Hata] Model çıktısı geçerli bir JSON değil: {e}")
        print(f"Ham Çıktı:\n{raw_response if 'raw_response' in locals() else 'Alınamadı'}")
        raise
    except Exception as e:
        print(f"\n[Hata] Beklenmeyen hata: {e}")
        raise

def clean_think_tags(text):
    if not isinstance(text, str):
        return text
    # Remove <think>...</think> block including the tags
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove stray tags if any
    cleaned = cleaned.replace('<think>', '').replace('</think>', '')
    return cleaned.strip()

def process_item(item, url, model, index, total):
    # Clean raw/dirty think tags from input
    instruction = clean_think_tags(item.get("instruction", ""))
    input_text = clean_think_tags(item.get("input", ""))
    output_text = clean_think_tags(item.get("output", ""))
    
    prompt = f"""Sen dilbilgisi uzmanı ve eğitimci bir Türkçe yapay zeka asistanısın.
Görevin: Sana verilen Alpaca formatındaki Türkçe soru-cevap çiftini (instruction, input, output) düzenlemek, imla/dilbilgisi hatalarından temizlemek ve cevabını (output) daha öğretici, zengin ve pedagojik olarak genişletmektir.

Kurallar:
1. Türkçe imla, yazım kuralları ve noktalama hatalarını tamamen düzelt.
2. Konuşmalarda yer alan gereksiz sistem gürültülerini, HTML/web kalıntılarını veya yarım kalmış ifadeleri temizle.
3. Cevap (output) kısmını pedagojik olarak daha açıklayıcı, net, zengin ve detaylı hale getirecek şekilde genişlet.
4. Ürettiğin JSON içindeki metin alanlarında asla kaçışsız çift tırnak (") kullanma. Cümle içinde çift tırnak kullanman gerekirse bunun yerine kesinlikle tek tırnak (') kullan.
5. Çıktıyı kesinlikle aşağıdaki JSON formatında ver. JSON dışında hiçbir açıklama, ön söz veya son söz yazma.

Giriş verisi:
- instruction: {instruction}
- input: {input_text}
- output: {output_text}

İstenen JSON formatı:
{{
  "instruction": "Düzenlenmiş talimat",
  "input": "Düzenlenmiş girdi",
  "output": "Temizlenmiş ve genişletilmiş açıklayıcı/pedagojik cevap"
}}
"""
    
    print(f"\r[{index}/{total}] İşleniyor...", end="", flush=True)
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            cleaned = call_ollama(url, model, prompt)
            # Validate structure
            if "instruction" in cleaned and "output" in cleaned:
                if "input" not in cleaned:
                    cleaned["input"] = ""
                # Make sure we clean think tags from output in case the model generated new ones
                cleaned["instruction"] = clean_think_tags(cleaned.get("instruction", ""))
                cleaned["input"] = clean_think_tags(cleaned.get("input", ""))
                cleaned["output"] = clean_think_tags(cleaned.get("output", ""))
                return cleaned
            else:
                print(f"\n[Uyarı] Geçersiz JSON yapısı (Eksik anahtarlar). Tekrar deneniyor... ({attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"\n[Uyarı] İstek hatası: {e}. Tekrar deneniyor... ({attempt + 1}/{max_retries})")
            time.sleep(2)
            
    print(f"\n[Hata] Öğe işlenemedi, orijinal öğe temizlenerek korunuyor.")
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "processing_failed": True
    }

def main():
    parser = argparse.ArgumentParser(description="Clean and expand Alpaca dataset using Ollama ornith:35b")
    parser.add_argument("--input", default="/Users/hakankilicaslan/Downloads/anythingllm-chats-7_5_2026-alpaca.json", help="Path to input json")
    parser.add_argument("--output", default="data/anythingllm_chats_cleaned.json", help="Path to output json")
    parser.add_argument("--ollama-url", default="http://192.168.1.14:11434", help="Ollama API URL")
    parser.add_argument("--model", default="ornith:35b", help="Ollama model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if available")
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"Hata: Girdi dosyası bulunamadı: {args.input}")
        sys.exit(1)
        
    print(f"Girdi dosyası yükleniyor: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)
        
    if not isinstance(items, list):
        print("Hata: Girdi dosyası bir liste (JSON array) olmalıdır.")
        sys.exit(1)
        
    total_items = len(items)
    if args.limit:
        items = items[:args.limit]
        total_items = len(items)
        print(f"Limit uygulandı. Toplam {total_items} adet öğe işlenecek.")
    else:
        print(f"Toplam {total_items} adet öğe bulundu.")
        
    processed_items = []
    start_idx = 0
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Resume functionality
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                processed_items = json.load(f)
            start_idx = len(processed_items)
            print(f"Önceki ilerleme yüklendi. {start_idx} adet öğe zaten işlenmiş. Buradan devam ediliyor.")
        except Exception as e:
            print(f"Önceki çıktı dosyası okunamadı, sıfırdan başlanıyor: {e}")
            processed_items = []
            
    if start_idx >= total_items:
        print("Tüm öğeler zaten işlenmiş!")
        sys.exit(0)
        
    print(f"Ollama sunucusuna bağlanılıyor: {args.ollama_url} (Model: {args.model})")
    
    # Quick connectivity check
    try:
        req = urllib.request.Request(
            f"{args.ollama_url.rstrip('/')}/api/tags",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            pass
        print("Ollama sunucusuna başarıyla bağlanıldı.")
    except Exception as e:
        print(f"Ollama sunucusuna veya modeline bağlanılamadı: {e}")
        print("Lütfen Ollama'nın çalıştığından ve modelin yüklü olduğundan emin olun.")
        sys.exit(1)

    print("İşlem başlatılıyor...")
    start_time = time.time()
    
    for idx in range(start_idx, total_items):
        item = items[idx]
        cleaned = process_item(item, args.ollama_url, args.model, idx + 1, total_items)
        processed_items.append(cleaned)
        
        # Incremental save after each item to prevent loss
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(processed_items, f, ensure_ascii=False, indent=2)
            
    duration = time.time() - start_time
    print(f"\n\nİşlem tamamlandı! Süre: {duration:.2f} saniye.")
    print(f"Temizlenmiş veri seti kaydedildi: {args.output}")

if __name__ == "__main__":
    main()
