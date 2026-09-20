#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/llm/prompt_contract.py

Kanonik Kristal-Vektörel Prompt ve Örnek Sözleşmesi (Sürüm 2).
T-0013 uyarınca tüm zarf ve prompt üretimini tek bir deterministik sözleşmede toplar.

MİMARİ PRENSİPLER (CLAUDE.md & T-0013):
  1. Kanonik Renderer'ı Yeniden Yazma: `KristalTokenizer._render_structured_prompt`
     dondurulmuş olup tek kanonik render referansıdır. Tüm işlevler doğrudan ona delege eder.
  2. Tip İpuçları Zorunlu, Saf Fonksiyonlar, Global Değiştirilebilir Durum YOK.
  3. Literal <BOS> Eklenmez: `tokenizer.encode()` metnin başına kendisi <BOS> (id 2) ekler.
     Prompt dizgelerine literal <BOS> yazılması ÇİFT <BOS> ([2, 2, ...]) hatası üretir.
  4. SAPMA 5 (BELGELENMİŞ): tokenizer.py:138 ve 140'ta `if input_text or input_text == "":`
     totolojisi (Python str için her zaman doğru) bulunmaktadır. `tokenizer.py` dondurulmuş
     olduğundan değiştirilmez; sözleşme bu "girdi boş da olsa <INPUT> etiketini her zaman bas"
     davranışını kanonik kabul eder.
  5. Servis Uyarısı: literal_entity_mode servis yolunda KAPALIDIR (servis checkpoint'i
     entity mimarisiyle eğitilmemiştir). T-0088: bu ilke cümlesi eskiden SİLİNMİŞ bir
     checkpoint adını anıyordu; ad kaldırıldı, ilke korundu.
"""

from dataclasses import dataclass
import os
import json
from typing import Dict, Any, Optional
import torch

from src.llm.tokenizer import KristalTokenizer, Vocabulary


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer ve sözlük konfigürasyonu. Açık ve loglanabilir parametreler."""
    # FAIL-CLOSED (T-0088): varsayilan KALDIRILDI. Once `data/vocab.json` idi ve BAYAT bir
    # sozlugu (31.357) isaret ediyordu; guncel kulliyat sozlugu 33.114 girişlidir. Sessiz
    # bir varsayilana dusmek yerine yoklugu ACIKCA durdurulur. Varsayilan BASKA BIR ADA da
    # tasinmadi (T-0085 emsali): uydurma bir ad, hangi kulliyata ait oldugu belirsiz bir
    # okuma hedefi yaratirdi.
    vocab_path: Optional[str] = None
    literal_entity_mode: bool = False

    def validate(self) -> None:
        if self.vocab_path is None:
            raise ValueError(
                "TokenizerConfig.vocab_path VERILMEDI. Sözlük yolu ZORUNLUDUR: bayat bir "
                "varsayilana dusmek yerine duruyorum (T-0088).")
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Sözlük dosyası bulunamadı: {self.vocab_path}")
        if self.literal_entity_mode:
            # Servis modelinin güvenliği için açık uyarı. T-0088: eski metin adı silinmiş
            # Kristal checkpoint'ini anıyordu; ad kaldırıldı, uyarı korundu.
            print("[SOZLESME_UYARI] literal_entity_mode=True aktif. Servis checkpoint'i "
                  "bu modda eğitilmemiştir; dikkatli kullanınız.")


def render_example(instruction: str, input_text: str, output_text: str) -> str:
    """
    Tam örnek zarfı (eğitim / tam sekans).
    KristalTokenizer._render_structured_prompt'a delege eder.
    Çıktı: '<INSTRUCTION> ... </INSTRUCTION> <INPUT> ... </INPUT> <OUTPUT> ... </OUTPUT>'
    """
    item = {
        "instruction": instruction.strip() if instruction else "",
        "input": input_text.strip() if input_text else "",
        "output": output_text.strip() if output_text else ""
    }
    return KristalTokenizer._render_structured_prompt(None, item)


def render_prompt(instruction: str, input_text: str) -> str:
    """
    Kanonik PROMPT zarfı (çıkarsama ve nedensel maskeleme sınırı).
    KristalTokenizer._render_structured_prompt'a delege eder.
    
    Özellikler:
      - Literal <BOS> İÇERMEZ (tokenizer.encode otomatik ekler).
      - instruction boşsa <INSTRUCTION> bloğu atlanır.
      - <INPUT> bloğu HER ZAMAN basılır (boş olsa dahi).
      - <OUTPUT> etiketi ile BİTER (modelin üretimi bu noktadan başlar).
    """
    item = {
        "instruction": instruction.strip() if instruction else "",
        "input": input_text.strip() if input_text else "",
        "output": ""
    }
    # _render_structured_prompt sonuna '<OUTPUT>  </OUTPUT>' ekler
    rendered = KristalTokenizer._render_structured_prompt(None, item)
    # <OUTPUT>'ta keserek kanonik prompt elde et
    if "<OUTPUT>" in rendered:
        return rendered.rsplit("<OUTPUT>", 1)[0] + "<OUTPUT>"
    return rendered + " <OUTPUT>"


def build_rag_input(doc_text: str, query: str) -> str:
    """
    Kanonik RAG girdisi kurucu.
    Biçim: '<BELGE> {doc_text} </BELGE> {query}'
    """
    d_clean = doc_text.strip() if doc_text else ""
    q_clean = query.strip() if query else ""
    return f"<BELGE> {d_clean} </BELGE> {q_clean}".strip()


def describe(config: Optional[TokenizerConfig] = None) -> str:
    """Sözleşme durumunu standart formatta döner.

    T-0088: `config` artik ZORUNLUDUR. Once argumansiz cagri sessizce BAYAT varsayilana
    (`data/vocab.json`, 31.357) dusuyordu; kapi cagri anina tasindi.
    """
    if config is None:
        raise ValueError("describe(): config VERILMEDI. Sözlük yolu çağırandan gelmelidir; "
                         "bayat bir varsayilana dusulmez (T-0088).")
    cfg = config
    cfg.validate()
    token_count = 0
    if os.path.exists(cfg.vocab_path):
        try:
            with open(cfg.vocab_path, "r", encoding="utf-8") as f:
                v_data = json.load(f)
                token_count = len(v_data.get("stoi", {}))
        except Exception:
            token_count = -1

    return (
        f"[SOZLESME] zarf=etiket surum=2 "
        f"vocab={cfg.vocab_path}({token_count} token) "
        f"literal_entity={cfg.literal_entity_mode}"
    )


def resize_state_dict(model: torch.nn.Module, old_state_dict: dict) -> dict:
    """
    Model embedding ve linear katmanlarını mevcut sözlük boyutuna uyarlar.
    Boyut uyuşmazlığı durumunda kaç satır kırpıldığı veya doldurulduğu
    [SOZLESME_UYARI] ile açıkça bildirilir (sessiz bozulma engellenir).
    """
    new_state_dict = model.state_dict()
    for k, v in old_state_dict.items():
        if k in new_state_dict:
            if v.shape != new_state_dict[k].shape:
                old_rows = v.shape[0]
                new_rows = new_state_dict[k].shape[0]
                diff = new_rows - old_rows
                action = f"{diff} satır dolduruldu (padding)" if diff > 0 else f"{-diff} satır kırpıldı (clipping)"
                print(
                    f"[SOZLESME_UYARI] Parametre '{k}' boyut uyumsuzluğu: "
                    f"eski {tuple(v.shape)} -> yeni {tuple(new_state_dict[k].shape)} ({action})"
                )
                if len(v.shape) == 2:
                    min_0 = min(v.shape[0], new_state_dict[k].shape[0])
                    min_1 = min(v.shape[1], new_state_dict[k].shape[1])
                    new_state_dict[k][:min_0, :min_1] = v[:min_0, :min_1]
                elif len(v.shape) == 1:
                    min_0 = min(v.shape[0], new_state_dict[k].shape[0])
                    new_state_dict[k][:min_0] = v[:min_0]
            else:
                new_state_dict[k] = v
    return new_state_dict
