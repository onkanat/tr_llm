import json
from typing import List, Dict
from src.llm.prompt_contract import render_example, render_prompt

class PedagogicalPhase:
    INFANCY = "INFANCY"       # Güdümsüz Keşif, Zıtlıklar (Positive/Negative boundaries)
    PARENTING = "PARENTING"   # Güdümlü Öğrenme, Skill-based SFT chunking
    SOCIALIZATION = "SOCIALIZATION" # Uzmanlaşma/RLHF, Domain specific fine-tuning

class CurriculumGenerator:
    """Generates synthetic Alpaca-format training data for Vector Rover's 3-phase pedagogy."""
    
    def generate_infancy_data(self) -> List[Dict[str, str]]:
        """
        Phase 1: Basic concept boundaries using positive/negative contrastive examples.
        Helps the model establish semantic boundaries instead of just statistical co-occurrence.
        """
        return [
            {
                "instruction": "Kavram sınırlarını belirle (Geçerli/Geçersiz).",
                "input": "Top",
                "output": "Pozitif: Topa vurulur, top seker. Negatif: *Top içilir, *Top okunur."
            },
            {
                "instruction": "Kavram sınırlarını belirle (Geçerli/Geçersiz).",
                "input": "Kitap",
                "output": "Pozitif: Kitap okunur, kitap yazılır. Negatif: *Kitap yenir, *Kitap uçar."
            }
        ]

    def generate_parenting_data(self) -> List[Dict[str, str]]:
        """
        Phase 2: Task-based micro-chunking (Skill-Based SFT).
        Guided learning using the rules of the Kristal Compiler.
        """
        return [
            {
                "instruction": "Cümledeki eylemin zamanını tespit et.",
                "input": "Yarın okula gideceğim.",
                "output": "TENSE_FUT (Gelecek Zaman)"
            },
            {
                "instruction": "Kelimedeki kök morfemini bul.",
                "input": "kitaplarda",
                "output": "ROOT: kitap"
            }
        ]

    def generate_specialization_data(self, domain: str) -> List[Dict[str, str]]:
        """
        Phase 3: Domain-specific fine-tuning ("Marangoz YZ" concept).
        Focused dataset that restricts the model's traversal to specific vector islands.
        """
        if domain == "marangoz":
            return [
                {
                    "instruction": "Ahşap yüzey pürüzsüzleştirme işlemi nasıl yapılır?",
                    "input": "",
                    "output": "Ahşap yüzey zımpara kağıdı (örneğin 120 numara ile başlayıp 220 numaraya geçerek) kullanılarak lif yönünde pürüzsüzleştirilir."
                },
                {
                    "instruction": "Hangi ahşap türü dış mekan mobilyaları için daha dayanıklıdır?",
                    "input": "",
                    "output": "Dış mekan için nem ve çürümeye karşı doğal yağlar barındıran tik (teak), sedir veya iroko gibi dayanıklı ağaç türleri tercih edilmelidir."
                }
            ]
        return []

    def render_example(self, item: Dict[str, str]) -> str:
        """
        Converts a structured curriculum item into a full plain text example.
        Delegates to canonical prompt_contract.render_example.
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        return render_example(instruction, input_text, output_text)

    def render_prompt(self, item: Dict[str, str]) -> str:
        """
        Converts a structured curriculum item into a plain text prompt stopping at <OUTPUT>.
        Delegates to canonical prompt_contract.render_prompt.
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        return render_prompt(instruction, input_text)

    def render_prompt_template(self, item: Dict[str, str]) -> str:
        """
        Converts a structured curriculum item into a plain text prompt template.
        For full training example with <OUTPUT>...</OUTPUT>, delegates to render_example.
        """
        return self.render_example(item)

    def export_to_prompt_file(self, data: List[Dict[str, str]], filepath: str):
        """Exports the generated curriculum as plain text prompt templates."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(self.render_prompt_template(item) + '\n')

    def export_to_jsonl(self, data: List[Dict[str, str]], filepath: str):
        """
        Exports the generated curriculum to JSONL format suitable for SFT/RLHF
        pipelines.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
