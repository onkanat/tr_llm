import json
from typing import List, Dict

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

    def render_prompt_template(self, item: Dict[str, str]) -> str:
        """
        Converts a structured curriculum item into a plain text prompt
        template.
        """
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()

        parts = []
        if instruction:
            parts.extend(["<INSTRUCTION>", instruction, "</INSTRUCTION>"])
        parts.extend(["<INPUT>", input_text, "</INPUT>"])
        parts.extend(["<OUTPUT>", output_text, "</OUTPUT>"])
        return " ".join(parts).strip()

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
