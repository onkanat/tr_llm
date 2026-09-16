"""src/eval/conditioning_score.py

İçerik Kökü Kapsamı ve Tekrar Cezası (Lexical Content Recall with Repetition Penalty)

Formüller T-0036 raporunun YAZILI tanımından birebir alınmıştır
(data/eval/t0036_kosullanma_olcutu_2026-09-15.json ->
 k1_metrik_tasarimi_ve_kendini_dogrulama.metrik_tanimi.formuller):

    Recall(P, O) = sum(min(C_P(r), C_O(r))) / len(C_P)
    RP(O)        = min(1.0, (len(unique(C_O)) / len(C_O)) / 0.70)
    Score(P, O)  = Recall(P, O) * RP(O)

C_P / C_O = içerik kökleri: aşağıdaki öğeler elenir:
  - Kontrol belirteçleri: <BOS> <EOS> <PAD> <UNK> <PROPER_NOUN>
                          <INSTRUCTION> </INSTRUCTION> <INPUT> </INPUT> <OUTPUT>
  - Noktalama kümesi
  - Biçimbirim ek önekleri: CASE_ POSS_ DERIV_ TENSE_ COPULA_ PART_
                             GERUND_ PLURAL PERSON_ VOICE_ REL_ INF_
  - Durak kelimeler

T-0036 F3 bulgusu: payda `len(C_P)` tokenizer sözlüğüne göre değişiyor
(temel 55, marangoz 44). Hangi sözlükle ölçüldüğü çağıran tarafından
raporlanmalıdır (vocab_path + sha256).

Kullanım:
    from src.eval.conditioning_score import conditioning_score
    result = conditioning_score(prompt_text, output_text, tokenizer, vocab)
    print(result["score"])  # float [0.0, 1.0]
"""
from __future__ import annotations

from typing import Any, Dict, List

# ── Sabitler ─────────────────────────────────────────────────────────────────

# Elenen kontrol belirteçleri (T-0036 raporundan)
CONTROL_TOKENS: frozenset = frozenset({
    "<BOS>", "<EOS>", "<PAD>", "<UNK>", "<PROPER_NOUN>",
    "<INSTRUCTION>", "</INSTRUCTION>",
    "<INPUT>", "</INPUT>",
    "<OUTPUT>",
})

# Elenen noktalama kümesi (T-0036 probe'undan)
PUNCT: frozenset = frozenset({
    ".", ",", ";", ":", "?", "!", "-", "(", ")", '"', "'", "\u2019",
    "\u2014", "\u2013", "\u2026",
})

# Elenen biçimbirim ek önekleri (T-0036 raporundan)
AFFIX_PREFIXES: tuple = (
    "CASE_", "POSS_", "DERIV_", "TENSE_", "COPULA_", "PART_",
    "GERUND_", "PLURAL", "PERSON_", "VOICE_", "REL_", "INF_",
)

# Durak kelimeler (T-0036 probe'undan)
STOPWORDS: frozenset = frozenset({
    "ve", "ile", "bu", "şu", "o", "bir", "de", "da", "ne", "nasıl",
    "için", "diye", "gibi", "var", "yok", "ise", "ki", "neler",
    "mu", "mi", "mü", "mı",
})


# ── İçerik kökü çıkarımı ─────────────────────────────────────────────────────

def _extract_content_roots(text: str, tokenizer: Any, vocab: Any) -> List[str]:
    """Metinden içerik köklerini çıkarır; kontrol belirteçleri, noktalama,
    biçimbirim ekleri ve durak kelimeler elenir.

    Args:
        text: Ham metin.
        tokenizer: KristalTokenizer örneği (encode metodu ile).
        vocab: Vocabulary örneği (decode metodu ile).

    Returns:
        İçerik köklerinin küçük harfli listesi (multiset — tekrar eden elemanlar korunur).
    """
    ids: List[int] = tokenizer.encode(text)
    tags: List[str] = [vocab.decode(i) for i in ids]
    roots: List[str] = []
    for tag in tags:
        # Kontrol belirteçlerini ele
        if tag in CONTROL_TOKENS:
            continue
        # Noktalama ele
        if tag in PUNCT:
            continue
        # Biçimbirim ek önekleri ele
        if any(tag.startswith(p) for p in AFFIX_PREFIXES):
            continue
        # Durak kelimeler ele
        tag_lower = tag.lower()
        if tag_lower in STOPWORDS:
            continue
        roots.append(tag_lower)
    return roots


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def conditioning_score(
    prompt_text: str,
    output_text: str,
    tokenizer: Any,
    vocab: Any,
) -> Dict[str, Any]:
    """İstem-çıktı koşullanma skoru hesaplar.

    Formüller T-0036 raporunun YAZILI tanımından birebir alınmıştır:
        Recall(P, O) = sum(min(C_P(r), C_O(r))) / len(C_P)
        RP(O)        = min(1.0, (len(unique(C_O)) / len(C_O)) / 0.70)
        Score(P, O)  = Recall(P, O) * RP(O)

    Args:
        prompt_text: İstem metni (ham; zarflı veya düz).
        output_text: Model çıktısı (ham metin).
        tokenizer: KristalTokenizer örneği.
        vocab: Vocabulary örneği. Hangi vocab kullanıldığı çağıran tarafından
               (vocab_path + sha256 ile) raporlanmalıdır (T-0036 F3).

    Returns:
        Dict[str, Any] — en az şu anahtarlar:
            score (float): Conditioning_Score(P, O) ∈ [0.0, 1.0]
            recall (float): Recall(P, O)
            rep_penalty (float): RP(O)
            common_roots (List[str]): Ortak kökler (multiset)
            prompt_roots (List[str]): C_P içerik kökleri
            output_roots (List[str]): C_O içerik kökleri
            bos (bool): C_P veya C_O boşsa True
    """
    c_p: List[str] = _extract_content_roots(prompt_text, tokenizer, vocab)
    c_o: List[str] = _extract_content_roots(output_text, tokenizer, vocab)

    # Boş küme kontrolü
    if not c_p or not c_o:
        return {
            "score": 0.0,
            "recall": 0.0,
            "rep_penalty": 1.0,
            "common_roots": [],
            "prompt_roots": c_p,
            "output_roots": c_o,
            "bos": True,
        }

    # Multiset sayaçları
    p_counts: Dict[str, int] = {}
    for r in c_p:
        p_counts[r] = p_counts.get(r, 0) + 1

    o_counts: Dict[str, int] = {}
    for r in c_o:
        o_counts[r] = o_counts.get(r, 0) + 1

    # Recall(P, O) = sum(min(C_P(r), C_O(r))) / len(C_P)
    common_count: int = sum(
        min(p_counts[r], o_counts.get(r, 0)) for r in p_counts
    )
    recall: float = common_count / len(c_p)

    # RP(O) = min(1.0, (len(unique(C_O)) / len(C_O)) / 0.70)
    unique_o: int = len(set(c_o))
    rep_ratio: float = unique_o / len(c_o)
    rep_penalty: float = min(1.0, rep_ratio / 0.70)

    # Score(P, O) = Recall(P, O) * RP(O)
    score: float = recall * rep_penalty

    # Ortak kökler listesi (açıklama amaçlı)
    common_roots: List[str] = [
        r for r in p_counts for _ in range(min(p_counts[r], o_counts.get(r, 0)))
    ]

    return {
        "score": score,
        "recall": recall,
        "rep_penalty": rep_penalty,
        "common_roots": common_roots,
        "prompt_roots": c_p,
        "output_roots": c_o,
        "bos": False,
    }
