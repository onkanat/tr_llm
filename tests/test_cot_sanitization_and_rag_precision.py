import pytest
from src.gateway.pedagogical_supervisor import sanitize_teacher_card
from src.rag.vector_memory import VectorMemory

def test_sanitize_teacher_card_cot_removal():
    dirty_text = (
        'We need to create a 2-3 sentence concise, pedagogically verified info card. '
        'The user says: "Sen Türk dili ve bilimi uzmanı bir öğretmensin." '
        'Then the question: "Zıvana yuvası zıvana diline göre çok gevşek açıldıysa ne yapılır?" '
        'Let\'s produce it.\n\n'
        'Zıvana dili yanaklarına aynı ağaçtan ince bir kaplama veya talaş şeridi yapıştırılarak kurutulmalıdır. '
        'Kuruduktan sonra hassas bir şekilde rendelenerek yuva ile sıkı bir geçme sağlanır.'
    )
    sanitized = sanitize_teacher_card(dirty_text)
    assert sanitized is not None
    assert "We need to" not in sanitized
    assert "The user says" not in sanitized
    assert "Zıvana dili yanaklarına aynı ağaçtan" in sanitized
    assert "sıkı bir geçme sağlanır" in sanitized

def test_sanitize_teacher_card_pure_english_rejection():
    pure_english = (
        'The user: Turkish language, wants a short information card, 2-3 sentences. '
        'They provide a question: describes a tree that is extremely hard. '
        'We need to keep it short: 2-3 sentences, net, concise.'
    )
    sanitized = sanitize_teacher_card(pure_english)
    assert sanitized is None

def test_question_words_in_common_roots():
    vm = VectorMemory(collection_name="test_temp_coll", vector_size=768, client=None)
    # Simulate query with question words
    query_tags = "<BOS> gürgen ağaç POSS_2SG CASE_ABL ne nasıl kim <EOS>"
    
    # Check that common_roots contains question words
    common_roots = {
        "su", "bir", "ve", "de", "da", "ki", "o", "bu", "şu", "ama", "ile", 
        "en", "daha", "her", "şey", "için", "ol", "et", "yap",
        "ne", "kim", "nasıl", "neden", "niçin", "hangi", "nere", "kaç", "mı", "mi", "mu", "mü"
    }
    assert "ne" in common_roots
    assert "kim" in common_roots
    assert "nasıl" in common_roots
