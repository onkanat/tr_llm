#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tests/test_challenger_m4_dual_parser_and_vault.py
=================================================
Milestone M4 Challenger 2 Adversarial Stress Test Suite:
Dual Output Parser (extract_cot_and_card) & CoT Vault Security (ALLOW_PROD_VAULT_WRITE).

Dimensions Tested:
1. extract_cot_and_card on extreme card-only inputs (NO phantom thought traces).
2. extract_cot_and_card on thought-only inputs (clean_card is None, thought cleanly extracted).
3. extract_cot_and_card on malformed and unclosed XML tags (<think>, <BILGI_KARTI>, etc.).
4. extract_cot_and_card on multiline procedural cards with numbered steps (steps preserved, meta CoT filtered).
5. ALLOW_PROD_VAULT_WRITE boolean parsing fail-closed security (falsy, empty, unset, path traversal).
6. Canonical production vault checksum invariance.
"""

import os
import sys
import json
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from src.gateway.pedagogical_supervisor import (
    extract_cot_and_card,
    sanitize_teacher_card,
    record_to_cot_vault,
    _CANONICAL_PROD_VAULT
)

EXPECTED_VAULT_SHA256 = "97b06fe5befd6613b520d28dc20a74deaeb30e4e8f7e6675a23720a019ff5619"


class TestAdversarialCardOnlyNoPhantomThought(unittest.TestCase):
    """Stress tests verifying that card-only inputs NEVER synthesize phantom thought traces."""

    def test_empty_tag_block(self):
        # The exact regression bug: '<BILGI_KARTI>\n</BILGI_KARTI>'
        text = "<BILGI_KARTI>\n</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None on empty tags, got {repr(thought)}")
        self.assertIsNone(card, f"Expected card=None on empty tags, got {repr(card)}")

    def test_empty_tag_block_with_whitespace(self):
        text = "   \n\n<BILGI_KARTI>   \n\t   </BILGI_KARTI>\n\n  "
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought)
        self.assertIsNone(card)

    def test_single_line_card_only(self):
        text = "<BILGI_KARTI>Gürgen ağacı yüksek mukavemeti ve sert dokusu nedeniyle mobilyacılıkta tercih edilir.</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None on single-line card-only, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Gürgen ağacı", card)
        self.assertNotIn("<BILGI_KARTI>", card)
        self.assertNotIn("</BILGI_KARTI>", card)

    def test_multiline_card_only(self):
        text = (
            "<BILGI_KARTI>\n"
            "Meşe kerestesi yüksek yoğunluğu ve su direnci sebebiyle dış cephe kaplamalarında kullanılır.\n"
            "Doğru fırınlanıp kurutulmadığında eğilme ve çatlama riski taşır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None on multiline card-only, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Meşe kerestesi", card)
        self.assertIn("çatlama riski", card)
        self.assertNotIn("<BILGI_KARTI>", card)
        self.assertNotIn("</BILGI_KARTI>", card)

    def test_card_only_with_leading_label_inside_tags(self):
        text = "<BILGI_KARTI>[Bilgi Kartı]: Çam ağacı reçineli yapısı sebebiyle emprenye edilmeden dış mekanda kullanılmaz.</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertTrue(card.startswith("Çam ağacı"))
        self.assertNotIn("[Bilgi Kartı]", card)

    def test_card_only_turkish_dotted_bilgi_karti(self):
        text = "<BİLGİ_KARTI>Dişbudak ağacı esnek yapısıyla spor aletleri ve bükme mobilyalarda yaygın olarak kullanılır.</BİLGİ_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None on Turkish dotted tag, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Dişbudak ağacı", card)
        self.assertNotIn("<BİLGİ_KARTI>", card)

    def test_card_only_case_insensitivity(self):
        for open_tag, close_tag in [
            ("<bilgi_karti>", "</bilgi_karti>"),
            ("<Bilgi_Karti>", "</Bilgi_Karti>"),
            ("<BILGI_KARTI>", "</bilgi_karti>")
        ]:
            text = f"{open_tag}\nIhlamur ağacı yumuşak dokusu sayesinde ahşap oyma işlerinde tercih edilir.\n{close_tag}"
            thought, card = extract_cot_and_card(text)
            self.assertIsNone(thought, f"Tag variation {open_tag} leaked thought: {repr(thought)}")
            self.assertIsNotNone(card)
            self.assertIn("Ihlamur ağacı", card)

    def test_card_only_short_content_below_threshold(self):
        # Short card < 15 chars should yield card=None, and NO thought trace
        text = "<BILGI_KARTI>Kısa metin.</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None, got {repr(thought)}")
        self.assertIsNone(card, f"Expected card=None for short content, got {repr(card)}")

    def test_card_only_nested_duplicate_tags(self):
        text = (
            "<BILGI_KARTI><BILGI_KARTI>\n"
            "Ceviz ağacı dekoratif deseni ve işlenme kolaylığıyla lüks mobilyalarda öne çıkar.\n"
            "</BILGI_KARTI></BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None on nested duplicate tags, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Ceviz ağacı", card)
        self.assertNotIn("<BILGI_KARTI>", card)


class TestAdversarialThoughtOnlyCleanExtraction(unittest.TestCase):
    """Stress tests verifying that thought-only inputs extract thought cleanly and leave clean_card=None."""

    def test_clean_dusunce_only(self):
        text = (
            "<DUSUNCE>\n"
            "Öğrenciye gürgen ağacının mekanik mukavemetini ve mobilya sektöründeki kullanımını anlatmamız gerekiyor.\n"
            "Pedagojik bağlamda 2-3 cümlelik net bir açıklama kurgulayalım.\n"
            "</DUSUNCE>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("gürgen ağacının mekanik mukavemetini", thought)
        self.assertNotIn("<DUSUNCE>", thought)
        self.assertNotIn("</DUSUNCE>", thought)
        self.assertIsNone(card, f"clean_card must be None on thought-only input, got {repr(card)}")

    def test_clean_think_tag_only(self):
        text = (
            "<think>\n"
            "The user wants a technical explanation of dovetail joints.\n"
            "We need to focus on wood grain alignment and mechanical interlocking resistance.\n"
            "</think>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("technical explanation of dovetail", thought)
        self.assertNotIn("<think>", thought)
        self.assertNotIn("</think>", thought)
        self.assertIsNone(card, f"clean_card must be None on English think tag, got {repr(card)}")

    def test_thought_tag_only(self):
        text = (
            "<thought>\n"
            "Düşünce adımları: Öncelikle bitişimli dillerdeki morfolojik durum makinesi analiz edilmelidir.\n"
            "Kök ve ek dizilim kuralları deterministik olarak incelenmelidir.\n"
            "</thought>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("morfolojik durum makinesi", thought)
        self.assertIsNone(card, f"clean_card must be None, got {repr(card)}")

    def test_reasoning_tag_only(self):
        text = (
            "<reasoning>\n"
            "Detailed botanical analysis of tomato late blight fungus Phytophthora infestans.\n"
            "Dosage calculation of copper sulfate Bordeaux mixture.\n"
            "</reasoning>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("Phytophthora infestans", thought)
        self.assertIsNone(card, f"clean_card must be None, got {repr(card)}")

    def test_contractual_dusunce_with_turkish_literature_content(self):
        text = (
            "<DUSUNCE>\n"
            "Açık istiare teşbihin bir türüdür ve sadece kendisine benzetilen öge kullanılır.\n"
            "Öğrencinin seviyesine uygun olarak edebi terimler basitleştirilmelidir.\n"
            "</DUSUNCE>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("Açık istiare teşbihin bir türüdür", thought)
        self.assertNotIn("<DUSUNCE>", thought)
        self.assertIsNone(card, f"clean_card must be None on DUSUNCE-only input, got {repr(card)}")

    @unittest.expectedFailure
    def test_turkish_dotted_dusunce_only(self):
        # Adversarial probe: While <DUSUNCE> is the contract tag, if a model outputs dotted <DÜŞÜNCE>,
        # Step 1 currently only searches for ASCII <DUSUNCE>, causing the thought to fall through to sanitizer.
        text = (
            "<DÜŞÜNCE>\n"
            "Açık istiare teşbihin bir türüdür ve sadece kendisine benzetilen öge kullanılır.\n"
            "Öğrencinin seviyesine uygun olarak edebi terimler basitleştirilmelidir.\n"
            "</DÜŞÜNCE>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIsNone(card)

    def test_thought_containing_xml_code_snippets(self):
        text = (
            "<DUSUNCE>\n"
            "XML şeması gereğince <tag> ve </tag> etiketleri çift yönlü doğrulanmalıdır.\n"
            "Kod içerisindeki <item id='1'> ayrıştırma mantığı denetleniyor.\n"
            "</DUSUNCE>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIn("<tag>", thought)
        self.assertIsNone(card)


class TestAdversarialMalformedAndUnclosedXMLTags(unittest.TestCase):
    """Stress tests on unclosed, mismatched, and pathological XML markup."""

    def test_unclosed_think_tag_followed_by_closed_bilgi_karti(self):
        text = (
            "<think>\n"
            "Burada ahşap analizi yapıyoruz ve meşe kerestesinin nem oranını hesaplıyoruz.\n"
            "<BILGI_KARTI>\n"
            "Meşe kerestesi dış mekan mobilyalarında ve parke üretiminde yaygın olarak kullanılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought, "thought_trace must be extracted from unclosed think prefix")
        self.assertIsNotNone(card, "clean_card must be extracted from closed BILGI_KARTI")
        self.assertIn("ahşap analizi yapıyoruz", thought)
        self.assertIn("Meşe kerestesi dış mekan", card)
        self.assertNotIn("<think>", thought)
        self.assertNotIn("<BILGI_KARTI>", card)
        self.assertNotIn("</BILGI_KARTI>", card)

    def test_closed_think_tag_followed_by_unclosed_bilgi_karti(self):
        text = (
            "<think>\n"
            "Burada ahşap analizi yapıyoruz ve meşe kerestesinin nem oranını hesaplıyoruz.\n"
            "</think>\n"
            "<BILGI_KARTI>\n"
            "Meşe kerestesi dış mekan mobilyalarında ve parke üretiminde yaygın olarak kullanılır."
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("ahşap analizi yapıyoruz", thought)
        self.assertIn("Meşe kerestesi dış mekan", card)
        self.assertNotIn("<think>", thought)
        self.assertNotIn("</think>", thought)
        self.assertNotIn("<BILGI_KARTI>", card)

    def test_unclosed_dusunce_followed_by_closed_bilgi_karti(self):
        text = (
            "<DUSUNCE>\n"
            "Ahşap birleştirme yöntemlerinden kırlangıç kuyruğunun mukavemetini açıklıyoruz.\n"
            "<BILGI_KARTI>\n"
            "Kırlangıç kuyruğu birleştirme tekniği, çekme kuvvetine direnç sağlayan trapez dişleriyle çekmecelerde kullanılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("kırlangıç kuyruğunun mukavemetini", thought)
        self.assertIn("Kırlangıç kuyruğu birleştirme tekniği", card)
        self.assertNotIn("<DUSUNCE>", thought)
        self.assertNotIn("<BILGI_KARTI>", card)

    def test_unclosed_bilgi_karti_only_without_closing_tag(self):
        text = "<BILGI_KARTI>\nGürgen ağacı yüksek mukavemeti ve sert dokusu nedeniyle mobilyacılıkta tercih edilir."
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Gürgen ağacı yüksek mukavemeti", card)
        self.assertNotIn("<BILGI_KARTI>", card)

    def test_unclosed_think_only_with_english_cot(self):
        text = "<think>\nWe need to analyze the question carefully.\nHere is our thought step."
        thought, card = extract_cot_and_card(text)
        # Without closing tag, unclosed English CoT cannot produce a valid Turkish card
        self.assertIsNone(card, f"clean_card must be None, got {repr(card)}")

    def test_mismatched_opening_and_closing_tags(self):
        text = (
            "<think>\n"
            "Burada içsel akıl yürütme adımları yer almaktadır.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Ahşap oyma işlerinde ıhlamur ağacı yumuşak dokusuyla en ideal malzemedir.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("içsel akıl yürütme", thought)
        self.assertIn("ıhlamur ağacı", card.lower())

    def test_only_closing_tag_residue(self):
        text = "Ahşap oyma işlerinde ıhlamur ve ceviz ağaçları tercih edilir.\n</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("Ahşap oyma işlerinde", card)
        self.assertNotIn("</BILGI_KARTI>", card)

    def test_none_empty_and_non_string_inputs(self):
        for bad_input in [None, "", "   \n\t  ", 12345, [], {}]:
            thought, card = extract_cot_and_card(bad_input)
            self.assertIsNone(thought, f"Expected None for thought on {repr(bad_input)}")
            self.assertIsNone(card, f"Expected None for card on {repr(bad_input)}")


class TestAdversarialMultilineProceduralCards(unittest.TestCase):
    """Stress tests on multiline procedural cards with numbered steps."""

    def test_four_step_carpentry_procedure(self):
        text = (
            "<BILGI_KARTI>\n"
            "Adım 1: Gürgen ağacı gönyesinde hassas biçimde kesilir.\n"
            "Adım 2: Fırınlanmış kereste rendelenerek pürüzsüz yüzey elde edilir.\n"
            "Adım 3: Zıvana delikleri ıskarpela yardımıyla açılır.\n"
            "Adım 4: Parçalar tutkallanarak işkencelerle sıkıştırılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought, f"Expected thought=None, got {repr(thought)}")
        self.assertIsNotNone(card)
        self.assertIn("Adım 1: Gürgen ağacı", card)
        self.assertIn("Adım 2: Fırınlanmış kereste", card)
        self.assertIn("Adım 3: Zıvana delikleri", card)
        self.assertIn("Adım 4: Parçalar tutkallanarak", card)

    def test_dot_notation_steps(self):
        text = (
            "<BILGI_KARTI>\n"
            "Adım 1. Kırlangıç kuyruğu zıvana çizgileri markalama bıçağıyla çizilir.\n"
            "Adım 2. İnce dişli japon testeresi ile işaret çizgisi takip edilerek kesim yapılır.\n"
            "Adım 3. Iskarpela ile yuvanın dip köşelerindeki talaşlar temizlenir.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("Adım 1.", card)
        self.assertIn("Adım 2.", card)
        self.assertIn("Adım 3.", card)

    def test_numerical_prefix_steps(self):
        text = (
            "<BILGI_KARTI>\n"
            "1. Adım: Aşılama yapılacak fidanın gövdesi temiz bir bıçakla T şeklinde yarılır.\n"
            "2. Adım: Uygun anaçtan alınan aşı gözü kabuğun altına dikkatlice yerleştirilir.\n"
            "3. Adım: Aşı bağı ve macunu ile yara yeri hava almayacak şekilde sarılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("1. Adım:", card)
        self.assertIn("2. Adım:", card)
        self.assertIn("3. Adım:", card)

    def test_high_step_count_procedure(self):
        steps = [f"Adım {i}: Ahşap işleme operasyonu aşama {i} gerçekleştirilir." for i in range(1, 11)]
        text = "<BILGI_KARTI>\n" + "\n".join(steps) + "\n</BILGI_KARTI>"
        thought, card = extract_cot_and_card(text)
        self.assertIsNone(thought)
        self.assertIsNotNone(card)
        for i in range(1, 11):
            self.assertIn(f"Adım {i}: Ahşap işleme", card)

    def test_dual_output_with_thought_and_procedural_card(self):
        text = (
            "<DUSUNCE>\n"
            "Marangozlukta zıvana açma sürecini pedagojik ve net adımlarla açıklayalım.\n"
            "İş güvenliği ve alet kullanımı kuralları hatırlatılmalıdır.\n"
            "</DUSUNCE>\n"
            "<BILGI_KARTI>\n"
            "Adım 1: Ahşap parçalar markalama pleyti üzerinde işaretlenir.\n"
            "Adım 2: Zıvana delikleri hassas tezgah testeresinde açılır.\n"
            "Adım 3: Yüzeyler bezir yağı ile parlatılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought)
        self.assertIsNotNone(card)
        self.assertIn("zıvana açma sürecini", thought)
        self.assertIn("Adım 1: Ahşap parçalar", card)
        self.assertIn("Adım 2: Zıvana delikleri", card)
        self.assertIn("Adım 3: Yüzeyler bezir yağı", card)

    def test_adversarial_discrimination_meta_cot_steps_vs_real_procedural_steps(self):
        # Mixed input: Meta CoT steps MUST be filtered into thought, while real procedural steps MUST be kept in card
        text = (
            "<BILGI_KARTI>\n"
            "Adım 1: Soru kökünü analiz edelim.\n"
            "Adım 1: Ahşap gönyeye alınır ve işaretlenir.\n"
            "Adım 2: Pedagojik kontrol sağlayalım.\n"
            "Adım 2: Zıvana testeresi ile hassas kesim yapılır.\n"
            "</BILGI_KARTI>"
        )
        thought, card = extract_cot_and_card(text)
        self.assertIsNotNone(thought, "Meta CoT steps should be captured as thought")
        self.assertIsNotNone(card, "Real procedural steps should remain in clean_card")
        # Meta CoT steps in thought
        self.assertIn("Soru kökünü analiz edelim", thought)
        self.assertIn("Pedagojik kontrol sağlayalım", thought)
        # Real steps in card
        self.assertIn("Ahşap gönyeye alınır ve işaretlenir", card)
        self.assertIn("Zıvana testeresi ile hassas kesim yapılır", card)
        # Meta CoT steps NOT in card
        self.assertNotIn("Soru kökünü analiz edelim", card)
        self.assertNotIn("Pedagojik kontrol sağlayalım", card)


class TestAdversarialAllowProdVaultWriteSecurity(unittest.TestCase):
    """Stress tests verifying fail-closed enforcement of ALLOW_PROD_VAULT_WRITE."""

    def setUp(self):
        self.orig_env = os.environ.get("ALLOW_PROD_VAULT_WRITE")
        self.orig_test_env = os.environ.get("PYTEST_CURRENT_TEST")
        os.environ["PYTEST_CURRENT_TEST"] = "test_adversarial_vault_security"

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["ALLOW_PROD_VAULT_WRITE"] = self.orig_env
        else:
            os.environ.pop("ALLOW_PROD_VAULT_WRITE", None)

        if self.orig_test_env is not None:
            os.environ["PYTEST_CURRENT_TEST"] = self.orig_test_env
        else:
            os.environ.pop("PYTEST_CURRENT_TEST", None)

    def test_non_whitelisted_falsy_values_strictly_raise_runtime_error(self):
        falsy_values = [
            "0",
            "false",
            "no",
            "",
            None,  # unset
            "00",
            "False",
            "FALSE",
            "no",
            "NO",
            "none",
            "None",
            "null",
            "disabled",
            "off",
            "2",
            "-1",
            "fake_true",
            "trueish",
            "0; rm -rf /",
            "undefined"
        ]

        for val in falsy_values:
            if val is None:
                os.environ.pop("ALLOW_PROD_VAULT_WRITE", None)
            else:
                os.environ["ALLOW_PROD_VAULT_WRITE"] = val

            with self.assertRaises(
                RuntimeError,
                msg=f"ALLOW_PROD_VAULT_WRITE={repr(val)} MUST raise RuntimeError when targeting canonical vault"
            ):
                record_to_cot_vault(
                    query="Güvenlik deneme sorgusu",
                    instruction="Güvenlik talimatı",
                    thought_trace="Düşünce zinciri doğrulaması en az 15 karakter",
                    final_answer="Bilgi kartı doğrulaması en az 15 karakter",
                    vault_path=_CANONICAL_PROD_VAULT
                )

    def test_path_traversal_and_relative_paths_to_canonical_vault_blocked(self):
        # Verify that relative paths, './', and '../' pointing to the canonical vault are caught by realpath
        os.environ["ALLOW_PROD_VAULT_WRITE"] = "0"

        relative_paths = [
            "data/pedagogy/cot_vault.jsonl",
            "./data/pedagogy/cot_vault.jsonl",
            os.path.join(".", "data", "pedagogy", "..", "pedagogy", "cot_vault.jsonl"),
            os.path.relpath(_CANONICAL_PROD_VAULT, os.getcwd())
        ]

        for rel_path in relative_paths:
            with self.assertRaises(
                RuntimeError,
                msg=f"Target path {rel_path} pointing to canonical vault MUST be blocked"
            ):
                record_to_cot_vault(
                    query="Güvenlik deneme sorgusu",
                    instruction="Güvenlik talimatı",
                    thought_trace="Düşünce zinciri doğrulaması en az 15 karakter",
                    final_answer="Bilgi kartı doğrulaması en az 15 karakter",
                    vault_path=rel_path
                )

    def test_default_path_unset_blocked_when_falsy(self):
        # When vault_path=None and COT_VAULT_PATH is unset, defaults to _CANONICAL_PROD_VAULT
        os.environ.pop("ALLOW_PROD_VAULT_WRITE", None)
        old_cot = os.environ.pop("COT_VAULT_PATH", None)
        try:
            with self.assertRaises(RuntimeError, msg="Default canonical vault MUST be blocked when unset"):
                record_to_cot_vault(
                    query="Default path deneme",
                    instruction="Yönerge",
                    thought_trace="Düşünce adımı güvenlik doğrulaması",
                    final_answer="Nihai yanıt güvenlik doğrulaması",
                    vault_path=None
                )
        finally:
            if old_cot is not None:
                os.environ["COT_VAULT_PATH"] = old_cot

    def test_isolated_tempfile_always_allowed_regardless_of_env(self):
        # Non-canonical temporary files MUST be writable even if ALLOW_PROD_VAULT_WRITE='0'
        os.environ["ALLOW_PROD_VAULT_WRITE"] = "0"
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_vault = os.path.join(tmpdir, "isolated_vault.jsonl")
            res = record_to_cot_vault(
                query="İzole sorgu",
                instruction="Yönerge",
                thought_trace="İzole düşünce adımı 15 karakter",
                final_answer="İzole yanıt metni 15 karakter",
                vault_path=temp_vault
            )
            self.assertTrue(res)
            self.assertTrue(os.path.exists(temp_vault))
            with open(temp_vault, "r", encoding="utf-8") as f:
                data = json.loads(f.readline())
                self.assertEqual(data["query"], "İzole sorgu")

    def test_canonical_vault_checksum_invariant(self):
        # Ensure that no test in the entire run modified data/pedagogy/cot_vault.jsonl
        self.assertTrue(os.path.exists(_CANONICAL_PROD_VAULT))
        with open(_CANONICAL_PROD_VAULT, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            h, EXPECTED_VAULT_SHA256,
            f"Canonical vault hash was corrupted! Expected {EXPECTED_VAULT_SHA256}, got {h}"
        )


if __name__ == "__main__":
    unittest.main()
