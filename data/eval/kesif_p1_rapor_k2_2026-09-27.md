# T-0149 FAZ-1 KEŞİF-RAPORU — P1 kalan-6 derin-mekanizma tanısı

**Hüküm:** **KEŞIF_TAMAM** (betikten; elle sayı YOK)
**Damga:** 2026-09-27T18:01:56Z (UTC, `time.gmtime`) — koşum sonu
**İlan:** `data/eval/kesif_p1_ilan2_2026-09-27.md` (koşum ÖNCESİ; sha256 `80ddbe600dc0f75a633b55315e371aeaea129d00047a66ebb7e23b89ba2679a8`)

## 1. Hüküm kapıları (betikten)

| Ayraç | Ölçülen | Hüküm |
|---|---|---|
| K1 SABİT-İMZA: 5 YOL_0 yüzeyi compile() → 0 analiz (İLAN-3 çıpa 6428b180… SABİT; beklenmedik yol = çıpa-çürütme) | `{"zeyrek": 0, "meçhul": 0, "nakil": 0, "güç": 0, "hacir": 0}` | GEÇTİ |
| K2 DEC-YÜZEY-BİREBİR: decompile_tags([lemma, affix]) == koşum-2 yüzeyi ×5 birebir | `{"zeyrek": "zeyrekin", "meçhul": "meçhlun", "nakil": "naklin", "güç": "güçüm", "hacir": "hacrimiz"}` | GEÇTİ |
| K3 ENVANTER-DOLU (İLAN-2 pos-ÇOKLU-SATIR): 5 örneğin HER TSV-satırı için ölçüm-alan-şeması dolu (transition-envanter + template? resolve-ikili + split-adaylar / gecis_yok ölçülmüş-imza) | `{"zeyrek": {"tsv_satir_sayi": 2, "satir_olcumleri_dolu": [true, true]}, "meçhul": {"tsv_satir_sayi": 2, "satir_olcumleri_dolu": [true, true]}, "nakil": {"tsv_satir_sayi": 1, "satir_olcumleri_dolu": [true]}, "güç": {"tsv_satir_sayi": 3, "satir_olcumleri_dolu": [true, true, true]}, "hacir": {"tsv_satir_sayi": 1, "satir_olcumleri_dolu": [true]}}` | GEÇTİ |
| K4 BILBI-KIRILIMI: compile('biler') ≥1 analiz + seçilen-yol lemma 'Bi' + geri-decompile 'Bi'ler' (İLAN-3 kanıt-birebir) + kırılım-alanları dolu | `{"analiz_sayisi": 7, "secilen_token_vector": ["Bi", "PLURAL"], "geri_decompile": "Bi'ler", "stems_surface_lemma_sirasi": ["bi", "bil", "bile", "bile", "bile", "bile"]}` | GEÇTİ |
| K5 İSTİSNA: tanı-koşumu istisna FIRLATMAZ (istisna == 0) | `0` | GEÇTİ |

## 2. YOL_0 tanı-kırılımı (birebir ölçüm)

### zeyrek + CASE_GEN → zeyrekin

```json
{
 "affix": "CASE_GEN",
 "compile_analiz_sayisi": 0,
 "compile_istisna": null,
 "dec_istisna": null,
 "dec_yuzey": "zeyrekin",
 "lemma": "zeyrek",
 "satir_olcumleri": [
  {
   "pos": "ADJ",
   "resolve_nok_attrs": null,
   "resolve_tsv_attrs": null,
   "split_adaylari": null,
   "template": null,
   "template_yok": true,
   "transition_envanteri": {
    "affix_gecisleri": [],
    "start_state": "S_ADJ_ROOT"
   },
   "tsv_attrs": "-",
   "tsv_is_case_alias": null
  },
  {
   "pos": "NOUN",
   "resolve_nok_attrs": {
    "affix_surface": "in",
    "mutated_stem": "zeyrek",
    "new_string": "zeyrekin",
    "startswith_yuzey": true
   },
   "resolve_tsv_attrs": {
    "affix_surface": "in",
    "mutated_stem": "zeyreğ",
    "new_string": "zeyreğin",
    "startswith_yuzey": false
   },
   "split_adaylari": [
    {
     "aday_prefix": "ze",
     "affix_surface": "nin",
     "birebir_yuzey": false,
     "kalan": "yrekin",
     "mutated": "ze",
     "new_string": "zenin"
    },
    {
     "aday_prefix": "zeyrek",
     "affix_surface": "in",
     "birebir_yuzey": true,
     "kalan": "in",
     "mutated": "zeyrek",
     "new_string": "zeyrekin"
    },
    {
     "aday_prefix": "zeyrek",
     "affix_surface": "in",
     "birebir_yuzey": false,
     "kalan": "in",
     "mutated": "zeyreğ",
     "new_string": "zeyreğin"
    }
   ],
   "template": "(n)In",
   "template_yok": false,
   "transition_envanteri": {
    "affix_gecisleri": [
     {
      "affix_id": "CASE_GEN",
      "attributes": "-",
      "template": "(n)In",
      "to_state": "S_NOUN_POST_CASE"
     }
    ],
    "start_state": "S_NOUN_ROOT"
   },
   "tsv_attrs": "VOICING",
   "tsv_is_case_alias": null
  }
 ],
 "stems_lemma": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ze",
   "matched_prefix": "ze",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "zeyrek",
   "matched_prefix": "zeyrek",
   "pos": "ADJ"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "zeyrek",
   "matched_prefix": "zeyrek",
   "pos": "NOUN"
  }
 ],
 "stems_yuzey": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ze",
   "matched_prefix": "ze",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "zeyrek",
   "matched_prefix": "zeyrek",
   "pos": "ADJ"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "zeyrek",
   "matched_prefix": "zeyrek",
   "pos": "NOUN"
  }
 ],
 "tsv_satir": [
  {
   "attributes": "-",
   "lemma": "zeyrek",
   "pos": "ADJ"
  },
  {
   "attributes": "VOICING",
   "lemma": "zeyrek",
   "pos": "NOUN"
  }
 ],
 "yuzey": "zeyrekin"
}
```

### meçhul + CASE_GEN → meçhlun

```json
{
 "affix": "CASE_GEN",
 "compile_analiz_sayisi": 0,
 "compile_istisna": null,
 "dec_istisna": null,
 "dec_yuzey": "meçhlun",
 "lemma": "meçhul",
 "satir_olcumleri": [
  {
   "pos": "ADJ",
   "resolve_nok_attrs": null,
   "resolve_tsv_attrs": null,
   "split_adaylari": null,
   "template": null,
   "template_yok": true,
   "transition_envanteri": {
    "affix_gecisleri": [],
    "start_state": "S_ADJ_ROOT"
   },
   "tsv_attrs": "VOWEL_DROP",
   "tsv_is_case_alias": null
  },
  {
   "pos": "NOUN",
   "resolve_nok_attrs": {
    "affix_surface": "un",
    "mutated_stem": "meçhul",
    "new_string": "meçhulun",
    "startswith_yuzey": false
   },
   "resolve_tsv_attrs": {
    "affix_surface": "un",
    "mutated_stem": "meçhl",
    "new_string": "meçhlun",
    "startswith_yuzey": true
   },
   "split_adaylari": [
    {
     "aday_prefix": "me",
     "affix_surface": "nin",
     "birebir_yuzey": false,
     "kalan": "çhlun",
     "mutated": "me",
     "new_string": "menin"
    },
    {
     "aday_prefix": "meç",
     "affix_surface": "in",
     "birebir_yuzey": false,
     "kalan": "hlun",
     "mutated": "meç",
     "new_string": "meçin"
    },
    {
     "aday_prefix": "meçhl",
     "affix_surface": "in",
     "birebir_yuzey": false,
     "kalan": "un",
     "mutated": "meçhl",
     "new_string": "meçhlin"
    },
    {
     "aday_prefix": "meçhl",
     "affix_surface": "in",
     "birebir_yuzey": false,
     "kalan": "un",
     "mutated": "meçhl",
     "new_string": "meçhlin"
    }
   ],
   "template": "(n)In",
   "template_yok": false,
   "transition_envanteri": {
    "affix_gecisleri": [
     {
      "affix_id": "CASE_GEN",
      "attributes": "-",
      "template": "(n)In",
      "to_state": "S_NOUN_POST_CASE"
     }
    ],
    "start_state": "S_NOUN_ROOT"
   },
   "tsv_attrs": "VOWEL_DROP",
   "tsv_is_case_alias": null
  }
 ],
 "stems_lemma": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "me",
   "matched_prefix": "me",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "meç",
   "matched_prefix": "meç",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "meçhul",
   "matched_prefix": "meçhul",
   "pos": "ADJ"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "meçhul",
   "matched_prefix": "meçhul",
   "pos": "NOUN"
  }
 ],
 "stems_yuzey": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "me",
   "matched_prefix": "me",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "meç",
   "matched_prefix": "meç",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "meçhul",
   "matched_prefix": "meçhl",
   "pos": "ADJ"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "meçhul",
   "matched_prefix": "meçhl",
   "pos": "NOUN"
  }
 ],
 "tsv_satir": [
  {
   "attributes": "VOWEL_DROP",
   "lemma": "meçhul",
   "pos": "ADJ"
  },
  {
   "attributes": "VOWEL_DROP",
   "lemma": "meçhul",
   "pos": "NOUN"
  }
 ],
 "yuzey": "meçhlun"
}
```

### nakil + CASE_GEN → naklin

```json
{
 "affix": "CASE_GEN",
 "compile_analiz_sayisi": 0,
 "compile_istisna": null,
 "dec_istisna": null,
 "dec_yuzey": "naklin",
 "lemma": "nakil",
 "satir_olcumleri": [
  {
   "pos": "NOUN",
   "resolve_nok_attrs": {
    "affix_surface": "in",
    "mutated_stem": "nakil",
    "new_string": "nakilin",
    "startswith_yuzey": false
   },
   "resolve_tsv_attrs": {
    "affix_surface": "in",
    "mutated_stem": "nakl",
    "new_string": "naklin",
    "startswith_yuzey": true
   },
   "split_adaylari": [
    {
     "aday_prefix": "na",
     "affix_surface": "nın",
     "birebir_yuzey": false,
     "kalan": "klin",
     "mutated": "na",
     "new_string": "nanın"
    },
    {
     "aday_prefix": "nakl",
     "affix_surface": "ın",
     "birebir_yuzey": false,
     "kalan": "in",
     "mutated": "nakl",
     "new_string": "naklın"
    }
   ],
   "template": "(n)In",
   "template_yok": false,
   "transition_envanteri": {
    "affix_gecisleri": [
     {
      "affix_id": "CASE_GEN",
      "attributes": "-",
      "template": "(n)In",
      "to_state": "S_NOUN_POST_CASE"
     }
    ],
    "start_state": "S_NOUN_ROOT"
   },
   "tsv_attrs": "VOWEL_DROP",
   "tsv_is_case_alias": null
  }
 ],
 "stems_lemma": [
  {
   "attributes": "-",
   "is_case_alias": true,
   "lemma": "Na",
   "matched_prefix": "na",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "nakil",
   "matched_prefix": "nakil",
   "pos": "NOUN"
  }
 ],
 "stems_yuzey": [
  {
   "attributes": "-",
   "is_case_alias": true,
   "lemma": "Na",
   "matched_prefix": "na",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "nakil",
   "matched_prefix": "nakl",
   "pos": "NOUN"
  }
 ],
 "tsv_satir": [
  {
   "attributes": "VOWEL_DROP",
   "lemma": "nakil",
   "pos": "NOUN"
  }
 ],
 "yuzey": "naklin"
}
```

### güç + POSS_1SG → güçüm

```json
{
 "affix": "POSS_1SG",
 "compile_analiz_sayisi": 0,
 "compile_istisna": null,
 "dec_istisna": null,
 "dec_yuzey": "güçüm",
 "lemma": "güç",
 "satir_olcumleri": [
  {
   "pos": "ADJ",
   "resolve_nok_attrs": null,
   "resolve_tsv_attrs": null,
   "split_adaylari": null,
   "template": null,
   "template_yok": true,
   "transition_envanteri": {
    "affix_gecisleri": [],
    "start_state": "S_ADJ_ROOT"
   },
   "tsv_attrs": "-",
   "tsv_is_case_alias": null
  },
  {
   "pos": "ADV",
   "resolve_nok_attrs": null,
   "resolve_tsv_attrs": null,
   "split_adaylari": null,
   "template": null,
   "template_yok": true,
   "transition_envanteri": {
    "affix_gecisleri": [],
    "start_state": "S_ADV_ROOT"
   },
   "tsv_attrs": "-",
   "tsv_is_case_alias": null
  },
  {
   "pos": "NOUN",
   "resolve_nok_attrs": {
    "affix_surface": "üm",
    "mutated_stem": "güç",
    "new_string": "güçüm",
    "startswith_yuzey": true
   },
   "resolve_tsv_attrs": {
    "affix_surface": "üm",
    "mutated_stem": "güc",
    "new_string": "gücüm",
    "startswith_yuzey": false
   },
   "split_adaylari": [
    {
     "aday_prefix": "güç",
     "affix_surface": "üm",
     "birebir_yuzey": true,
     "kalan": "üm",
     "mutated": "güç",
     "new_string": "güçüm"
    },
    {
     "aday_prefix": "güç",
     "affix_surface": "üm",
     "birebir_yuzey": true,
     "kalan": "üm",
     "mutated": "güç",
     "new_string": "güçüm"
    },
    {
     "aday_prefix": "güç",
     "affix_surface": "üm",
     "birebir_yuzey": false,
     "kalan": "üm",
     "mutated": "güc",
     "new_string": "gücüm"
    }
   ],
   "template": "(I)m",
   "template_yok": false,
   "transition_envanteri": {
    "affix_gecisleri": [
     {
      "affix_id": "POSS_1SG",
      "attributes": "-",
      "template": "(I)m",
      "to_state": "S_NOUN_POST_POSSESSIVE"
     }
    ],
    "start_state": "S_NOUN_ROOT"
   },
   "tsv_attrs": "VOICING",
   "tsv_is_case_alias": null
  }
 ],
 "stems_lemma": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "ADJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "ADV"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "NOUN"
  }
 ],
 "stems_yuzey": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "ADJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "ADV"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "güç",
   "matched_prefix": "güç",
   "pos": "NOUN"
  }
 ],
 "tsv_satir": [
  {
   "attributes": "-",
   "lemma": "güç",
   "pos": "ADJ"
  },
  {
   "attributes": "-",
   "lemma": "güç",
   "pos": "ADV"
  },
  {
   "attributes": "VOICING",
   "lemma": "güç",
   "pos": "NOUN"
  }
 ],
 "yuzey": "güçüm"
}
```

### hacir + POSS_1PL → hacrimiz

```json
{
 "affix": "POSS_1PL",
 "compile_analiz_sayisi": 0,
 "compile_istisna": null,
 "dec_istisna": null,
 "dec_yuzey": "hacrimiz",
 "lemma": "hacir",
 "satir_olcumleri": [
  {
   "pos": "NOUN",
   "resolve_nok_attrs": {
    "affix_surface": "imiz",
    "mutated_stem": "hacir",
    "new_string": "hacirimiz",
    "startswith_yuzey": false
   },
   "resolve_tsv_attrs": {
    "affix_surface": "imiz",
    "mutated_stem": "hacr",
    "new_string": "hacrimiz",
    "startswith_yuzey": true
   },
   "split_adaylari": [
    {
     "aday_prefix": "ha",
     "affix_surface": "mız",
     "birebir_yuzey": false,
     "kalan": "crimiz",
     "mutated": "ha",
     "new_string": "hamız"
    },
    {
     "aday_prefix": "ha",
     "affix_surface": "mız",
     "birebir_yuzey": false,
     "kalan": "crimiz",
     "mutated": "ha",
     "new_string": "hamız"
    },
    {
     "aday_prefix": "ha",
     "affix_surface": "mız",
     "birebir_yuzey": false,
     "kalan": "crimiz",
     "mutated": "ha",
     "new_string": "hamız"
    },
    {
     "aday_prefix": "ha",
     "affix_surface": "mız",
     "birebir_yuzey": false,
     "kalan": "crimiz",
     "mutated": "ha",
     "new_string": "hamız"
    },
    {
     "aday_prefix": "ha",
     "affix_surface": "mız",
     "birebir_yuzey": false,
     "kalan": "crimiz",
     "mutated": "ha",
     "new_string": "hamız"
    },
    {
     "aday_prefix": "hac",
     "affix_surface": "ımız",
     "birebir_yuzey": false,
     "kalan": "rimiz",
     "mutated": "hac",
     "new_string": "hacımız"
    },
    {
     "aday_prefix": "hacr",
     "affix_surface": "ımız",
     "birebir_yuzey": false,
     "kalan": "imiz",
     "mutated": "hacr",
     "new_string": "hacrımız"
    }
   ],
   "template": "(I)mIz",
   "template_yok": false,
   "transition_envanteri": {
    "affix_gecisleri": [
     {
      "affix_id": "POSS_1PL",
      "attributes": "-",
      "template": "(I)mIz",
      "to_state": "S_NOUN_POST_POSSESSIVE"
     }
    ],
    "start_state": "S_NOUN_ROOT"
   },
   "tsv_attrs": "VOWEL_DROP",
   "tsv_is_case_alias": null
  }
 ],
 "stems_lemma": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "CONJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "INTERJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "NOUN"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "VERB"
  },
  {
   "attributes": "-",
   "is_case_alias": true,
   "lemma": "Ha",
   "matched_prefix": "ha",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "hac",
   "matched_prefix": "hac",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "hacir",
   "matched_prefix": "hacir",
   "pos": "NOUN"
  }
 ],
 "stems_yuzey": [
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "CONJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "INTERJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "NOUN"
  },
  {
   "attributes": "VOICING",
   "is_case_alias": false,
   "lemma": "ha",
   "matched_prefix": "ha",
   "pos": "VERB"
  },
  {
   "attributes": "-",
   "is_case_alias": true,
   "lemma": "Ha",
   "matched_prefix": "ha",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "hac",
   "matched_prefix": "hac",
   "pos": "NOUN"
  },
  {
   "attributes": "VOWEL_DROP",
   "is_case_alias": false,
   "lemma": "hacir",
   "matched_prefix": "hacr",
   "pos": "NOUN"
  }
 ],
 "tsv_satir": [
  {
   "attributes": "VOWEL_DROP",
   "lemma": "hacir",
   "pos": "NOUN"
  }
 ],
 "yuzey": "hacrimiz"
}
```

## 3. bil/Bi iki-farklı-lemma kırılımı (birebir ölçüm)

```json
{
 "analyses": [
  {
   "morpheme_ids": [
    "Bi",
    "PLURAL"
   ],
   "score": 8.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "bil",
    "TENSE_AORIST_VOWEL"
   ],
   "score": 8.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "bile",
    "TENSE_AORIST"
   ],
   "score": 8.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "Bi",
    "DERIV_lA",
    "TENSE_AORIST"
   ],
   "score": 7.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "bil",
    "TENSE_AORIST_VOWEL",
    "PERSON_3SG"
   ],
   "score": 7.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "bile",
    "TENSE_AORIST",
    "PERSON_3SG"
   ],
   "score": 7.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX",
    "AFFIX"
   ]
  },
  {
   "morpheme_ids": [
    "Bi",
    "DERIV_lA",
    "TENSE_AORIST",
    "PERSON_3SG"
   ],
   "score": 6.0,
   "surface_form": "biler",
   "types": [
    "ROOT",
    "AFFIX",
    "AFFIX",
    "AFFIX"
   ]
  }
 ],
 "compile_analiz_sayisi": 7,
 "compile_istisna": null,
 "dec_Bi_istisna": null,
 "dec_Bi_tags": "Bi'ler",
 "dec_bil_istisna": null,
 "dec_bil_tags": "biler",
 "geri_decompile_secilen": "Bi'ler",
 "needs_disambiguation": true,
 "secilen_token_vector": [
  "Bi",
  "PLURAL"
 ],
 "stems_surface": [
  {
   "attributes": "-",
   "is_case_alias": true,
   "lemma": "Bi",
   "matched_prefix": "bi",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "bil",
   "matched_prefix": "bil",
   "pos": "VERB"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "bile",
   "matched_prefix": "bile",
   "pos": "ADV"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "bile",
   "matched_prefix": "bile",
   "pos": "CONJ"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "bile",
   "matched_prefix": "bile",
   "pos": "POSTP"
  },
  {
   "attributes": "-",
   "is_case_alias": false,
   "lemma": "bile",
   "matched_prefix": "bile",
   "pos": "VERB"
  }
 ],
 "surface": "biler",
 "tsv_satirlar": [
  {
   "attributes": "-",
   "lemma": "Bi",
   "pos": "NOUN"
  },
  {
   "attributes": "-",
   "lemma": "bil",
   "pos": "VERB"
  }
 ]
}
```

## 4. Tam SHA-256 digest tablosu

| Dosya | SHA-256 |
|---|---|
| `data/lexicon/roots.tsv` | `fe3005e5e2a594f09cbcfc3286e2c8812953ae6614333815ab87a7e3a6763598` |
| `data/eval/kesif_p1_ilan2_2026-09-27.md` | `80ddbe600dc0f75a633b55315e371aeaea129d00047a66ebb7e23b89ba2679a8` |
| `scripts/kesif_p1_yol0_derin_mekanizma.py` | `af0bcec5019e2e543bb877b3d3fc7de878ac3b795ebe268c4c67e72c27a2bfa6` |
| `data/eval/kesif_p1_hukum_k2_2026-09-27.json` | `7527767639aa371cff5dd7fc034ac85d5158c8c9d8b16d87b56ce444b1557b16` |

