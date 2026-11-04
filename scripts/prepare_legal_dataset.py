#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P2.5 FAZ-1 — HUKUK SFT VERİ SETİ ÜRETİCİ (LEGAL CANONICAL SFT)
================================================================
`data/legal_korpus/` korpusunu okuyup `data/pedagogy/legal_canonical_sft.jsonl`
kütüğünü üretir. Kalıp: scripts/prepare_carpenter_specialization_dataset.py
kart biçimi ({instruction, input, output}).

GÖREV TİPLERİ:
  (a) kavram tanımı      — tanım bentleri ('Kavram: ...') doğrudan korpus tanım
                           maddelerinden; alıntı + kanun/madde kaynağı.
  (b) madde sorusu       — kanun başlıklı maddeler ('Madde başlığı' satırları)
                           üzerinden madde alıntısı.
  (c) kavram ayrımı      — iki gerçek korpus tanımı yan yana + kaynak.
  (d) prosedür adımları  — her adım gerçek bir madde alıntısıyla.

UYDURMA KAPILARI (her kayıt):
  - Her alıntı kurulum gereği korpus madde metninden alınır (satır-satır);
    çözümlenemeyen terim/prosedür DÜŞÜRÜLÜR ve drop sayaçlarına yazılır.
  - Mülga maddeler ve boş metinler elenir.
  - Şema doğrulama + dedupe + token/UNK ölçümü meta kütüğüne yazılır.

Kullanım:
    venv/bin/python scripts/prepare_legal_dataset.py [--dry-run]
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.compiler.lexicon import LexiconManager
from src.compiler.morphotactics import build_default_graph
from src.compiler.core import CrystalCompiler
from src.llm.tokenizer import KristalTokenizer, Vocabulary

CORPUS_DIR = os.path.join(REPO, "data", "legal_korpus")
MANIFEST = os.path.join(CORPUS_DIR, "korpus_manifest.json")
OUT_JSONL = os.path.join(REPO, "data", "pedagogy", "legal_canonical_sft.jsonl")
OUT_META = OUT_JSONL + ".meta.json"
VOCAB_PATH = os.path.join(REPO, "data", "rebuild", "vocab_anka_r1_33114.json")

MADDE_HEAD = re.compile(r"^(Geçici\s+|Ek\s+)?Madde\s*(\d+)\s*[-–—]\s*(.*)$")
MADDE_REF = re.compile(r"(?:Geçici\s+|Ek\s+)?(?:Madde|madde)\s*(?:\d+)")
BENT_RE = re.compile(
    r"^\(?([a-zçğıöşüâ]{1,2})\)\s*"
    r"(?:\((?:Değişik|Ek|Mülga|İptal)[^)]*\)\s*)*"
    r"([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşüâî\s./-]{2,60}):\s+(.+)$"
)
ANNOT_LINE = re.compile(
    r"^\s*\((?:Ek(?:\s+fıkra|\s+cümle)?|Mülga[^)]*|Değişik[^)]*|İptal[^)]*)\)\s*\.?\s*$"
)
INLINE_ANNOT = re.compile(r"\s*\((?:Değişik|Ek|İptal|Mülga)[^)]*md\.\)\s*")
MULGA_MARK = re.compile(r"^\s*\(?\s*Mülga\b", re.I)

QUOTE_CAP = 700          # madde alıntısı tavanı (karakter)
DEF_CAP = 500            # tanım alıntısı tavanı
MIN_MADDE_CHARS = 60     # daha kısa madde SFT'ye girmez

# (b) tipi için kanun başına tavan — denge (kalite>nicelik, ~300-600 hedef)
PER_LAW_CAP = {
    657: 18, 2577: 14, 2872: 12, 3194: 14, 4708: 14, 4721: 30, 4857: 22,
    5237: 24, 5271: 20, 5378: 10, 5510: 16, 6098: 26, 6100: 20, 6331: 14,
    6446: 14, 6502: 16, 6698: 14,
}

LAW_NAMES = {
    657: "Devlet Memurları Kanunu",
    2577: "İdari Yargılama Usulü Kanunu",
    2872: "Çevre Kanunu",
    3194: "İmar Kanunu",
    4708: "Yapı Denetimi Hakkında Kanun",
    4721: "Türk Medeni Kanunu",
    4857: "İş Kanunu",
    5237: "Türk Ceza Kanunu",
    5271: "Ceza Muhakemeleri Kanunu",
    5378: "Engelliler Hakkında Kanun",
    5510: "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    6098: "Türk Borçlar Kanunu",
    6100: "Hukuk Muhakemeleri Kanunu",
    6331: "İş Sağlığı ve Güvenliği Kanunu",
    6446: "Elektrik Piyasası Kanunu",
    6502: "Tüketicinin Korunması Hakkında Kanun",
    6698: "Kişisel Verilerin Korunması Kanunu",
}

# ---------------------------------------------------------------------------
# (c) KAVRAM AYRIMLARI — her iki kavram korpus tanımından çözümlenir;
# çözülemeyen çift düşürülür. ayrim cümlesi İKİ alıntının gerçek içeriğine
# sadık, elle yazılmış kontrast cümlesidir.
# ---------------------------------------------------------------------------
KAVRAM_AYRIMLARI = [
    # (kanun, x, y, ayrim, x_no, x_terms, y_no, y_terms) — no'lu hedef kapısı:
    # bent tanımı yoksa hedef maddede terim etrafından satır-odaklı alıntı alınır.
    {"kanun": 5378, "x": "Doğrudan ayrımcılık", "y": "Dolaylı ayrımcılık",
     "ayrim": "Doğrudan ayrımcılık engelliliği açıkça gerekçe edinen farklı muameleyken, dolaylı ayrımcılık görünüşte tarafsız eylem ve uygulamaların sonuç itibarıyla dezavantajlı konum üretmesidir."},
    {"kanun": 4857, "x": "İşveren", "y": "İşveren vekili",
     "ayrim": "İşveren işçiyi çalıştıran asıl taraftır; işveren vekili ise işveren adına hareket eden, işçilere karşı işlem ve yükümlülüklerinden doğrudan işverenin sorumlu olduğu kimsedir.",
     "x_no": 2, "x_terms": ["işveren"], "y_no": 2, "y_terms": ["işveren vekili"]},
    {"kanun": 6331, "x": "Çalışan", "y": "Çalışan temsilcisi",
     "ayrim": "Çalışan istihdam edilen gerçek kişidir; çalışan temsilcisi ise iş sağlığı ve güvenliği çalışmalarına katılma ve çalışanları temsil etmeye yetkili çalışandır."},
    {"kanun": 6698, "x": "Kişisel veri", "y": "Veri sorumlusu",
     "ayrim": "Kişisel veri kimliği belirli veya belirlenebilir kişiye ilişkin bilgiyi adlandırır; veri sorumlusu ise bu verilerin işlenme amaç ve vasıtalarını belirleyen tarafı adlandırır."},
    {"kanun": 6698, "x": "Veri sorumlusu", "y": "Veri işleyen",
     "ayrim": "Veri sorumlusu verilerin işlenme amaçlarını ve vasıtalarını belirleyen taraftır; veri işleyen ise veri sorumlusunun verdiği yetkiye dayanarak onun adına veri işleyen taraftır."},
    {"kanun": 6446, "x": "Dağıtım", "y": "İletim",
     "ayrim": "Kanun gerilim seviyesiyle ayırır: dağıtım 36 kV ve altındaki hatlar üzerinden, iletim bu seviyenin üstündeki hatlar üzerinden elektrik enerjisinin naklidir."},
    {"kanun": 6502, "x": "Tüketici", "y": "Satıcı",
     "ayrim": "Tüketici ticari veya mesleki amaçlarla değil, kendisi ve ailesi gibi yakın çevresi için hareket eden edinendir; satıcı ise mal veya hizmeti sunan ticari/mesleki taraftır."},
    {"kanun": 5237, "x": "Kast", "y": "Taksir",
     "ayrim": "Kast suçun kanuni unsurunu bilerek ve isteyerek gerçekleştirmedir; taksir ise dikkat ve özen yükümlülüğüne aykırı davranış sonucu failin istemediği sonucun meydana gelmesidir.",
     "x_no": 21, "x_terms": ["Kast"], "y_no": 22, "y_terms": ["Taksir"]},
    {"kanun": 4857, "x": "Süreksiz iş", "y": "Sürekli iş",
     "ayrim": "Ölçüt işin süresidir: nitelikleri bakımından en çok otuz iş günü süren işler süreksiz, bundan fazla devam eden işler sürekli sayılır.",
     "x_no": 10, "x_terms": ["süreksiz"], "y_no": 10, "y_terms": ["sürekli"]},
    {"kanun": 2577, "x": "İptal davası", "y": "Tam yargı davası",
     "ayrim": "İptal davası idari işlemin hukuka aykırılığının denetimi ve iptaliyle sınırlıdır; tam yargı davası ise idari eylem ve işlemlerden doğan kişisel hakların tazmini yoludur.",
     "x_no": 2, "x_terms": ["iptal davaları"], "y_no": 2, "y_terms": ["tam yargı davaları"]},
    {"kanun": 4721, "x": "Ölüm karinesi", "y": "Gaiplik",
     "ayrim": "Ölüm karinesinde kişi ölümüne kesin gözüyle bakılan durumlar içinde kaybolmuştur ve ölmüş sayılır; gaiplikte ise ölüm kesin olmayıp kuvvetli olasılık vardır ve mahkeme gaipliğe karar verir.",
     "x_no": 31, "x_terms": ["ölüm karinesi", "ölmüş sayılır"], "y_no": 32, "y_terms": ["gaipliğine karar"]},
]

# ---------------------------------------------------------------------------
# (d) PROSEDÜRLER — her adım (kanun, arama terimleri) ile çözümlenir;
# tek adım bile çözümlenemezse TÜM prosedür düşürülür.
# ---------------------------------------------------------------------------
PROSEDURLER = [
    # adım = (kanun_no, hedef_madde_no, [doğrulama terimleri], adım başlığı)
    {"ad": "İptal davası açma prosedürü (İdari Yargılama Usulü Kanunu)",
     "giris": "Bir idari işlemin iptali için idare mahkemesinde izlenecek adımlar",
     "adımlar": [
        (2577, 2, ["İdari dava türleri"], "Dava türü: iptal ve tam yargı ayrımı"),
        (2577, 7, ["Dava açma süresi", "altmış"], "Dava açma süresi: idare mahkemelerinde altmış gün"),
        (2577, 48, ["Temyiz istemleri"], "Kanun yolu: temyiz dilekçesi usulü"),
     ]},
    {"ad": "Kişisel veri güvenliği ihlali prosedürü (KVKK)",
     "giris": "Kişisel veri güvenliğinin ihlali hâlinde izlenecek adımlar",
     "adımlar": [
        (6698, 12, ["güvenliğine ilişkin yükümlülükler"], "Güvenlik tedbirleri ve ihlal hâlinde bildirim yükümlülüğü"),
        (6698, 15, ["Şikâyet üzerine veya resen inceleme"], "Kurum incelemesi: şikâyet üzerine veya resen"),
     ]},
    {"ad": "Yapı denetimi prosedürü (4708 sayılı Kanun)",
     "giris": "Yapı ruhsatı aşamasında yapı denetiminin izlediği adımlar",
     "adımlar": [
        (4708, 1, ["yapı denetimini sağlamak"], "Yapı denetim hizmetinin konusu"),
        (4708, 2, ["yapı denetim kuruluşu", "görevleri"], "Yapı denetim kuruluşlarının görevleri"),
        (4708, 5, ["hizmet sözleşmesi"], "Hizmet sözleşmesi usulü"),
     ]},
    {"ad": "Yıllık ücretli izin prosedürü (4857 sayılı İş Kanunu)",
     "giris": "İşçinin yıllık ücretli izne ilişkin izlenecek adımlar",
     "adımlar": [
        (4857, 53, ["yıllık ücretli izin"], "İzin hakkının doğması"),
        (4857, 54, ["hak kazanma"], "İzne hak kazanma ve izni kullanma dönemi"),
        (4857, 56, ["yıllık ücretli iznin"], "İznin uygulanması"),
     ]},
    {"ad": "İşe iade prosedürü (4857 sayılı İş Kanunu)",
     "giris": "Belirsiz süreli iş sözleşmesi feshinde işe iade yolu",
     "adımlar": [
        (4857, 20, ["işe iade", "bir ay içinde"], "Arabulucuya başvuru ve bir aylık süre"),
        (4857, 21, ["işe başlatmak", "tazminat"], "İşe başlatmama hâlinde tazminat"),
     ]},
    {"ad": "Tüketici hakem heyeti prosedürü (6502 sayılı Kanun)",
     "giris": "Tüketici sorunlarında hakem heyeti başvuru adımları",
     "adımlar": [
        (6502, 66, ["hakem heyetleri", "görev alanı"], "Kuruluş ve görev alanı"),
        (6502, 68, ["hakem heyetleri"], "İnceleme ve haklar"),
     ]},
    {"ad": "Çevresel Etki Değerlendirme (ÇED) süreci (2872 sayılı Çevre Kanunu)",
     "giris": "Faaliyetlerden çevreye etki öncesi izleme adımları",
     "adımlar": [
        (2872, 10, ["çevresel etki değerlendirilmesi"], "ÇED zorunluluğu"),
        (2872, 2, ["çevresel etki"], "ÇED tanımı (Tanımlar maddesi)"),
     ]},
    {"ad": "Elektrik üretim lisansı başvurusu (6446 sayılı Kanun)",
     "giris": "Elektrik üretim faaliyeti için lisans adımları",
     "adımlar": [
        (6446, 5, ["Lisans, bu Kanun"], "Lisansın tanımı ve kapsamı"),
        (6446, 6, ["lisans başvurusunda"], "Başvuru ve önlisans esasları"),
     ]},
    {"ad": "Kanunun kabulden yürürlüğe girişi (4857 sayılı İş Kanunu örneği)",
     "giris": "Kanunun yürürlük ve yürütme hükümleri",
     "adımlar": [
        (4857, 121, ["yürürlüğe girer"], "Yürürlük maddesi"),
        (4857, 122, ["Bakanlar Kurulu yürütür"], "Yürütme maddesi"),
     ]},
]

# ---------------------------------------------------------------------------
# (b) MADDE SORUSU — elle doğrulanmış hedefli sorular (terim→madde otomatik çözülür)
# ---------------------------------------------------------------------------
HEDEFLI_MADDE_SORULARI = [
    # (kanun_no, soru, [doğrulama terimleri — hedef maddede geçmeli], hedef_madde_no)
    (4857, "4857 sayılı İş Kanunu'na göre haftalık normal çalışma süresi kaç saattir?",
     ["haftada en çok kırkbeş"], 63),
    (4857, "4857 sayılı İş Kanunu'nda deneme süreli iş sözleşmesinde tarafların hakları nedir?",
     ["deneme süresi"], 15),
    (4857, "4857 sayılı İş Kanunu'na göre fesih bildirim süreleri (ihbar süreleri) nasıl kademelenir?",
     ["bildirim sürelerine"], 17),
    (4857, "4857 sayılı İş Kanunu'na göre fazla çalışmanın tanımı ve sınırı nedir?",
     ["fazla çalışma"], 41),
    (4857, "4857 sayılı İş Kanunu'nda ulusal bayram ve genel tatil günlerinde çalışma nasıl düzenlenir?",
     ["bayram ve genel tatil"], 44),
    (4857, "4857 sayılı İş Kanunu'nda yıllık ücretli izin hakkı ne zaman doğar?",
     ["yıllık ücretli izin"], 53),
    (4857, "4857 sayılı İş Kanunu'na göre asgari ücret nasıl belirlenir ve hangi dönemde yürürlüğe girer?",
     ["asgari ücret"], 39),
    (4857, "4857 sayılı İş Kanunu'nda işe iade talebi hangi yolla ve ne zaman ileri sürülür?",
     ["işe iade"], 20),
    (4857, "4857 sayılı İş Kanunu hangi tarihte yürürlüğe girmiştir?",
     ["yürürlüğe girer"], 121),
    (5237, "5237 sayılı Türk Ceza Kanunu'na göre taksirli davranış hangi koşulda cezalandırılır?",
     ["taksir", "dikkat ve özen"], 22),
    (5237, "5237 sayılı Türk Ceza Kanunu'nda meşru savunma hangi unsurları içerir?",
     ["meşru savunma"], 25),
    (5237, "5237 sayılı Türk Ceza Kanunu'nda kanunun hükmü ve amirin emri ceza sorumluluğunu nasıl etkiler?",
     ["kanunun hükmü"], 24),
    (6098, "6098 sayılı Türk Borçlar Kanunu'na göre genel (on yıllık) zamanaşımı kuralı nedir?",
     ["on yıllık zamanaşımı"], 146),
    (6098, "6098 sayılı Türk Borçlar Kanunu'nda borçlunun temerrüdü ne zaman doğar?",
     ["temerrüde düşer"], 117),
    (6331, "6331 sayılı İş Sağlığı ve Güvenliği Kanunu'na göre işverenin risk değerlendirmesi yükümlülüğü nedir?",
     ["risk değerlendirmesi"], 10),
    (6331, "6331 sayılı Kanuna göre sağlık gözetimi hangi amaçlarla yapılır?",
     ["sağlık gözetimi"], 15),
    (6698, "6698 sayılı KVKK'da kişisel verilerin açık rıza olmaksızın işlenmesi hangi şartlara bağlıdır?",
     ["işlenme şartları"], 5),
    (6698, "6698 sayılı KVKK'da veri sorumlusunun aydınlatma yükümlülüğü neleri kapsar?",
     ["aydınlatma"], 10),
    (6698, "6698 sayılı KVKK'ya göre veri güvenliğine ilişkin yükümlülükler nelerdir?",
     ["güvenliğine ilişkin yükümlülükler"], 12),
    (6502, "6502 sayılı Kanuna göre taksitle satış sözleşmesinde cayma hakkı süresi nedir?",
     ["yedi gün"], 18),
    (6502, "6502 sayılı Kanunda tüketici hakem heyetlerinin kuruluş ve görev alanı nasıldır?",
     ["hakem heyetleri", "görev alanı"], 66),
    (6446, "6446 sayılı Elektrik Piyasası Kanunu'nda lisans nedir, hangi faaliyetleri kapsar?",
     ["Lisans, bu Kanun"], 5),
    (6446, "6446 sayılı Kanunda üretim lisansı başvurusu hangi adımları izler?",
     ["lisans başvurusunda"], 6),
    (4708, "4708 sayılı Kanunda yapı denetim hizmeti hangi sözleşme ile yürütülür?",
     ["yapı denetim hizmeti"], 2),
    (3194, "3194 sayılı İmar Kanunu'na göre yapı kullanma izni ne zaman ve kim tarafından verilir?",
     ["Yapı tamamen bittiği"], 30),
    (2577, "2577 sayılı Kanuna göre idare ve vergi mahkemelerinde dava açma süresi kaştır?",
     ["Dava açma süresi", "altmış"], 7),
    (2577, "2577 sayılı Kanunda idari dava türleri (iptal ve tam yargı) nasıl ayrılır?",
     ["İdari dava türleri"], 2),
    (4721, "4721 sayılı Türk Medeni Kanunu'na göre ergin olmayanların evlenmesi hangi koşula bağlıdır?",
     ["onaltı yaşını"], 124),
    (6100, "6100 sayılı HMK'ya göre dava dilekçesinde hangi hususlar bulunur?",
     ["Dava dilekçesinin içeriği"], 119),
    (6100, "6100 sayılı HMK'ya göre ispat yükü kuralı nasıldır?",
     ["İspat yükü, kanunda"], 190),
    (657, "657 sayılı Devlet Memurları Kanunu'nda aday memurun adaylık devresi nasıl düzenlenir?",
     ["adaylık"], 56),
    (5271, "5271 sayılı CMK'ya göre arama kararı hangi koşul ve usulde verilebilir?",
     ["arama"], 119),
    (2872, "2872 sayılı Çevre Kanunu'na göre çevresel etki değerlendirmesi hangi faaliyetler için zorunludur?",
     ["çevresel etki değerlendirilmesi"], 10),
    (5378, "5378 sayılı Kanunda engellilerin istihdamı hangi esaslarla korunur?",
     ["istihdam"], 14),
    (5510, "5510 sayılı Kanunda işveren ve işveren vekili nasıl tanımlanır?",
     ["hizmet akdiyle"], 12),
]

B_SABLONLARI = [
    "Mevzuat uzmanı: {kanun}'da '{baslik}' hükmü ne der?",
    "Hukuk danışmanı olarak: {kanun} çerçevesinde '{baslik}' düzenlemesini açıklar mısın?",
    "Mevzuat sorusu: {kanun}'nda '{baslik}' hangi maddede ve hükmü nedir?",
    "{kanun} metnine göre '{baslik}' neyi öngörür?",
    "Kanun hükmü sorusu: {kanun} '{baslik}' başlıklı düzenlemeyi nasıl taşır?",
]
A_SABLONLARI = [
    "Mevzuat tanım sorusu: {kanun}'nda '{kavram}' nasıl tanımlanmıştır?",
    "Hukuk kavramı: '{kavram}' kavramı {kanun}'da hangi tanımla geçer?",
    "Mevzuat uzmanı: {kanun} uyarınca '{kavram}' ne anlama gelir?",
    "{kanun} tanımlarına göre '{kavram}' kim/ne olarak tanımlanmıştır?",
]
HEDEFLI_SABLON = "{soru}"  # hedefli sorular hazır cümledir


# ---------------------------------------------------------------------------
# Korpus kayıt defteri
# ---------------------------------------------------------------------------
class Madde:
    __slots__ = ("law", "label", "no", "gecici", "baslik", "lines")

    def __init__(self, law, label, no, gecici):
        self.law = law
        self.label = label          # 'Madde 63 -' kurallı etiket
        self.no = no
        self.gecici = gecici
        self.baslik = ""
        self.lines = []             # gövde satırları (madde başlığı hariç)

    @property
    def text(self):
        return " ".join(l.strip() for l in self.lines if l.strip())

    def is_mulga(self):
        head = self.lines[0] if self.lines else ""
        return bool(MULGA_MARK.match(head)) or (
            len(self.text) < 40 and "mülga" in self.text.lower()
        )

    def clean_quote(self, cap):
        """Madde gövdesinden açıklama-dipnotları atılmış alıntı kurar."""
        parts, size = [], 0
        for ln in self.lines:
            ln = ln.strip()
            if not ln or ANNOT_LINE.match(ln):
                continue
            ln = INLINE_ANNOT.sub(" ", ln)
            ln = re.sub(r"^\(\d+\)\s*", "", ln)  # fıkra numarası
            ln = re.sub(r"\s+", " ", ln).strip()
            if not ln:
                continue
            parts.append(ln)
            size += len(ln) + 1
            if size >= cap:
                break
        q = " ".join(parts)
        if len(q) > cap:
            cut = q.rfind(". ", DEF_CAP - 120, cap)
            q = (q[: cut + 1] + "…") if cut > 0 else q[: cap - 1] + "…"
        if q.endswith(".…"):
            q = q[:-1]  # cümle zaten nokta ile bitmiş: üç noktayı düşür
        return q.strip()

    def quote_around(self, term, cap=500):
        """Terimin geçtiği satırdan başlayan, dipnot-temiz alıntı kurar."""
        key = term.lower()
        for i, ln in enumerate(self.lines):
            if key in ln.lower():
                parts, size = [], 0
                for l2 in self.lines[i:]:
                    l2 = l2.strip()
                    if not l2 or ANNOT_LINE.match(l2):
                        continue
                    l2 = INLINE_ANNOT.sub(" ", l2)
                    l2 = re.sub(r"^\(\d+\)\s*", "", l2)
                    l2 = re.sub(r"\s+", " ", l2).strip()
                    if not l2:
                        continue
                    parts.append(l2)
                    size += len(l2) + 1
                    if size >= cap:
                        break
                q = " ".join(parts)
                if len(q) > cap:
                    cut = q.rfind(". ", 120, cap)
                    q = (q[: cut + 1] + "…") if cut > 0 else q[: cap - 1] + "…"
                if q.endswith(".…"):
                    q = q[:-1]
                return q
        return self.clean_quote(cap)


def load_corpus():
    laws = {}
    for f in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt"))):
        no = int(os.path.basename(f)[:4])
        raw = open(f, encoding="utf-8").read()
        lines = [l.rstrip() for l in raw.split("\n")]
        mads = []
        cur = None
        prev = ""
        for ln in lines:
            if ln.startswith("#"):
                continue
            m = MADDE_HEAD.match(ln)
            if m:
                # Etiket yalnız kurallı başlık ('Madde 63 -'); başlık satırındaki
                # gövde metni m.group(3) ilk satır olarak Madde'ye yazılır.
                prefix = (m.group(1) or "").strip().title()  # 'Geçici' / 'Ek'
                label = f"{prefix + ' ' if prefix else ''}Madde {int(m.group(2))} -"
                cur = Madde(no, label, int(m.group(2)), bool(m.group(1)))
                head_text = m.group(3).strip()
                if head_text:
                    cur.lines.append(head_text)
                # başlık: önceki satır kısa, temiz, madde değilse
                if (prev and len(prev) < 90 and not MADDE_HEAD.match(prev)
                        and re.match(r"^[A-ZÇĞİÖŞÜa-zçğıöşüâ0-9\s.,;:()/'-]+$", prev)):
                    cur.baslik = prev.strip()
                mads.append(cur)
            elif cur is not None:
                cur.lines.append(ln)
            prev = ln
        laws[no] = mads
    return laws


def find_madde(laws, law_no, terms, must=None):
    """Terim listesinin tümünü barındıran en iyi maddeyi bulur (terim sıklığı skoru)."""
    cands = laws.get(law_no, [])
    best, best_score = None, -1.0
    for m in cands:
        t = m.text
        low = t.lower()
        if all(term.lower() in low for term in terms):
            score = sum(low.count(term.lower()) for term in terms)
            if m.baslik and any(term.lower() in m.baslik.lower() for term in terms):
                score += 1.0
            if score > best_score:
                best, best_score = m, score
    return best


def find_definition(laws, law_no, kavram):
    """'Kavram:' bentli tanımı, tanım maddeleri içinde arar."""
    key = kavram.lower()
    for m in laws.get(law_no, []):
        for ln in m.lines:
            mm = BENT_RE.match(ln.strip())
            if mm and mm.group(2).strip().lower() == key:
                return m, mm.group(2).strip(), mm.group(3).strip()
    return None, None, None


def bent_definitions(laws):
    """(kanun, madde, kavram, tanım, ham_satırlar) bent tanımlarını toplar.
    ham_satırlar: korpus satırları birebir (bent öneki ayıklanmış) — alıntı bunlardan kurulur."""
    out = []
    BENT_PREFIX = re.compile(r"^\(?[a-zçğıöşüâ]{1,2}\)\s*")
    for no, mads in laws.items():
        for m in mads:
            cur_def = None
            for ln in m.lines:
                s = ln.strip()
                mm = BENT_RE.match(s)
                if mm:
                    if cur_def:
                        out.append(cur_def)
                    kavram, tanim = mm.group(2).strip(), mm.group(3).strip()
                    raw = BENT_PREFIX.sub("", s, count=1).strip()
                    cur_def = [no, m, kavram, tanim, [raw]]
                elif cur_def is not None:
                    if re.match(r"^[a-zçğıöşüâ]{1,2}\)", s) and not mm:
                        if cur_def:
                            out.append(cur_def)
                        cur_def = None
                        continue
                    if (not ANNOT_LINE.match(s) and len(s) > 2
                            and len(cur_def[3]) < DEF_CAP):
                        cur_def[3] = cur_def[3] + " " + s
                        cur_def[4].append(s)
            if cur_def:
                out.append(cur_def)
    return out


# ---------------------------------------------------------------------------
# Kayıt kurucular
# ---------------------------------------------------------------------------
def rec_a(no, name, m, kavram, tanim, idx, raw_quote=""):
    sablon = A_SABLONLARI[idx % len(A_SABLONLARI)]
    if raw_quote:  # korpus ham satırı — birebir alıntı
        tanim_q = raw_quote[:DEF_CAP]
        if len(raw_quote) > DEF_CAP:
            cut = tanim_q.rfind(",", DEF_CAP - 120, DEF_CAP)
            tanim_q = ((raw_quote[: cut + 1] + "…") if cut > 0
                       else raw_quote[: DEF_CAP - 1] + "…")
    else:
        tanim_q = tanim[:DEF_CAP]
        if len(tanim) > DEF_CAP:
            cut = tanim_q.rfind(",", DEF_CAP - 100, DEF_CAP)
            tanim_q = (tanim[: cut + 1] + "…") if cut > 0 else tanim[: DEF_CAP - 1] + "…"
    instruction = sablon.format(kanun=name, kavram=kavram)
    quote_body = tanim_q if raw_quote else f"{kavram}: {tanim_q}"
    output = (
        f"Kanun Tanımı: {kavram} — {name} (Kanun No. {no}), {m.label.rstrip(' -')}: "
        f"«{quote_body}»"
    )
    return {"instruction": instruction, "input": "", "output": output}


def rec_b(no, name, m, idx):
    sablon = B_SABLONLARI[idx % len(B_SABLONLARI)]
    quote = m.clean_quote(QUOTE_CAP)
    instruction = sablon.format(kanun=name, baslik=m.baslik)
    output = f"Mevzuat Hükmü — {name} (Kanun No. {no}), {m.label.rstrip(' -')} ({m.baslik}): «{quote}»"
    return {"instruction": instruction, "input": "", "output": output}


def rec_b_hedefli(no, name, m, soru):
    quote = m.clean_quote(QUOTE_CAP)
    output = (
        f"Mevzuat Yanıtı — {name} (Kanun No. {no}), {m.label.rstrip(' -')}: "
        f"«{quote}» Kaynak madde: {m.label.rstrip(' -')}."
    )
    return {"instruction": soru, "input": "", "output": output}


def rec_c(no_name, kanun_no, mx, kx, dx, my, ky, dy, ayrim):
    name = no_name
    (nx, nn) = (mx.label, mx.law)
    out = (
        f"Kavram Ayrımı: (1) {kx} — {LAW_NAMES.get(mx.law, '')}, {mx.label.rstrip(' -')}: «{dx[:DEF_CAP]}» "
        f"(2) {ky} — {LAW_NAMES.get(my.law, '')}, {my.label.rstrip(' -')}: «{dy[:DEF_CAP]}». "
        f"Ayrım: {ayrim}"
    )
    ins = (
        f"Hukuk kavram ayrımı: '{kx}' ile '{ky}' arasındaki farkı ilgili kanun "
        f"tanımlarıyla açıkla."
    )
    return {"instruction": ins, "input": "", "output": out}


def rec_d(prosedur, steps):
    lines = [f"Prosedür — {prosedur['ad']}: {prosedur['giris']}."]
    for i, (baslik, m, quote) in enumerate(steps, 1):
        lines.append(
            f"{i}) {baslik} — {LAW_NAMES.get(m.law, '')}, {m.label.rstrip(' -')}: «{quote}»"
        )
    ins = f"Yordam sorusu: {prosedur['giris']} için mevzuattaki adımları madde alıntılarıyla sırala."
    return {"instruction": ins, "input": "", "output": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Doğrulama kapıları
# ---------------------------------------------------------------------------
def validate_record(r, drops, i):
    try:
        if set(r.keys()) != {"instruction", "input", "output"}:
            return False
        if not all(isinstance(v, str) for v in r.values()):
            return False
        if not r["instruction"].strip() or not r["output"].strip():
            return False
        if len(r["instruction"]) > 400 or len(r["output"]) > 2600:
            return False
        if "Madde" not in r["output"] and "Kanun No." not in r["output"]:
            return False
        # alıntı cümleleri korpus kökenli mi: en az bir «» bloğu ve madde etiketi
        if "«" not in r["output"] or "»" not in r["output"]:
            return False
        return True
    except Exception as e:  # noqa: BLE001
        drops.append((i, str(e)))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print(" P2.5 FAZ-1 — LEGAL CANONICAL SFT ÜRETİMİ (marangoz kart kalıbı)")
    print("=" * 72)

    laws = load_corpus()
    print(f"Korpus: {len(laws)} kanun, "
          f"{sum(len(v) for v in laws.values()):,} madde satırı")
    drops = []
    records = []
    per_type = Counter()

    # ---- (a) Kavram tanımları -------------------------------------------------
    defs = bent_definitions(laws)
    print(f"(a) Bent tanımı bulundu: {len(defs)}")
    kept_defs = []
    for (no, m, kavram, tanim, raw_lines) in defs:
        if len(kavram) < 3 or len(tanim) < 30:
            continue
        if len(kavram) > 48 or len(kavram.split()) > 5:
            continue  # bent değil, madde-içi sayıltı liste başlığı
        if m.is_mulga():
            continue
        if "yerine işlenmiştir" in tanim and len(tanim) < 400:
            continue  # mezar-taşı bent
        # kapı: ham satır maddede birebir durmalı (kurulum gereği doğrulama)
        if raw_lines[0] not in m.text:
            continue
        kept_defs.append((no, m, kavram, tanim, raw_lines))
    # kanun başına tanım dengesi (tanım maddeleri şişmesin)
    seen_kavram = set()
    def_by_law = defaultdict(int)
    for (no, m, kavram, tanim, raw_lines) in kept_defs:
        key = (no, kavram.lower())
        if key in seen_kavram:
            continue
        if def_by_law[no] >= 28:
            continue
        seen_kavram.add(key)
        def_by_law[no] += 1
        records.append(rec_a(no, LAW_NAMES[no], m, kavram, tanim, len(records),
                             " ".join(raw_lines)))
        per_type["a_kavram_tanimi"] += 1
    print(f"(a) kayıt: {per_type['a_kavram_tanimi']}")

    # ---- (b1) Başlıklı madde soruları ----------------------------------------
    for no, mads in laws.items():
        eligible = [m for m in mads
                    if not m.gecici and m.baslik and not m.is_mulga()
                    and len(m.text) >= MIN_MADDE_CHARS
                    and m.text.count("yerine işlenmiştir") < 2
                    and not ("yerine işlenmiştir" in m.text and len(m.text) < 400)
                    and not m.baslik.lower().startswith(("birinci", "ikinci"))
                    and m.baslik.lower() not in
                    ("genel hükümler", "kapsam", "amaç", "yürürlük", "kabul")]
        cap = PER_LAW_CAP.get(no, 12)
        if len(eligible) > cap:
            step = len(eligible) / cap
            eligible = [eligible[int(i * step)] for i in range(cap)]
        for m in eligible:
            records.append(rec_b(no, LAW_NAMES[no], m, len(records)))
            per_type["b_madde_sorusu"] += 1
    print(f"(b) kayıt (başlıklı madde): {per_type['b_madde_sorusu']}")

    # ---- (b2) Hedefli madde soruları (hedef madde + terim doğrulama kapısı) ----
    def resolve_target(no, target_no, terms):
        """Kayıt defterinden hedef maddeyi bulur, terimlerin varlığını doğrular."""
        for m in laws.get(no, []):
            if m.no == target_no and not m.gecici:
                hay = (m.text + " " + m.baslik).lower()
                if all(t.lower() in hay for t in terms):
                    return m
                return None  # hedef madde var ama terim doğrulanamadı → kapı
        return None

    for (no, soru, terms, target_no) in HEDEFLI_MADDE_SORULARI:
        m = resolve_target(no, target_no, terms)
        if m is None:
            m = find_madde(laws, no, terms)  # terim-aramalı yedek
        if m is None or m.is_mulga() or len(m.text) < 40:
            drops.append((no, f"hedefli soru çözülmedi: {soru[:60]}"))
            continue
        records.append(rec_b_hedefli(no, LAW_NAMES[no], m, soru))
        per_type["b_hedefli_soru"] += 1
    print(f"(b) kayıt (hedefli soru): {per_type['b_hedefli_soru']}")

    # ---- (c) Kavram ayrımları -------------------------------------------------
    for pair in KAVRAM_AYRIMLARI:
        if pair.get("ayrim") is None:
            drops.append((pair["kanun"], f"ayrım cümlesi yok, atlandı: {pair['x']}"))
            continue
        mx, kx, dx = find_definition(laws, pair["kanun"], pair["x"])
        my, ky, dy = find_definition(laws, pair["kanun"], pair["y"])
        if mx is not None:
            dx = f"{kx}: {dx}"  # bent alıntısı korpus biçimi: 'Kavram: tanım'
        if my is not None:
            dy = f"{ky}: {dy}"
        if mx is None and pair.get("x_no"):
            m2 = resolve_target(pair["kanun"], pair["x_no"], pair.get("x_terms", [pair["x"]]))
            if m2 is not None:
                mx, kx = m2, pair["x"]
                dx = m2.quote_around(pair.get("x_terms", [pair["x"]])[0], DEF_CAP)
        if my is None and pair.get("y_no"):
            m2 = resolve_target(pair["kanun"], pair["y_no"], pair.get("y_terms", [pair["y"]]))
            if m2 is not None:
                my, ky = m2, pair["y"]
                dy = m2.quote_around(pair.get("y_terms", [pair["y"]])[0], DEF_CAP)
        if mx is None or my is None or not dx or not dy:
            drops.append((pair["kanun"], f"kavram ayrımı çözülmedi: {pair['x']} / {pair['y']}"))
            continue
        dx = dx[:DEF_CAP] + ("…" if len(dx) > DEF_CAP else "")
        dy = dy[:DEF_CAP] + ("…" if len(dy) > DEF_CAP else "")
        records.append(rec_c(LAW_NAMES[pair["kanun"]], pair["kanun"], mx, kx, dx,
                              my, ky, dy, pair["ayrim"]))
        per_type["c_kavram_ayrimi"] += 1
    print(f"(c) kayıt: {per_type['c_kavram_ayrimi']}")

    # ---- (d) Prosedürler ------------------------------------------------------
    for p in PROSEDURLER:
        steps = []
        ok = True
        for (no, target_no, terms, baslik) in p["adımlar"]:
            m = resolve_target(no, target_no, terms)
            if m is None:
                m = find_madde(laws, no, terms)
            if m is None:
                ok = False
                drops.append((no, f"prosedür adımı çözülmedi: {baslik[:50]}"))
                break
            quote = m.quote_around(terms[0], 320)
            if len(quote) < 40:
                ok = False
                break
            steps.append((baslik, m, quote))
        if ok and len(steps) >= 2:
            records.append(rec_d(p, steps))
            per_type["d_prosedur"] += 1
    print(f"(d) kayıt: {per_type['d_prosedur']}")

    # ---- Şema doğrulama + dedupe ----------------------------------------------
    valid, seen = [], set()
    for i, r in enumerate(records):
        if not validate_record(r, drops, i):
            drops.append((i, "şema dışı kayıt"))
            continue
        h = hashlib.sha256((r["instruction"] + "||" + r["output"]).encode()).hexdigest()
        if h in seen:
            per_type["dup"] = per_type.get("dup", 0) + 1
            continue
        seen.add(h)
        valid.append(r)
    print(f"Şema sonrası: {len(valid)} kayıt (dedupe düşüş: {per_type.get('dup', 0)})")

    # ---- Token/UNK ölçümü ------------------------------------------------------
    print("Token ölçümü (anka 33.114, literal_entity_mode=True)…")
    vocab = Vocabulary()
    vocab.load(VOCAB_PATH)
    lexicon = LexiconManager()
    lexicon.load_from_tsv(os.path.join(REPO, "data", "lexicon", "roots.tsv"))
    tok = KristalTokenizer(
        CrystalCompiler(lexicon, build_default_graph()), vocab,
        literal_entity_mode=True, verbose=False,
    )
    unk_id = vocab.stoi["<UNK>"]
    tot_tok = unk_tok = 0
    len_hist = []
    for r in valid:
        ids = tok.encode(json.dumps(r, ensure_ascii=False))
        tot_tok += len(ids)
        unk_tok += sum(1 for i in ids if i == unk_id)
        len_hist.append(len(ids))
    print(f"Toplam token: {tot_tok:,}  UNK: {unk_tok:,} (%{100*unk_tok/max(1,tot_tok):.2f})  "
          f"ort. kayıt: {tot_tok/max(1,len(valid)):.0f} token")

    meta = {
        "uretim_tarihi": time.strftime("%Y-%m-%d %H:%M"),
        "kaynak_korpus": "data/legal_korpus/ (17 kanun, mevzuat.gov.tr)",
        "vocab": os.path.basename(VOCAB_PATH),
        "literal_entity_mode": True,
        "kayit_sayisi": len(valid),
        "tip_dagilimi": dict(per_type),
        "token_olcumu": {
            "toplam_token": tot_tok,
            "unk_token": unk_tok,
            "unk_yuzde": round(100 * unk_tok / max(1, tot_tok), 3),
            "ortalama_kayit_token": round(tot_tok / max(1, len(valid)), 1),
            "maks_kayit_token": max(len_hist) if len_hist else 0,
        },
        "dusenler": drops,
    }

    if args.dry_run:
        print("[dry-run] yazım yapılmadı.")
        for r in valid[:6]:
            print("ÖRNEK:", json.dumps(r, ensure_ascii=False)[:220])
        return

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"Yazıldı: {OUT_JSONL} ({len(valid)} kayıt)")
    print(f"Meta: {OUT_META}")
    print("=" * 72)


if __name__ == "__main__":
    main()