#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: TEMEL LİSE EĞİTİMİ VERİ SETİ OLUŞTURUCU
===================================================================
Bu betik; lise müfredatının temel derslerini (Türk Dili ve Edebiyatı,
Fizik, Kimya, Biyoloji, Tarih, Coğrafya, Matematik & Mantık) kapsayan
yüksek kaliteli, pedagojik SFT soru-cevap veri setini oluşturur.
Çıktı: `data/pedagogy/high_school_foundation_dataset.jsonl`
"""

import os
import json
import random

# =====================================================================
# LİSE MÜFREDAT KÜTÜPHANESİ
# =====================================================================

LITERATURE_ITEMS = [
    # Edebi Sanatlar
    ("Teşbih nedir?", "Teşbih, aralarında benzerlik ilgisi kurulan iki varlıktan nitelikçe zayıf olanın güçlü olana benzetilmesidir. Dört ögesi vardır: benzeyen, kendisine benzetilen, benzetme yönü ve benzetme edatı."),
    ("Teşhis ve intak sanatı arasındaki fark nedir?", "Teşhis insan dışındaki canlı veya cansız varlıklara insani özellikler verme sanatıdır. İntak ise teşhis yapılan bu varlıkların konuşturulması sanatıdır. Her intakta bir teşhis bulunur ancak her teşhiste intak bulunmaz."),
    ("Mecaz-ı mürsel ne demektir?", "Mecaz-ı mürsel (ad aktarması), bir sözcüğün benzetme amacı güdülmeden, parça-bütün, iç-dış, yazar-eser gibi bir ilgiyle başka bir sözcük yerine kullanılmasıdır."),
    ("Tezat sanatı nedir?", "Tezat sanatı, aralarında anlamca karşıtlık bulunan iki duygu, düşünce veya durumun aynı ifadede karşıtlık ilişkisiyle bir arada kullanılmasıdır."),
    # Metin Türleri
    ("Makale türünün temel özellikleri nelerdir?", "Makale; belirli bir konuda bilgi vermek, bir tezi savunmak veya bir gerçeği kanıtlamak amacıyla nesnel, bilimsel ve ciddi bir üslupla yazılan gazete ve dergi yazısıdır."),
    ("Deneme türü nasıl tanımlanır?", "Deneme, yazarın herhangi bir konu üzerindeki kişisel görüş ve düşüncelerini kesin kurallara ve kanıtlama zorunluluğuna bağlı kalmadan samimi bir dille anlattığı düzyazı türüdür."),
    ("Trajedi ve komedi arasındaki temel farklar nelerdir?", "Trajedi, hayatın acıklı ve korkunç yönlerini ele alıp seyircide acıma ve korku duyguları uyandırmayı hedeflerken; komedi, insanların ve toplumun gülünç yönlerini ele alarak düşündürmeyi hedefler."),
    # Şiir Bilgisi
    ("Redif ile kafiye arasındaki fark nedir?", "Kafiye (uyak), dize sonlarındaki farklı anlam ve görevdeki ses benzerlikleridir. Redif ise kafiyeden sonra gelen, aynı görev ve anlamdaki ekler veya tekrarlanan aynı sözcüklerdir."),
    ("Zengin kafiye nedir?", "Dize sonlarında üç veya daha fazla ses benzerliğine dayanan kafiye türüne zengin kafiye denir."),
    # Cümle Bilgisi
    ("Sıralı cümle nedir?", "Sıralı cümle, birden fazla bağımsız yargının birbirine virgül ya da noktalı virgülle bağlandığı cümle yapısıdır. Ögeleri ortak olursa bağımlı sıralı, olmazsa bağımsız sıralı cümle denir."),
    ("Birleşik cümle nasıl oluşur?", "Bir temel cümle ile ona bağlı bir veya birden fazla yan cümlecikten (fiilimsi, şart, ki'li veya iç içe) oluşan cümlelere birleşik cümle denir.")
]

PHYSICS_ITEMS = [
    # Newton Yasaları
    ("Newton'ın eylemsizlik yasası nedir?", "Eylemsizlik yasası (1. Yasa); bir cisim üzerine etki eden net kuvvet sıfır ise cismin mevcut hareket durumunu korumasıdır. Cisim duruyorsa durmaya, hareket ediyorsa sabit hızla doğrusal hareketine devam eder."),
    ("Newton'ın temel yasası F=ma neyi ifade eder?", "Temel yasa (2. Yasa); bir cisme etki eden net kuvvetin, cismin kütlesi ile ivmesinin çarpımına eşit olduğunu ifade eder. Net kuvvet arttıkça cismin ivmesi doğru orantılı olarak artar."),
    ("Etki-tepki yasası ne anlama gelir?", "Etki-tepki yasası (3. Yasa); bir cisim diğer bir cisme kuvvet uyguladığında, ikinci cismin de birinci cisme eşit büyüklükte ve zıt yönde bir kuvvet uygulamasıdır."),
    # Enerji & İş
    ("Fiziksel anlamda iş nasıl tanımlanır?", "Fiziksel iş, bir cisme kuvvet uygulanması ve cismin uygulanan kuvvet doğrultusunda yer değiştirmesi sonucu gerçekleşir. İş, kuvvet ile yer değiştirmenin skaler çarpımıdır ve birimi joule'dür."),
    ("Kinetik enerji ile potansiyel enerji arasındaki fark nedir?", "Kinetik enerji bir cismin hareketinden (hızından) dolayı sahip olduğu enerjidir. Potansiyel enerji ise cismin konumundan veya esneklik durumundan dolayı depoladığı enerjidir."),
    ("Mekanik enerjinin korunumu ne demektir?", "Sürtünmesiz bir ortamda dışarıdan sisteme net bir kuvvet etki etmediğinde, kinetik ve potansiyel enerjilerin toplamı olan mekanik enerji sabit kalır ve birbirine dönüşebilir."),
    # Isı ve Sıcaklık
    ("Isı ile sıcaklık arasındaki fark nedir?", "Sıcaklık, bir maddedeki moleküllerin ortalama kinetik enerjisinin bir göstergesidir ve termometre ile ölçülür. Isı ise sıcaklık farkından dolayı bir maddeden diğerine aktarılan toplam termal enerjidir ve kalorimetre ile hesaplanır.")
]

CHEMISTRY_ITEMS = [
    # Atom ve Periyodik Sistem
    ("Atomun temel parçacıkları nelerdir?", "Atomun temel parçacıkları; çekirdekte bulunan pozitif yüklü protonlar, yüksüz nötronlar ve çekirdeğin etrafındaki yörüngelerde dolanan negatif yüklü elektronlardır."),
    ("Periyodik tabloda grup ve periyot neyi gösterir?", "Periyodik tabloda yatay sıralara periyot denir ve atomun katman (yörünge) sayısını gösterir. Düşey sütunlara grup denir ve benzer kimyasal özellik gösteren elementleri gruplar."),
    ("Metaller ile ametaller arasındaki temel farklar nelerdir?", "Metaller elektron verme eğiliminde olup ısı ve elektriği iyi iletir, tel ve levha haline gelebilir. Ametaller ise elektron alma eğilimindedir, elektriği iletmezler ve kırılgandırlar."),
    # Kimyasal Bağlar
    ("İyonik bağ nasıl oluşur?", "İyonik bağ, metal atomunun elektron vererek pozitif yüklü katyon, ametal atomunun elektron alarak negatif yüklü anyon oluşturması ve aralarındaki elektrostatik çekim kuvvetiyle meydana gelir."),
    ("Kovalent bağ nedir?", "Kovalent bağ, genellikle ametal atomları arasında elektronların ortaklaşa kullanılması sonucu oluşan güçlü bir kimyasal bağ türüdür."),
    # Asit ve Bazlar
    ("Asitlerin ve bazların temel özellikleri nelerdir?", "Asitler suda çözündüğünde H+ iyonu verir, tatları ekşidir ve turnusol kağıdını kırmızıya çevirir (pH < 7). Bazlar ise OH- iyonu verir, tatları acıdır, ele kayganlık hissi verir ve turnusolu maviye çevirir (pH > 7).")
]

BIOLOGY_ITEMS = [
    # Hücre ve Organeller
    ("Hücre teorisinin temel ilkeleri nelerdir?", "Tüm canlılar bir ya da birden fazla hücreden oluşur. Hücre canlılığın en temel yapısal ve işlevsel birimidir. Yeni hücreler var olan hücrelerin bölünmesiyle oluşur."),
    ("Mitokondri ve kloroplastın görevleri nelerdir?", "Mitokondri, oksijenli solunum yaparak hücrenin ihtiyacı olan ATP enerjisini üretir. Kloroplast ise bitki hücrelerinde fotosentez yaparak ışık enerjisini kimyasal bağ enerjisine (besine) dönüştürür."),
    ("DNA ile RNA arasındaki farklar nelerdir?", "DNA çift zincirli olup genetik bilgiyi taşır, deoksiriboz şekeri ve timin bazı içerir. RNA ise tek zincirlidir, protein sentezinde görev alır, riboz şekeri ve urasil bazı içerir."),
    # Hücre Bölünmeleri
    ("Mitoz bölünme ile mayoz bölünme arasındaki farklar nelerdir?", "Mitoz bölünme vücut hücrelerinde görülür, kromozom sayısı değişmez, 2 özdeş hücre oluşur ve büyüme-onarımı sağlar. Mayoz bölünme üreme ana hücrelerinde görülür, kromozom sayısı yarıya iner, 4 genetik olarak farklı hücre oluşur ve çeşitliliği sağlar."),
    # Canlıların Ortak Özellikleri
    ("Canlıların ortak özellikleri nelerdir?", "Hücresel yapı, beslenme, hücresel solunum, boşaltım, hareket, uyarılara tepki, metabolizma, homeostazi (iç denge), uyum, üreme ve büyüme-gelişme tüm canlıların ortak özellikleridir.")
]

HISTORY_ITEMS = [
    ("Orta Asya ilk Türk devletlerinde töre nedir?", "Töre, ilk Türk devletlerinde sosyal hayatı, adaleti, ahlakı ve devlet yönetimini düzenleyen, değişmez temel prensipleri olan yazısız hukuk kuralları bütünüdür."),
    ("Metehan'ın Türk ve dünya askeri tarihine en önemli katkısı nedir?", "Metehan, Türk ordusunu onluk sisteme (onbaşı, yüzbaşı, binbaşı, tümen) göre teşkilatlandırmış ve bu hiyerarşik yapı günümüz modern dünya ordularının temelini oluşturmuştur."),
    ("Malazgirt Meydan Muharebesi'nin Türk tarihindeki önemi nedir?", "1071 Malazgirt Zaferi ile Bizans İmparatorluğu mağlup edilmiş ve Anadolu'nun kapıları Türklere kesin olarak açılarak Anadolu Türk yurdu haline gelmiştir."),
    ("Osmanlı Devleti'nde tımar sisteminin faydaları nelerdir?", "Tımar sistemi; toprağın sürekli işlenmesini sağlamış, devlet kasasından para çıkmadan büyük bir sipahi ordusu beslenmesini temin etmiş ve kırsal güvenliği sağlamıştır."),
    ("Lozan Barış Antlaşması'nın Türk milleti için önemi nedir?", "Lozan Barış Antlaşması, Türkiye Cumhuriyeti'nin bağımsızlığını ve sınırlarını uluslararası alanda tescilleyen, kapitülasyonları kaldıran kurucu antlaşmadır.")
]

GEOGRAPHY_ITEMS = [
    ("Türkiye'nin matematiksel (mutlak) konumu nedir?", "Türkiye, 36°-42° Kuzey paralelleri ile 26°-45° Doğu meridyenleri arasında yer alır. Kuzey Yarımküre'de ve ılıman kuşakta bulunur."),
    ("Türkiye'de görülen üç temel iklim tipi hangileridir?", "Türkiye'de yazları sıcak ve kurak, kışları ılık ve yağışlı olan Akdeniz iklimi; her mevsim yağışlı ve ılıman olan Karadeniz iklimi; yazları sıcak ve kurak, kışları soğuk ve kar yağışlı olan Karasal iklim görülür."),
    ("Fiziki haritalarda renklerin anlamı nedir?", "Fiziki haritalarda renkler yükselti basamaklarını gösterir. Yeşil deniz seviyesine yakın alçak yerleri (0-500 m), sarı orta yükseltileri (500-1000 m), kahverengi yüksek dağlık bölgeleri (1500 m üzeri) temsil eder."),
    ("Türkiye'de heyelan olayının en çok Karadeniz bölgesinde görülmesinin sebepleri nelerdir?", "Karadeniz bölgesinde bol yağış, dik ve engebeli yamaçlar ve killi toprak yapısının bir arada bulunması heyelan riskini en yüksek seviyeye çıkarır.")
]

MATH_LOGIC_ITEMS = [
    ("Mantıkta önerme ne demektir?", "Doğru ya da yanlış kesin bir hüküm bildiren ifadelere önerme denir. Soru, istek, ünlem veya emir cümleleri bir hüküm bildirmediği için önerme kabul edilmez."),
    ("Koşullu önermede (p ise q) doğruluk değeri ne zaman sıfır olur?", "Koşullu önerme yalnızca birinci önermenin doğru (1) ve ikinci önermenin yanlış (0) olduğu durumda yanlış (0) değerini alır. Diğer tüm durumlarda doğrudur (1)."),
    ("Fonksiyon kavramı matematikte nasıl tanımlanır?", "A kümesinden B kümesine tanımlı bir bağıntıda, A kümesindeki her bir elemanın B kümesindeki yalnız ve yalnız bir elemanla eşleşmesi kuralına fonksiyon denir."),
    ("Doğru orantı ile ters orantı arasındaki fark nedir?", "Doğru orantıda iki çokluktan biri artarken diğeri de aynı oranda artar (oranları sabittir). Ters orantıda ise iki çokluktan biri artarken diğeri aynı oranda azalır (çarpımları sabittir).")
]


def generate_high_school_dataset(output_path: str = "data/pedagogy/high_school_foundation_dataset.jsonl", samples_multiplier: int = 5):
    """
    Generates structured instruction-tuning dataset covering core high school subjects.
    Uses diversified phrasing and instruction templates.
    """
    print("=" * 65)
    print(" TEMEL LİSE EĞİTİMİ (HIGH SCHOOL FOUNDATION) VERİ SETİ GENERATÖRÜ")
    print("=" * 65)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset = []

    domain_configs = [
        ("Lise edebiyat dersi kapsamında açıkla.", LITERATURE_ITEMS),
        ("Lise fizik dersi kapsamında açıkla.", PHYSICS_ITEMS),
        ("Lise kimya dersi kapsamında açıkla.", CHEMISTRY_ITEMS),
        ("Lise biyoloji dersi kapsamında açıkla.", BIOLOGY_ITEMS),
        ("Lise tarih dersi kapsamında açıkla.", HISTORY_ITEMS),
        ("Lise coğrafya dersi kapsamında açıkla.", GEOGRAPHY_ITEMS),
        ("Lise matematik ve mantık dersi kapsamında açıkla.", MATH_LOGIC_ITEMS),
    ]

    instruction_variants = [
        "Lise seviyesinde açıkla ve temel kavramı belirt.",
        "Kavramı tanımla ve temel özelliğini belirt.",
        "Öğrencinin anlayacağı şekilde pedagojik ve net olarak cevapla.",
        "Konuyu lise müfredatına uygun bilimsel doğrulukla açıkla."
    ]

    total_base_items = sum(len(items) for _, items in domain_configs)
    print(f"Tanımlı Temel Konu Sayısı: {total_base_items}")

    for domain_inst, items in domain_configs:
        for q, a in items:
            # 1. Direct Domain Instruction
            dataset.append({
                "instruction": domain_inst,
                "input": q,
                "output": a
            })

            # 2. Diversified variants with pedagogical phrasing
            for var_idx in range(samples_multiplier):
                chosen_inst = random.choice(instruction_variants)
                dataset.append({
                    "instruction": chosen_inst,
                    "input": f"{q} Lütfen detaylı açıklar mısın?",
                    "output": a
                })
                dataset.append({
                    "instruction": domain_inst,
                    "input": f"Soru: {q}",
                    "output": f"Cevap: {a}"
                })

    # Shuffle dataset
    random.seed(42)
    random.shuffle(dataset)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Toplam Üretilen Lise Müfredat Kaydı: {len(dataset):,}")
    print(f"Kayıt Dosyası: {output_path}")
    return len(dataset)


if __name__ == "__main__":
    generate_high_school_dataset()
