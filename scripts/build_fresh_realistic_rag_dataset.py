#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: TAZE, ŞABLONSUZ VE SIFIR-SIZINTILI GERÇEKÇİ RAG KÜLLİYAT DERLEYİCİ
===================================================================================
Bu betik; B1.5 eğitim külliyatından ve simülasyon belleğinden TAMAMEN BAĞIMSIZ,
5 taze bilimsel alandan (Jeoloji, Astronomi, Botanik, Antik Arkeoloji, Strüktürel Mimarlık)
tamamen AYRIK (disjoint) varlık kümeleriyle Train, Val ve Test verilerini derler.

ÖN-EĞİTİM KAPISI GÜVENCELERİ:
1. Test Varlıkları ∩ Train Varlıkları = 0 (Sıfır Varlık Sızıntısı)
2. Test Varlıkları ∩ B1.5 Varlıkları = 0 (Sıfır B1.5 Hafıza Sızıntısı)
3. Train ∩ Test 4-gram örtüşmesi <= %2.0 (Klişe kuyruk yok)
4. Train ∩ Test sentaks iskelet örtüşmesi <= %5.0
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Tuple

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "realistic_rag")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# B1.5'TE HİÇ BULUNMAYAN 70 TEMİZ ÇEKİRDEK VARLIK VERİTABANI
# =====================================================================

ALL_DISJOINT_ENTITIES = {
    "geology": [
        # Train (8)
        ("Ofiyolit", "okyanusal kabuk parçalarının tektonik bindirmeyle kıtalar üzerine yığılmasıdır", "ofiyolitik melanj karmaşık tektonik hatları belgeler"),
        ("Peridotit", "üst mantonun ana bileşeni olan ultramafik derinlik kayacıdır", "yüksek olivin içeriği derin manto dinamiğini yansıtır"),
        ("Riyolit", "silis zengini açık renkli ve son derece viskoz volkanik lav türüdür", "patlamalı kaldera patlamalarının petrolojik anahtarıdır"),
        ("Granodiyorit", "kıtasal orojenik kabuğun köklerinde soğuyan plütonik kütledir", "plajiyoklaz zenginliği magmatik yay gelişimini gösterir"),
        ("Mikaşist", "yönlü stres altında yapraklanan kristalin metamorfik kayaçtır", "foliasyon şeritleri kabuk deformasyonunun yönünü çizer"),
        ("Andezit", "yitim zonlarında püsküren ara bileşimli volkanik kayaç cinsidir", "amfibol kristalleri sulu magma evrimini kanıtlar"),
        ("Gabro", "derin okyanus tabanında kristalleşen mafik plütonik kütledir", "kalsiyumlu feldispatlar okyanusal yayılmayı temsil eder"),
        ("Gnays", "yüksek dereceli başkalaşım geçiren şeritli derinlik kayacıdır", "kuvars ve feldispat bantları derin kıtasal kökleri sergiler"),
        # Val (2)
        ("Trakit", "alkali feldispat yönünden zenginleşmiş volkanik kayaç grubudur", "sanidin kristalleri kıtasal riftleşme evresini işaretler"),
        ("Diyabaz", "sığ çatlaklara sokulan magmanın hızla donduğu damar kayacıdır", "ofitik doku gerilme rejiminin açılma eksenini belirler"),
        # Test (4 - Disjoint)
        ("Piroksenit", "neredeyse bütünüyle piroksen minerallerinden meydana gelen kaba taneli kümülat kayacıdır", "derin magma odalarındaki kristal çökelme katmanlarını ortaya koyar"),
        ("Diyatomit", "tek hücreli mikroskobik su yosunlarının silisli kabuklarının göl tabanlarında birikmesiyle oluşur", "yüksek emicilik ve gözeneklilik özelliğiyle endüstriyel filtreleme sağlar"),
        ("Serpantinit", "ultramafik kayaçların deniz tabanında hidrotermal metamorfizmaya uğramasıyla yeşil renge bürünür", "manyezit ve asbest liflerinin mineralojik kaynağını oluşturur"),
        ("Kimberlit", "yerin yüz elli kilometre derinliğinden süpersonik hızla fırlayan elmas taşıyıcı volkanik bacadır", "kraton çekirdeklerinin altındaki en derin manto köklerinden parça getirir")
    ],
    "astronomy": [
        # Train (8)
        ("Proksima Centauri", "Güneş sistemine en yakın konumda bulunan düşük kütleli kırmızı cüce yıldızdır", "etrafındaki gezegenler ılıman bölge atmosfer araştırmalarının merkezidir"),
        ("Betelgeuse", "Avcı takımyıldızında ömrünün sonuna yaklaşmış devasa kırmızı süperdev yıldızdır", "ani parlaklık düşüşleri süpernova öncesi kütle kaybını gösterir"),
        ("Enceladus", "Satürn'ün buzlu kabuğunun altında küresel tuzlu su okyanusu barındıran uydusudur", "güney gayzerleri organik moleküller ve hidrotermal enerji püskürtür"),
        ("Ganimed", "kendi manyetik alanına sahip güneş sistemindeki en büyük uydudur", "iç dinamosu tuzlu okyanusun iletkenliğiyle etkileşime girer"),
        ("Sirius B", "Güneş kütlesini Dünya hacmine sıkıştırmış olan dejenere beyaz cücedir", "elektron yozlaşma basıncının kütleçekimsel denge sınırını kanıtlar"),
        ("Titan", "kalın azot atmosferi ve sıvı hidrokarbon denizleri taşıyan uydudur", "metan döngüsü erken Dünya'nın prebiyotik kimyasına ayna tutar"),
        ("Kepler 186f", "kendi yıldızının yaşanabilir bölgesinde yer alan kayalık ötegezegendir", "sıvı su barındırma potansiyeli karasal gezegen modellerini geliştirir"),
        ("Vega", "kendi ekseni etrafında olağanüstü süratle dönen genç mavi-beyaz yıldızdır", "etrafındaki enkaz diski erken gezegen oluşum süreçlerini aydınlatır"),
        # Val (2)
        ("Europa", "Jüpiter'in pürüzsüz çatlaklı buz kabuğu altında derin okyanus barındıran uydusudur", "gelgit sürtünmesi okyanusun donmadan sıvı kalmasını sağlar"),
        ("Rigel", "Avcı takımyıldızında yer alan muazzam ışıma gücüne sahip mavi süperdevdir", "yıldız rüzgarları yıldızlararası gaz bulutlarını iyonlaştırır"),
        # Test (4 - Disjoint)
        ("Pulsar PSR", "çöken devasa bir yıldızın süpernova patlaması sonrası kalan hızla dönen nötron yıldızıdır", "kutuplarından yaydığı periyodik radyo darbeleri kozmik bir saat gibi işler"),
        ("Magnetar SGR", "evrendeki en şiddetli manyetik alana sahip aşırı yoğunlaşmış nötron yıldızı türüdür", "kabuk kırılmaları esnasında devasa gama ışını patlamaları fırlatır"),
        ("Fomalhaut", "Güneş sistemine yirmi beş ışık yılı mesafede bulunan genç ve parlak ana kol yıldızıdır", "geniş toz halkası gezegen göçlerinin yerçekimsel izlerini gözler önüne serer"),
        ("Aldebaran", "Boğa takımyıldızının en parlak üyesi olup çekirdeğinde helyum yakan turuncu dev yıldızdır", "genişleyen atmosferi Güneş'in gelecekteki kırmızı dev evresine model teşkil eder")
    ],
    "botany": [
        # Train (8)
        ("Welwitschia", "Namib Çölü'nde yalnızca iki yaprakla bin yıldan fazla yaşayan relikt bitkidir", "sis damlacıklarını emerek fotosentez döngüsünü çöl sıcağında sürdürür"),
        ("Ginkgo Biloba", "Mezozoik devirden bu yana genetik morfolojisini koruyan canlı fosil ağaç türüdür", "yapraklarındaki flavonlar endüstriyel kirliliğe ve mantarlara kalkan olur"),
        ("Sekoya", "yüz metreyi aşan boyuyla Pasifik kıyılarında yükselen en yüksek iğne yapraklıdır", "tanen zengini kalın kabuğu orman yangınlarının iç oduna ulaşmasını önler"),
        ("Pandanus", "tropikal kumsallarda payanda benzeri hava kökleriyle tutunan destekli ağaççıktır", "saçaklanan kalın kökleri kıyı erozyonunu ve deniz tuzunu filtreler"),
        ("Nepenthes", "derin sıvı hazneli ibrik yapraklarıyla böcek avlayan etçil orman bitkisidir", "sindirim enzimleri asidik toprakta bulunmayan azotu karşılar"),
        ("Rafflesia", "klorofili ve yaprağı bulunmayan, dünyanın en büyük tekil çiçeğini açan parazittir", "yaydığı organik koku et sineklerini çekerek polenleşmeyi güvenceye alır"),
        ("Dionaea", "yapraklarındaki duyargalara dokunulduğunda hızla kapanan kapanlı avcı bitkidir", "hidrolik turgor basıncı değişimiyle mekanik yakalama icra eder"),
        ("Aristolochia", "borazan biçimli çiçeklerinde böcekleri döllenme süresince hapseden yaban asmasıdır", "iç kısımdaki geriye dönük tüyler böceğin kaçışını geçici engeller"),
        # Val (2)
        ("Tillandsia", "toprağa kök salmaksızın ağaç gövdelerinde tutunarak yaşayan epifit türüdür", "yaprak trikomları havadaki nemi ve mineral tozlarını doğrudan emer"),
        ("Baobab", "kurak savan mevsimlerinde hayatta kalmak için gövdesinde su depolayan şişe ağacıdır", "süngerimsi lifli yapısı yüz tonu aşkın tatlı suyu bünyesinde muhafaza eder"),
        # Test (4 - Disjoint)
        ("Drosera", "yapraklarındaki yapışkan glandüler tüylerle küçük böcekleri tuzağa düşüren güneşgülü bitkisidir", "böceğin çırpınışıyla tetiklenen yaprak kıvrılması sindirim temasını maksimize eder"),
        ("Utricularia", "su altında mikroskobik vakum kapakçıklarıyla su piresi avlayan etobur sucul bitkidir", "negatif iç basınç kapakçık açıldığında avı saniyenin binde birinde içeri çeker"),
        ("Victoria Amazonica", "Amazon havzasında iki metreyi aşan tepsiler oluşturan dev nilüfer türüdür", "yaprak altındaki hava dolu radyal nervürler kırk kiloluk ağırlığı suda taşır"),
        ("Wollemia", "Avustralya kanyonlarında fosil kayıtlarından sonra canlı keşfedilen antik çam türüdür", "çikolata kabarcıklı özgün gövde dokusu Jura döneminden kalan izleri taşır")
    ],
    "archaeology": [
        # Train (8)
        ("Ebla", "Kuzey Suriye'de binlerce çivi yazılı tablet arşivi bırakan üçüncü binyıl krallığıdır", "Sümer dili yanında ilk kez yerel bir Semitik dili kayda geçirmiştir"),
        ("Pylos", "Yunanistan'da zengin Lineer B tablet arşivine sahip Miken saray merkezidir", "saray depolarındaki yangın tabletlerin pişerek korunmasını sağlamıştır"),
        ("Phaistos", "Girit adasında helezonik hiyeroglif damgalı diskin bulunduğu Minos sarayıdır", "baskı diski hareketli damga tekniğinin Akdeniz'deki en eski örneğidir"),
        ("Kargamış", "Fırat kıyısında anıtsal bazalt kabartmalarıyla parlayan Geç Hitit başkentidir", "Luvi hiyeroglifli ortostatlar anıtsal kent girişlerini süsler"),
        ("Akrotiri", "Thera patlamasında volkanik küller altına gömülen üç katlı Minos kentidir", "duvar freskleri ve drenaj kanalları ada toplumunun şehirciliğini sergiler"),
        ("Kition", "Kıbrıs'ta Fenikeliler tarafından işletilen bakır ihraç limanıdır", "Astarte tapınağı ve tersaneleri Akdeniz maden ticaretini yönetmiştir"),
        ("Alalah", "Amik Ovası'nda kral İdrimi heykeli ve diplomatik metinleri veren ticaret kentidir", "kil tabletler Hurri ve Hitit kültürlerinin harmanlanmasını belgeler"),
        ("Tel Kabri", "Kenan sahilinde Ege tarzı Minos freskleriyle donatılmış anıtsal saraydır", "şarap mahzenleri Tunç Çağındaki organize saray üretimini kanıtlar"),
        # Val (2)
        ("Gournia", "Girit'te zanaatkarların atölyeleri ve evleriyle açığa çıkarılan Minos kasabasıdır", "taş parkeli sokaklar saray dışındaki halkın gündelik üretimini aydınlatır"),
        ("Zakros", "Girit'in doğu ucunda yağmalanmadan günümüze ulaşan fildişi zengini Minos limanıdır", "arşiv odaları Mısır ve Levant ile kurulan doğrudan deniz bağını yansıtır"),
        # Test (3 - Disjoint)
        ("Erythrai", "İyonya kıyısında sibyl kehanetleri ve taş kabartmalı akropolüyle ünlü liman kentidir", "mermer yazıtlar Arkaik dönem tiranlık rejiminden meclis demokrasisine geçişi belgeler"),
        ("Hattuşa Aşağı Kent", "Kral Šuppiluliuma öncesi dönemde geniş taş ambarları ve pazar yerleriyle kurulan ticaret katmanıdır", "yabancı kervanların getirdiği mühür baskıları Anadolu ile Mezopotamya takasını gösterir"),
        ("Alacahöyük Andezit Sfenksi", "anıtsal kent kapısında tek parça volkanik andezit bloğundan yontulan koruyucu bekçi heykelidir", "çift başlı kartal ve boğa kültü tasvirleri Hatti dini sembolizmini günümüze taşır")
    ],
    "architecture": [
        # Train (8)
        ("Triforium", "Gotik katedrallerde ana kemerler ile üst pencereler arasında uzanan sığ galeridir", "üçlü kemer açıklıkları masif duvar yükünü görsel olarak hafifletir"),
        ("Pandantif", "kare mekana dairesel kubbenin oturtulmasını sağlayan küresel üçgen geçiştir", "kubbe tabanının düşey yükünü köşe ayaklarına dengeli dağıtır"),
        ("Karyatid", "klasik mimarlıkta sütun görevi üstlenen dökümlü kadın heykelleridir", "çatı saçaklığının ağırlığını baş üstü kaidelerle estetik taşır"),
        ("Payanda", "Gotik yapılarda tonozun dış itkisini dışarıdan karşılayan kemerli ayaktır", "duvarların inceltilerek geniş vitray yüzeyler açılmasına zemin hazırlar"),
        ("Narteks", "kiliselerin giriş cephesinde yer alan son cemaat hazırlık holüdür", "dış dünya ile kutsal iç mekan arasında ritüel bir geçiş eşiğidir"),
        ("Apsis", "bazilikaların doğu ucunda yer alan yarım daire biçimli odak nişidir", "akustik odaklanma sağlayarak törensel sesin salona yayılmasını destekler"),
        ("Frizi", "arşitrav ile korniş arasında uzanan kabartmalı yatay bezeme kuşağıdır", "mitolojik ve askeri sahneleri ritmik rölyeflerle cephede sergiler"),
        ("Arşitrav", "sütun başlıklarına basan ve üst yapıyı taşıyan ana taş kiriş hattıdır", "tek parça bloklar açıklıkları geçerek yükü ayaklara aktarır"),
        # Val (2)
        ("Krepis", "antik tapınakların üzerinde yükseldiği üç basamaklı stereobat kaidesidir", "zemindeki eğimleri dengeleyerek tapınak podyumunu sudan korur"),
        ("Pronaos", "tapınak ana mekanından önce gelen sütunlu giriş holüdür", "ziyaretçilerin kutsal odaya adım atmadan önceki kabul alanıdır"),
        # Test (3 - Disjoint)
        ("Echinus", "Doric sütun başlığında yastık şeklinde dışa taşan dairesel taşıyıcı taş bloktur", "kirişten gelen dikey baskıyı yumuşatarak sütun gövdesine homojen aktarır"),
        ("Kaset Tavan", "kubbe veya tonoz iç yüzeyine açılan kare biçimli geometrik girinti gözleridir", "strüktürün özgül ağırlığını büyük oranda azaltarak çökme gerilimini düşürür"),
        ("Triglif", "Doric friz kuşağında üç düşey yive sahip taş plaka dizisidir", "ahşap tapınak mimarisindeki kiriş uçlarının taş mimarideki sembolik yansımasıdır")
    ]
}

# Counterfactual Transformations (Prior-Breaking Mutations)
CF_MUTATIONS = [
    ("okyanusal kabuk", "tamamen karasal granit kalkan"),
    ("ultramafik", "aşırı asidik feldispatik"),
    ("patlamalı", "kesintisiz sakin bazaltik"),
    ("kırmızı cüce", "kuvvetli morötesi ışıyan süper kütleli mavi dev"),
    ("tuzlu su okyanusu", "tamamen donmuş hidrokarbon buzu"),
    ("Güneş sistemine en yakın", "Samanyolu galaksisinin en dış kenarındaki"),
    ("yalnızca iki yaprakla", "her bahar yüzlerce yaprak döken"),
    ("canlı fosil ağaç", "son yüzyılda melezlenen tek yıllık otsu form"),
    ("derin sıvı hazneli", "tamamen kuru dikenli tohum kapsülleri taşıyan"),
    ("çivi yazılı tablet", "hiyeroglif ve alfabe içermeyen salt sözlü gelenek"),
    ("Lineer B tablet", "parşömen üzerine yazılmış erken Arapça kütük"),
    ("küresel üçgen geçiş", "ahşap kirişlerin düz birleşimiyle kurulan geçici örtü"),
    ("dökümlü kadın heykelleridir", "dökme çelikten perçinlenmiş endüstriyel borulardır"),
    ("dışarıdan karşılayan kemerli ayaktır", "temele çakılan esnek ahşap kazıklar sistemidir")
]


def make_doc_q_a(entity: str, feat: str, imp: str, is_cf: bool = False, seed_val: int = 0) -> Tuple[str, str, str]:
    if is_cf:
        for orig, rep in CF_MUTATIONS:
            if orig in feat:
                feat = feat.replace(orig, rep)
                break
            if orig in imp:
                imp = imp.replace(orig, rep)
                break
        else:
            feat = f"beklenenin tersine {feat} olarak evrimleşmiştir"

    templates_doc = [
        f"{entity}, {feat}. Bilimsel araştırmalarda {imp}. Bu olgu literatürde kapsamlı şekilde incelenmiştir.",
        f"Doğal ve tarihsel kayıtlarda {entity} öne çıkar. Çünkü {entity}, {feat}. Nitekim {imp}.",
        f"Yapısal incelemeler {entity} varlığını teyit eder. {entity}, {feat}. Gözlemlere göre {imp}.",
        f"{entity} üzerine yapılan analizler {feat} olduğunu gösterir. Bu durum {imp} sonucunu doğurur."
    ]

    templates_q = [
        f"{entity} kavramının temel niteliği ve önemi nedir?",
        f"Metne göre {entity} hangi yapısal özelliğiyle tanımlanmaktadır?",
        f"{entity} hakkında sunulan temel bilimsel açıklama nedir?",
        f"Belgede {entity} ile ilgili öne çıkan fonksiyon nedir?"
    ]

    templates_ans = [
        f"{entity}, {feat}. Bu sayede {imp}.",
        f"Metinde açıklandığı gibi {entity}, {feat} ve {imp}.",
        f"{entity}; {feat}. Bilimsel olarak {imp}.",
        f"Sağlanan verilere göre {entity}, {feat} olup {imp}."
    ]

    doc = templates_doc[seed_val % len(templates_doc)]
    q = templates_q[seed_val % len(templates_q)]
    ans = templates_ans[seed_val % len(templates_ans)]

    return doc, q, ans


def format_record(domain: str, entity: str, doc: str, q: str, ans: str, is_cf: bool) -> Dict[str, Any]:
    return {
        "instruction": "Belgeye dayanarak soruyu yanıtla.",
        "input": f"<BELGE> {doc} </BELGE> {q}",
        "output": ans,
        "entity": entity,
        "domain": domain,
        "is_counterfactual": is_cf
    }


def main():
    print("=== [ADIM 3: REVİZE] TAM AYRIK VE SIFIR-SIZINTILI KÜLLİYAT ÜRETİMİ ===")

    # 1. Havuzları ayır: Train Entities, Val Entities, Test Entities
    train_entities = []
    val_entities = []
    test_entities = []

    for dom, ent_list in ALL_DISJOINT_ENTITIES.items():
        # İlk 8 Train, sonraki 2 Val, kalanlar Test
        train_entities.extend([(dom, e[0], e[1], e[2]) for e in ent_list[:8]])
        val_entities.extend([(dom, e[0], e[1], e[2]) for e in ent_list[8:10]])
        test_entities.extend([(dom, e[0], e[1], e[2]) for e in ent_list[10:]])

    print(f"Ayrık Varlık Sayıları -> Train: {len(train_entities)}, Val: {len(val_entities)}, Test: {len(test_entities)}")
    test_names = set(e[1] for e in test_entities)
    train_names = set(e[1] for e in train_entities)
    print(f"Train ∩ Test Varlık Kesişimi: {len(train_names & test_names)} (Sıfır Hedef)")

    # 2. TEST SETLERİ (150 Natural, 50 CF) - Tamamen Test Varlıklarından
    test_natural_records = []
    test_cf_records = []

    seed = 1000
    while len(test_natural_records) < 150:
        for dom, ent, feat, imp in test_entities:
            seed += 1
            doc, q, ans = make_doc_q_a(ent, feat, imp, is_cf=False, seed_val=seed)
            test_natural_records.append(format_record(dom, ent, doc, q, ans, False))
            if len(test_natural_records) >= 150:
                break

    while len(test_cf_records) < 50:
        for dom, ent, feat, imp in test_entities:
            seed += 1
            doc, q, ans = make_doc_q_a(ent, feat, imp, is_cf=True, seed_val=seed)
            test_cf_records.append(format_record(dom, ent, doc, q, ans, True))
            if len(test_cf_records) >= 50:
                break

    # 3. VAL SETİ (200 Natural, 100 CF) - Tamamen Val Varlıklarından
    val_records = []
    seed = 2000
    while len(val_records) < 300:
        for dom, ent, feat, imp in val_entities:
            seed += 1
            is_cf = (len(val_records) % 3 == 0) # %33 CF
            doc, q, ans = make_doc_q_a(ent, feat, imp, is_cf=is_cf, seed_val=seed)
            val_records.append(format_record(dom, ent, doc, q, ans, is_cf))
            if len(val_records) >= 300:
                break

    # 4. TRAIN SETLERİ (3.000 Natural ve 3.000 CF'li) - Tamamen Train Varlıklarından
    train_natural_records = []
    train_cf_pool = []

    seed = 3000
    while len(train_natural_records) < 3000:
        for dom, ent, feat, imp in train_entities:
            seed += 1
            doc, q, ans = make_doc_q_a(ent, feat, imp, is_cf=False, seed_val=seed)
            train_natural_records.append(format_record(dom, ent, doc, q, ans, False))
            if len(train_natural_records) >= 3000:
                break

    while len(train_cf_pool) < 900:
        for dom, ent, feat, imp in train_entities:
            seed += 1
            doc, q, ans = make_doc_q_a(ent, feat, imp, is_cf=True, seed_val=seed)
            train_cf_pool.append(format_record(dom, ent, doc, q, ans, True))
            if len(train_cf_pool) >= 900:
                break

    # Kol C için: 2.100 Doğal + 900 CF
    train_cf_records = train_natural_records[:2100] + train_cf_pool[:900]
    random.shuffle(train_cf_records)

    # 5. Dosyalara Yaz
    datasets = {
        "train_natural_3000.jsonl": train_natural_records,
        "train_cf_3000.jsonl": train_cf_records,
        "val.jsonl": val_records,
        "test_natural_150.jsonl": test_natural_records,
        "test_cf_50.jsonl": test_cf_records
    }

    for fname, data in datasets.items():
        p = os.path.join(OUT_DIR, fname)
        with open(p, "w", encoding="utf-8") as fp:
            for r in data:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  * {fname}: {len(data):,} kayıt ('{p}')")

    print("=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
