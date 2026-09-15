#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: NÖTRLEŞTİRİLMİŞ KARŞI-OLGUSAL (COUNTERFACTUAL) KÜLLİYAT ÜRETİCİ
================================================================================
Bu betik; Counterfactual veri üretimindeki tüm istatistiksel işaretçileri ("beklenenin tersine",
"evrimleşmiştir" vb.) tamamen sıfırlayarak Doğal (Natural) ve Karşı-Olgusal (CF) metinleri
aynı şablon, sentaks, ton, kelime sayısı ve karakter uzunluğu simetrisinde üretir.

Doğrulama Güvencesi:
Tek özellikli (single-feature) sınıflandırıcılar (karakter uzunluğu, kelime sayısı, şablon kimliği,
virgül/noktalama sayısı, sözlüksel belirteçler) üzerinde ROC-AUC ≈ 0.500 olduğunu kanıtlar.
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "realistic_rag")
os.makedirs(OUT_DIR, exist_ok=True)

# 68 Tekil Varlık için Birebir Simetrik ve Uzunluk-Dengeli (Doğal vs. Karşı-Olgusal) Veritabanı
# Her kayıt: (entity, feat_nat, imp_nat, feat_cf, imp_cf)
SYMMETRIC_ENTITIES = {
    "geology": [
        # Train (8)
        ("Ofiyolit", 
         "okyanusal kabuk parçalarının tektonik bindirmeyle kıtalar üzerine yığılmasıdır",
         "ofiyolitik melanj karmaşık tektonik hatları belgeler",
         "kıtasal granit kabuk bloklarının derin yarıklara batıp yitimle erimesidir",
         "kristalin masifler homojen iç platformları eksiksiz belgeler"),
        ("Peridotit",
         "üst mantonun ana bileşeni olan ultramafik derinlik kayacıdır",
         "yüksek olivin içeriği derin manto dinamiğini yansıtır",
         "yeryüzü tortullarının yüzeyde birikmesiyle oluşan asidik kırıntılı bir kayaçtır",
         "yüksek kuvars içeriği sığ nehir akıntılarını yansıtır"),
        ("Riyolit",
         "silis zengini açık renkli ve son derece viskoz volkanik lav türüdür",
         "patlamalı kaldera patlamalarının petrolojik anahtarıdır",
         "demir zengini koyu renkli ve son derece akışkan bir lav türüdür",
         "sakin tabaka akıntılarının anahtarıdır"),
        ("Granodiyorit",
         "kıtasal orojenik kabuğun köklerinde soğuyan plütonik kütledir",
         "plajiyoklaz zenginliği magmatik yay gelişimini gösterir",
         "okyanus ortası sırtlarda hızla püsküren camsı magmatik kütledir",
         "olivin zenginliği riftleşme gelişimini açıkça gösterir"),
        ("Mikaşist",
         "yönlü stres altında yapraklanan kristalin metamorfik kayaçtır",
         "foliasyon şeritleri kabuk deformasyonunun yönünü çizer",
         "durağan göl çanaklarında tabakasız çökelen tortul kayaçtır",
         "çatlak izleri yüzey erozyonunun yönünü ayrıntıyla çizer"),
        ("Andezit",
         "yitim zonlarında püsküren ara bileşimli volkanik kayaç cinsidir",
         "amfibol kristalleri sulu magma evrimini kanıtlar",
         "derin kraton çekirdeğinde katılaşan bazik derinlik kayacı cinsidir",
         "piroksen kristalleri kuru magma evrimini kanıtlar"),
        ("Gabro",
         "derin okyanus tabanında kristalleşen mafik plütonik kütledir",
         "kalsiyumlu feldispatlar okyanusal yayılmayı temsil eder",
         "sığ lagün diplerinde çökelen kireçli kimyasal bir tortul kütledir",
         "kalsit taneleri havza buharlaşma döngüsünü temsil eder"),
        ("Gnays",
         "yüksek dereceli başkalaşım geçiren şeritli derinlik kayacıdır",
         "kuvars ve feldispat bantları derin kıtasal kökleri sergiler",
         "düşük sıcaklıkta yüzeyde katılaşan süngersi gözenekli lav kayacıdır",
         "gaz boşlukları yüzey püskürme köklerini açıkça sergiler"),
        # Val (2)
        ("Trakit",
         "alkali feldispat yönünden zenginleşmiş volkanik kayaç grubudur",
         "sanidin kristalleri kıtasal riftleşme evresini işaretler",
         "kalsiyum sülfat yönünden zenginleşmiş çökel kayaç grubudur",
         "jips kristalleri havza kuruma evresini işaretler"),
        ("Diyabaz",
         "sığ çatlaklara sokulan magmanın hızla donduğu damar kayacıdır",
         "ofitik doku gerilme rejiminin açılma eksenini belirler",
         "açık deniz tabanına yayılan lavın yavaşça soğuduğu örtü kayacıdır",
         "amorf doku sıkışma rejiminin kapanma eksenini belirler"),
        # Test (4 - Disjoint)
        ("Piroksenit",
         "neredeyse bütünüyle piroksen minerallerinden meydana gelen kaba taneli kümülat kayacıdır",
         "derin magma odalarındaki kristal çökelme katmanlarını ortaya koyar",
         "neredeyse bütünüyle kalsit minerallerinden meydana gelen ince taneli karstik kayaçtır",
         "sığ deniz kollarındaki kimyasal çökelme katmanlarını ortaya koyar"),
        ("Diyatomit",
         "tek hücreli mikroskobik su yosunlarının silisli kabuklarının göl tabanlarında birikmesiyle oluşur",
         "yüksek emicilik ve gözeneklilik özelliğiyle endüstriyel filtreleme sağlar",
         "kaba taneli volkanik patlama küllerinin geniş akarsu deltalarında çimentolaşmasıyla meydana gelir",
         "düşük emicilik ve geçirimsizlik niteliğiyle ağır endüstriyel yalıtım sağlar"),
        ("Serpantinit",
         "ultramafik kayaçların deniz tabanında hidrotermal metamorfizmaya uğramasıyla yeşil renge bürünür",
         "manyezit ve asbest liflerinin mineralojik kaynağını oluşturur",
         "asidik kumtaşlarının karasal çöllerde rüzgar erozyonuna uğramasıyla sarı renge bürünür",
         "kuvarsit ve feldspat tanelerinin mineralojik kaynağını oluşturur"),
        ("Kimberlit",
         "yerin yüz elli kilometre derinliğinden süpersonik hızla fırlayan elmas taşıyıcı volkanik bacadır",
         "kraton çekirdeklerinin altındaki en derin manto köklerinden parça getirir",
         "yerin bir kilometre derinliğinde durağan katılaşan kükürt taşıyıcı kireçtaşı yarığıdır",
         "genç dağ kuşaklarının altındaki en sığ kabuk köklerinden parça getirir")
    ],
    "astronomy": [
        # Train (8)
        ("Proksima Centauri",
         "Güneş sistemine en yakın konumda bulunan düşük kütleli kırmızı cüce yıldızdır",
         "etrafındaki gezegenler ılıman bölge atmosfer araştırmalarının merkezidir",
         "Samanyolu galaksisinin en dış sarmalında yer alan süper kütleli mavi dev yıldızdır",
         "etrafındaki gaz bulutları iyonize şok dalgalarının temel merkezidir"),
        ("Betelgeuse",
         "Avcı takımyıldızında ömrünün sonuna yaklaşmış devasa kırmızı süperdev yıldızdır",
         "ani parlaklık düşüşleri süpernova öncesi kütle kaybını gösterir",
         "Kuğu takımyıldızında ömrünün hemen başında bulunan kararlı sarı cüce yıldızdır",
         "düzenli parlaklık artışları hidrojen füzyonunun başlangıcını gösterir"),
        ("Enceladus",
         "Satürn'ün buzlu kabuğunun altında küresel tuzlu su okyanusu barındıran uydusudur",
         "güney gayzerleri organik moleküller ve hidrotermal enerji püskürtür",
         "Mars'ın kuru kabuğunun altında kalın bazaltik lav tabakaları barındıran uydusudur",
         "kuzey kraterleri silikat tozları ve metalik parçacıklar püskürtür"),
        ("Ganimed",
         "kendi manyetik alanına sahip güneş sistemindeki en büyük uydudur",
         "iç dinamosu tuzlu okyanusun iletkenliğiyle etkileşime girer",
         "hiçbir manyetik alanı bulunmayan güneş sistemindeki en küçük uydudur",
         "dış kabuğu güneş rüzgarlarının sürtünmesiyle etkileşime girer"),
        ("Sirius B",
         "Güneş kütlesini Dünya hacmine sıkıştırmış olan dejenere beyaz cücedir",
         "elektron yozlaşma basıncının kütleçekimsel denge sınırını kanıtlar",
         "Jüpiter kütlesini Güneş hacmine yaymış olan geniş hidrojen gaz bulutudur",
         "termal genişleme basıncının seyreltik denge sınırını kanıtlar"),
        ("Titan",
         "kalın azot atmosferi ve sıvı hidrokarbon denizleri taşıyan uydudur",
         "metan döngüsü erken Dünya'nın prebiyotik kimyasına ayna tutar",
         "tamamen havasız taş yüzeyi ve donmuş kükürt kraterleri taşıyan uydudur",
         "karbon döngüsü soğuk Venüs'ün volkanik kimyasına ayna tutar"),
        ("Kepler 186f",
         "kendi yıldızının yaşanabilir bölgesinde yer alan kayalık ötegezegendir",
         "sıvı su barındırma potansiyeli karasal gezegen modellerini geliştirir",
         "kendi yıldızının kavurucu koronasında dönen akkor halinde gaz ötegezegenidir",
         "ağır metal buharlaşması dev gezegen modellerini ayrıntıyla geliştirir"),
        ("Vega",
         "kendi ekseni etrafında olağanüstü süratle dönen genç mavi-beyaz yıldızdır",
         "etrafındaki enkaz diski erken gezegen oluşum süreçlerini aydınlatır",
         "kendi ekseni etrafında son derece yavaş dönen yaşlı kahverengi cücedir",
         "etrafındaki toz halesi geç yıldız sönme süreçlerini aydınlatır"),
        # Val (2)
        ("Europa",
         "Jüpiter'in pürüzsüz çatlaklı buz kabuğu altında derin okyanus barındıran uydusudur",
         "gelgit sürtünmesi okyanusun donmadan sıvı kalmasını sağlar",
         "Neptün'ün engebeli kraterli kaya kabuğu altında kuru çekirdek barındıran uydusudur",
         "radyoaktif bozunma çekirdeğin ısınmadan katı kalmasını sağlar"),
        ("Rigel",
         "Avcı takımyıldızında yer alan muazzam ışıma gücüne sahip mavi süperdevdir",
         "yıldız rüzgarları yıldızlararası gaz bulutlarını iyonlaştırır",
         "Boğa takımyıldızında yer alan son derece sönük ışıma gücüne sahip kırmızı cücedir",
         "manyetik fırtınalar gezegenlerarası toz bulutlarını iyonlaştırır"),
        # Test (4 - Disjoint)
        ("Pulsar PSR",
         "çöken devasa bir yıldızın süpernova patlaması sonrası kalan hızla dönen nötron yıldızıdır",
         "kutuplarından yaydığı periyodik radyo darbeleri kozmik bir saat gibi işler",
         "genişleyen devasa bir yıldızın nükleer füzyonu sonrası kalan durağan gaz devidir",
         "ekvatorundan yaydığı sürekli kızılötesi ışık kozmik bir fener gibi işler"),
        ("Magnetar SGR",
         "evrendeki en şiddetli manyetik alana sahip aşırı yoğunlaşmış nötron yıldızı türüdür",
         "kabuk kırılmaları esnasında devasa gama ışını patlamaları fırlatır",
         "galaksideki en zayıf manyetik alana sahip aşırı seyrelmiş helyum bulutu türüdür",
         "çekirdek birleşmeleri esnasında düşük enerjili radyo dalgaları fırlatır"),
        ("Fomalhaut",
         "Güneş sistemine yirmi beş ışık yılı mesafede bulunan genç ve parlak ana kol yıldızıdır",
         "geniş toz halkası gezegen göçlerinin yerçekimsel izlerini gözler önüne serer",
         "Güneş sistemine binlerce ışık yılı mesafede bulunan yaşlı ve sönük beyaz cücedir",
         "dar plazma halkası yıldız çarpışmalarının manyetik izlerini gözler önüne serer"),
        ("Aldebaran",
         "Boğa takımyıldızının en parlak üyesi olup çekirdeğinde helyum yakan turuncu dev yıldızdır",
         "genişleyen atmosferi Güneş'in gelecekteki kırmızı dev evresine model teşkil eder",
         "Başak takımyıldızının en sönük üyesi olup çekirdeğinde demir biriktiren mavi cücedir",
         "büzülen atmosferi Güneş'in geçmişteki ilk oluşum evresine model teşkil eder")
    ],
    "botany": [
        # Train (8)
        ("Welwitschia",
         "Namib Çölü'nde yalnızca iki yaprakla bin yıldan fazla yaşayan relikt bitkidir",
         "sis damlacıklarını emerek fotosentez döngüsünü çöl sıcağında sürdürür",
         "Kuzey Tundrası'nda yüzlerce yaprakla her yaz yeniden filizlenen relikt bitkidir",
         "kar suyunu emerek büyüme döngüsünü kutup soğuğunda kararlılıkla sürdürür"),
        ("Ginkgo Biloba",
         "Mezozoik devirden bu yana genetik morfolojisini koruyan canlı fosil ağaç türüdür",
         "yapraklarındaki flavonlar endüstriyel kirliliğe ve mantarlara kalkan olur",
         "geçtiğimiz yüzyılın başında botanik bahçelerinde melezlenen kısa ömürlü bir çalı türüdür",
         "kabuğundaki yapışkan reçineler aşırı kuraklığa ve zararlı böceklere kalkan olur"),
        ("Sekoya",
         "yüz metreyi aşan boyuyla Pasifik kıyılarında yükselen en yüksek iğne yapraklıdır",
         "tanen zengini kalın kabuğu orman yangınlarının iç oduna ulaşmasını önler",
         "otuz santimi geçmeyen boyuyla kurak dağ eteklerinde sürünen bodur çalı cinsidir",
         "klorofil fakiri ince zarı kış donlarının köklere ulaşmasını tümüyle önler"),
        ("Pandanus",
         "tropikal kumsallarda payanda benzeri hava kökleriyle tutunan destekli ağaççıktır",
         "saçaklanan kalın kökleri kıyı erozyonunu ve deniz tuzunu filtreler",
         "kireçli yaylalarda tek bir ana kazık kökle toprağa inen destekli ağaççıktır",
         "yapraklarındaki tüyler dağ rüzgarlarını ve deniz tuzunu filtreler"),
        ("Nepenthes",
         "derin sıvı hazneli ibrik yapraklarıyla böcek avlayan etçil orman bitkisidir",
         "sindirim enzimleri asidik toprakta bulunmayan azotu karşılar",
         "kuru tohum taşıyan dikenli sürgünleriyle kuşları besleyen otsu kır bitkisidir",
         "kök bakterileri alkali toprakta bulunmayan fosforu karşılar"),
        ("Rafflesia",
         "klorofili ve yaprağı bulunmayan, dünyanın en büyük tekil çiçeğini açan parazittir",
         "yaydığı organik koku et sineklerini çekerek polenleşmeyi güvenceye alır",
         "klorofil zengini geniş yapraklarıyla dünyanın en küçük çiçeklerini açan sucul türdür",
         "yaydığı tatlı nektar bal arılarını çekerek polenleşmeyi güvenceye alır"),
        ("Dionaea",
         "yapraklarındaki duyargalara dokunulduğunda hızla kapanan kapanlı avcı bitkidir",
         "hidrolik turgor basıncı değişimiyle mekanik yakalama icra eder",
         "gövdesindeki dikenlere temas edildiğinde zehir salgılayan durağan çalı türüdür",
         "osmotik tuz dengesi değişimiyle kimyasal savunma icra eder"),
        ("Aristolochia",
         "borazan biçimli çiçeklerinde böcekleri döllenme süresince hapseden yaban asmasıdır",
         "iç kısımdaki geriye dönük tüyler böceğin kaçışını geçici engeller",
         "küresel biçimli meyvelerinde tohumları olgunlaşma süresince koruyan çam türüdür",
         "dış kısımdaki sert pullar kuşların yemlenmesini geçici engeller"),
        # Val (2)
        ("Tillandsia",
         "toprağa kök salmaksızın ağaç gövdelerinde tutunarak yaşayan epifit türüdür",
         "yaprak trikomları havadaki nemi ve mineral tozlarını doğrudan emer",
         "derin balçığa kök salarak göl kenarlarında kümelenen bataklık sazıdır",
         "kök uçları sudaki çamuru ve organik atıkları doğrudan emer"),
        ("Baobab",
         "kurak savan mevsimlerinde hayatta kalmak için gövdesinde su depolayan şişe ağacıdır",
         "süngerimsi lifli yapısı yüz tonu aşkın tatlı suyu bünyesinde muhafaza eder",
         "nemli yağmur ormanlarında hızla büyümek için gövdesinde reçine biriktiren kavak türüdür",
         "yoğun odunsu yapısı yüksek oranda minerali bünyesinde muhafaza eder"),
        # Test (4 - Disjoint)
        ("Drosera",
         "yapraklarındaki yapışkan glandüler tüylerle küçük böcekleri tuzağa düşüren güneşgülü bitkisidir",
         "böceğin çırpınışıyla tetiklenen yaprak kıvrılması sindirim temasını maksimize eder",
         "sürgünlerindeki keskin sert dikenlerle otçul hayvanları uzaklaştıran sütleğen bitkisidir",
         "kuru rüzgarın esişiyle tetiklenen hızlı yaprak dökümü gövde içi nemi maksimize eder"),
        ("Utricularia",
         "su altında mikroskobik vakum kapakçıklarıyla su piresi avlayan etobur sucul bitkidir",
         "negatif iç basınç kapakçık açıldığında avı saniyenin binde birinde içeri çeker",
         "toprak üstünde yapışkan koku damlacıklarıyla karınca çeken zehirli kara bitkisidir",
         "pozitif osmotik basınç damlacık kuruduğunda tohumu çevreye hızla fırlatır"),
        ("Victoria Amazonica",
         "Amazon havzasında iki metreyi aşan tepsiler oluşturan dev nilüfer türüdür",
         "yaprak altındaki hava dolu radyal nervürler kırk kiloluk ağırlığı suda taşır",
         "Alp göllerinde on santimi geçmeyen minik kadehler oluşturan cüce nilüfer türüdür",
         "yaprak altındaki hava dolu ince nervürler bir kiloluk ağırlığı suda taşır"),
        ("Wollemia",
         "Avustralya kanyonlarında fosil kayıtlarından sonra canlı keşfedilen antik çam türüdür",
         "çikolata kabarcıklı özgün gövde dokusu Jura döneminden kalan izleri taşır",
         "Avrupa parklarında modern ıslah yöntemleriyle üretilen melez süs servi türüdür",
         "düz çizgili pürüzsüz kabuk dokusu yakın geçmişte geliştirilen izleri taşır")
    ],
    "archaeology": [
        # Train (8)
        ("Ebla",
         "Kuzey Suriye'de binlerce çivi yazılı tablet arşivi bırakan üçüncü binyıl krallığıdır",
         "Sümer dili yanında ilk kez yerel bir Semitik dili kayda geçirmiştir",
         "Güney Arabistan'da hiçbir yazılı belge bırakmayan göçebe bir çöl konfederasyonudur",
         "sözlü şiir geleneği yanında ilk kez kabile soyağaçlarını kayda geçirmiştir"),
        ("Pylos",
         "Yunanistan'da zengin Lineer B tablet arşivine sahip Miken saray merkezidir",
         "saray depolarındaki yangın tabletlerin pişerek korunmasını sağlamıştır",
         "İtalya'da hiçbir arşiv barındırmayan müstahkem Etrüsk savunma merkezidir",
         "kale surlarındaki yangın kerpiçlerin pişerek korunmasını sağlamıştır"),
        ("Phaistos",
         "Girit adasında helezonik hiyeroglif damgalı diskin bulunduğu Minos sarayıdır",
         "baskı diski hareketli damga tekniğinin Akdeniz'deki en eski örneğidir",
         "Rodos adasında geometrik desenli boyalı vazoların bulunduğu Minos sarayıdır",
         "baskı diski hareketli damga tekniğinin Akdeniz'deki en geç örneğidir"),
        ("Kargamış",
         "Fırat kıyısında anıtsal bazalt kabartmalarıyla parlayan Geç Hitit başkentidir",
         "Luvi hiyeroglifli ortostatlar anıtsal kent girişlerini süsler",
         "Dicle kıyısında kerpiç kabartmalarıyla korunan Geç Asur başkentidir",
         "Luvi hiyeroglifli ortostatlar anıtsal saray girişlerini süsler"),
        ("Akrotiri",
         "Thera patlamasında volkanik küller altına gömülen üç katlı Minos kentidir",
         "duvar freskleri ve drenaj kanalları ada toplumunun şehirciliğini sergiler",
         "ani bir tektonik depremde deniz suları altına batan tek katlı ahşap balıkçı kentidir",
         "ahşap iskeleler ve balık ağları yerel kıyı toplumunun şehirciliğini sergiler"),
        ("Kition",
         "Kıbrıs'ta Fenikeliler tarafından işletilen bakır ihraç limanıdır",
         "Astarte tapınağı ve tersaneleri Akdeniz maden ticaretini yönetmiştir",
         "Girit'te Akalar tarafından işletilen zeytinyağı ihraç limanıdır",
         "Zeus tapınağı ve tersaneleri Akdeniz maden ticaretini yönetmiştir"),
        ("Alalah",
         "Amik Ovası'nda kral İdrimi heykeli ve diplomatik metinleri veren ticaret kentidir",
         "kil tabletler Hurri ve Hitit kültürlerinin harmanlanmasını belgeler",
         "Konya Ovası'nda mermer ana tanrıça idolleri ve stelleri veren ticaret kentidir",
         "duvar resimleri avcı ve toplayıcı klanların harmanlanmasını belgeler"),
        ("Tel Kabri",
         "Kenan sahilinde Ege tarzı Minos freskleriyle donatılmış anıtsal saraydır",
         "şarap mahzenleri Tunç Çağındaki organize saray üretimini kanıtlar",
         "Ürdün vadisinde Mezopotamya tarzı kerpiç zigguratla donatılmış tapınaktır",
         "tahıl depoları Tunç Çağındaki organize tapınak üretimini kanıtlar"),
        # Val (2)
        ("Gournia",
         "Girit'te zanaatkarların atölyeleri ve evleriyle açığa çıkarılan Minos kasabasıdır",
         "taş parkeli sokaklar saray dışındaki halkın gündelik üretimini aydınlatır",
         "Mora'da soyluların mezarları ve hazineleriyle açığa çıkarılan Minos kasabasıdır",
         "altın maskeli odalar saray dışındaki halkın gündelik üretimini aydınlatır"),
        ("Zakros",
         "Girit'in doğu ucunda yağmalanmadan günümüze ulaşan fildişi zengini Minos limanıdır",
         "arşiv odaları Mısır ve Levant ile kurulan doğrudan deniz bağını yansıtır",
         "Girit'in batı ucunda korsanlarca yıkılan ve terk edilen bronz fakiri dağ karakoludur",
         "gözetleme kuleleri anakara ile sürdürülen doğrudan savunma hattını yansıtır"),
        # Test (3 - Disjoint)
        ("Erythrai",
         "İyonya kıyısında sibyl kehanetleri ve taş kabartmalı akropolüyle ünlü liman kentidir",
         "mermer yazıtlar Arkaik dönem tiranlık rejiminden meclis demokrasisine geçişi belgeler",
         "Likya dağlarında kaya mezarları ve ahşap kuleli kalesiyle ünlü dağ kentidir",
         "bronz sikkeler Arkaik dönem tiranlık rejiminden krallık vergisine geçişi belgeler"),
        ("Hattuşa Aşağı Kent",
         "Kral Šuppiluliuma öncesi dönemde geniş taş ambarları ve pazar yerleriyle kurulan ticaret katmanıdır",
         "yabancı kervanların getirdiği mühür baskıları Anadolu ile Mezopotamya takasını gösterir",
         "Geç Roma imparatorluk döneminde mermer hamamları ve pazar yerleriyle kurulan ticaret katmanıdır",
         "yerli ustaların getirdiği mühür baskıları Anadolu ile Mezopotamya takasını gösterir"),
        ("Alacahöyük Andezit Sfenksi",
         "anıtsal kent kapısında tek parça volkanik andezit bloğundan yontulan koruyucu bekçi heykelidir",
         "çift başlı kartal ve boğa kültü tasvirleri Hatti dini sembolizmini günümüze taşır",
         "saray iç avlusunda pişmiş kilden kalıba dökülen geniş süslemeli dekoratif saksı kaidesidir",
         "lotus çiçeği ve kanatlı aslan avı tasvirleri Pers dini sembolizmini günümüze taşır")
    ],
    "architecture": [
        # Train (8)
        ("Triforium",
         "Gotik katedrallerde ana kemerler ile üst pencereler arasında uzanan sığ galeridir",
         "üçlü kemer açıklıkları masif duvar yükünü görsel olarak hafifletir",
         "Roma tiyatrolarında sahne binası ile seyirci basamakları arasında uzanan sığ galeridir",
         "üçlü kemer açıklıkları masif sahne yükünü görsel olarak dengeler"),
        ("Pandantif",
         "kare mekana dairesel kubbenin oturtulmasını sağlayan küresel üçgen geçiştir",
         "kubbe tabanının düşey yükünü köşe ayaklarına dengeli dağıtır",
         "dikdörtgen mekana düz ahşap tavanın çakılmasını sağlayan küresel üçgen geçiştir",
         "çatı tabanının düşey yükünü köşe ayaklarına dengeli dağıtır"),
        ("Karyatid",
         "klasik mimarlıkta sütun görevi üstlenen dökümlü kadın heykelleridir",
         "çatı saçaklığının ağırlığını baş üstü kaidelerle estetik taşır",
         "barok mimarlıkta sütun görevi üstlenen dökümlü kadın heykelleridir",
         "çatı saçaklığının ağırlığını baş üstü kaidelerle estetik taşır"),
        ("Payanda",
         "Gotik yapılarda tonozun dış itkisini dışarıdan karşılayan kemerli ayaktır",
         "duvarların inceltilerek geniş vitray yüzeyler açılmasına zemin hazırlar",
         "Selçuklu yapılarında tonozun iç itkisini içeriden karşılayan kemerli ayaktır",
         "duvarların kalınlaştırılarak dar vitray yüzeyler açılmasına zemin hazırlar"),
        ("Narteks",
         "kiliselerin giriş cephesinde yer alan son cemaat hazırlık holüdür",
         "dış dünya ile kutsal iç mekan arasında ritüel bir geçiş eşiğidir",
         "sarayların arka bahçesinde yer alan son dinlenme hazırlık holüdür",
         "dış dünya ile kutsal iç mekan arasında ritüel bir geçiş eşiğidir"),
        ("Apsis",
         "bazilikaların doğu ucunda yer alan yarım daire biçimli odak nişidir",
         "akustik odaklanma sağlayarak törensel sesin salona yayılmasını destekler",
         "kalelerin batı ucunda yer alan yarım daire biçimli odak nişidir",
         "akustik odaklanma sağlayarak törensel sesin salona yayılmasını destekler"),
        ("Frizi",
         "arşitrav ile korniş arasında uzanan kabartmalı yatay bezeme kuşağıdır",
         "mitolojik ve askeri sahneleri ritmik rölyeflerle cephede sergiler",
         "taban ile duvar arasında uzanan kabartmalı yatay bezeme kuşağıdır",
         "mitolojik ve askeri sahneleri ritmik rölyeflerle cephede sergiler"),
        ("Arşitrav",
         "sütun başlıklarına basan ve üst yapıyı taşıyan ana taş kiriş hattıdır",
         "tek parça bloklar açıklıkları geçerek yükü ayaklara aktarır",
         "zemin döşemesine oturan ve üst yapıyı taşıyan ana taş kiriş hattıdır",
         "tek parça bloklar açıklıkları geçerek yükü ayaklara aktarır"),
        # Val (2)
        ("Krepis",
         "antik tapınakların üzerinde yükseldiği üç basamaklı stereobat kaidesidir",
         "zemindeki eğimleri dengeleyerek tapınak podyumunu sudan korur",
         "antik limanların üzerinde yükseldiği üç basamaklı stereobat kaidesidir",
         "zemindeki eğimleri dengeleyerek liman podyumunu sudan korur"),
        ("Pronaos",
         "tapınak ana mekanından önce gelen sütunlu giriş holüdür",
         "ziyaretçilerin kutsal odaya adım atmadan önceki kabul alanıdır",
         "hamam ana mekanından önce gelen sütunlu giriş holüdür",
         "ziyaretçilerin sıcak odaya adım atmadan önceki kabul alanıdır"),
        # Test (3 - Disjoint)
        ("Echinus",
         "Doric sütun başlığında yastık şeklinde dışa taşan dairesel taşıyıcı taş bloktur",
         "kirişten gelen dikey baskıyı yumuşatarak sütun gövdesine homojen aktarır",
         "İyonik sütun tabanında yastık şeklinde içe taşan dairesel taşıyıcı taş bloktur",
         "kirişten gelen dikey baskıyı yumuşatarak sütun gövdesine homojen aktarır"),
        ("Kaset Tavan",
         "kubbe veya tonoz iç yüzeyine açılan kare biçimli geometrik girinti gözleridir",
         "strüktürün özgül ağırlığını büyük oranda azaltarak çökme gerilimini düşürür",
         "dış cephe saçak altına açılan kare biçimli geometrik girinti gözleridir",
         "strüktürün özgül ağırlığını büyük oranda azaltarak devrilme gerilimini düşürür"),
        ("Triglif",
         "Doric friz kuşağında üç düşey yive sahip taş plaka dizisidir",
         "ahşap tapınak mimarisindeki kiriş uçlarının taş mimarideki sembolik yansımasıdır",
         "Korint korniş kuşağında üç düşey yive sahip taş plaka dizisidir",
         "antik çeşme mimarisindeki kiriş uçlarının taş mimarideki sembolik yansımasıdır")
    ]
}

TEMPLATES_DOC = [
    "{entity}, {feat}. Bilimsel araştırmalarda {imp}. Bu olgu literatürde kapsamlı şekilde incelenmiştir.",
    "Doğal ve tarihsel kayıtlarda {entity} öne çıkar. Çünkü {entity}, {feat}. Nitekim {imp}.",
    "Yapısal incelemeler {entity} varlığını teyit eder. {entity}, {feat}. Gözlemlere göre {imp}.",
    "{entity} üzerine yapılan analizler {feat} olduğunu gösterir. Bu durum {imp} sonucunu doğurur."
]

TEMPLATES_Q = [
    "{entity} kavramının temel niteliği ve önemi nedir?",
    "Metne göre {entity} hangi yapısal özelliğiyle tanımlanmaktadır?",
    "{entity} hakkında sunulan temel bilimsel açıklama nedir?",
    "Belgede {entity} ile ilgili öne çıkan fonksiyon nedir?"
]

TEMPLATES_ANS = [
    "{entity}, {feat}. Bu sayede {imp}.",
    "Metinde açıklandığı gibi {entity}, {feat} ve {imp}.",
    "{entity}; {feat}. Bilimsel olarak {imp}.",
    "Sağlanan verilere göre {entity}, {feat} olup {imp}."
]


def make_record(domain: str, entity: str, feat: str, imp: str, is_cf: bool, seed_idx: int) -> Dict[str, Any]:
    t_doc = TEMPLATES_DOC[seed_idx % len(TEMPLATES_DOC)]
    t_q = TEMPLATES_Q[seed_idx % len(TEMPLATES_Q)]
    t_ans = TEMPLATES_ANS[seed_idx % len(TEMPLATES_ANS)]

    doc = t_doc.format(entity=entity, feat=feat, imp=imp)
    q = t_q.format(entity=entity)
    ans = t_ans.format(entity=entity, feat=feat, imp=imp)

    return {
        "instruction": "Belgeye dayanarak soruyu yanıtla.",
        "input": f"<BELGE> {doc} </BELGE> {q}",
        "output": ans,
        "entity": entity,
        "domain": domain,
        "is_counterfactual": is_cf
    }


def rankdata_average(a: np.ndarray) -> np.ndarray:
    """Computes average ranks for ties (standard Mann-Whitney / ROC-AUC)."""
    unique, counts = np.unique(a, return_counts=True)
    ranks = np.empty(len(a), dtype=float)
    idx = 0
    for val, count in zip(unique, counts):
        avg_rank = (idx + 1 + idx + count) / 2.0
        ranks[a == val] = avg_rank
        idx += count
    return ranks


def compute_single_feature_auc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = np.array([1 if r.get("is_counterfactual") else 0 for r in records])
    if len(np.unique(labels)) < 2:
        return {"error": "Need both classes"}

    features = {
        "input_char_len": np.array([len(r["input"]) for r in records], dtype=float),
        "input_word_count": np.array([len(r["input"].split()) for r in records], dtype=float),
        "output_char_len": np.array([len(r["output"]) for r in records], dtype=float),
        "output_word_count": np.array([len(r["output"].split()) for r in records], dtype=float),
        "comma_count": np.array([r["input"].count(",") for r in records], dtype=float),
        "semicolon_count": np.array([r["output"].count(";") for r in records], dtype=float),
        "q_char_len": np.array([len(r["input"].split("</BELGE> ")[-1]) for r in records], dtype=float)
    }

    # Belirteç anahtar kelimeler
    MARKER_WORDS = ["beklenenin", "tersine", "evrimleşmiştir", "aslında", "oysa", "yanılsama"]
    for mw in MARKER_WORDS:
        features[f"marker_{mw}"] = np.array([1.0 if mw in (r["input"] + " " + r["output"]) else 0.0 for r in records])

    def calc_auc(y_true, y_score):
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        ranks = rankdata_average(y_score)
        u_stat = np.sum(ranks[y_true == 1]) - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))

    auc_results = {}
    for feat_name, scores in features.items():
        raw_auc = calc_auc(labels, scores)
        effective_auc = max(raw_auc, 1.0 - raw_auc)
        auc_results[feat_name] = {
            "raw_auc": round(raw_auc, 4),
            "effective_auc": round(effective_auc, 4),
            "is_neutral": abs(effective_auc - 0.50) <= 0.05
        }

    return auc_results


def main():
    print("=================================================================")
    print("   NÖTRLEŞTİRİLMİŞ KARŞI-OLGUSAL (CF) KÜLLİYAT ÜRETİMİ (SIMETRİK)  ")
    print("=================================================================\n")

    train_entities = []
    val_entities = []
    test_entities = []

    for dom, ent_list in SYMMETRIC_ENTITIES.items():
        train_entities.extend([(dom, e[0], e[1], e[2], e[3], e[4]) for e in ent_list[:8]])
        val_entities.extend([(dom, e[0], e[1], e[2], e[3], e[4]) for e in ent_list[8:10]])
        test_entities.extend([(dom, e[0], e[1], e[2], e[3], e[4]) for e in ent_list[10:]])

    print(f"Kardinalite (Tekil Öğe) -> Train: {len(train_entities)} (40 Tekil Öğe), Val: {len(val_entities)}, Test: {len(test_entities)} (18 Tekil Öğe)")

    # 1. TEST-CF (50 CF Örnek - Test Varlıklarından)
    test_cf_records = []
    seed = 5000
    while len(test_cf_records) < 50:
        for dom, ent, feat_nat, imp_nat, feat_cf, imp_cf in test_entities:
            seed += 1
            rec = make_record(dom, ent, feat_cf, imp_cf, is_cf=True, seed_idx=seed)
            test_cf_records.append(rec)
            if len(test_cf_records) >= 50:
                break

    with open(os.path.join(OUT_DIR, "test_cf_50.jsonl"), "w", encoding="utf-8") as f:
        for r in test_cf_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Test-CF (N=50) kaydedildi: {os.path.join(OUT_DIR, 'test_cf_50.jsonl')}")

    # 2. VAL SETİ (200 Doğal + 100 CF)
    val_records = []
    seed = 6000
    while len(val_records) < 300:
        for dom, ent, feat_nat, imp_nat, feat_cf, imp_cf in val_entities:
            seed += 1
            is_cf = (len(val_records) % 3 == 0) # %33 CF
            feat = feat_cf if is_cf else feat_nat
            imp = imp_cf if is_cf else imp_nat
            rec = make_record(dom, ent, feat, imp, is_cf=is_cf, seed_idx=seed)
            val_records.append(rec)
            if len(val_records) >= 300:
                break

    with open(os.path.join(OUT_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Val Seti (N=300: 200 Doğal + 100 CF) kaydedildi: {os.path.join(OUT_DIR, 'val.jsonl')}")

    # 3. TRAIN-CF (3.000 Örnek: 2.100 Doğal + 900 CF)
    train_nat_file = os.path.join(OUT_DIR, "train_natural_3000.jsonl")
    with open(train_nat_file, "r", encoding="utf-8") as f:
        train_nat_all = [json.loads(l) for l in f]

    train_cf_combined = []
    # 2100 Doğal
    train_cf_combined.extend(train_nat_all[:2100])

    # 900 Nötrleştirilmiş CF
    cf_generated = []
    seed = 7000
    while len(cf_generated) < 900:
        for dom, ent, feat_nat, imp_nat, feat_cf, imp_cf in train_entities:
            seed += 1
            rec = make_record(dom, ent, feat_cf, imp_cf, is_cf=True, seed_idx=seed)
            cf_generated.append(rec)
            if len(cf_generated) >= 900:
                break

    train_cf_combined.extend(cf_generated)
    random.seed(RANDOM_SEED)
    random.shuffle(train_cf_combined)

    cf_train_file = os.path.join(OUT_DIR, "train_cf_3000.jsonl")
    with open(cf_train_file, "w", encoding="utf-8") as f:
        for r in train_cf_combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Train-CF (N=3.000: 2.100 Doğal + 900 Nötr CF) kaydedildi: {cf_train_file}")

    # 4. TEK ÖZELLİKLİ SINIFLANDIRICI AUC DOĞRULAMASI
    print("\n--- TEK ÖZELLİKLİ SINIFLANDIRICI AUC DOĞRULAMASI (Hedef: AUC ≈ 0.500) ---")
    auc_report = compute_single_feature_auc(train_cf_combined)
    
    all_neutral = True
    for feat_name, res in auc_report.items():
        status = "NÖTR [PASS]" if res["is_neutral"] else "AYIRT EDİLEBİLİR [ALARM]"
        if not res["is_neutral"]:
            all_neutral = False
        print(f"  {feat_name:22s}: Ham AUC={res['raw_auc']:.4f} | Efektif AUC={res['effective_auc']:.4f} -> {status}")

    report_path = os.path.join(OUT_DIR, "cf_neutralization_report.json")
    final_output = {
        "timestamp": "2026-09-14",
        "cardinality": {
            "train_unique_entities": len(train_entities),
            "test_unique_entities": len(test_entities),
            "val_unique_entities": len(val_entities),
            "total_records": len(train_cf_combined),
            "natural_records": 2100,
            "cf_records": 900
        },
        "all_features_neutral": all_neutral,
        "feature_aucs": auc_report
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"\nRapor kaydedildi: '{report_path}'")
    print(f"Nihai Hüküm: {'TÜM ÖZELLİKLER NÖTR (AUC ≈ 0.500) - BAŞARILI' if all_neutral else 'BAZI ÖZELLİKLER SIZINTI YAPIYOR'}\n")


if __name__ == "__main__":
    main()
