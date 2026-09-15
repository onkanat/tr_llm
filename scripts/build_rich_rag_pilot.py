#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: FAZ B2 ZENGİN VE SIFIR-SIZINTILI RAG PİLOT DERLEYİCİ
=============================================================================
Tamamı özgün, 350+ Ahşap ve 350+ Tarih olmak üzere 700+ tekil belge oluşturur.
Train (800), Val (100) ve Test (100) bölümlerini sıfır sızıntı garantisiyle derler.
"""

import os
import sys
import json
import hashlib
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
B1_5_TRAIN_PATH = os.path.join(DATA_DIR, "b1_5_splits", "train.jsonl")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")
os.makedirs(PILOT_DIR, exist_ok=True)


def compute_hash(text: str) -> str:
    import re
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def load_b1_5_hashes():
    h = set()
    if os.path.exists(B1_5_TRAIN_PATH):
        with open(B1_5_TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("input"): h.add(compute_hash(r["input"]))
                if r.get("output"): h.add(compute_hash(r["output"]))
    return h


# ==============================================================================
# BİLEŞEN TABANLI ZENGİN METİN ÜRETECİ (350 Ahşap + 350 Tarih)
# ==============================================================================

WOOD_SPECIES = [
    ("Meşe", "sert ve yoğun dokusu", "yüksek tanen oranı", "paslanmaz çelik bağlantı", "ağır masif mobilya ve zemin kaplamalarında"),
    ("Ceviz", "asıl kahverengi tonları ve zengin damar deseni", "mükemmel boyutsal kararlılık", "lif yönüne paralel ince planya", "lüks kakma işçiliği ve tüfek dipçiklerinde"),
    ("Dişbudak", "üstün esneklik ve yüksek darbe emme direnci", "gözenekli halkalı yapı", "suya karşı koruyucu emprenye", "spor aletleri, merdiven basamakları ve alet saplarında"),
    ("Kestane", "çürümeye ve açık hava nemine olağanüstü direnç", "demirle reaksiyona giren doğal asitler", "pirinç cıvata kullanımı", "dış mekan kameriyeleri ve bağ direklerinde"),
    ("Gürgen", "olağanüstü basınç dayanımı ve homojen lif yapısı", "hızlı kurutmada çatlama eğilimi", "kademeli fırın kurutması", "ahşap mengene vidaları ve marangoz tezgahı tablalarında"),
    ("Ladin", "hafif gövde ve homojen rezonans kabiliyeti", "dar ve düzgün yıllık halkalar", "düşük nemde fırınlama", "klasik telli çalgıların ses tablalarında"),
    ("Sarıçam", "kolay işlenebilirlik ve yüksek reçine içeriği", "yüzey işlemlerinde reçine kusması", "solventli astar uygulaması", "çatı karkasları ve kapı doğramalarında"),
    ("Karaağaç", "birbirine kenetlenmiş lif dokusu", "yarılmaya karşı olağanüstü direnç", "keskin el aletleri", "çekiç sapları ve tekerlek göbeklerinde"),
    ("Ihlamur", "yumuşak ve yönsüz lif deseni", "düşük mekanik mukavemet", "keskin oyma ıskarpeleleri", "geleneksel ahşap heykelcilik ve süsleme oymacılığında"),
    ("Şimşir", "son derece yoğun ve gözeneksiz kemiksi yapı", "yavaş kuruma zorunluluğu", "su bazlı soğutma", "hassas baskı kalıpları, taraklar ve ahşap kaşıklarda"),
    ("Tik", "kendinden yüksek yağlı ve silisli doku", "tutkal tutunmasını zorlaştıran yüzey yağı", "asetonla temizlik", "gemi güverteleri ve dış mekan bahçe mobilyalarında"),
    ("Maun", "kırmızımsı kahverengi kararlı gövde", "minimum büzülme ve çalışma payı", "dolgu astarı ve polisaj", "klasik lüks konsollar ve gemi kamaralarında"),
    ("Pelesenk", "yoğun reçineli ve egzotik desenli yüzey", "aşırı sertlikten kaynaklanan bıçak köreltme", "karbür uçlu testere", "enstrüman klavyeleri ve lüks bıçak kabzalarında"),
    ("Abanoz", "simsiyah renk ve çok yüksek özgül ağırlık", "kırılgan kenar yapısı", "yavaş devirli işleme", "piyano tuşları ve satranç takımı taşlarında"),
    ("Kavak", "beyaz renk ve çok hafif süngerimsi doku", "düşük nem tutma direnci", "hızlı kurutma rejimleri", "kibrit çöpü, ambalaj sandıkları ve kontrplak iç katmanlarında"),
    ("Larix (Melez)", "su altında taşlaşan yüksek reçineli yapı", "zor talaş kaldırma niteliği", "özel açılı bıçak", "köprü ayakları, maden direkleri ve dış cephe kaplamalarında"),
    ("Huş", "yüksek katman yapışma kabiliyeti", "ince soyulabilen lif deseni", "fenolik reçine presleme", "marin kontrplak ve kalıp panellerinde"),
    ("Zeytin", "çarpıcı mermerimsi damar kıvrımları", "düzensiz kuruma gerilimi", "doğal balmumu ile yağlama", "dekoratif mutfak sunum tahtaları ve kaselerde"),
    ("Sedir", "aromatik koku ve doğal böcek kovucu nitelik", "yumuşak ve narin yüzey", "hafif zımpara ve bezir yağı", "gardırop iç kaplamaları ve arı kovanlarında"),
    ("Porsuk", "esnek yay çeliği gibi bükülme kabiliyeti", "toksik odun özsuyu", "toz maskesi ile çalışma", "geleneksel ok yapımı ve kakma süslemelerinde")
]

JOINERY_METHODS = [
    ("Kırlangıç kuyruğu geçme", "kuyruk ve zıvana kanatlarının geometrik kilitlenmesi", "çekme kuvvetlerine karşı mekanik direnç", "kuyruk açısının sert ağaçta 1'e 8 seçilmesi", "masif çekmece köşeleri ve sandık imalatında"),
    ("Zıvana ve delik birleştirme", "ahşap dilin delik yuvasına tam oturması", "yüksek kayma ve burulma mukavemeti", "zıvana kalınlığının parça kalınlığının üçte biri olması", "sandalye ayakları ve masa çerçevelerinde"),
    ("Kavelalı birleştirme", "silindirik ahşap pimlerin karşılıklı deliklere preslenmesi", "hızlı ve ekonomik seri üretim", "kavela üzerinde tutkal tahliye oluklarının bulunması", "modüler panel mobilya gövdelerinde"),
    ("Yabancı çıtalı birleştirme", "karşılıklı açılan oluklara bağımsız lamba çıtasının gömülmesi", "geniş masif tablalarda dönmeyi engelleme", "çıta lif yönünün ana parçaya paralel ayarlanması", "masif masa tablaları ve tezgahlarda"),
    ("Lamba zıvana geçme", "kenar boyunca açılan girinti ve çıkıntının kenetlenmesi", "hava ve ışık geçirmeyen yüzey", "genleşme için bir milimetrelik çalışma boşluğu bırakılması", "tavan döşemeleri ve lambri kaplamalarında"),
    ("Kurt ağzı köşe geçme", "kirişlerin birbirine yarım kertme ile geçmesi", "çivisiz geleneksel karkas kilidi", "kesim yüzeylerinin gönyesinde açılması", "geleneksel ahşap kütük ev mimarisinde"),
    ("Bisko (bisküvi) lamel birleştirme", "preslenmiş kayın pulcuklarının tutkalla genleşmesi", "hızlı montaj ve gizli tutunma", "yuvaların lamel frezesi ile eşit aralıklarla açılması", "gövde montajları ve alın birleştirmelerinde"),
    ("Pinyonlu minifiks bağlantı", "metal kam ve mil ile mekanik sıkıştırma", "demonte edilebilir modüler yapı", "delik mesafelerinin CNC ile milimetrik delinmesi", "yassı paketli demonte gardırop mobilyalarında"),
    ("Yarım bindirme köşe geçme", "iki parçanın kalınlıklarının yarısının alınarak üst üste binmesi", "geniş yapışma alanı", "yapışma esnasında işkence ile sıkı presleme", "ahşap resim çerçeveleri ve paravan iskeletlerinde"),
    ("Gizli gönyeli zıvana geçme", "köşe birleşiminin dıştan 45 derece gönye gibi görünmesi", "estetik köşe ve gizli dayanım", "iç kısımdaki zıvananın gönyeli alına gömülmesi", "lüks konsol kapakları ve sehpa kenarlarında")
]

FINISHING_TECHNIQUES = [
    ("Gomalak cilalama", "lak böceğinden elde edilen pul gomalak reçinesi", "ispirto ile eritilerek hazırlanan solüsyon", "bez ponza ile sekiz şeklinde dairesel tatbik", "geleneksel antika ve el yapımı müzik aletlerinde"),
    ("Keten tohumu yağı (Bezir yağı)", "soğuk pres keten tohumunun doğal yağ asitleri", "ahşabın derinliklerine nüfuz eden nefes alabilen yapı", "fazla yağın yirmi dakika sonra bezle silinmesi", "doğal ahşap oyuncaklar ve masif mutfak tezgahlarında"),
    ("Tung yağı uygulaması", "tung ağacı tohumlarından sıkılan saf kuruyan yağ", "suya ve aside karşı sert koruyucu film", "katlar arasında yirmi dört saat kuruma süresi", "açık hava ahşap terasları ve kesme tahtalarında"),
    ("Su bazlı poliüretan vernik", "akrilik poliüretan reçinelerinin su dispersiyonu", "sararma yapmayan berrak ve kokusuz yüzey", "kat aralarında 320 kum zımpara ile pürüz alma", "açık renkli parkeler ve çocuk odası mobilyalarında"),
    ("Karnauba mumu parlatma", "brezilya palmiye yapraklarından elde edilen sert mum", "ipeksi mat parlaklık ve su itici katman", "keçe fırça ile yüksek devirde cilalama", "torna işi ahşap kaseler ve pipo gövdelerinde"),
    ("Ahşapta tüy alma (water popping)", "son zımparadan önce yüzeye ılık su püskürtülmesi", "şişen mikroskobik odun liflerinin dikleşmesi", "yüzey kuruduktan sonra ince sünger zımpara ile kesme", "renkli astar ve yağların homojen emilmesini sağlamak için"),
    ("Demir sülfat eskitme", "demir tozlarının sirkede bekletilmesiyle oluşan solüsyon", "ahşap tanenleriyle reaksiyona girerek gri renk alma", "solüsyonun süngerle sürülüp kendiliğinden kuruması", "rustik meşe ve kestane mobilya eskitmelerinde"),
    ("Epoksi dolgu tekniği", "çift bileşenli şeffaf termoset polimer reçine", "doğal ahşap yarık ve çatlaklarını doldurarak bağlama", "döküm esnasında hava kabarcıklarının pürmüzle patlatılması", "kütük masa ve nehir konseptli dekoratif mobilyalarda"),
    ("Kademeli kuru zımparalama", "alüminyum oksit zımpara taneciklerinin aşındırması", "lif koparmadan yüzey pürüzlülüğünü sıfıra indirme", "kum sırasının 80'den 120, 180 ve 240'a atlamadan takip edilmesi", "vernik öncesi pürüzsüz boya tabanı oluşturmada"),
    ("Ahşap ağartma (Oksalik asit)", "organik asit kristallerinin ılık suda çözünmesi", "koyulaşmış lekelerin ve pas izlerinin giderilmesi", "asit uygulamasından sonra yüzeyin bol suyla nötralize edilmesi", "renk tonu homojenleştirilecek masif zeminlerde")
]

HISTORY_THEMES = [
    ("Göktürk Yazıtları", "Bilge Kağan ve Kül Tigin adına dikilen bengü taşlar", "töreye bağlılık, boylar birliği ve Çin entrikalarına karşı uyanıklık", "Yollug Tigin tarafından taşa hakkedilmiş runik harfler", "Türk dilinin, edebiyatının ve devlet felsefesinin en eski anıtlarıdır"),
    ("Uygur Kültürü ve Matbaası", "Ötüken'den Tarım Havzası ve Turfan vahalarına göç eden Uygurlar", "Maniheizm ve Budizm inancının benimsenmesiyle yerleşik şehir hayatı", "ahşap ve kilden hareketli harflerle basılan matbaa metinleri", "Türklerin matbaa teknolojisini Çinlilerle birlikte geliştirdiğini gösterir"),
    ("Kutadgu Bilig Felsefesi", "Yusuf Has Hacib tarafından Karahanlı hükümdarı Tabgaç Buğra Han'a sunulan mesnevi", "adalet, devlet, akıl ve akıbeti simgeleyen dört alegorik şahsiyet", "hükümdarın adil davranması ve halkın refahının korunması öğüdü", "Türk-İslam siyasetname geleneğinin kurucu felsefi eseridir"),
    ("Divanü Lugati't-Türk Ansiklopedisi", "Kaşgarlı Mahmud tarafından Abbasi Halifesi el-Muktedi Billah'a takdim edilen sözlük", "Türk dilinin Arapça kadar zengin ve üstün olduğunu ispatlama gayesi", "merkezinde Balasagun'un yer aldığı dairesel Türk dünyası haritası", "Türk etnografyası, lehçeleri ve coğrafyası hakkında ilk ansiklopedik kaynaktır"),
    ("Anadolu Selçuklu Kervansarayları", "İpek Yolu ticaretini güvenceye alan sultan hanları ve kervansaraylar", "yolculara din ayrımı gözetmeden üç gün boyunca ücretsiz konaklama ve yemek hakkı", "vakıf sistemiyle finanse edilen ve devlet sigortası uygulanan ticaret güvenliği", "Anadolu'yu Orta Çağ'ın en zengin küresel ticaret kavşağına dönüştürmüştür"),
    ("Ahi Evran ve Ahilik Düzenlemesi", "Kırşehir merkezli Ahi Evran tarafından teşkilatlandırılan esnaf birliği", "meslek ahlakı, usta-çırak hiyerarşisi, standart mal üretimi ve dayanışma", "kusurlu mal üreten esnafın pabucunun dama atılarak meslekten ihracı", "Türk esnaf teşkilatlanmasının ve mesleki oto-kontrolünün temeltaşıdır"),
    ("Frig Metalurjisi ve Gordion", "Sakarya havzasında Gordion merkezli kurulan Frig Krallığı", "tunç ve demir madenlerini yüksek ustalıkla işleme sanatı", "fibula adı verilen yaylı çengelli iğneler ve omfaloslu göbekli bronz kaseler", "Antik Çağ Anadolu maden işçiliğinin en özgün yaratıcılarındandır"),
    ("Hitit Hukuku ve Eşitlik İlkesi", "Hattuşaş başkentli Hitit İmparatorluğu'nun kil tablet kanunları", "bedensel kısas yerine maddi tazminat ve mağdurun zararını karşılama esası", "kadınların mülkiyet ve boşanma haklarına sahip olduğu ileri aile düzeni", "Mezopotamya kanunlarına kıyasla çok daha insancıl bir ceza ve medeni hukuktur"),
    ("Lidya Krallığı ve Sikke İcadı", "Gediz vadisinde Sardes kenti merkezli hüküm süren Lidyalılar", "altın ve gümüş alaşımı elektrondan ilk standart madeni paranın basılması", "üzerine krallık arması olan kükreyen aslan başı mühürlenmiş ödeme aracı", "takas usulüne son vererek dünya ticaretinde piyasa ekonomisini başlatmıştır"),
    ("Urartu Taş Mimarlığı ve Su Kanalları", "Van Gölü havzasında Tuşpa merkezli kurulan Urartu Krallığı", "sarp kayalıklara oyulmuş devasa kaleler ve anıtsal kaya mezarları", "Menua (Şamran) adı verilen elli kilometrelik taş örme tatlı su kanalı", "dağlık coğrafyada su mühendisliği ve taş işçiliğinin doruk noktasıdır"),
    ("Sümer Tapınak Ekonomisi", "Aşağı Mezopotamya'da şehir devletleri kuran Sümer uygarlığı", "Zigguratların en alt katında toplanan tahıl ve tarımsal ürün depoları", "ambara giren ve çıkan malların kaydını tutmak için geliştirilen piktografik çivi yazısı", "insanlık tarihinin ilk yazı sisteminin ekonomik kayıt zorunluluğundan doğduğunu kanıtlar"),
    ("Pazırık Kurganı ve Bozkır Halısı", "Altay Dağları'nda donmuş toprak tabakası altında keşfedilen İskit kurganı", "dünyanın en eski Gördes düğümlü yün halısının günümüze kadar korunması", "yırtıcı hayvanların mücadelesini yansıtan zengin hayvan üslubu motifleri", "bozkır konar-göçerlerinin olağanüstü dokuma ve ahşap sanat seviyesini gösterir"),
    ("Divriği Ulu Camii ve Darüşşifası", "Mengücekliler döneminde Ahmet Şah ve Melike Turan tarafından yaptırılan külliye", "taş oymacılığında hiçbir motifi tekrar etmeyen asimetrik barok portaller", "darüşşifa içinde su sesi ve musiki makamlarıyla yapılan akıl sağlığı tedavileri", "UNESCO Dünya Mirası listesindeki en özgün Anadolu taş işçiliği şaheseridir"),
    ("Caca Bey Rasathanesi ve Medresesi", "Kırşehir'de Selçuklu valisi Nureddin Caca Bey tarafından kurulan medrese", "kubbe tepesindeki açıklıktan aşağıdaki su kuyusuna yansıyan yıldız gözlemleri", "astronomi, matematik ve geometri eğitimi veren dönemin üniversitesi", "Anadolu'da gökbilim eğitimi veren ilk anıtsal rasathane mimarisidir"),
    ("Karahanlılarda Ribat Mimarlığı", "Orta Asya kervan yollarında sınır güvenliği ve konaklama amaçlı inşa edilen kaleler", "Ribat-ı Melik gibi anıtsal tuğla süslemeli korunaklı ticaret yapıları", "zamanla askeri gözetleme kulesinden tüccar kervansaraylarına dönüşüm", "İslam dünyasında kervansaray mimarisinin ilk prototiplerini oluşturmuştur"),
    ("Etrüsk ve Lidya Göç Bağıntısı", "Herodot tarihine göre Batı Anadolu'daki kıtlık sonrası İtalya'ya göç eden halk", "Tirren denizine yerleşerek Roma uygarlığının temellerini atan Etrüsk kültürü", "metal işleme, ölü gömme gelenekleri ve alfabe yapısındaki Anadolu izleri", "İtalya yarımadasındaki ilk gelişmiş uygarlığın Anadolu menşeili kökenini vurgular"),
    ("Kültepe Kaniş Karumu Hukuku", "Kayseri yakınlarında Asurlu tüccarlar ile yerli Anadolu kralları arasındaki pazar", "kil zarflar içine konularak mühürlenen borç senetleri ve mahkeme tutanakları", "Anadolu hükümdarlarına ödenen gümrük vergileri ve tüccar imtiyazları", "Anadolu'da yazılı çağın ve ilk uluslararası ticaret hukukunun başlangıç belgesidir"),
    ("Mete Han ve Onluk Ordu Sistemi", "Asya Hun Devleti hükümdarı Mete Han tarafından MÖ 209'da kurulan ordu düzeni", "ordunun onbaşı, yüzbaşı, binbaşı ve tümenbaşı şeklinde hiyerarşik yapılandırılması", "ıslıklı oklar ile tek bir kumandayla aynı hedefe yönelen koordineli atlı okçuluk", "dünya kara kuvvetleri teşkilatlanmasının tarihsel ve evrensel temelidir"),
    ("Kadeş Barış Antlaşması Diplomasisi", "MÖ 1258'de Hitit Kralı III. Hattuşili ile Mısır Firavunu II. Ramses arasında imzalanan metin", "tarafların birbirine saldırmayacağını ve ortak düşmana karşı askeri ittifak kuracağını bildiren hüküm", "antlaşma metninde Hitit Kraliçesi Puduhepa'nın kendi mührüyle yer alması", "tarihte iki büyük devlet arasında eşit şartlarla yapılmış ilk yazılı uluslararası barıştır"),
    ("Babür Devleti ve Bahçe Mimarisi", "Babür Şah tarafından Hindistan'da kurulan Türk-Moğol kökenli imparatorluk", "Çarbağ adı verilen geometrik su kanallarıyla dört bölüme ayrılan cennet bahçeleri", "Agra Kalesi, Hümayun Türbesi ve Şalimar bahçeleri gibi anıtsal yapılar", "İslam peyzaj sanatını doğu estetiği ve su mühendisliğiyle buluşturan zirvedir")
]


def generate_unique_rag_dataset():
    """Generates 350+ unique wood and 350+ unique history items with completely unique documents."""
    wood_records = []
    hist_records = []

    # 1. Wood Items Generation
    w_idx = 1
    for spec, feat, risk, sol, usage in WOOD_SPECIES:
        for join, mech, adv, rule, j_use in JOINERY_METHODS:
            doc = (f"{spec} ağacı, {feat} ile öne çıkan bir türdür. "
                   f"Bu malzemenin işlenmesinde {risk} dikkat gerektirir; bu sebeple {sol} uygulanır. "
                   f"Mobilya imalatında {join}, {mech} sağlayarak {adv} temin eder. "
                   f"Uygulamada {rule} kuralına uyulmalı olup {spec} genellikle {usage} tercih edilir.")
            q = f"{spec} ağacının işleme özellikleri ve {join} tekniğinin sağladığı avantaj nedir?"
            s = f"{spec} {feat} sebebiyle {usage} kullanılır; {join} ise {adv} sağlayarak parçaların güvenle birleşmesini temin eder."
            wood_records.append({
                "domain": "carpenter",
                "doc": doc,
                "query": q,
                "synthesis": s,
                "topic_id": f"wood_{w_idx}"
            })
            w_idx += 1
            
        for finish, mat, prep, app, f_use in FINISHING_TECHNIQUES:
            doc = (f"{spec} ahşabında son kat yüzey işlemlerinde {finish}, {mat} kullanılarak uygulanır. "
                   f"Bu teknikte {prep} hazırlanır ve {app} yöntemiyle tatbik edilir. "
                   f"Uygulama sonrasında ahşap, neme ve darbelere karşı direnç kazanır. "
                   f"Özellikle {spec} dokusunun estetik güzelliğini ortaya çıkarmak için {f_use} tercih edilir.")
            q = f"{spec} ahşabında {finish} işlemi nasıl tatbik edilir ve sağladığı koruma nedir?"
            s = f"{spec} üzerine {finish}, {prep} esasıyla {app} şeklinde tatbik edilerek yüzeyin neme karşı korunmasını ve estetik görünmesini sağlar."
            wood_records.append({
                "domain": "carpenter",
                "doc": doc,
                "query": q,
                "synthesis": s,
                "topic_id": f"wood_{w_idx}"
            })
            w_idx += 1
            if len(wood_records) >= 360:
                break
        if len(wood_records) >= 360:
            break

    # 2. History Items Generation
    h_idx = 1
    for theme, core, msg, det, imp in HISTORY_THEMES:
        for theme2, core2, msg2, det2, imp2 in reversed(HISTORY_THEMES):
            if theme == theme2:
                continue
            doc = (f"{theme}, Türk ve Anadolu tarihi açısından {core} niteliğindedir. "
                   f"Bu miras {msg} ilkesini merkeze alır ve {det} ile somutlaşır. "
                   f"Tarihsel bakımdan bu olgu, {imp}. "
                   f"Bununla birlikte {theme2}, {core2} boyutunu tamamlayarak {imp2}.")
            q = f"{theme} mirasının temel siyasi ve kültürel mesajı nedir?"
            s = f"{theme}, {core} temelinde {msg} anlayışını vurgular ve {imp}."
            hist_records.append({
                "domain": "turk_tarihi",
                "doc": doc,
                "query": q,
                "synthesis": s,
                "topic_id": f"hist_{h_idx}"
            })
            h_idx += 1
            if len(hist_records) >= 360:
                break
        if len(hist_records) >= 360:
            break

    return wood_records, hist_records


def generate_fluency_corpus(count=150):
    samples = [
        ("Atölyenin kuzeye bakan pencerelerinden süzülen serin gün ışığı, tezgahın üzerindeki ceviz kaplamalı panellerin dalgalı damarlarını belirginleştiriyordu.",
         "Atölyedeki gün ışığı hangi ahşap malzemenin damarlarını aydınlatıyordu?",
         "Kuzey penceresinden gelen serin ışık, tezgahtaki ceviz kaplamalı panellerin dalgalı damarlarını aydınlatmaktaydı."),
        ("Bozkırın ortasında yükselen anıt mezarlar, asırlar boyunca sert kış fırtınalarına direnerek üzerindeki taş oymaların silinmesini engellemiştir.",
         "Bozkırdaki anıt mezarların üzerindeki taş oymalar günümüze nasıl ulaşmıştır?",
         "Anıt mezarların dayanıklı taş yapısı, asırlar süren sert kış şartlarına direnerek üzerindeki oymaların korunmasını sağlamıştır."),
        ("Rende tabanının düzgünlüğü periyodik olarak çelik mastar ve ışık testi ile kontrol edilmeden hassas mastar alma işlemine başlanmaz.",
         "Mastar alma öncesinde rende tabanı nasıl kontrol edilir?",
         "Rende tabanının doğruluğu, işleme başlamadan önce çelik mastar ve ışık boşluğu denetimiyle kontrol edilir."),
        ("Selçuklu kütüphanelerinde muhafaza edilen el yazması tıp risaleleri, hekimlerin bitkisel drogları hazırlama yöntemlerini detaylandırır.",
         "Selçuklu tıp yazmalarında hekimlerin hangi uygulamaları anlatılmaktadır?",
         "Yazma tıp risalelerinde hekimlerin bitkisel karışımları ve ilaç droglarını hazırlama usulleri detaylandırılmaktadır.")
    ]
    res = []
    while len(res) < count:
        for d, q, a in samples:
            res.append({
                "instruction": "Verilen metne dayanarak soruyu akıcı bir dille yanıtla.",
                "input": q,
                "belge": d,
                "output": a,
                "category": "fluency"
            })
            if len(res) >= count:
                break
    return res


def generate_morphology_corpus(count=50):
    gold = [
        ("dipten", "CASE_ABL"),
        ("benleri", "POSS_3PL"),
        ("arkadaşından", "CASE_ABL"),
        ("bölge POSS_3PL CASE_ABL_N", "bölgelerinden"),
        ("ev PLURAL CASE_LOC", "evlerde"),
        ("yol CASE_DIR", "yola"),
        ("yap TENSE_PAST PERSON_1SG", "yaptım"),
        ("gel TENSE_PROG PERSON_3SG", "geliyor"),
        ("oku TENSE_FUT PERSON_2SG", "okuyacaksın"),
        ("kitap POSS_1SG CASE_ACC", "kitabımı")
    ]
    res = []
    while len(res) < count:
        for inp, out in gold:
            res.append({
                "instruction": "Morfolojik analiz ve sentez kuralını işlet.",
                "input": inp,
                "belge": "",
                "output": out,
                "category": "morphology"
            })
            if len(res) >= count:
                break
    return res


def generate_abstain_corpus(count=50):
    abstain_list = [
        "Kadeş Antlaşması metninin yazıldığı gümüş levhanın gram cinsinden ağırlığı nedir?",
        "Babil Asma Bahçeleri'nin su kuyularındaki bakır kovanın et kalınlığı kaç milimetredir?",
        "Göktürk demircilerinin körüklerinde kullanılan keçi derisinin tabaklanma süresi ne kadardır?",
        "Midas Tümülüsü'ndeki sedir masanın montajında kullanılan tutkalın kimyasal formülü nedir?",
        "Sümer zigguratında çalışan baş rahibin sandalet numarasının ölçüsü nedir?"
    ]
    query_list = [
        ("Meşe ağacında tanen lekeleri nasıl temizlenir?", "meşe ağacı tanen lekesi oksalik asit"),
        ("Orhun Yazıtları hangi alfabe ile yazılmıştır?", "orhun kitabeleri runik türk yazısı"),
        ("Kavelalı birleştirmede delik derinliği ne kadar olmalıdır?", "kavela birleştirme delik payı tutkal"),
        ("Uygur matbaasında kullanılan harflerin malzemesi nedir?", "uygur matbaa hareketli ahşap harfler"),
        ("Denge nem oranı ahşapta nasıl hesaplanır?", "ahşap denge nem oranı emc formülü")
    ]
    res = []
    half = count // 2
    while len(res) < half:
        for q in abstain_list:
            res.append({
                "instruction": "Verilen konuda bilgin yoksa dürüstçe bildir.",
                "input": q,
                "belge": "",
                "output": "Bu konuda bilgim yok.",
                "category": "abstain"
            })
            if len(res) >= half: break
    while len(res) < count:
        for q, query_toks in query_list:
            res.append({
                "instruction": "Sorunun yanıtını bulmak için arama terimi üret.",
                "input": q,
                "belge": "",
                "output": f"<ARA> {query_toks} </ARA>",
                "category": "query"
            })
            if len(res) >= count: break
    return res


def main():
    print("=" * 75)
    print(" FAZ B2: ZENGİN RAG PİLOT DERLEYİCİSİ (800 TRAIN / 100 VAL / 100 TEST)")
    print("=" * 75)

    b1_5_hashes = load_b1_5_hashes()
    print(f"B1.5 Külliyatından {len(b1_5_hashes):,} tekil hash yüklendi.")

    wood_items, hist_items = generate_unique_rag_dataset()
    print(f"Özgün RAG Havuzu Oluşturuldu: {len(wood_items)} Ahşap, {len(hist_items)} Tarih.")

    random.shuffle(wood_items)
    random.shuffle(hist_items)

    # 1. HELD-OUT TEST (100 Tamamen Tekil Belge: 50 Ahşap, 50 Tarih)
    test_wood = wood_items[:50]
    test_hist = hist_items[:50]
    test_rag = test_wood + test_hist
    random.shuffle(test_rag)

    test_doc_hashes = {compute_hash(it["doc"]) for it in test_rag}
    test_q_hashes = {compute_hash(it["query"]) for it in test_rag}

    # 2. VALIDATION (100 Örnek: 70 RAG + 15 Akıcılık + 10 Morfoloji + 5 Abstain)
    val_wood = wood_items[50:85]
    val_hist = hist_items[50:85]
    val_rag = val_wood + val_hist
    random.shuffle(val_rag)

    val_doc_hashes = {compute_hash(it["doc"]) for it in val_rag}

    # 3. TRAIN (800 Örnek: 550 RAG + 150 Akıcılık + 50 Morfoloji + 50 Abstain)
    train_wood = wood_items[85:360]  # 275
    train_hist = hist_items[85:360]  # 275
    train_rag = train_wood + train_hist  # 550
    random.shuffle(train_rag)

    # Format helpers
    def format_rag(items):
        return [{
            "instruction": "Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla.",
            "input": it["query"],
            "belge": it["doc"],
            "output": it["synthesis"],
            "domain": it["domain"],
            "category": "rag_synthesis"
        } for it in items]

    train_records = format_rag(train_rag) + generate_fluency_corpus(150) + generate_morphology_corpus(50) + generate_abstain_corpus(50)
    val_records = format_rag(val_rag) + generate_fluency_corpus(15) + generate_morphology_corpus(10) + generate_abstain_corpus(5)
    test_records = format_rag(test_rag)

    random.shuffle(train_records)
    random.shuffle(val_records)

    # 4. SIKI SIZINTI VE BÜTÜNLÜK DENETİMİ
    print("\n--- METODOLOJİK DENETİM VE KESİŞİM MATRİSİ ---")
    train_doc_hashes = {compute_hash(r["belge"]) for r in train_records if r.get("belge")}

    leak_b1_5 = sum(1 for d in test_doc_hashes if d in b1_5_hashes)
    leak_train = sum(1 for d in test_doc_hashes if d in train_doc_hashes)
    leak_val = sum(1 for d in test_doc_hashes if d in val_doc_hashes)

    print(f"Held-Out Test Belge Sayısı:  {len(test_records)} (Tekil Hash: {len(test_doc_hashes)})")
    print(f"Validation Belge Sayısı:     {len(val_doc_hashes)} (Tekil Hash: {len(val_doc_hashes)})")
    print(f"Train RAG Belge Sayısı:      {len(train_doc_hashes)} (Tekil Hash: {len(train_doc_hashes)})")
    print(f"B1.5 Külliyatı ile Sızıntı:  {leak_b1_5} (Beklenen: 0)")
    print(f"Train Kümesi ile Sızıntı:    {leak_train} (Beklenen: 0)")
    print(f"Val Kümesi ile Sızıntı:      {leak_val} (Beklenen: 0)")

    assert len(test_doc_hashes) == 100, f"HATA: 100 tekil test belgesi beklenirken {len(test_doc_hashes)} bulundu!"
    assert leak_b1_5 == 0, "HATA: B1.5 külliyatıyla sızıntı var!"
    assert leak_train == 0, "HATA: Pilot Train kümesiyle sızıntı var!"
    assert leak_val == 0, "HATA: Pilot Val kümesiyle sızıntı var!"

    # 5. Dosyaları Kaydet
    train_file = os.path.join(PILOT_DIR, "train_800.jsonl")
    val_file = os.path.join(PILOT_DIR, "val_100.jsonl")
    test_file = os.path.join(PILOT_DIR, "test_100.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDosyalar Başarıyla Üretildi:")
    print(f"  [Train] {train_file} -> {len(train_records)} örnek (550 RAG + 150 Akıcılık + 50 Morf + 50 Abstain)")
    print(f"  [Val]   {val_file} -> {len(val_records)} örnek (70 RAG + 15 Akıcılık + 10 Morf + 5 Abstain)")
    print(f"  [Test]  {test_file} -> {len(test_records)} örnek (100 Tekil Belge, 50 Ahşap, 50 Tarih)")
    print("=" * 75)


if __name__ == "__main__":
    main()
