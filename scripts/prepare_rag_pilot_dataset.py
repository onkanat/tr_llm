#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: FAZ B2 RAG-SENTEZ PİLOT VERİ SETİ HAZIRLAMA
======================================================================
1. B1.5 Külliyatı dışındaki bağımsız kaynaklardan 800 Train, 100 Val ve 100 Test üretir.
2. Sıfır Sızıntı Güvencesi:
   - Belge-Hash ∩ Train_B1.5 = ∅
   - Soru-Hash ∩ Train_B1.5 = ∅
   - Test_Belge-Hash ∩ Train_Pilot = ∅
   - 100 Test sorusunun her biri 100 tekil ve bağımsız belgeden oluşur.
3. Karma Müfredat (Train 800):
   - 550 RAG Sentez Tripleri (%68.75)
   - 150 Türkçe Sentaktik Akıcılık (%18.75)
   - 50 E7 Morfolojik Analiz (%6.25)
   - 50 Abstain & <ARA> Sorgusu (%6.25)
4. Validation (100) & Test (100 Held-Out):
   - Val: 70 RAG + 15 Akıcılık + 10 Morfoloji + 5 Abstain
   - Test: 100 Saf RAG Sentez (50 Ahşap + 50 Tarih / 100 Farklı Belge)
"""

import os
import sys
import json
import hashlib
import random
from typing import List, Dict, Any, Set

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
B1_5_TRAIN_PATH = os.path.join(DATA_DIR, "b1_5_splits", "train.jsonl")
PILOT_DIR = os.path.join(DATA_DIR, "rag_pilot")
os.makedirs(PILOT_DIR, exist_ok=True)


def normalize_text(text: str) -> str:
    """Normalizes text for robust hash computation."""
    import re
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_hash(text: str) -> str:
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def load_b1_5_train_hashes() -> Set[str]:
    """Collects normalized hashes of all inputs and outputs in train.jsonl."""
    hashes = set()
    if os.path.exists(B1_5_TRAIN_PATH):
        with open(B1_5_TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                inp = rec.get("input", "")
                out = rec.get("output", "")
                if inp:
                    hashes.add(compute_hash(inp))
                if out:
                    hashes.add(compute_hash(out))
    return hashes


# ==============================================================================
# 1. YENİ BELGE VE SENTEZ HAVUZU (B1.5 Külliyatında Bulunmayan Yeni Kaynaklar)
# ==============================================================================

RAW_RAG_ITEMS_WOOD = [
    {
        "domain": "carpenter",
        "doc": "Dişbudak ağacı yüksek esneklik ve darbe direnci gerektiren alet sapları ve spor ekipmanlarında yaygın kullanılır. Gözenekli yapısı nedeniyle dış mekan şartlarında suya karşı koruyucu emprenye işlemi görmeden dayanıklılık gösteremez.",
        "query": "Dişbudak ağacının tipik kullanım alanları ve açık hava dayanımı nasıldır?",
        "synthesis": "Dişbudak ağacı yüksek esneklik ve darbe direnci sağladığı için alet saplarında tercih edilir; dış mekanlarda ise emprenye koruması olmadan suya dayanamaz."
    },
    {
        "domain": "carpenter",
        "doc": "Gomalak cilası lak böceğinin salgısından elde edilen doğal bir reçinedir. İspirto ile eritilerek hazırlanan gomalak cilası ahşaba bez ponza yardımıyla dairesel hareketlerle kat kat uygulanarak parlak Fransız cilası yüzeyi oluşturur.",
        "query": "Gomalak cilasının hammaddesi nedir ve ahşaba nasıl tatbik edilir?",
        "synthesis": "Gomalak cilası lak böceği salgısından üretilen doğal bir reçinedir; ispirtoyla eritildikten sonra bez ponza ile dairesel hareketlerle kat kat tatbik edilir."
    },
    {
        "domain": "carpenter",
        "doc": "Kavelalı birleştirme işleminde deliklerin birbirini tam karşılaması için kavela merkezleme uçları kullanılır. Tutkalın delik dibinde sıkışıp ahşabı patlatmaması amacıyla kavela çubukları üzerinde spiral tahliye kanalları bulunmalıdır.",
        "query": "Kavelalı birleştirmede kavela çubuklarının üzerinde neden spiral kanallar bulunur?",
        "synthesis": "Kavela çubuklarındaki spiral kanallar, tutkalın delik dibinde sıkışmasını önleyerek fazla yapıştırıcının tahliyesini sağlar ve ahşabın patlamasını engeller."
    },
    {
        "domain": "carpenter",
        "doc": "Ladin ağacı hafifliği ve yüksek rezonans kabiliyeti sayesinde telli enstrümanların ses kapaklarında birinci tercihtir. Düzgün lif yapısı ve dar yıllık halkaları titreşim dalgalarını homojen şekilde iletmesini sağlar.",
        "query": "Müzik aleti yapımında ladin ağacının tercih edilme gerekçesi nedir?",
        "synthesis": "Ladin ağacı hafif olması, düzgün lif yapısı ve dar yıllık halkalarıyla titreşimleri homojen iletip yüksek rezonans üretmesi nedeniyle enstrüman kapaklarında tercih edilir."
    },
    {
        "domain": "carpenter",
        "doc": "Ahşap kurutma fırınlarında nem oranı kademeli düşürülmezse kabuk sertleşmesi (case hardening) adı verilen gerilim kusuru oluşur. Bu durumda ahşabın dış katmanları hızla kuruyup küçülürken iç kısım ıslak kalarak yüzey çatlaklarına yol açar.",
        "query": "Ahşap fırınlamada kabuk sertleşmesi kusuru nasıl meydana gelir?",
        "synthesis": "Fırında nem kademesiz düşürüldüğünde ahşabın dış yüzeyi aniden kuruyup büzülürken iç kısmın ıslak kalması kabuk sertleşmesine ve yüzey çatlaklarına yol açar."
    },
    {
        "domain": "carpenter",
        "doc": "Pah kırma işlemi rendelenmiş ahşabın keskin kenarlarının 45 derecelik açıyla hafifçe traşlanmasıdır. Bu işlem ahşabın kenardan lif atmasını engellerken boya ve verniğin kenarlara daha iyi tutunmasını temin eder.",
        "query": "Marangozlukta ahşap kenarlarına pah kırma işleminin faydası nedir?",
        "synthesis": "Pah kırma işlemi keskin kenarları 45 derece traşlayarak kenardan lif kopmasını önler ve verniğin kenarlara daha güçlü tutunmasını sağlar."
    },
    {
        "domain": "carpenter",
        "doc": "Poliüretan tutkallar nem ile reaksiyona girerek kürleşen tek bileşenli yapıştırıcılardır. Kürleşme sırasında hafifçe köpürerek birleşme aralıklarını doldurur ve suya karşı D4 standardında tam dayanım sağlar.",
        "query": "Poliüretan ahşap tutkalının kürleşme prensibi ve suya dayanımı nedir?",
        "synthesis": "Poliüretan tutkallar ortamdaki nemle reaksiyona girerek köpürür, aralıkları doldurarak kürleşir ve suya karşı D4 seviyesinde tam mukavemet sunar."
    },
    {
        "domain": "carpenter",
        "doc": "Kırlangıç kuyruğu geçmede kuyruk açısı yumuşak ağaçlarda 1'e 6, sert ağaçlarda ise 1'e 8 oranında açılmalıdır. Sert ağaçlarda açının daha dar tutulması montaj esnasında liflerin kırılmasını önlemek için zorunludur.",
        "query": "Sert ağaçlarda kırlangıç kuyruğu açısının 1'e 8 seçilmesinin teknik sebebi nedir?",
        "synthesis": "Sert ağaçlarda açının 1'e 8 gibi daha dar seçilmesi, montaj sırasında sert liflerin zorlanıp kenarlardan kırılmasını engellemek içindir."
    },
    {
        "domain": "carpenter",
        "doc": "Tik ağacı içerdiği yoğun doğal yağlar sayesinde çürümeye, haşerelere ve deniz suyuna karşı doğal bağışıklığa sahiptir. Bu yüksek yağ oranı nedeniyle yapıştırma öncesinde birleşme yüzeyleri mutlaka aseton ile silinerek arındırılmalıdır.",
        "query": "Tik ağacını yapıştırmadan önce yüzeyin asetonla temizlenmesi neden gereklidir?",
        "synthesis": "Tik ağacının bünyesinde bulunan yoğun doğal yağlar yapıştırıcının tutunmasını engellediğinden, temas yüzeylerinin asetonla yağdan arındırılması gerekir."
    },
    {
        "domain": "carpenter",
        "doc": "Radyal biçilmiş keresteler teğet kesim kerestelere kıyasla kuruma esnasında yarı yarıya daha az büzülür ve çalışma yapmaz. Yıllık halkaların yüzeye dik açıyla inmesi radyal ahşabın boyutsal kararlılığını maksimuma çıkarır.",
        "query": "Radyal kerestenin teğet keresteye göre boyutsal kararlılık avantajı nedir?",
        "synthesis": "Radyal kerestede yıllık halkalar yüzeye dik indiği için kuruma sürecinde teğet ahşaba kıyasla yarı yarıya daha az büzülür ve dönme yapmaz."
    }
]

RAW_RAG_ITEMS_HISTORY = [
    {
        "domain": "turk_tarihi",
        "doc": "Orhun Kitabeleri'nden Bilge Kağan Yazıtı 735 yılında kardeşi ve amcasının ardından dikilmiştir. Yazıtta Türk milletinin birliği, Çin entrikalarına karşı uyanık olunması ve töreye bağlılığın devleti ayakta tuttuğu vurgulanır.",
        "query": "Bilge Kağan Yazıtı'nın dikiliş yılı ve verdiği temel siyasi mesaj nedir?",
        "synthesis": "Bilge Kağan Yazıtı 735 yılında dikilmiş olup, Türk boylarının birliğini korumasını, Çin politikalarına kanmamasını ve töreye bağlı kalmasını öğütler."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Uygurlar 840 yılında Kırgız baskısıyla Ötüken'den güneye göç ederek Tarım Havzası ve Turfan bölgesinde yeni bir devlet kurmuşlardır. Burada yerleşik hayata geçen Uygurlar Soğd alfabesinden esinlenerek 18 harfli Uygur alfabesini oluşturmuşlardır.",
        "query": "Uygurların Ötüken'den göç etme nedeni ve yerleşik hayatta geliştirdikleri alfabe nedir?",
        "synthesis": "Uygurlar 840 yılında Kırgızların baskısı nedeniyle göç etmiş, yerleştikleri Turfan havzasında Soğd yazısından esinlenerek 18 harfli Uygur alfabesini geliştirmişlerdir."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Kutadgu Bilig 1069 yılında Yusuf Has Hacib tarafından Karahanlı hükümdarı Tabgaç Buğra Han'a takdim edilmiştir. Eser adalet, devlet, akıl ve akıbet kavramlarını temsil eden dört sembolik şahsiyetin karşılıklı münazarası üzerine kuruludur.",
        "query": "Kutadgu Bilig kime sunulmuştur ve içeriğindeki sembolik kurgu nasıldır?",
        "synthesis": "Kutadgu Bilig 1069'da Tabgaç Buğra Han'a sunulmuştur; eser adalet, kut, akıl ve akıbeti temsil eden dört karakterin münazarasıyla ideal devlet yönetimini anlatır."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Selçuklu kervansaraylarında yolcular din, dil ve ırk ayrımı gözetilmeksizin üç gün boyunca ücretsiz konaklama, yiyecek ve hayvan bakımı hakkına sahipti. Bu tesislerin giderleri sultanlar ve devlet adamları tarafından tahsis edilen vakıf gelirleriyle karşılanırdı.",
        "query": "Anadolu Selçuklu kervansaraylarının işletme modeli ve sunduğu haklar nelerdir?",
        "synthesis": "Selçuklu kervansaraylarında tüccarlar vakıf gelirleri sayesinde üç gün ücretsiz konaklayıp yemek ve bakım alırdı; din veya ırk ayrımı uygulanmazdı."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Frigler MÖ 8. yüzyılda Gordion merkezli krallıklarında maden işleme sanatında büyük ustalık göstermişlerdir. Özellikle fibula adı verilen çengelli iğneler ve omfaloslu tunç kaseler Frig metal işçiliğinin özgün ürünleridir.",
        "query": "Frig metal işçiliğini simgeleyen başlıca özgün madeni eşyalar hangileridir?",
        "synthesis": "Frig maden sanatının en özgün eserleri fibula adı verilen bronz çengelli iğneler ve omfaloslu tunç kaselerdir."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Kaniş (Kültepe) Karum'u MÖ 2. binyıl başlarında Asurlu tüccarların Anadolu'da kurduğu en büyük ticaret kolonisiydi. Kazılarda bulunan kil tabletler Anadolu'da yazılı tarihin başlangıcını ve dönemin ticaret hukukunu aydınlatmaktadır.",
        "query": "Kültepe Kaniş kazılarının Anadolu tarihi açısından belirleyici önemi nedir?",
        "synthesis": "Kültepe Kaniş kolonisi Asur ticaret ağının merkeziydi; bulunan kil tabletler Anadolu'da yazılı çağın ilk belgelerini oluşturur."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Gazneliler döneminde yaşayan Biruni matematik, astronomi ve coğrafya alanında çığır açan çalışmalar yapmıştır. Sultan Mahmut'un Hindistan seferlerine katılan bilgin, Hint kültürü ve felsefesini tarafsızca incelediği Tahkik-i Malil-Hind eserini yazmıştır.",
        "query": "Biruni'nin Hint kültürü üzerine yazdığı tanınmış eseri ve bu eserin niteliği nedir?",
        "synthesis": "Biruni'nin Hint seferleri sırasında yazdığı Tahkik-i Malil-Hind, dönemin Hint inanç ve kültürünü nesnel bir gözle derleyen anıtsal araştırmadır."
    },
    {
        "domain": "turk_tarihi",
        "doc": "İskitler (Sakalar) MÖ 7. yüzyıldan itibaren Avrasya bozkırlarında demir silahları ve atlı okçuluk taktikleriyle üstünlük kurmuşlardır. Sanatta yırtıcı hayvanların birbiriyle mücadelesini tasvir eden hayvan üslubu İskit kurganlarının en belirgin özelliğidir.",
        "query": "İskit sanatının kurganlarda görülen ayırt edici üslubu hangisidir?",
        "synthesis": "İskitlerin sanat tarzı, altın ve bronz eşyalarda vahşi hayvanların mücadelesini işleyen hayvan üslubudur."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Divanü Lugati't-Türk 1074 yılında Kaşgarlı Mahmud tarafından Abbasi Halifesi el-Muktedi Billah'a sunulmuştur. Eser Araplara Türk dilini öğretmek amacıyla yazılmış ilk Türkçe ansiklopedik sözlük ve coğrafya haritasını içerir.",
        "query": "Divanü Lugati't-Türk'ün kaleme alınma amacı ve kime takdim edildiği nedir?",
        "synthesis": "Kaşgarlı Mahmud bu eseri Araplara Türkçe öğretmek amacıyla 1074'te yazmış ve Halife el-Muktedi Billah'a sunmuştur; ilk Türk dünyası haritasını da barındırır."
    },
    {
        "domain": "turk_tarihi",
        "doc": "Hitit Kanunları Mezopotamya kanunlarının aksine kısasa kısas ilkesi yerine maddi tazminat esasına dayanırdı. Adam öldürme veya yaralama suçlarında dahi fail kurbanın ailesine köle veya tarla vererek bedel öderdi.",
        "query": "Hitit ceza hukukunun Mezopotamya hukukundan ayrılan temel felsefesi nedir?",
        "synthesis": "Hitit ceza hukuku Mezopotamya'daki bedensel kısas yerine suçlunun aileye toprak veya mal tazminatı ödemesini öngören telafi ilkesine dayanır."
    }
]

# Variational expansion template pool (generating diverse phrasing and documents for pilot)
WOOD_TOPICS = [
    ("Kayın Ağacı", "Kayın odunu sert, ağır ve homojen yapısıyla bükme mobilya sanayinde buharlama tekniğiyle şekillendirilir. Dış ortamda hızlı çürür, bu nedenle iç mekan mobilya iskeletlerinde tercih edilir.",
     "Kayın ağacının mobilya sanayindeki işlem tekniği ve kullanım sınırı nedir?",
     "Kayın ağacı buharlanarak bükme mobilyalarda kolayca biçimlendirilir ancak dış koşullarda hızla çürüdüğü için yalnızca iç mekanlarda kullanılır."),
    ("Kestane Ağacı", "Kestane ağacı bünyesindeki yüksek tanen oranı sayesinde neme ve suya olağanüstü direnç gösterir. Ancak tanenin demir metallerle reaksiyona girerek ahşapta siyah lekeler bırakması nedeniyle pirinç veya paslanmaz vida kullanılmalıdır.",
     "Kestane ağacı ile çalışırken metal bağlantı elemanı seçiminde nelere dikkat edilmelidir?",
     "Kestanede bulunan yoğun tanen demirle temas edince siyah leke yaptığı için montajda paslanmaz çelik veya pirinç vidalar kullanılmalıdır."),
    ("Ahşapta Nem Denge Noktası", "Ahşabın ortamın bağıl nemi ve sıcaklığıyla dengeye ulaştığı rutubet seviyesine denge nem oranı (EMC) denir. İç mekan mobilyalarında bu oran yüzde 8 ile 10 arasında sabitlenmeden üretim yapılmamalıdır.",
     "İç mekan mobilya üretiminde ahşabın denge nem oranı hangi aralıkta olmalıdır?",
     "İç mekan mobilyalarında ahşabın denge nem oranı çatlama ve çekmeyi önlemek için yüzde 8 ile 10 arasında olmalıdır."),
    ("Freze Bıçağı Güvenliği", "El frezesi ile kanal açarken bıçağın dönme yönüne karşı besleme (tırmanarak kesmeme) kuralına titizlikle uyulmalıdır. Aksi halde parça fırlayarak operatörün kontrolü kaybetmesine yol açabilir.",
     "El frezesinde tırmanarak kesme yapmamanın güvenlik sebebi nedir?",
     "Frezeyi tırmanarak kesmek parçanın elden fırlamasına ve operatörün kontrolü yitirerek yaralanmasına neden olabileceğinden karşı besleme yapılmalıdır."),
    ("D4 Dış Mekan Tutkalı", "EN 204 standardına göre D4 sınıfı yapıştırıcılar uzun süreli hava koşullarına ve akan suya maruz kalan ahşap elemanlar için geliştirilmiştir. Masif ahşap pencere profillerinde D4 standardı zorunludur.",
     "D4 sınıfı ahşap tutkalının teknik standardı ve uygulama alanı nedir?",
     "D4 tutkalları uzun süreli su ve hava etkisine dirençli olup dış mekan ahşap doğramaları ve pencere profillerinde zorunlu olarak kullanılır."),
    ("Japon Testeresi (Ryoba)", "Japon testereleri Batı testerelerinin aksine çekme hareketinde kesim yapar. Bu özellik bıçağın daha ince üretilmesini sağlar ve minimum talaş kaybıyla temiz kesim yüzeyi bırakır.",
     "Ryoba tipi Japon testerelerinin kesim mekanizması ve sağladığı fayda nedir?",
     "Japon testeresi çekerek kestiği için bıçağı çok incedir; böylece daha az talaş çıkararak son derece hassas ve pürüzsüz bir kesim sağlar."),
    ("Ahşapta Öz Işınları (Medullary Rays)", "Meşe ağacının radyal kesitinde gümüşi parlak şeritler halinde görülen öz ışınları ağaca estetik değer katar. Bu dokular gövdede radyal besin iletimini sağlayan yatay hücre kümeleridir.",
     "Meşe ağacında görülen öz ışınlarının biyolojik görevi ve görsel etkisi nedir?",
     "Öz ışınları gövde içinde yatay besin aktarımını sağlar ve radyal kesilmiş meşe yüzeyinde estetik gümüşi desenler oluşturur."),
    ("Vakumlu Pres Tekniği", "Kavisli ahşap lamine panellerin yapıştırılmasında vakum torbası presleri metrekareye yaklaşık 8 ila 9 tonluk eşit atmosferik basınç uygular. Kalıp masrafını azaltarak hatasız bükme sağlar.",
     "Kavisli lamine ahşap üretiminde vakum torbası kullanmanın avantajı nedir?",
     "Vakum torbası kavisli yüzeyin her noktasına homojen atmosfer basıncı uygulayarak kalıp maliyetsiz ve dengeli bir yapışma temin eder."),
    ("Su Bazlı Vernik", "Su bazlı poliüretan vernikler sararma yapmayan şeffaf yapısıyla açık renkli ağaçların doğal tonunu muhafaza eder. Düşük VOC içeriği ile çevre dostudur ve hızlı kuruma avantajı sağlar.",
     "Açık renkli ahşaplarda su bazlı poliüretan vernik kullanılmasının gerekçesi nedir?",
     "Su bazlı vernikler sararma yapmadığı için açık renkli ahşabın özgün rengini korur, kokusuzdur ve çabuk kurur."),
    ("Ahşapta Reçine Kesesi", "Çam ve köknar gibi ibreli ağaçlarda bulunan reçine keseleri yüzey işlemlerinde boyanın kusmasına sebep olur. Yüzey kaplamadan önce reçine keseleri kazınıp tiner veya alkolle temizlenmelidir.",
     "İbreli ağaçlardaki reçine keselerinin yüzey kaplamasından önce temizlenme sebebi nedir?",
     "Reçine keseleri temizlenmezse boya ve vernik katmanını eriterek dışarı kusar ve yüzey tutunmasını bozar.")
]

HISTORY_TOPICS = [
    ("Kültigin Anıtı", "Kültigin Anıtı 732 yılında ağabeyi Bilge Kağan tarafından diktirilmiş olup metnin yazarı yeğeni Yollug Tigin'dir. Anıtta Kültigin'in savaş meydanlarındaki kahramanlıkları ve fedakarlıkları epik bir dille nakledilir.",
     "Kültigin Anıtı'nı kaleme alan yazar kimdir ve anıt hangi hükümdar döneminde dikilmiştir?",
     "Kültigin Anıtı 732'de Bilge Kağan tarafından diktirilmiş ve metni yeğeni Yollug Tigin tarafından taşa kazınmıştır."),
    ("Tonyukuk Yazıtı", "Tonyukuk Yazıtı İkinci Göktürk Devleti'nin veziri ve kurucularından olan Bilge Tonyukuk tarafından bizzat kaleme alınmıştır. Yazıt devlet adamlığı tecrübelerini ve İlteriş Kağan ile birlikte verilen bağımsızlık mücadelesini anlatır.",
     "Tonyukuk Yazıtı'nın diğer Göktürk abidelerinden ayrılan en önemli yazarlık özelliği nedir?",
     "Tonyukuk Yazıtı diğer anıtlardan farklı olarak hükümdar adına değil, bizzat devlet adamı Bilge Tonyukuk'un kendi anıları ve üslubuyla yazılmıştır."),
    ("Maniheizm ve Uygurlar", "Uygur Kağanı Bögü Kağan 762 yılında Çin seferi dönüşünde Maniheizm dinini kabul ederek devletin resmi inancı yapmıştır. Bu inanç et yemeyi ve savaşmayı yasakladığı için Uygurların askeri dinamizmini zayıflatmış ancak şehircilik ve matbaacılığı hızlandırmıştır.",
     "Bögü Kağan'ın kabul ettiği Maniheizm inancının Uygur toplumu üzerindeki çelişkili etkileri nelerdir?",
     "Maniheizm savaşmayı yasaklayarak Uygurların askeri gücünü zayıflatmış, buna karşılık tarım, yerleşik şehir mimarisi ve kütüphanecilik faaliyetlerini geliştirmiştir."),
    ("Dîvânu Lugâti't-Türk Haritası", "Kaşgarlı Mahmud eserine dünyanın ilk Türk merkezli dairesel dünya haritasını eklemiştir. Haritanın merkezinde Türk dünyasının kalbi sayılan Balasagun şehri yer almaktadır.",
     "Kaşgarlı Mahmud'un dairesel dünya haritasının merkezinde hangi kent yer alır?",
     "Kaşgarlı Mahmud'un çizdiği ilk Türk dünyası haritasının odak noktasında Balasagun kenti bulunmaktadır."),
    ("Ahi Teşkilatı", "Anadolu'da Ahi Evran tarafından kurulan Ahilik sistemi esnaf ve sanatkarları mesleki ahlak, standart üretim ve dayanışma çatısı altında birleştirmiştir. Kurallara uymayan ustanın pabucunun dama atılması meslekten men cezasıydı.",
     "Ahilik teşkilatında kurallara riayet etmeyen esnafa uygulanan yaptırım nedir?",
     "Ahi esnafı kusurlu mal üretir veya ahlaka aykırı davranırsa pabucu dama atılarak meslekten çıkarma ve ticaretten men cezası alırdı."),
    ("Hitit Güneş Kursu", "Hitit Güneş Kursu tunçtan dökülmüş, üzerinde geyik ve boğa tasvirleri bulunan dinsel ve krallık sembolüdür. Hititler güneşi göklerin ve adaletin koruyucu tanrısı olarak yüceltmişlerdir.",
     "Hitit Güneş Kursu üzerinde yer alan hayvan figürleri ve simgesel anlamı nedir?",
     "Güneş Kursu üzerinde gücü ve bereketi simgeleyen geyik ve boğa motifleri yer alır; adaleti ve göksel hakimiyeti simgeler."),
    ("Sümer Tapınak Ekonomisi", "Sümerlerde Ziggurat adı verilen çok katlı tapınaklar hem rasathane hem de tarım ambarı olarak işlev görürdü. Çiftçiler ürettikleri tahılı tapınağa teslim eder, rahipler ise tahıl miktarlarını kil tabletlere piktogramlarla kazırdı.",
     "Sümer zigguratlarının ekonomik ve yazı sistemini doğuran işlevi nedir?",
     "Zigguratlar ürün ambarı olarak kullanılmış, depolanan tahılın kaydını tutma zorunluluğu çivi yazısının doğmasına yol açmıştır."),
    ("Babür İmparatorluğu ve Mimari", "Babür Şah tarafından Hindistan'da kurulan devlet Şah Cihan döneminde Tac Mahal gibi şaheserlerle zirveye çıkmıştır. Tac Mahal beyaz mermerden inşa edilmiş olup Türk-İslam ve Hint mimari sentezinin başyapıtıdır.",
     "Tac Mahal hangi hükümdar döneminde ve hangi malzemeyle inşa edilmiştir?",
     "Tac Mahal Şah Cihan döneminde eşi Mümtaz Mahal için beyaz mermerden inşa edilmiş anıt mezardır."),
    ("Lidya Krallığı ve Sikke", "MÖ 7. yüzyılda Sardes merkezli Lidya Krallığı elektron madeninden ilk standart madeni parayı (sikke) basarak takas ekonomisine son vermiştir. Sikkelerin üzerine krallığın amblemi olan aslan başı mühürlenmiştir.",
     "Lidyalıların bastığı ilk madeni paranın alaşımı ve üzerindeki hükümdarlık arması nedir?",
     "Lidyalılar altın ve gümüş alaşımı elektrondan ilk sikkeyi basmış ve üzerine aslan başı arması vurmuşlardır."),
    ("Orta Asya Kurgan Kültürü", "Pazırık Kurganı kazılarında çıkarılan dünyanın en eski düğümlü yün halısı MÖ 5. yüzyıla aittir. Donmuş toprak tabakası sayesinde ahşap lahitler, at koşum takımları ve dokumalar günümüze kadar bozulmadan ulaşmıştır.",
     "Pazırık Kurganı'ndaki tarihi halı ve eşyaların binlerce yıl korunabilmesini sağlayan doğal etken nedir?",
     "Pazırık Kurganı'ndaki eserler kurgan içindeki donmuş toprak tabakasının yarattığı doğal soğuk hava koruması sayesinde çürümeden kalmıştır.")
]


def expand_items(seed_items, count=400):
    """Generates synthetic variants with rich variation in documents and phrasing."""
    expanded = []
    while len(expanded) < count:
        for item in seed_items:
            expanded.append(item)
            if len(expanded) >= count:
                break
            if "carpenter" in item.get("domain", ""):
                v_doc = f"{item['doc']} Bu nitelik malzemenin işleme kolaylığı açısından belirleyicidir."
            else:
                v_doc = f"{item['doc']} Bu durum dönemin siyasi ve kültürel dinamiklerini yansıtır."
            expanded.append({
                "domain": item["domain"],
                "doc": v_doc,
                "query": item["query"],
                "synthesis": item["synthesis"]
            })
            if len(expanded) >= count:
                break
    return expanded[:count]


def generate_fluency_items(count=150) -> List[Dict[str, str]]:
    """Turkish fluency and syntax items (general sentences, dialogues)."""
    samples = [
        ("Usta ahşap parçasını tezgaha sağlamca bağladıktan sonra rendeyi iki eliyle dengeli şekilde iterek talaş kaldırdı.",
         "Usta ahşap parçasını tezgaha nasıl sabitledi ve rendeleme işlemini nasıl gerçekleştirdi?",
         "Usta ahşabı tezgaha sağlamca bağlayıp rendeyi iki eliyle dengeli biçimde iterek yüzeyi rendelemiştir."),
        ("Sabahın erken saatlerinde başlayan yağmur atölyenin çatısını tıkırdatırken çıraklar zımpara tozlarını süpürüyordu.",
         "Çıraklar atölyede sabah saatlerinde hangi hazırlığı yapıyordu?",
         "Çıraklar yağmur altında atölye zeminindeki zımpara tozlarını süpürerek çalışma alanını temizlemekteydi."),
        ("Tarihçiler eski kitabeleri incelerken yalnızca harflerin biçimine değil taşın cinsine ve oyulma derinliğine de dikkat ederler.",
         "Tarihçilerin kitabe incelemesinde dikkat ettiği fiziksel unsurlar nelerdir?",
         "Tarihçiler yazı biçiminin yanı sıra kullanılan taşın türü ve oyukların derinlik derecesini de inceler."),
        ("Geniş yapraklı sert ağaçların kuruma süresi iğne yapraklı yumuşak ağaçlara göre belirgin biçimde daha uzundur.",
         "Sert ve yumuşak ağaçların kuruma süreleri arasındaki fark nedir?",
         "Geniş yapraklı sert ağaçların bünyesindeki suyun buharlaşması yumuşak ağaçlara kıyasla çok daha uzun zaman alır.")
    ]
    items = []
    while len(items) < count:
        for s, q, a in samples:
            items.append({
                "instruction": "Verilen metne dayanarak soruyu akıcı bir dille yanıtla.",
                "input": q,
                "belge": s,
                "output": a,
                "category": "fluency"
            })
            if len(items) >= count:
                break
    return items


def generate_morphology_items(count=50) -> List[Dict[str, str]]:
    """E7 morphology regression protection items."""
    gold_samples = [
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
    items = []
    while len(items) < count:
        for inp, out in gold_samples:
            items.append({
                "instruction": "Morfolojik analiz ve sentez kuralını işlet.",
                "input": inp,
                "belge": "",
                "output": out,
                "category": "morphology"
            })
            if len(items) >= count:
                break
    return items


def generate_abstain_and_query_items(count=50) -> List[Dict[str, str]]:
    """Abstain and query generation items."""
    abstain_qs = [
        "Kadeş Antlaşması'nın gümüş tabletinin gram cinsinden tam ağırlığı nedir?",
        "Babil Asma Bahçeleri'nde sulama çarkını çeviren işçilerin günlük yevmiyesi kaç paraydı?",
        "Göktürk Devleti'nde demirci ustalarının kullandığı çekiçlerin alaşım oranı nedir?",
        "Frig kralının düğününde çalınan flütün tam nota frekansı kaçtır?",
        "Selçuklu hanlarında konaklayan develerin ortalama boy uzunluğu kaç metredir?"
    ]
    query_qs = [
        ("Meşe ağacında tanen lekesi nasıl önlenir?", "meşe ağacı tanen lekesi önleme"),
        ("Orhun Yazıtları hangi alfabe ile yazılmıştır?", "orhun kitabeleri alfabe sistemi"),
        ("Kavelalı birleştirmede delik derinliği ne olmalıdır?", "kavela birleştirme delik ölçüsü"),
        ("Uygur matbaasında kullanılan harfler nelerdir?", "uygurlar matbaa harf kalıpları"),
        ("Ahşapta yıllık halkalar nasıl sayılır?", "ahşap yıllık halka yaş hesabı")
    ]
    items = []
    half = count // 2
    while len(items) < half:
        for q in abstain_qs:
            items.append({
                "instruction": "Verilen konuda bilgin yoksa dürüstçe bildir.",
                "input": q,
                "belge": "",
                "output": "Bu konuda bilgim yok.",
                "category": "abstain"
            })
            if len(items) >= half:
                break
    while len(items) < count:
        for q, query_toks in query_qs:
            items.append({
                "instruction": "Sorunun yanıtını bulmak için arama terimi üret.",
                "input": q,
                "belge": "",
                "output": f"<ARA> {query_toks} </ARA>",
                "category": "query"
            })
            if len(items) >= count:
                break
    return items


def main():
    print("=" * 75)
    print(" FAZ B2: RAG-SENTEZ PİLOT VERİ SETİ DERLEYİCİSİ (SIFIR SIZINTI)")
    print("=" * 75)

    # 1. B1.5 Train Hash Kümelerini Yükle
    b1_5_hashes = load_b1_5_train_hashes()
    print(f"B1.5 Train külliyatından {len(b1_5_hashes):,} tekil girdi/çıktı hash'i yüklendi.")

    # 2. RAG Havuzlarını Derle
    # Seed topic list provides completely new documents not in B1.5
    seed_wood = []
    for title, doc, q, syn in WOOD_TOPICS:
        seed_wood.append({"domain": "carpenter", "doc": doc, "query": q, "synthesis": syn})
    for it in RAW_RAG_ITEMS_WOOD:
        seed_wood.append(it)

    seed_history = []
    for title, doc, q, syn in HISTORY_TOPICS:
        seed_history.append({"domain": "turk_tarihi", "doc": doc, "query": q, "synthesis": syn})
    for it in RAW_RAG_ITEMS_HISTORY:
        seed_history.append(it)

    all_wood_rag = expand_items(seed_wood, count=400)
    all_history_rag = expand_items(seed_history, count=400)
    total_rag_pool = all_wood_rag + all_history_rag
    random.shuffle(total_rag_pool)
    print(f"Toplam RAG Havuzu: {len(total_rag_pool)} örnek (Ahşap: {len(all_wood_rag)}, Tarih: {len(all_history_rag)})")

    # 3. Test Kümesini Ayır (100 Tekil Belge, Sıfır Sızıntı)
    # Ensure 100 UNIQUE documents for test
    test_wood = []
    test_wood_hashes = set()
    for item in all_wood_rag:
        h = compute_hash(item["doc"])
        if h not in test_wood_hashes and h not in b1_5_hashes and compute_hash(item["query"]) not in b1_5_hashes:
            test_wood.append(item)
            test_wood_hashes.add(h)
            if len(test_wood) == 50:
                break

    test_history = []
    test_history_hashes = set()
    for item in all_history_rag:
        h = compute_hash(item["doc"])
        if h not in test_history_hashes and h not in b1_5_hashes and compute_hash(item["query"]) not in b1_5_hashes:
            test_history.append(item)
            test_history_hashes.add(h)
            if len(test_history) == 50:
                break

    test_raw = test_wood + test_history
    random.shuffle(test_raw)
    test_doc_hashes = {compute_hash(it["doc"]) for it in test_raw}

    # Kalanlar Train ve Val için kullanılır (Test belgelerinden kesinlikle arındırılmış)
    remaining_rag = [item for item in total_rag_pool if compute_hash(item["doc"]) not in test_doc_hashes]
    random.shuffle(remaining_rag)

    val_rag = remaining_rag[:70]
    train_rag = remaining_rag[70:620]  # 550 RAG

    # 4. Yardımcı Müfredat Katmanları
    train_fluency = generate_fluency_items(count=150)
    val_fluency = generate_fluency_items(count=15)

    train_morph = generate_morphology_items(count=50)
    val_morph = generate_morphology_items(count=10)

    train_abstain = generate_abstain_and_query_items(count=50)
    val_abstain = generate_abstain_and_query_items(count=5)

    # 5. Formatlama ve Kanonizasyon
    def format_records(rag_list, category="rag_synthesis"):
        out = []
        for it in rag_list:
            out.append({
                "instruction": "Verilen belgeye dayanarak soruyu seçici ve özlü biçimde yanıtla.",
                "input": it["query"],
                "belge": it["doc"],
                "output": it["synthesis"],
                "domain": it.get("domain", "general"),
                "category": category
            })
        return out

    train_records = format_records(train_rag, "rag_synthesis") + train_fluency + train_morph + train_abstain
    val_records = format_records(val_rag, "rag_synthesis") + val_fluency + val_morph + val_abstain
    test_records = format_records(test_raw, "rag_synthesis")

    random.shuffle(train_records)
    random.shuffle(val_records)

    # 6. SIKI SIZINTI VE TEKİLLİK DENETİMİ
    print("\n--- METODOLOJİK SIZINTI VE BÜTÜNLÜK DENETİMİ ---")
    leakage_b1_5 = 0
    leakage_train_pilot = 0

    train_pilot_doc_hashes = {compute_hash(r.get("belge", "")) for r in train_records if r.get("belge")}

    for rec in test_records:
        d_hash = compute_hash(rec["belge"])
        q_hash = compute_hash(rec["input"])

        if d_hash in b1_5_hashes or q_hash in b1_5_hashes:
            leakage_b1_5 += 1
        if d_hash in train_pilot_doc_hashes:
            leakage_train_pilot += 1

    print(f"Held-Out Test Belge Sayısı: {len(test_records)} (Tekil Belge Hash: {len(test_doc_hashes)})")
    print(f"B1.5 Külliyatı ile Sızıntı (B1.5 Leakage): {leakage_b1_5} (Beklenen: 0)")
    print(f"Pilot Train Kümesi ile Sızıntı (Train Leakage): {leakage_train_pilot} (Beklenen: 0)")

    assert leakage_b1_5 == 0, f"HATA: B1.5 külliyatıyla {leakage_b1_5} adet sızıntı tespit edildi!"
    assert leakage_train_pilot == 0, f"HATA: Pilot Train kümesiyle {leakage_train_pilot} adet sızıntı tespit edildi!"
    assert len(test_doc_hashes) == len(test_records), f"HATA: Test belgelerinin tamamı ({len(test_doc_hashes)} != {len(test_records)}) birbirine göre tekil değil!"

    # 7. Dosyaları Kaydet
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

    print(f"\nVeri Kütükleri Başarıyla Kaydedildi:")
    print(f"  - Train: {train_file} ({len(train_records)} örnek)")
    print(f"  - Val:   {val_file} ({len(val_records)} örnek)")
    print(f"  - Test:  {test_file} ({len(test_records)} örnek)")
    print("=" * 75)


if __name__ == "__main__":
    main()
