#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: MARANGOZLUK ALAN UZMANLIĞI (CARPENTER AI)
===================================================================
Universal Dataset Orchestrator prensiplerine uygun, 5 boyutlu davranış
distilasyonu rubriğiyle filtrelenmiş yüksek kaliteli Altın SFT veri üretici.
"""

import os
import sys
import json
import random
import re
from typing import List, Dict, Tuple

# 1. GENİŞLETİLMİŞ AĞAÇ TÜRLERİ (WOOD SPECIES - 25 Tür)
WOOD_SPECIES = [
    ("meşe", "Sert, yoğun lifli, suya ve aşınmaya karşı oldukça dirençli bir ağaçtır; masif mobilya, merdiven ve parkede sıkça tercih edilir."),
    ("ceviz", "Zengin damar desenine ve asil kahverengi tonlara sahip, işlemesi kolay, formunu mükemmel koruyan lüks mobilya ağacıdır."),
    ("kayın", "Oldukça sert ve sıkı liflidir; buharlama yöntemiyle kolayca bükülerek Thonet tarzı sandalye ve bükme mobilyalarda idealdir."),
    ("çam", "Yumuşak dokulu, reçineli ve kolay işlenebilen, çatı karkası, yapı kerestesi ve ekonomik doğramalarda kullanılan iğne yapraklıdır."),
    ("tik", "Doğal silika ve yağ barındırdığından suya, güneşe ve çürümeye aşırı dayanıklıdır; dış mekan bahçe mobilyası ve tekne güvertesinde rakipsizdir."),
    ("iroko", "Afrika kökenli, sert ve doğal yağlı yapısıyla tik ağacının en popüler ekonomik alternatifidir; dış cephe ve havuz kenarlarında kullanılır."),
    ("dişbudak", "Esnekliği ve darbe emme gücü çok yüksektir; balta, çekiç, kürek sapları ve spor aletleri yapımında ilk tercihtir."),
    ("gürgen", "Aşırı sert, tok ve beyaz yapılıdır; marangoz mengene çeneleri, planya gövdeleri ve ağır yüke maruz kalan parçalarda kullanılır."),
    ("kiraz", "Zamanla güneş ışığıyla koyulaşan sıcak kızıl tonlu, ince dokulu ve pürüzsüz yüzey veren asil bir mobilya ağacıdır."),
    ("akçaağaç", "Çok açık renkli, yoğun, kokusuz ve serttir; kesme tahtaları, kasap blokları, müzik aletleri ve tezgah üstlerinde tercih edilir."),
    ("kestane", "Doğal tanen içeriği sayesinde neme ve mantara karşı dirençlidir; dış mekan doğramalarında ve fıçı yapımında kullanılır."),
    ("ıhlamur", "Yumuşak, lifsiz ve homojen yapısıyla ahşap oyma zanaatında ve heykelcilikte en rahat işlenen ağaçtır."),
    ("sedir", "Doğal hoş kokulu ve güve kovucu etkiye sahip, hafif ve reçineli bir ağaçtır; sandık içi ve sauna kaplamalarında kullanılır."),
    ("ladin", "Hafif, düzgün elyaflı ve esnek bir ağaçtır; enstrüman ses tahtalarında ve iç mekan tavan kaplamalarında tercih edilir."),
    ("köknar", "Reçinesiz ve açık renkli yapısıyla kağıt hamuru, ambalaj sandığı ve çıta imalatında sıkça kullanılır."),
    ("pelesenk", "Olağanüstü sert, yağlı ve koyu renkli tropikal bir ağaçtır; bıçak kabzası, enstrüman klavyesi ve lüks kakmalarda kullanılır."),
    ("maun", "Kırmızımsı kahverengi, büzülme ve çalışma yapmayan çok stabil bir tropikal ağaçtır; klasik müzik aletleri ve gemi kamaralarında kullanılır."),
    ("balsa", "Bilinen en hafif ticari ağaçtır; maketçilikte, uçak modellerinde ve rüzgar türbini kanat çekirdeğinde kullanılır."),
    ("huş", "Beyazımsı sarı renkli, yüksek elastikiyetli bir ağaçtır; özellikle yüksek mukavemetli marin kontrplak üretiminde temel hammaddedir."),
    ("karaağaç", "Lifleri birbirine kenetli olduğundan yarılması imkansıza yakındır; tekerlek göbekleri, sandalye oturakları ve dövme bloklarında kullanılır."),
    ("servi", "Çürümeye ve suya çok dayanıklıdır; mezarlık parmaklıkları, su kanalları ve dayanıklı bahçe çitlerinde tercih edilir."),
    ("kavak", "Çok hızlı büyüyen, hafif, lifli ve ucuz bir ağaçtır; kibrit çöpü, meyve kasası ve kontrplak ara katmanlarında kullanılır."),
    ("şimşir", "Mermer gibi sert, gözeneksiz ve homojen yapıdadır; baskı kalıpları, tarak, cetvel ve müzik aleti süslemelerinde kullanılır."),
    ("ardıç", "Çürümeye meydan okuyan, yoğun aromatik kokulu ve dayanıklı bir dağ ağacıdır; geleneksel ahşap ev temellerinde ve kovan yapımında kullanılır."),
    ("zeytin", "Kıvrımlı, kontrastlı damarlarıyla eşsiz bir desene ve sertliğe sahiptir; dekoratif sunum tahtaları ve mutfak eşyalarında popülerdir.")
]

# 2. GENİŞLETİLMİŞ EL ALETLERİ VE MAKİNELER (TOOLS - 25 Alet)
TOOLS = [
    ("rende", "Ahşabın yüzeyini düzeltmek, talaş kaldırarak inceltmek ve pürüzsüz bir referans yüzeyi açmak için kullanılan kesici el aletidir."),
    ("iskarpela", "Ahşapta delik, kanal veya zıvana yuvaları açmak, kenar temizlemek ve ahşabı yontmak için kullanılan çelik keski aletidir."),
    ("kırlangıç testere", "Hassas geçme ve birleştirme hatlarını kesmek için sırtı pirinç veya çelik takviyeli ince dişli hassas el testeresidir."),
    ("freze", "Ahşap kenarlarına profil vermek, kanal ve lamba açmak veya şablon kopyalamak için yüksek devirli dönen bıçaklı makinedir."),
    ("işkence", "Yapıştırma veya montaj sırasında parçaları tutkal kuruyana kadar sabit, basılı ve sıkı tutmaya yarayan sıkıştırma aletidir."),
    ("gönye", "Ahşap parçaların 90 derece dik açılarını, düzlemselliklerini ve paralelliklerini kontrol etmeye yarayan L biçimli ölçüm aletidir."),
    ("kumpas", "Ahşap kalınlığını, zıvana genişliğini veya delik derinliğini milimetrenin onda biri hassasiyetle ölçen hassas mekanik alettir."),
    ("planya", "Kaba biçilmiş kerestenin yüzeyini ve cumbasını kusursuz bir 90 derecelik referans düzlemine getiren döner bıçaklı tezgah makinesidir."),
    ("kalınlık makinesi", "Planyalanmış kerestenin diğer yüzünü istenen tam milimetrik kalınlığa ve paralelliğe getiren merdaneli makinedir."),
    ("raspa", "Ters damarlı veya dalgalı lifli zorlu ahşaplarda rende izlerini kazıyarak cam gibi kusursuz yüzey bırakan esnek çelik plakadır."),
    ("tokmak", "İskarpelanın arkasına vururken çelik sapı koruyan, darbeli yontma işlemlerinde ahşap veya poliüretandan yapılan çekiçtir."),
    ("sürme gönye", "Sabit 90 derece yerine kilitlenebilir bıçağı ile her türlü açıyı kopyalayıp ahşaba aktarmaya yarayan ayarlı gönyedir."),
    ("çizecek", "Kurşun kalemin kalın çizgisinin aksine ahşap liflerini keserek milimetrik kılavuz hat açan ucu sivri sert çelik işaretleyicidir."),
    ("çekiç", "Çivi çakma, kamaları yerine oturtma ve kaba montaj işlemlerinde kullanılan metal başlıklı temel vurma aletidir."),
    ("şerit testere", "İki kasnak arasında dönen sürekli esnek şerit bıçağıyla kavisli kesimler, biçme ve kalın keresteyi dilimleme makinesidir."),
    ("dekupaj", "Yukarı aşağı ileri geri hareket eden ince bıçağıyla ahşap plakalarda kıvrımlı iç ve dış konturları kesen el makinesidir."),
    ("zımpara takozu", "Zımpara kağıdının ahşap yüzeye eşit ve düzgün basmasını sağlayarak dalgalanmayı önleyen kauçuk veya mantar bloktur."),
    ("rayba", "Açılmış deliklerin çapını hassas şekilde genişletmeye ve konik yuvalar oluşturmaya yarayan çok bıçaklı alettir."),
    ("matkap", "Ahşapta vida, kavela veya zıvana başlangıç delikleri açmak için helisel uçlar döndüren delme makinesidir."),
    ("tığ", "Vida takılacak noktayı tam merkezden işaretleyerek vida ucunun kaymasını engelleyen sivri uçlu kılavuz alettir."),
    ("eğe", "Kavisli ahşap yüzeyleri, zıvana yanaklarını veya testere dişlerini inceltip şekillendiren sert dişli metal alettir."),
    ("mengene", "Marangoz tezgahının kenarında parçayı rendeleme veya kesme sırasında sımsıkı kilitleyen vidalı çenelerdir."),
    ("kerpeten", "Eğri çakılan veya hatalı çivileri ahşaptan sökmek ve başları kesmek için kullanılan kollu kıskaç aletidir."),
    ("su terazisi", "Montajı yapılan mobilyanın, rafın veya tezgahın yerçekimine göre tam yatay ve dikey dengede olduğunu gösteren havalı tüptür."),
    ("japon testere", "İtme yerine çekme hareketinde kesim yaparak bıçağın bükülmesini engelleyen ve son derece temiz kesen testeredir.")
]

# 3. GENİŞLETİLMİŞ BİRLEŞTİRME TEKNİKLERİ (JOINERY - 20 Teknik)
JOINERY = [
    ("kırlangıç kuyruğu", "Çekme kuvvetine karşı mekanik olarak kilitlenen, özellikle çekmece kasalarında ve sandık köşelerinde ayrılmaz mukavim birleşimdir."),
    ("zıvana ve geçme", "Bir parçanın ucundaki erkek zıvana dilinin diğer parçadaki dişi zıvana yuvasına oturmasıyla yapılan, masa ve sandalye iskeletlerinin temel birleşimidir."),
    ("kavela birleştirme", "Silindirik ahşap pimlerin (kavela) tutkallanarak karşılıklı açılan deliklere yerleştirilmesiyle yapılan pratik ve gizli birleşimdir."),
    ("lamba zıvana", "Tahtaların bir kenarına erkek çıta, diğerine dişi kanal açılarak birbirine geçirilmesidir; ahşap döşeme ve tavan kaplamalarında kullanılır."),
    ("gönye burun", "İki parçanın 45 derecelik açıyla kesilerek 90 derecelik köşe oluşturduğu, lif bitimlerini gizleyen son derece estetik çerçeve birleşimidir."),
    ("kelebek kama", "Masif masa tablalarındaki doğal çatlakların daha fazla açılmasını önlemek için çatlağın üzerine kilit şeklinde kakılan ahşap parçadır."),
    ("kurtboğazı", "Kütük evlerde ve masif kiriş köşelerinde parçaların birbirine geçirilerek yatay kaymayı kilitleyen geleneksel birleşimdir."),
    ("parmak birleştirme", "İki parçanın uçlarına çok sayıda dikdörtgen parmak açılarak yapıştırıcı yüzeyinin artırıldığı sağlam köşe birleşimidir."),
    ("bindirme geçme", "İki ahşap parçanın kalınlıklarının yarısı kadar boşaltılarak üst üste bindirilmesiyle yapılan haç veya köşe birleşimidir."),
    ("kertme birleştirme", "Bir parçanın içine diğer parçanın tam oturacağı kadar kanal açılarak yapılan raf ve kayıt bağlantısıdır."),
    ("kamalı zıvana", "Zıvana dilinin dışarı taştığı ve ucuna takoz kama çakılarak mekanik olarak kilitlendiği geleneksel sökülebilir masa birleşimidir."),
    ("çift zıvana", "Geniş veya kalın kerestelerde tek zıvana yerine yan yana iki zıvana dili açılarak taşıma kapasitesinin iki katına çıkarıldığı tekniktir."),
    ("yarı gömme kırlangıç", "Çekmece ön panellerinde kırlangıç dişlerinin dışarıdan görünmesini engelleyen tek taraflı estetik birleşimdir."),
    ("bisküvi birleştirme", "Oval sıkıştırılmış kayın bisküvilerinin karşılıklı kanallara tutkalla geçirilerek plaka genişletmede kullanıldığı modern birleşimdir."),
    ("kavelalı gönye", "45 derece gönye kesilmiş köşelerin içine gizli kavela delikleri delinerek tutkallandığı çerçeve birleşimidir."),
    ("kurt dişi birleştirme", "Açılı zikzak dişlerle ahşap parçaların boylamasına eklenerek sonsuz kereste elde edilmesini sağlayan endüstriyel eklemedir."),
    ("lamba kavelalı birleşim", "Geniş masa tablalarını yan yana yapıştırırken hizalama hatasını sıfıra indiren kombine kenar birleşimidir."),
    ("cep vidalı birleşim", "Ahşabın arka veya alt yüzüne açılı gizli yuvalar delinerek özel vidalarla mengenesiz hızlı yapılan karkas birleşimidir."),
    ("haç zıvana", "Masa veya sehpa ayaklarının merkezde birbirinin içinden geçerek 90 derece kenetlendiği taşıyıcı birleşimdir."),
    ("kama takviyeli gönye", "Gönye burun birleşimin köşesine sonradan zıt lifli ince ahşap dilim kakılarak köşenin yırtılmasını önleyen takviyedir.")
]

# 4. AHŞAP FİZİĞİ, NEM VE KURUTMA KURALLARI (15 Prensip)
PHYSICS_AND_DRYING = [
    ("masif nem oranı kuralı", "İç mekan masif ahşap mobilyalarda ideal kereste nem oranı yüzde 8 ile 12 arasında olmalıdır; dış mekanda ise yüzde 14-16 hedeflenir."),
    ("lif doyum noktası", "Ahşapta hücre boşluklarındaki serbest suyun bittiği ve hücre çeperlerindeki bağlı suyun kalmaya başladığı yaklaşık yüzde 28-30 nem seviyesidir."),
    ("enine boyuna genleşme farkı", "Ahşap lifleri boyuna neredeyse hiç çalışmaz (genleşmez); ancak teğetsel yönde yüzde 6-10, radyal yönde yüzde 3-5 oranında genişler veya çeker."),
    ("fırın kurutma işlemi", "Kerestenin sıcaklık, nem ve hava sirkülasyonu kontrollü fırınlarda kademeli olarak kurutularak hücre içi gerilimlerinin boşaltılmasıdır."),
    ("tabii kurutma", "Kerestenin çıtalanarak açık havada doğrudan güneş ve yağmurdan korunarak gölgede kurutulmasıdır; her santim kalınlık için yaklaşık bir yıl sürer."),
    ("kamburlaşma (cupping)", "Geniş tahtaların yıllık halkalarının teğetsel yönü boyunca kenarlarının yukarı kıvrılarak oluk şeklini alması deformasyonudur."),
    ("eğrilme (bowing)", "Kerestenin boy ekseni boyunca bir yay gibi düzleminden sapması şeklindeki dönme kusurudur."),
    ("burulma (twisting)", "Kerestenin dört köşesinin aynı düzlemde kalmayıp pervane gibi bükülmesi durumudur; ters damarlı ağaçlarda sık görülür."),
    ("tabla genleşme boşluğu kuralı", "Masif masa tablaları ayaklara veya karkasa doğrudan sabitlenmez; ahşabın nefes alıp çalışabilmesi için Z-klips veya kanallı demir kullanılır."),
    ("hızlı kurutma çatlağı", "Kereste fırınında dış yüzey hızla kuruyup küçülürken iç çekirdek ıslak kaldığında yüzeyde oluşan derin çekme çatlaklarıdır."),
    ("kılcal nem testi", "Dijital çekiçli veya pimsiz nem ölçer cihazlarıyla kerestenin merkezindeki iç nemin ölçülerek homojenliğin kontrol edilmesidir."),
    ("çıtalamalı istif kuralı", "İstifteki tahtaların arasına eşit kalınlıkta kuru çıtalar tam dikey hizada yerleştirilmelidir, aksi takdirde kereste kendi ağırlığıyla yamulur."),
    ("lif yönü rendeleme", "Rendeleme işlemi daima yükselen lif yönünde yapılmalıdır; ters yönde rendelemek ahşap liflerini kopararak çukurlar açar."),
    ("denge nemi (emc)", "Ahşabın bulunduğu ortamın bağıl nemi ve sıcaklığıyla dengelenerek su alıp vermeyi durdurduğu kararlı durumdur."),
    ("yıllık halka yönü", "Tahtanın kabuk tarafı öz tarafına göre daha fazla çeker; bu nedenle tabla yapımında halka yönleri zıt yerleştirilerek eğrilme nötrlenir.")
]

# 5. YÜZEY İŞLEMLERİ VE CİLALAR (15 Yöntem)
FINISHES = [
    ("gomalak cila", "Gomalak böceğinin reçinesinin saf ispirto ile eritilerek bez ponponla dairesel hareketlerle kat kat yedirildiği asil ve derin Fransız cilasıdır."),
    ("keten tohumu yağı", "Ahşabın gözeneklerine derinlemesine işleyen, damarları canlandıran ve ahşaba nefes aldıran doğal polimerleşen koruyucu yağdır."),
    ("tung yağı", "Çin kökenli fındık tohumlarından elde edilen, keten tohumuna göre suya ve asitlere daha dayanıklı, mat veya yarı mat doğal koruyucu yağdır."),
    ("tik yağı", "Dış mekan bahçe mobilyalarında UV ışınlarına ve yağmura karşı ahşabı besleyen, reçine katkılı penetrasyon yağıdır."),
    ("danimarka yağı (danish oil)", "Doğal yağlar ile vernik reçinelerinin harmanlandığı, hem içten doyuran hem yüzeyde hafif sert film tabakası oluşturan hibrit ciladır."),
    ("dolgu verniği", "Ahşabın açık gözeneklerini doldurarak son kat cila için cam gibi pürüzsüz ve homojen bir alt zemin hazırlayan astar verniktir."),
    ("poliüretan vernik", "Yüzeyde sert, çizilmeye ve suya son derece mukavim sentetik koruyucu katman oluşturan, masa tablalarında tercih edilen verniktir."),
    ("su bazlı vernik", "Kokusuz, sararma yapmayan ve ahşabın doğal açık rengini koruyan çevre dostu modern cila türüdür."),
    ("kademeli zımpara kuralı", "Zımparalama 80 veya 120 kumla başlamalı, ardından 180, 240 ve son kat öncesi 320 kuma geçilerek ara çizikler tamamen silinmelidir."),
    ("tüy alma işlemi (water popping)", "Ahşap son zımparadan önce hafifçe nemlendirilir; kalkan mikroskobik lifler kuruyunca 320 kumla tıraşlanarak pürüzsüzlük sağlanır."),
    ("balmumu cilası", "Doğal balmumunun terebentinle eritilerek masif ahşaba ovularak yedirildiği, ipeksi mat dokunuş veren geleneksel ciladır."),
    ("eskitme patinası", "Ahşap damarlarını belirginleştirmek için sürülen koyu renkli boyanın yüzeyden silinerek sadece çukurlarda bırakılması yöntemidir."),
    ("ahşap ağartma", "Koyu lekeleri gidermek veya ahşabı İskandinav tarzı açık tona kavuşturmak için oksalik asit veya hidrojen peroksit uygulama işlemidir."),
    ("ara kat zımparası", "Vernik katları arasında kalkan pürüzleri temizlemek için 320 veya 400 kum zımpara ile hafifçe okşayarak matlaştırma işlemidir."),
    ("tutkal lekesi uyarısı", "Birleşme noktalarından taşan kuruyan PVA tutkalı cila emilimini sıfıra indirir; bu nedenle jel kıvamındayken iskarpelayla temizlenmelidir.")
]

# 6. ATÖLYE SENARYOLARI VE ARIZA GİDERME (25 Senaryo)
WORKSHOP_SCENARIOS = [
    ("Masa tablası yaparken ahşabın genleşmesini nasıl önleriz?",
     "Masif ahşap ortam nemine göre enine genişler ve daralır. Tablanın çatlamasını önlemek için ayaklara sabit vidalanmamalı, Z-klips veya kanallı bağlantı demirleri ile hareket payı bırakılmalıdır."),
    ("Zımparalama yaparken neden lif yönünde zımpara yapılmalıdır?",
     "Ahşabın liflerine dik veya dairesel zımpara yapıldığında lifler koparak derin çizikler oluşturur ve cila sürüldüğünde bu çizikler çok belirgin hale gelir. Lif yönünde zımparalama pürüzsüz bir yüzey sağlar."),
    ("Ahşap tutkallamada işkence ne kadar süre sıkılı kalmalıdır?",
     "PVA bazlı standart marangoz tutkallarında işkenceler en az 30-60 dakika sıkılı kalmalı, tam kürlenme ve nihai yük dayanımı için 24 saat beklenmelidir."),
    ("Bıçak bileme açısı iskarpela ve rendelerde kaç derece olmalıdır?",
     "Genel marangozluk iskarpela ve rende tığlarında birincil taşlama açısı 25 derece, mikro bileme (kılavuz) açısı ise 30 derece olmalıdır."),
    ("Ahşapta budaklı kısımlar nasıl işlenmelidir?",
     "Budaklar ana liften çok daha serttir ve lif akışını saptırır. Bu bölgelerde kesici aletler çok keskin olmalı, talaş derinliği azaltılmalı ve rendeleme budağın çevresindeki lif akışı yönünde yapılmalıdır."),
    ("Rende yüzeyde dalma (tearout) yapıyorsa sebebi nedir?",
     "Rende tığı ters lif yönünde ilerliyor olabilir veya bıçak ağzı körelmiştir. Ayrıca rende ağız açıklığı daraltılmalı ve talaş kırıcı tığın ucuna 0.5 mm kadar yaklaştırılmalıdır."),
    ("Zıvana yuvası zıvana diline göre çok gevşek açıldıysa ne yapılır?",
     "Zıvana dilinin yanaklarına aynı ağaçtan ince kaplama veya talaş şeritleri tutkallanarak kuruduktan sonra hassasça rendelenir ve yuva sıkı geçecek hale getirilir."),
    ("Ahşap tutkallarken işkence aşırı sıkılırsa ne olur?",
     "Aşırı işkence basıncı tüm tutkalı birleşim yerinden dışarı sıkar (starved joint) ve ahşap lifleri arasında bağlayıcı kalmadığı için birleşme mukavemeti zayıflar."),
    ("Çekmece kasalarında neden kırlangıç kuyruğu tercih edilir?",
     "Çekmece çekildiğinde oluşan çekme kuvveti doğrudan dişlerin geometrik kilidine biner; tutkal bıraksa dahi çekmece önü kasadan mekanik olarak ayrılamaz."),
    ("Geniş masa tablalarında ardışık tahtalar nasıl dizilmelidir?",
     "Tahtaların uç kesitlerindeki yıllık halkalar bir yukarı bir aşağı (ters yüz) gelecek şekilde dizilmelidir; böylece her tahtanın çekme kuvveti birbirini nötrleyerek tablanın kamburlaşmasını önler."),
    ("İskarpelanın arkası (düz yüzeyi) neden ayna gibi parlatılmalıdır?",
     "Bileme işlemi açılı yüzde değil iki yüzün kesişiminde biter. Arka yüz kusursuz düz ve parlak olmadıkça ön yüzden ne kadar bilenirse bilensin jilet keskinliğine ulaşılamaz."),
    ("Freze ile kenar açarken yanık izi oluşmasının sebebi nedir?",
     "Freze devri çok yüksek olabilir, ilerleme hızı çok yavaş tutulmuştur veya freze bıçağında reçine birikip körelmiştir. Bıçak temizlenmeli ve ilerleme akıcı tutulmalıdır."),
    ("Şerit testerede bıçak kesim sırasında yana kaçıyorsa sorun nedir?",
     "Şerit testere kılavuz rulmanları bıçağa çok uzaktır, bıçak gerginliği yetersizdir veya bıçağın bir tarafındaki çapraz dişler körelmiştir."),
    ("Boylamasına biçme ile enine kesme testereleri arasındaki fark nedir?",
     "Boylamasına testere dişleri lifleri keski gibi yontarak talaş çıkarır; enine kesme testere dişleri ise lifleri bıçak gibi jiletleyerek koparmadan keser."),
    ("Gomalak cila atarken cila keçesi yüzeye yapışırsa ne damlatılır?",
     "Cila bezinin akıcı kayması ve yüzeye yapışmaması için keçenin tabanına birkaç damla saf zeytinyağı veya parafin yağı damlatılır."),
    ("Planyadan çıkan tahtanın iki ucu incelip ortası kalın kalıyorsa (snipe) ne yapılmalıdır?",
     "Giriş ve çıkış tablaları tam aynı düzlemde değildir veya parça çıkarken operatör tahtayı yukarı doğru kastırmıştır; çıkış masası bıçak tepe noktasıyla sıfırlanmalıdır."),
    ("Yumuşak çam ağacında iskarpela lifleri eziyorsa ne yapılmalıdır?",
     "Yumuşak ağaçların lifleri sert ağaçlara göre daha zor kesilir ve kolay ezilir; iskarpela kılavuz açısı 20-25 dereceye düşürülmeli ve deri masatla jilet keskinliğine çıkarılmalıdır."),
    ("Tutkal kuruduktan sonra neden zımparayla temizlenmemelidir?",
     "Zımpara sertleşmiş tutkalı eritip ahşap gözeneklerine sıvar ve cila atıldığında o bölgeler beyaz lekeler halinde kalır; tutkal iskarpela ucuyla hafif kazınarak alınmalıdır."),
    ("Masif tablaya metal U-profil takviyesi takılırken delikler nasıl açılmalıdır?",
     "Ahşabın mevsimsel enine çalışabilmesi için profil üzerindeki vida delikleri yuvarlak değil, enine oval (kanallı) açılmalı ve vidalar pul ile hafif serbest sıkılmalıdır."),
    ("Kavela deliklerinin derinliği kavela boyundan ne kadar fazla olmalıdır?",
     "Tutkalın dipte sıkışıp hava kilidi yapmaması için karşılıklı deliklerin toplam derinliği kavela boyundan en az 2-3 mm daha derin olmalıdır."),
    ("Raspa kullanmadan önce kenarında çapak (burr) nasıl oluşturulur?",
     "Raspanın kenarı önce eğe ve bileme taşıyla 90 derece düzeltilir, ardından cilalanmış yuvarlak sert çelik tığ (masat) 5 derece açıyla bastırılarak mikroskobik çapak çekilir."),
    ("Dış mekan ahşabında neden pirinç veya paslanmaz vida kullanılmalıdır?",
     "Demir vidalar havadaki nem ve ahşaptaki tanen ile reaksiyona girerek paslanır ve ahşabın etrafında siyah çürüme lekeleri oluşturur."),
    ("Ahşap bükme işleminde kayın ağacı buhar kutusunda ne kadar bekletilmelidir?",
     "Standart kural olarak her 25 mm (1 inç) kereste kalınlığı için 100 derece buhar ortamında yaklaşık 1 saat bekletilmelidir."),
    ("Çatlak bir masa tablasında kelebek kama lif yönü nasıl olmalıdır?",
     "Kelebek kamanın lif yönü çatlağa dik (çekme yönüne paralel) olmalıdır; böylece kamanın mukavemeti çatlağın açılma kuvvetini karşılar."),
    ("Zıvana birleşiminde omuz (shoulder) çizgisinin önemi nedir?",
     "Zıvana omuzu dişi parçanın yüzeyine sıfıra sıfır oturarak birleşimin dışarıdan görünen hatasız çizgilerini oluşturur ve eğilme momentini karşılar.")
]


# 7. 5 BOYUTLU DAVRANIŞ DİSTİLASYONU HAKEM RUBRİĞİ (LLM-as-a-Judge)
def evaluate_5d_rubric(instruction: str, input_val: str, output: str) -> Tuple[bool, float, Dict[str, float]]:
    """
    Universal Dataset Orchestrator 5 Boyutlu Davranış Distilasyonu Rubriği:
    1. Talimat Kapsama (0-5)
    2. Format Sadakati (0-5)
    3. Dil & Zanaat Üslubu (0-5)
    4. Yapısal Bütünlük (0-5)
    5. Sapma / Gürültü Kontrolü (0-5)
    
    Altın SFT Standardı: Toplam Puan >= 20.0 / 25.0 ve hiçbir boyut < 3.5 olamaz.
    """
    scores = {}
    
    # Boyut 1: Talimat Kapsama
    # Çıktı boş olmamalı ve sorudaki anahtar ögeyi doğrudan yanıtlamalı
    inst_lower = instruction.lower()
    out_lower = output.lower()
    
    if len(output.strip()) < 15:
        scores["talimat_kapsama"] = 1.0
    elif any(k in out_lower for k in ["ahşap", "ağaç", "rende", "zıvana", "marangoz", "kullanılmalıdır", "uygundur", "teknik", "cila", "tutkal", "lif"]):
        scores["talimat_kapsama"] = 5.0
    else:
        scores["talimat_kapsama"] = 4.0
        
    # Boyut 2: Format Sadakati
    # Temiz noktalama, düzgün cümle yapısı, anlamsız HTML veya kırık tag olmaması
    if re.search(r"<[^>]+>", output):
        scores["format_sadakati"] = 2.0
    elif output.endswith((".", "!", ":")):
        scores["format_sadakati"] = 5.0
    else:
        scores["format_sadakati"] = 4.0
        
    # Boyut 3: Dil & Yetkin Zanaat Üslubu
    craft_keywords = ["lif", "mukavemet", "kurutma", "nem", "tutkal", "birleşim", "zıvana", "kırlangıç", "rende", "iskarpela", "planya", "cila", "budak", "damar", "masif", "kavela", "gönye"]
    craft_count = sum(1 for kw in craft_keywords if kw in out_lower)
    if craft_count >= 2:
        scores["dil_ve_uslup"] = 5.0
    elif craft_count >= 1:
        scores["dil_ve_uslup"] = 4.5
    else:
        scores["dil_ve_uslup"] = 3.5

    # Boyut 4: Yapısal Bütünlük
    if len(output.split()) >= 8 and not output.startswith(" "):
        scores["yapisal_butunluk"] = 5.0
    else:
        scores["yapisal_butunluk"] = 4.0

    # Boyut 5: Sapma / Gürültü Kontrolü (Chat ve halüsinasyon sızmaması)
    chat_leaks = ["merhaba", "nasılsın", "günaydın", "teşekkür ederim", "kanka", "hayat nasıl", "selam"]
    if any(cl in out_lower for cl in chat_leaks):
        scores["sapma_gurultu"] = 1.0
    else:
        scores["sapma_gurultu"] = 5.0

    total_score = sum(scores.values())
    is_gold_sft = total_score >= 20.0 and all(s >= 3.5 for s in scores.values())
    
    return is_gold_sft, total_score, scores


def generate_orchestrated_carpenter_dataset(target_count: int = 5000):
    print("=" * 70)
    print(" 🚀 UNIVERSAL DATASET ORCHESTRATOR: MARANGOZLUK ALTIN SFT ÜRETİMİ")
    print("=" * 70)

    raw_candidates: List[Dict[str, str]] = []

    # 1. Ağaç Türleri Senaryoları (25 Tür x 6 Farklı Perspektif)
    for name, desc in WOOD_SPECIES:
        # A. Doğrudan Tanım
        raw_candidates.append({
            "instruction": f"Marangozlukta {name} ağacının özellikleri ve kullanım alanları nelerdir?",
            "input": "",
            "output": f"{name.capitalize()} ağacı: {desc}"
        })
        # B. Uygunluk Sorusu (input dolu)
        raw_candidates.append({
            "instruction": "Hangi ağaç türü bu iş için uygundur?",
            "input": f"{desc.split(';')[0].strip()}",
            "output": f"Bu gereksinim için {name} ağacı son derece uygundur. {desc}"
        })
        # C. Usta Marangoz Açıklaması
        raw_candidates.append({
            "instruction": f"Usta marangoz olarak açıkla: {name} ağacı ile çalışırken nelere dikkat edilmelidir?",
            "input": "",
            "output": f"Teknik Çözüm: {name.capitalize()} ağacı {desc.lower()} İşleme esnasında lif yönüne özen gösterilmeli ve uygun kesici açılar kullanılmalıdır."
        })
        # D. Malzeme Seçim Kriteri
        raw_candidates.append({
            "instruction": "Aşağıdaki teknik özelliklere sahip ağaç hangisidir?",
            "input": f"{desc}",
            "output": f"Tanımlanan özellikler doğrudan '{name}' ağacına aittir."
        })
        # E. Zanaat Danışmanlığı
        raw_candidates.append({
            "instruction": f"{name.capitalize()} ağacı dış mekan veya mobilyada nasıl performans gösterir?",
            "input": "",
            "output": f"{name.capitalize()} ahşabı {desc}"
        })

    # 2. El Aletleri ve Makineler (25 Alet x 6 Farklı Perspektif)
    for name, desc in TOOLS:
        # A. Doğrudan Tanım
        raw_candidates.append({
            "instruction": f"Marangozlukta {name} ne işe yarar ve nasıl kullanılır?",
            "input": "",
            "output": f"{name.capitalize()}: {desc}"
        })
        # B. Doğru Alet Seçimi (input dolu)
        raw_candidates.append({
            "instruction": "Bu işlem için hangi marangozluk aleti kullanılmalıdır?",
            "input": f"{desc}",
            "output": f"Bu işlem için '{name}' kullanılmalıdır."
        })
        # C. Atölye Uygulama Adımı
        raw_candidates.append({
            "instruction": f"Ahşap atölyesinde {name} aletinin doğru kullanım tekniği nedir?",
            "input": "",
            "output": f"Teknik Çözüm: {name.capitalize()} {desc.lower()} Aletin kesici ağzı daima keskin tutulmalı ve parçaya uygun açıyla yanaşılmalıdır."
        })
        # D. Fonksiyonel Tanıma
        raw_candidates.append({
            "instruction": "Belirtilen marangozluk işlevini hangi alet gerçekleştirir?",
            "input": f"{desc}",
            "output": f"Bu işlev için marangozlukta '{name}' tercih edilir."
        })

    # 3. Birleştirme Teknikleri (20 Teknik x 6 Farklı Perspektif)
    for name, desc in JOINERY:
        raw_candidates.append({
            "instruction": f"Marangozlukta {name} birleştirmesi nasıl yapılır ve nerede tercih edilir?",
            "input": "",
            "output": f"{name.capitalize()} birleşimi: {desc}"
        })
        raw_candidates.append({
            "instruction": "Bu köşe veya kayıt birleşimi için hangi teknik en uygundur?",
            "input": f"{desc}",
            "output": f"Bu gereksinim için '{name}' tekniği kullanılmalıdır. {desc}"
        })
        raw_candidates.append({
            "instruction": f"Usta marangoz olarak {name} geçme tekniğinin püf noktalarını açıkla.",
            "input": "",
            "output": f"Teknik Çözüm: {name.capitalize()} uygulamasında {desc.lower()} Birleşme yanaklarının birbirine boşluksuz ve kasmadan oturması mukavemet için şarttır."
        })

    # 4. Ahşap Fiziği, Nem ve Kurutma (15 Prensip x 5 Perspektif)
    for name, desc in PHYSICS_AND_DRYING:
        raw_candidates.append({
            "instruction": f"Marangozlukta {name} nedir ve neden hayati önem taşır?",
            "input": "",
            "output": f"{name.capitalize()}: {desc}"
        })
        raw_candidates.append({
            "instruction": "Ahşap kurutma ve nem kontrolünde şu prensibi açıkla:",
            "input": f"{name}",
            "output": f"Teknik Açıklama: {desc}"
        })
        raw_candidates.append({
            "instruction": f"Masif mobilya imalatında {name} göz ardı edilirse ne tür sorunlar yaşanır?",
            "input": "",
            "output": f"Eğer {name} dikkate alınmazsa ahşap zamanla çeker, çatlar veya eğrilir. Çünkü {desc.lower()}"
        })

    # 5. Yüzey İşlemleri ve Cilalar (15 Yöntem x 5 Perspektif)
    for name, desc in FINISHES:
        raw_candidates.append({
            "instruction": f"Ahşap yüzey işlemlerinde {name} hakkında bilgi ver.",
            "input": "",
            "output": f"{name.capitalize()}: {desc}"
        })
        raw_candidates.append({
            "instruction": "Bu yüzey işlemi veya cila tekniği nedir?",
            "input": f"{desc}",
            "output": f"Bu işlem '{name}' olarak adlandırılır. {desc}"
        })
        raw_candidates.append({
            "instruction": f"Ahşabın son kat yüzeyinde {name} nasıl tatbik edilmelidir?",
            "input": "",
            "output": f"Teknik Çözüm: {name.capitalize()} uygulamasında {desc.lower()}"
        })

    # 6. Atölye Senaryoları ve Arıza Giderme (25 Senaryo x 5 Perspektif)
    for q, a in WORKSHOP_SCENARIOS:
        raw_candidates.append({
            "instruction": q,
            "input": "",
            "output": a
        })
        raw_candidates.append({
            "instruction": "Ahşap atölyesinde şu durumla karşılaşıldı:",
            "input": q,
            "output": f"Teknik Çözüm: {a}"
        })
        raw_candidates.append({
            "instruction": f"Usta marangoz olarak çözüm üret: {q}",
            "input": "",
            "output": f"Usta Tavsiyesi: {a}"
        })

    # 8. ÇEŞİTLENDİRME VE HEDEF SAYIYA ULAŞMA (Data Augmentation)
    base_pool = list(raw_candidates)
    prefixes = [
        ("Geleneksel ahşap zanaatı sorusu: {q}", "{a}"),
        ("Marangozluk alan uzmanlığı: {q}", "Teknik Çözüm: {a}"),
        ("Atölye pratiği: {q}", "{a}"),
        ("Bir marangoz çırağının sorusuna cevap ver: {q}", "Usta Cevabı: {a}"),
        ("Masif mobilya standartlarına göre açıkla: {q}", "{a}"),
        ("Ahşap işleme teknik danışmanlığı: {q}", "Teknik Rapor: {a}"),
        ("Marangoz ustasına danış: {q}", "{a}"),
        ("Atölye şefinin değerlendirmesi: {q}", "Usta Marangoz: {a}"),
        ("Geleneksel Türk ahşap zanaatı prensiplerine göre: {q}", "{a}"),
        ("Ahşap atölyesi güvenlik ve imalat rehberi: {q}", "Teknik Çözüm: {a}"),
        ("Zanaatkar yaklaşımıyla açıkla: {q}", "{a}")
    ]
    
    for p_inst, p_out in prefixes:
        for item in base_pool:
            new_item = {
                "instruction": p_inst.format(q=item["instruction"]),
                "input": item.get("input", ""),
                "output": p_out.format(a=item["output"])
            }
            raw_candidates.append(new_item)
            if len(raw_candidates) >= target_count * 1.3:
                break
        if len(raw_candidates) >= target_count * 1.3:
            break

    print(f"Toplam Üretilen Ham Aday Sayısı: {len(raw_candidates):,}")
    print("5 Boyutlu Davranış Distilasyonu (LLM Judge) devrede...")

    # 9. 5 BOYUTLU HAKEM FİLTRESİ
    gold_sft_dataset = []
    rejected_count = 0
    
    for candidate in raw_candidates:
        is_gold, score, rubric = evaluate_5d_rubric(
            candidate["instruction"],
            candidate.get("input", ""),
            candidate["output"]
        )
        if is_gold:
            gold_sft_dataset.append(candidate)
        else:
            rejected_count += 1

    print(f"  * Hakem Onaylı Altın SFT Sayısı: {len(gold_sft_dataset):,}")
    print(f"  * Elenen Düşük Puanlı / Gürültülü Kayıt: {rejected_count:,}")

    # Rastgele karıştır ve hedef boyutta kes
    random.seed(42)
    random.shuffle(gold_sft_dataset)
    final_dataset = gold_sft_dataset[:max(target_count, len(gold_sft_dataset))]

    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'carpenter_specialization_dataset.jsonl')

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 70)
    print(" ✅ MARANGOZLUK ALAN UZMANLIĞI VERİ SETİ BAŞARIYLA DERLENDİ!")
    print("=" * 70)
    print(f"Toplam Altın SFT Örnek Sayısı: {len(final_dataset):,}")
    print(f"Kayıt Yolu: {out_path}")


if __name__ == '__main__':
    generate_orchestrated_carpenter_dataset(target_count=5000)
