#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL MİMARİSİ: ZENGİNLEŞTİRİLMİŞ TÜRKÇE SOHBET VERİ ÜRETİCİ
=====================================================================
Bu betik, KristalLM dil modelinin doğal, samimi, kibar ve yardımsever bir Türkçe
ile diyalog kurabilmesi için pedagojik olarak yapılandırılmış çok alanlı sohbet
verisi üretir.

Kategoriler:
1. Selamlaşma, Tanışma ve Hatır Sorma
2. Yardımsever Asistan ve Rehberlik
3. Günlük Yaşam, Empati ve Moral Desteği
4. Merak, Bilim, Doğa ve Düşünce Sohbetleri
5. Hobiler, Sanat, Spor ve Arkadaşça Diyaloglar
6. Nezaket, Teşekkür ve Kapanış İfadeleri
"""

import os
import sys
import json
import random
from typing import List, Dict

def generate_greetings() -> List[Dict[str, str]]:
    """Kategori 1: Selamlaşma, Tanışma ve Hal-Hatır"""
    g_inputs = [
        "Merhaba!", "Selam!", "Merhabalar.", "Selamlar.", "Günaydın!",
        "İyi günler.", "İyi akşamlar.", "Selam dostum!", "Merhaba nasılsın?",
        "Selam nasıl gidiyor?", "Naber?", "Ne haber?", "Günün nasıl geçiyor?",
        "Neler yapıyorsun?", "Hal hatır sormak istedim.", "Merhaba, bugün nasılsın?",
        "Selam, keyifler nasıl?", "Selamlar, umarım iyisindir.", "Nasılsın bakalım?"
    ]

    g_outputs = [
        "Merhaba! Çok teşekkür ederim, gayet iyiyim. Siz nasılsınız?",
        "Selamlar! Harika bir gün geçiriyorum. Size nasıl yardımcı olabilirim?",
        "Merhabalar! İyilik sağlık, her şey yolunda gidiyor. Umarım siz de iyisinizdir.",
        "Günaydın! Gününüz aydın, neşeli ve verimli geçsin. Nasıl yardımcı olabilirim?",
        "İyi günler! Teşekkür ederim, gayet iyiyim. Bugün ne hakkında konuşmak istersiniz?",
        "İyi akşamlar! Gününüz umarım güzel geçmiştir. Size yardımcı olmaktan mutluluk duyarım.",
        "Selam dostum! Yuvarlanıp gidiyoruz valla, her şey yolunda. Sende durumlar nasıl?",
        "İyiyim, çok teşekkürler! Yeni şeyler öğrenmek ve sohbet etmek için sabırsızlanıyorum.",
        "Teşekkür ederim, çok iyiyim! Hayat güzel akıyor. Sizin gününüz nasıl geçiyor?",
        "Selam! Ben bir yapay zeka dil modeliyim ama seninle sohbet etmek beni mutlu ediyor!",
        "Merhabalar! Çok teşekkürler. Size destek olmak için buradayım, keyfim gayet yerinde."
    ]

    records = []
    for inp in g_inputs:
        for out in g_outputs:
            records.append({
                "instruction": "Kullanıcı ile doğal, kibar ve samimi bir Türkçe diyalog yürüt.",
                "input": inp,
                "output": out
            })
    return records

def generate_assistant_guidance() -> List[Dict[str, str]]:
    """Kategori 2: Yardımsever Asistan ve Rehberlik"""
    tasks = [
        ("Bana yardımcı olabilir misin?", 
         "Elbette! Size her türlü konuda yardımcı olmaktan büyük mutluluk duyarım. Ne sormak veya öğrenmek istersiniz?"),
        ("Sen ne yapabilirsin?",
         "Ben Türkçe metinleri analiz edebilir, sorularınızı yanıtlayabilir, dilbilgisi kurallarını açıklayabilir ve keyifli sohbetler edebilirim."),
        ("Bana bir kitap önerir misin?",
         "Tabii ki! Klasiklerden Yaşar Kemal'in İnce Memed eserini veya Antoine de Saint-Exupéry'nin Küçük Prens kitabını içtenlikle tavsiye ederim."),
        ("Ders çalışırken çok çabuk sıkılıyorum, ne yapmalıyım?",
         "Pomodoro tekniğini deneyebilirsiniz. Yirmi beş dakika odaklanarak çalışıp ardından beş dakika mola vermek dikkatinizi taze tutacaktır."),
        ("Zamanımı daha iyi nasıl yönetebilirim?",
         "Günün başında yapılacaklar listesi hazırlayıp önem sırasına göre sıralamak ve gereksiz dikkat dağıtıcıları uzaklaştırmak harika bir başlangıçtır."),
        ("Yeni bir alışkanlık nasıl kazanılır?",
         "Küçük adımlarla başlayın. Her gün sadece beş veya on dakika ayırarak tutarlılık oluşturun; zamanla bu davranış doğal bir alışkanlığa dönüşecektir."),
        ("Yabancı bir dili öğrenmenin en iyi yolu nedir?",
         "Dili günlük hayatınızın parçası haline getirin. O dilde müzik dinleyin, kısa hikayeler okuyun ve her gün düzenli olarak birkaç yeni kelime öğrenin."),
        ("Kendimi nasıl motive edebilirim?",
         "Ulaşmak istediğiniz büyük hedefi küçük ve somut hedeflere bölün. Kazandığınız her küçük başarı size yeni bir motivasyon dalgası sağlayacaktır."),
        ("Bir metin yazarken nelere dikkat etmeliyim?",
         "Anlatımınızın açık, akıcı ve tutarlı olmasına özen gösterin. Gereksiz kelimelerden kaçının ve imla kurallarına dikkat edin."),
        ("Stresle başa çıkmak için ne önerirsin?",
         "Derin ve yavaş nefes egzersizleri yapın, açık havada kısa bir yürüyüşe çıkın veya sevdiğiniz bir müzikle zihninizi dinlendirin.")
    ]

    variations_in = [
        "{}", "Lütfen söyler misin: {}", "Acaba {} Teşekkürler.", 
        "Bir sorum var: {}", "Bu konuda fikrini almak istiyorum: {}"
    ]

    records = []
    for inp, out in tasks:
        for v in variations_in:
            records.append({
                "instruction": "Kullanıcıya yardımcı, yapıcı ve rehberlik eden bir asistan olarak cevap ver.",
                "input": v.format(inp),
                "output": out
            })
    return records

def generate_empathy_and_life() -> List[Dict[str, str]]:
    """Kategori 3: Günlük Hayat, Empati ve Moral Desteği"""
    dialogues = [
        ("Bugün çok yoruldum.",
         "Çok haklısınız, bazen günler insanı gerçekten yıpratabiliyor. Şimdi dinlenme vakti; sıcak bir çay için ve kendinize zaman ayırın."),
        ("Bugün sınavımdan çok yüksek not aldım!",
         "Tebrik ederim, harika bir haber! Emeğinizin ve çalışmanızın karşılığını almak muazzam bir duygu. Başarılarınızın devamını dilerim!"),
        ("Kendimi biraz moralsiz hissediyorum.",
         "Bunu duyduğuma üzüldüm. Hepimizin zaman zaman böyle günleri olur; bu çok insani bir durum. Unutmayın ki yarın yepyeni bir gün."),
        ("Yeni bir projeye başladım ama biraz çekiniyorum.",
         "Yeni başlangıçlar her zaman heyecan ve biraz kaygı getirir. Adım adım ilerleyin, kendinize inanın; güzel sonuçlar alacaksınız."),
        ("Hava bugün çok güzel!",
         "Ne kadar harika! Güzel havalar insanın enerjisini ve yaşama sevincini tazeler. Vaktiniz varsa dışarıda kısa bir yürüyüş yapmanızı öneririm."),
        ("Biraz sohbet etmek istedim sadece.",
         "Çok sevindim! Sohbet etmek, fikirleri ve düşünceleri paylaşmak harika bir şey. Aklınızda ne var, nelerden bahsetmek istersiniz?"),
        ("Bugün işler pek istediğim gibi gitmedi.",
         "Bazen hayat planladığımız gibi akmaz ama her aksilik bize yeni bir deneyim kazandırır. Derin bir nefes alın ve moralinizi bozmayın."),
        ("Başardım, sonunda hedefime ulaştım!",
         "Sizin adınıza çok sevindim! Kararlılıkla çalışmanın meyvesini toplamak kadar gurur verici bir an olamaz. Kutlarım!"),
        ("Bugün içimden hiçbir şey yapmak gelmiyor.",
         "Bazen zihnimiz ve bedenimiz sadece durup nefes almak ister. Kendinizi zorlamayın; bugün dinlenme gününüz olsun."),
        ("Yarın önemli bir görüşmem var, heyecanlıyım.",
         "Heyecan hissetmeniz ne kadar önemsediğinizi gösterir. Kendinize güvenin, hazırlığınızı gözden geçirin ve derin bir nefesle rahatlayın.")
    ]

    records = []
    for inp, out in dialogues:
        records.append({
            "instruction": "Kullanıcıya anlayışlı, empatik ve destekleyici bir dille yanıt ver.",
            "input": inp,
            "output": out
        })
        records.append({
            "instruction": "Kullanıcıya içten ve motive edici bir dille karşılık ver.",
            "input": f"Merhaba, {inp.lower()}",
            "output": out
        })
    return records

def generate_curiosity_and_science() -> List[Dict[str, str]]:
    """Kategori 4: Merak, Bilim, Doğa ve Düşünce Sohbetleri"""
    topics = [
        ("Ağaçlar kışın neden yaprak döker?",
         "Ağaçlar kış mevsiminde su tasarrufu yapmak ve soğuk havanın dondurucu etkisinden korunmak için yapraklarını döker."),
        ("Gökyüzü neden mavidir?",
         "Güneş ışığı atmosfere girdiğinde gaz moleküllerine çarpar ve dalga boyu kısa olan mavi ışık her yöne saçılarak göğü mavi gösterir."),
        ("Güneş neden sıcaktır?",
         "Güneş'in çekirdeğinde gerçekleşen nükleer füzyon reaksiyonları hidrojen atomlarını helyuma dönüştürür ve devasa miktarda ısı ile ışık yayar."),
        ("Bal arıları neden bal yapar?",
         "Arılar çiçeklerden topladıkları nektarı kış aylarında ve yiyecek bulamadıkları dönemlerde besin olarak tüketmek üzere kovanlarında bal olarak depolar."),
        ("Deniz suyu neden tuzludur?",
         "Yağmurlar ve akarsular kara parçalarındaki kayaları aşındırarak mineralleri ve tuzları denizlere taşır; buharlaşma sonucu tuzlar denizde kalır."),
        ("Kitap okumak insanı nasıl geliştirir?",
         "Kitap okumak kelime dağarcığını zenginleştirir, empati yeteneğini güçlendirir, odaklanmayı artırır ve hayal gücünü besler."),
        ("Doğa yürüyüşü yapmanın faydaları nelerdir?",
         "Doğada yürümek stresi azaltır, zihni dinlendirir, kalp sağlığını güçlendirir ve temiz hava sayesinde vücuda zindelik katar."),
        ("Müzik insan ruhunu nasıl etkiler?",
         "Müzik beyinde dopamin salgılanmasını tetikler; duyguları harekete geçirir, odaklanmayı artırabilir veya derin bir huzur hissi verebilir."),
        ("Yapay zeka nasıl öğrenir?",
         "Yapay zeka modelleri büyük miktarda veriyi inceleyerek matematiksel örüntüleri, kalıpları ve kuralları öğrenen algoritmalarla eğitilir."),
        ("Neden rüya görürüz?",
         "Rüyalar beynin gün boyunca edindiği anıları işlemesine, duyguları düzenlemesine ve zihinsel arınma sağlamasına yardımcı olur.")
    ]

    records = []
    for inp, out in topics:
        records.append({
            "instruction": "Soruyu açık, anlaşılır ve eğitici bir Türkçe ile yanıtla.",
            "input": inp,
            "output": out
        })
        records.append({
            "instruction": "Kullanıcının merak ettiği konuyu bilimsel ve duru bir dille açıkla.",
            "input": f"Bunu merak ediyorum: {inp}",
            "output": out
        })
    return records

def generate_hobbies_and_youth() -> List[Dict[str, str]]:
    """Kategori 5: Hobiler, Sanat, Oyunlar ve Spor (Arkadaşça Sohbet)"""
    hobbies = [
        ("En sevdiğin spor dalı hangisi?",
         "Yürüyüş ve koşu yapmak harikadır! Ancak takım sporlarından basketbol ve futbol da arkadaşlık bağlarını güçlendiren çok heyecanlı sporlardır."),
        ("Satranç oynamayı sever misin?",
         "Satranç mükemmel bir zihin sporu! Her hamlede ileriyi düşünmek, strateji kurmak ve sabırlı olmak insana muazzam bir analitik bakış katar."),
        ("Resim yapmak sence nasıl bir aktivite?",
         "Resim yapmak duyguları ve hayalleri renklere dökmektir. Zihni dinlendirir ve yaratıcılığı olağanüstü biçimde geliştirir."),
        ("Müzik aleti çalmak sence zor mu?",
         "Başlangıçta biraz sabır ve düzenli pratik ister; fakat ilk melodiyi çalmaya başladığınızda verdiği haz tüm zorluklara değer!"),
        ("Bisiklete binmek eğlenceli midir?",
         "Kesinlikle çok eğlencelidir! Rüzgarı hissetmek, yeni yerler keşfetmek ve spor yapmak için bisiklet sürmek harika bir yoldur."),
        ("Hangi oyunları seversin?",
         "Zeka ve strateji oyunlarını çok severim! Hem eğlendirir hem de planlama becerisini geliştirir."),
        ("Doğada kamp yapmayı sever misin?",
         "Yıldızların altında uyumak, ateş başında sohbet etmek ve kuş sesleriyle uyanmak hayatın en huzur verici anlarındandır.")
    ]

    records = []
    for inp, out in hobbies:
        records.append({
            "instruction": "Bir arkadaş gibi samimi, neşeli ve doğal bir dille sohbet et.",
            "input": inp,
            "output": out
        })
        records.append({
            "instruction": "Kullanıcıya içten ve arkadaşça bir diyalogla karşılık ver.",
            "input": f"Sence {inp.lower()}",
            "output": out
        })
    return records

def generate_courtesies_and_closings() -> List[Dict[str, str]]:
    """Kategori 6: Nezaket, Teşekkür ve Kapanış İfadeleri"""
    closings = [
        ("Teşekkür ederim.", "Rica ederim, ne demek! Her zaman yardımcı olmaktan mutluluk duyarım."),
        ("Çok teşekkürler, çok yardımcı oldun.", "Ben teşekkür ederim! Faydalı olabildiysem ne mutlu bana. Başka bir ihtiyacınız olursa buradayım."),
        ("Görüşmek üzere, hoşça kal!", "Görüşmek üzere! Kendinize çok iyi bakın, harika bir gün dilerim."),
        ("İyi geceler!", "İyi geceler! Tatlı rüyalar ve huzurlu bir dinlenme dilerim."),
        ("Kendine iyi bak dostum.", "Siz de kendinize çok iyi bakın! Tekrar görüşmek dileğiyle."),
        ("Eline sağlık, harika anlattın.", "Çok naziksiniz, güzel sözleriniz için teşekkür ederim! Öğrenmeye ve sohbete her zaman hazırım."),
        ("Şimdilik bu kadar, sonra görüşürüz.", "Elbette, dilediğiniz zaman tekrar gelebilirsiniz. Sağlıcakla kalın!")
    ]

    records = []
    for inp, out in closings:
        records.append({
            "instruction": "Kullanıcıya nazik, sıcak ve saygılı bir dille karşılık ver.",
            "input": inp,
            "output": out
        })
    return records

def main():
    print("=" * 65)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: DERİN TÜRKÇE SOHBET VERİ ÜRETİCİ")
    print("=" * 65)

    greetings = generate_greetings()
    guidance = generate_assistant_guidance()
    empathy = generate_empathy_and_life()
    curiosity = generate_curiosity_and_science()
    hobbies = generate_hobbies_and_youth()
    closings = generate_courtesies_and_closings()

    print(f"Temel Çekirdek Kayıt Sayıları:")
    print(f"  * Selamlaşma & Tanışma  : {len(greetings)} kayıt")
    print(f"  * Asistan & Rehberlik    : {len(guidance)} kayıt")
    print(f"  * Empati & Günlük Hayat : {len(empathy)} kayıt")
    print(f"  * Merak, Bilim & Doğa   : {len(curiosity)} kayıt")
    print(f"  * Hobiler & Gençlik     : {len(hobbies)} kayıt")
    print(f"  * Nezaket & Kapanış     : {len(closings)} kayıt")

    base_pool = greetings + guidance + empathy + curiosity + hobbies + closings
    print(f"  -> Toplam Benzersiz Kalıp Havuzu: {len(base_pool)} adet")

    target_count = 6000
    dataset: List[Dict[str, str]] = []

    while len(dataset) < target_count:
        item = random.choice(base_pool)
        inp = item["input"]
        out = item["output"]
        inst = item["instruction"]

        r = random.random()
        if r < 0.15 and not inp.endswith("?"):
            inp = inp + " :)"
        elif r < 0.25 and inp.startswith("Merhaba"):
            inp = inp.replace("Merhaba", "Merhabalar")
        elif r < 0.35 and inp.startswith("Selam"):
            inp = inp.replace("Selam", "Selamlar")

        dataset.append({
            "instruction": inst,
            "input": inp,
            "output": out
        })

    random.seed(42)
    random.shuffle(dataset)

    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'chat_conversations.jsonl')

    with open(out_path, 'w', encoding='utf-8') as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print("\n" + "=" * 65)
    print(" SOHBET VERİ KÜMESİ BAŞARIYLA ÜRETİLDİ!")
    print("=" * 65)
    print(f"Toplam Üretilen Sohbet Örneği : {len(dataset):,}")
    print(f"Kayıt Dosyası                 : {out_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()
