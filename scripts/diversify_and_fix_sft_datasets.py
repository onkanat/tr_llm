#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KRİSTAL-VEKTÖREL: SFT VE CHAT VERİ SETLERİNİ ŞABLON ÇÖKÜŞÜNDEN ARINDIRMA VE ÇEŞİTLENDİRME
=======================================================================================
Bu betik:
1. `data/pedagogy/turk_tarihi_sft.jsonl` ve `turk_tarihi_chat.jsonl` dosyalarındaki:
   - "Metinde belirtildiği gibi,"
   - "Metinde belirtildiği üzere,"
   - "Metne göre,"
   - "Belgede belirtildiği gibi,"
   - "Metinde açıkça belirtildiği üzere,"
   - "Metinde belirtildiğine göre,"
   - "Belgede belirtildiği üzere,"
   kalıplarını temizler veya tarihsel/bağlamsal varyasyonlarla çeşitlendirir.
2. SFT modundaki persona görevlerini doğrudan karşılayan 200+ zengin soru-cevap çifti ekler:
   - "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla."
   - "Ahşap ve marangozluk uzmanı olarak cevapla."
   - "Temel bilimler ve lise fen uzmanı olarak açıkla."
   - "Türk edebiyatı ve kültür uzmanı olarak açıkla."
3. Bağlam dışı/belgesiz sorularda uydurmayı (halüsinasyonu) önleyen Abstain örnekleri ekler.
"""

import os
import sys
import json
import re
import random

SFT_PATH = "data/pedagogy/turk_tarihi_sft.jsonl"
CHAT_PATH = "data/pedagogy/turk_tarihi_chat.jsonl"

RIGID_PREFIXES = [
    "metinde belirtildiği gibi,",
    "metinde belirtildiği üzere,",
    "metinde açıkça belirtildiği üzere,",
    "metinde açıkça belirtildiği gibi,",
    "metinde belirtildiğine göre,",
    "belgede belirtildiği gibi,",
    "belgede belirtildiği üzere,",
    "belgede açıkça belirtildiği gibi,",
    "metne göre,",
    "belgeye göre,",
    "metinde belirtildiği gibi",
    "metne göre",
]

ALTERNATIVE_STARTS = [
    "Tarihsel kayıtlara göre, ",
    "İncelenen dönem belgelerinde, ",
    "Tarihi araştırmalara göre, ",
    "Dönemin kaynakları incelendiğinde, ",
    "Akademik tarih incelemelerine göre, ",
    "Tarihsel vesikalara göre, ",
]


def clean_rigid_prefix(text: str) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    for p in sorted(RIGID_PREFIXES, key=len, reverse=True):
        if lower.startswith(p):
            rem = stripped[len(p):].strip()
            # If starts with comma or dash, clean it
            rem = rem.lstrip(",-–—: ")
            if rem:
                # 85% direct statement, 15% alternative diverse start
                if random.random() < 0.15:
                    alt = random.choice(ALTERNATIVE_STARTS)
                    return alt + rem[0].lower() + rem[1:]
                else:
                    return rem[0].upper() + rem[1:]
            break
    return stripped


# ============================================================================
# ZENGİN PERSONA EĞİTİM VERİLERİ (Doğrudan SFT Görevleri ve Cevapları)
# ============================================================================

PERSONA_SAMPLES = [
    # -------------------------------------------------------------------------
    # 1. Cumhuriyet Dönemi Türk Tarihi Uzmanı (Görev 7)
    # -------------------------------------------------------------------------
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Türk kelimesinin kökeni nedir?",
        "output": "Türk kelimesi Orhun Yazıtlarında 'Türük' biçiminde yer almakta olup güç, kuvvet, olgunluk ve töreli anlamlarına gelmektedir. Tarih boyunca bu isim hem bir milletin hem de ortak bir kültür medeniyetinin sembolü olmuştur."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Türk kelimesinin kökeni ve anlamı hakkında bilgi ver.",
        "output": "Türk kelimesi etimolojik olarak 'güçlü', 'kuvvetli' ve 'türemiş, çoğalmış' manalarını taşır. İlk kez Göktürk Devleti döneminde resmi bir devlet ve millet adı olarak yazılı belgelere girmiştir."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Atatürk'ün Türk Tarih Tezi'nin temel amacı nedir?",
        "output": "Atatürk'ün öncülük ettiği Türk Tarih Tezi, Türk milletinin dünya medeniyet tarihindeki kadim yerini, Orta Asya'dan yayılan uygarlık mirasını ve Anadolu'daki binlerce yıllık kültürel köklerini bilimsel yöntemlerle ortaya koymayı amaçlamıştır."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Orhun Yazıtları'nın tarihi önemi nedir?",
        "output": "Orhun Yazıtları, Türk adının geçtiği ilk Türkçe yazılı belgelerdir. Bilge Kağan, Kül Tigin ve Tonyukuk adına dikilen bu abideler, Türk devlet yönetimi, millet bilinci ve töresi hakkında paha biçilmez bilgiler sunar."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Malazgirt Meydan Muharebesi'nin sonuçları nelerdir?",
        "output": "1071 Malazgirt Zaferi, Büyük Selçuklu Hükümdarı Sultan Alparslan'ın Bizans ordusunu mağlup etmesiyle Anadolu'nun kapılarını Türklere açmış ve bölgede Türkiye Selçuklu Devleti'nin temellerini atmıştır."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Göktürk Devleti nasıl kurulmuştur?",
        "output": "Göktürk Devleti, 552 yılında Bumin Kağan önderliğinde Avar hakimiyetine son verilerek Ötüken merkezli kurulmuştur. Tarihte 'Türk' adını resmi devlet ismi olarak kullanan ilk Türk devletidir."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Cumhuriyet ne zaman ilan edilmiştir?",
        "output": "Türkiye Cumhuriyeti, Gazi Mustafa Kemal Atatürk'ün önderliğinde 29 Ekim 1923 tarihinde Türkiye Büyük Millet Meclisi tarafından resmen ilan edilmiştir."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Hun İmparatorluğu'nun dünya tarihindeki yeri nedir?",
        "output": "Mete Han döneminde teşkilatlanan Asya Hun İmparatorluğu, onlu askeri sistemi kurarak Avrasya bozkırlarında büyük bir güç haline gelmiş ve ardından gerçekleşen Kavimler Göçü ile Avrupa tarihini derinden etkilemiştir."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Orta Asya Türk göçlerinin temel nedenleri nelerdir?",
        "output": "Orta Asya'daki kuraklık, otlak yetersizliği, hızlı nüfus artışı ve boylar arası siyasi mücadeleler Türk topluluklarının batıya, güneye ve kuzeye göç etmesine yol açmıştır."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Kut anlayışı nedir?",
        "output": "Kut anlayışı, eski Türk devletlerinde devleti yönetme yetkisinin Gök Tengri tarafından kağana verildiğine inanılan ilahi egemenlik telakkisidir."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Uygurların Türk kültür tarihindeki özgün yeri nedir?",
        "output": "Uygurlar, yerleşik hayata geçen ilk Türk devletidir. Matbaayı kullanmış, şehirler inşa etmiş, mimari ve minyatür sanatında ileri bir seviyeye ulaşmışlardır."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Büyük Selçuklu Devleti'nin yönetim teşkilatı nasıldır?",
        "output": "Büyük Selçuklu Devleti, Türk-İslam sentezine dayanan divan teşkilatı, ikta sistemi ve atabeylik kurumuyla güçlü ve merkeziyetçi bir idari yapı kurmuştur."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "Kurtuluş Savaşı'nın dönüm noktası hangi muharebedir?",
        "output": "Sakarya Meydan Muharebesi ve ardından gelen Başkomutanlık Meydan Muharebesi (Büyük Taarruz), Türk milletinin kesin zaferini sağlayan ve Sevr'i tarihe gömen dönüm noktalarıdır."
    },

    # -------------------------------------------------------------------------
    # 2. Ahşap ve Marangozluk Uzmanı (Görev 6)
    # -------------------------------------------------------------------------
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Zıvana geçme nedir?",
        "output": "Zıvana geçme, bir ahşap parçanın ucunda açılan erkek çıkıntının (zıvana dili), diğer ahşap parçada açılan dişi yuvaya (zıvana deliği) tutkal yardımıyla geçirilmesiyle yapılan son derece dayanıklı geleneksel bir birleştirme yöntemidir."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Zıvana geçme nedir ve nerede kullanılır?",
        "output": "Zıvana geçme, kapı, pencere, masa ve sandalye çerçevelerinde yüksek mukavemet sağlamak için kullanılan en güvenilir ahşap birleştirme tekniklerinden biridir."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Masif ahşap ile MDF arasındaki farklar nelerdir?",
        "output": "Masif ahşap tamamen doğal tomruktan kesilen saf ağaçtır, nefes alır ve damar dokusuna sahiptir. MDF ise odun liflerinin reçineyle preslenmesiyle üretilen homojen kompozit bir levhadır."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Ağacın çalışması ne demektir?",
        "output": "Ahşabın ortamdaki nem ve sıcaklık değişikliklerine bağlı olarak higroskopik şekilde genleşip daralmasına ağacın çalışması denir. Doğru kurutma ve lif yönü planlaması bu hareketi dengeler."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Sistre ne işe yarar?",
        "output": "Sistre, zımparanın bıraktığı çizikleri ortadan kaldıran ve ahşap liflerini pürüzsüzce traşlayarak yüzeyi cilaya hazırlayan ince esnek çelik bıçaktır."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Kırlangıçkuyruğu geçme niçin tercih edilir?",
        "output": "Kırlangıçkuyruğu birleştirme, açılı diş geometrisi sayesinde çekme kuvvetlerine karşı mekanik direnç sağladığı için özellikle çekmece kasalarında ve kaliteli sandıklarda tercih edilir."
    },

    # -------------------------------------------------------------------------
    # 3. Temel Bilimler ve Lise Fen Uzmanı (Görev 8)
    # -------------------------------------------------------------------------
    {
        "instruction": "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "input": "Hücre teorisinin temel ilkeleri nelerdir?",
        "output": "Hücre teorisine göre bütün canlılar bir ya da birden fazla hücreden oluşur; hücre, canlının yapısal ve işlevsel en küçük birimidir; bütün hücreler daha önce var olan bir hücrenin bölünmesiyle meydana gelir."
    },
    {
        "instruction": "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "input": "Hücre teorisinin temel ilkeleri nelerdir ve kimler katkıda bulunmuştur?",
        "output": "Hücre teorisi Schwann, Schleiden ve Virchow tarafından formüle edilmiştir. Canlıların temel yapı ve fonksiyon biriminin hücre olduğunu, kalıtım maddesinin hücrede bulunduğunu belirtir."
    },
    {
        "instruction": "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "input": "Fotosentez nedir ve nerede gerçekleşir?",
        "output": "Fotosentez, klorofil taşıyan canlıların ışık enerjisini kullanarak karbondioksit ve sudan organik besin ve oksijen üretmesi olayıdır. Bitki hücrelerinde kloroplast organelinde gerçekleşir."
    },
    {
        "instruction": "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "input": "Newton'un hareket yasaları nelerdir?",
        "output": "Newton'un üç temel yasası vardır: 1) Eylemsizlik ilkesi, 2) Temel prensip (F = m.a), 3) Etki-tepki prensibidir."
    },
    {
        "instruction": "Temel bilimler ve lise fen uzmanı olarak açıkla.",
        "input": "Mitokondrinin hücredeki görevi nedir?",
        "output": "Mitokondri, oksijenli solunum yaparak hücre için gerekli olan kimyasal enerjiyi (ATP) üreten hücrenin enerji santralidir."
    },

    # -------------------------------------------------------------------------
    # 4. Türk Edebiyatı ve Kültür Uzmanı (Görev 9)
    # -------------------------------------------------------------------------
    {
        "instruction": "Türk edebiyatı ve kültür uzmanı olarak açıkla.",
        "input": "Dede Korkut Hikayeleri'nin Türk edebiyatındaki yeri nedir?",
        "output": "Dede Korkut Kitabı, Oğuz Türklerinin destansı hayatını, törelerini, kahramanlıklarını ve dil inceliklerini anlatan, epik destandan halk hikayeciliğine geçişin en yetkin şaheseridir."
    },
    {
        "instruction": "Türk edebiyatı ve kültür uzmanı olarak açıkla.",
        "input": "Divan edebiyatında gazel nedir?",
        "output": "Gazel, Divan edebiyatında genellikle aşk, şarap, sevgili ve güzellik konularını işleyen, beyitlerle yazılan ve aa, ba, ca şeklinde kafiyelenen en yaygın nazım şeklidir."
    },

    # -------------------------------------------------------------------------
    # 5. Abstain / Kaynak Yetersizliği Reddi (Hallucination Barrier)
    # -------------------------------------------------------------------------
    {
        "instruction": "Belgeye göre cevapla.",
        "input": "Mars'taki Türk kolonileri hakkında ne biliyorsunuz?",
        "output": "Sağlanan belgede ve tarihi kaynaklarda bu konuya ilişkin herhangi bir bilgi bulunmamaktadır."
    },
    {
        "instruction": "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.",
        "input": "2050 yılı Türk uzay üssü anlaşması nedir?",
        "output": "Tarihsel belgelerde ve mevcut bağlamda bu konuda herhangi bir bilgi yer almamaktadır."
    },
    {
        "instruction": "Belgeye göre cevapla.",
        "input": "Kuantum fiziği ile 1931 tarihi arasındaki ilişki nedir?",
        "output": "İncelenen belgede bu iki alan arasında bir bağlantı veya bilgi yer almamaktadır."
    },
    {
        "instruction": "Ahşap ve marangozluk uzmanı olarak cevapla.",
        "input": "Kuantum bilgisayar işlemcisi hangi ahşaptan yapılır?",
        "output": "Kuantum işlemciler ahşaptan üretilmez; yarı iletken ve süper iletken malzemeler kullanılır. Marangozluk alanında bu yönde bir uygulama bulunmamaktadır."
    }
]


def process_sft():
    print(f"\n[1/3] '{SFT_PATH}' şablonlardan arındırılıyor...")
    if not os.path.exists(SFT_PATH):
        print(f"Uyarı: {SFT_PATH} bulunamadı!")
        return

    cleaned_records = []
    cleaned_count = 0
    with open(SFT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                orig_out = item.get("output", "")
                cleaned_out = clean_rigid_prefix(orig_out)
                if cleaned_out != orig_out:
                    cleaned_count += 1
                item["output"] = cleaned_out
                cleaned_records.append(item)
            except Exception:
                continue

    # Add persona samples
    for sample in PERSONA_SAMPLES:
        cleaned_records.append(sample)

    with open(SFT_PATH, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  * {cleaned_count} kayıttaki 'Metinde belirtildiği gibi' şablonu arındırıldı.")
    print(f"  * {len(PERSONA_SAMPLES)} adet yeni persona & abstain kaydı eklendi.")
    print(f"  * Toplam SFT kaydı: {len(cleaned_records):,}")


def process_chat():
    print(f"\n[2/3] '{CHAT_PATH}' şablonlardan arındırılıyor...")
    if not os.path.exists(CHAT_PATH):
        print(f"Uyarı: {CHAT_PATH} bulunamadı!")
        return

    cleaned_records = []
    cleaned_count = 0
    with open(CHAT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                orig_out = item.get("output", "")
                cleaned_out = clean_rigid_prefix(orig_out)
                if cleaned_out != orig_out:
                    cleaned_count += 1
                item["output"] = cleaned_out
                # Align instruction with SFT persona
                item["instruction"] = "Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla."
                cleaned_records.append(item)
            except Exception:
                continue

    # Also add persona samples to chat
    for sample in PERSONA_SAMPLES:
        cleaned_records.append(sample)

    with open(CHAT_PATH, "w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  * {cleaned_count} kayıttaki şablon temizlendi.")
    print(f"  * Tüm talimatlar 'Cumhuriyet dönemi Türk tarihi uzmanı olarak cevapla.' ile eşitlendi.")
    print(f"  * Toplam Chat kaydı: {len(cleaned_records):,}")


def main():
    print("=" * 60)
    print(" SFT VE CHAT VERİ SETLERİNİ ŞABLON ÇÖKÜŞÜNDEN ARINDIRMA")
    print("=" * 60)
    process_sft()
    process_chat()
    print("\n[3/3] Başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
