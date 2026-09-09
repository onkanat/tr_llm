#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
from typing import List, Dict

WOOD_SPECIES = [
    ("meşe", "Sert, yoğun lifli, suya ve aşınmaya karşı oldukça dirençli bir ağaçtır; mobilya ve parkede sıkça tercih edilir."),
    ("ceviz", "Zengin damar desenine ve kahverengi tonlara sahip, işlemesi kolay ve çok kaliteli lüks mobilya ağacıdır."),
    ("kayın", "Oldukça sert ve sıkı liflidir; buharlama yöntemiyle bükülerek Thonet tarzı sandalye yapımında idealdir."),
    ("çam", "Yumuşak dokulu, reçineli ve kolay işlenebilen, yapı kerestesi ve ekonomik doğramalarda kullanılan bir iğne yapraklıdır."),
    ("tik", "Doğal silika ve yağ barındırdığından suya, güneşe ve çürümeye aşırı dayanıklıdır; dış mekan ve tekne güvertesinde rakipsizdir."),
    ("iroko", "Afrika kökenli, sert ve dayanıklı yapısıyla tik ağacının en popüler ekonomik alternatifidir."),
    ("dişbudak", "Esnekliği ve darbe emme gücü çok yüksektir; balta, çekiç sapları ve spor aletleri yapımında ilk tercihtir."),
    ("gürgen", "Aşırı sert ve tok yapılıdır; marangoz mengene çeneleri ve ağır yüke maruz kalan parçalarda kullanılır."),
    ("kiraz", "Zamanla güneş ışığıyla koyulaşan kızıl tonlu, pürüzsüz yüzey veren asil bir mobilya ağacıdır."),
    ("akçaağaç", "Çok açık renkli, yoğun ve serttir; kesme tahtaları, müzik aletleri ve tezgah üstlerinde tercih edilir.")
]

TOOLS = [
    ("rende", "Ahşabın yüzeyini düzeltmek, inceltmek ve pürüzsüz bir katman açmak için kullanılan kesici alettir."),
    ("iskarpela", "Ahşapta delik, kanal veya zıvana yuvaları açmak ve ahşabı yontmak için kullanılan çelik keski aletidir."),
    ("kırlangıç testere", "Hassas geçme ve birleştirme hatlarını kesmek için sırtı takviyeli ince dişli el testeresidir."),
    ("freze", "Ahşap kenarlarına profil vermek, kanal açmak veya şablon kopyalamak için yüksek devirli dönen kesici bıçaklı makinedir."),
    ("işkence", "Yapıştırma veya montaj sırasında parçaları kuruyana kadar sabit ve sıkı tutmaya yarayan sıkıştırma aletidir."),
    ("gönye", "Ahşap parçaların 90 derece dik açılarını ve paralelliklerini kontrol etmeye yarayan ölçüm aletidir."),
    ("kumpas", "Ahşap kalınlığını, zıvana genişliğini veya delik derinliğini milimetrenin onda biri hassasiyetle ölçer."),
    ("planya", "Kaba biçilmiş kerestenin yüzeyini ve cumbasını referans düzlemine getirmek için kullanılan tezgah makinesidir."),
    ("kalınlık makinesi", "Planyalanmış kerestenin diğer yüzünü istenen tam milimetrik kalınlığa getiren döner bıçaklı makinedir."),
    ("raspa", "Lif kıvrımlı veya ters damarlı zorlu ahşaplarda rende izlerini kazıyarak cam gibi yüzey bırakan esnek çelik plakadır.")
]

JOINERY = [
    ("kırlangıç kuyruğu", "Çekme kuvvetine karşı mekanik olarak kilitlenen, özellikle çekmece ve sandık köşelerinde hem estetik hem aşırı mukavim birleşimdir."),
    ("zıvana ve geçme", "Bir parçanın ucundaki erkek çıkıntının diğer parçadaki dişi yuvaya oturmasıyla yapılan, masa ve sandalye iskeletlerinin temel birleşimidir."),
    ("kavela birleştirme", "Silindirik ahşap pimlerin tutkallanarak karşılıklı açılan deliklere yerleştirilmesiyle yapılan pratik ve gizli birleşimdir."),
    ("lamba zıvana", "Tahtaların kenarlarına erkek-dişi kanallar açılarak yan yana geçirilmesi yöntemidir; döşeme ve tavan kaplamalarında kullanılır."),
    ("gönye burun", "İki parçanın 45 derecelik açıyla kesilerek 90 derecelik köşe oluşturduğu, lif bitimlerini gizleyen estetik birleşimdir."),
    ("kelebek kama", "Masif masa tablalarındaki doğal çatlakların daha fazla açılmasını önlemek için çatlağın üzerine kilit şeklinde kakılan ahşap parçadır.")
]

FINISHES = [
    ("gomalak cila", "Gomalak böceğinin reçinesinden alkolle eritilerek yapılan, ahşaba derin ve sıcak bir parlaklık kazandıran geleneksel Fransız cilasıdır."),
    ("keten tohumu yağı", "Ahşabın gözeneklerine nüfuz ederek içten koruma sağlayan, damar desenlerini canlandıran doğal kuruyan bir yağdır."),
    ("tung yağı", "Çin kökenli fındıktan elde edilen, keten tohumuna göre suya daha dayanıklı ve mat-yarı mat doğal koruyucu yağdır."),
    ("zımparalama kuralı", "Zımparalama işlemine kalın kumla (örn. 80-120) başlanmalı, lif yönünde yapılmalı ve aşama aşama ince kuma (180, 240, 320) geçilmelidir."),
    ("dolgu verniği", "Ahşabın açık gözeneklerini doldurarak son kat cila için pürüzsüz ve düzgün bir zemin hazırlayan astar verniktir.")
]

def generate_carpenter_dataset(target_count: int = 3000):
    print("=" * 60)
    print(" KRİSTAL-VEKTÖREL MİMARİSİ: MARANGOZLUK (CARPENTER AI) VERİ ÜRETİCİ")
    print("=" * 60)

    dataset: List[Dict[str, str]] = []

    # 1. Species Questions
    for name, desc in WOOD_SPECIES:
        dataset.append({
            "instruction": f"Marangozlukta {name} ağacının özellikleri ve kullanım alanları nelerdir?",
            "input": "",
            "output": f"{name.capitalize()} ağacı: {desc}"
        })
        dataset.append({
            "instruction": "Hangi ağaç türü bu iş için uygundur?",
            "input": f"{desc.split(';')[0]}",
            "output": f"Bu gereksinim için {name} ağacı son derece uygundur. {desc}"
        })

    # 2. Tool Questions
    for name, desc in TOOLS:
        dataset.append({
            "instruction": f"Marangozlukta {name} ne işe yarar ve nasıl kullanılır?",
            "input": "",
            "output": f"{name.capitalize()}: {desc}"
        })
        dataset.append({
            "instruction": "Bu işlem için hangi marangozluk aleti kullanılmalıdır?",
            "input": f"{desc}",
            "output": f"Bu işlem için '{name}' kullanılmalıdır."
        })

    # 3. Joinery Questions
    for name, desc in JOINERY:
        dataset.append({
            "instruction": f"Marangozlukta {name} birleştirmesi nasıl yapılır ve nerede tercih edilir?",
            "input": "",
            "output": f"{name.capitalize()} birleşimi: {desc}"
        })

    # 4. Finishing Questions
    for name, desc in FINISHES:
        dataset.append({
            "instruction": f"Ahşap yüzey işlemlerinde {name} hakkında bilgi ver.",
            "input": "",
            "output": f"{name.capitalize()}: {desc}"
        })

    # 5. Combinations & Procedural Scenarios
    scenarios = [
        ("Masa tablası yaparken ahşabın genleşmesini nasıl önleriz?",
         "Masif ahşap ortam nemine göre enine genişler ve daralır. Tablanın çatlamasını önlemek için ayaklara sabit vidalanmamalı, Z-klips veya kanallı bağlantı demirleri ile hareket payı bırakılmalıdır."),
        ("Zımparalama yaparken neden lif yönünde zımpara yapılmalıdır?",
         "Ahşabın liflerine dik veya dairesel zımpara yapıldığında lifler koparak derin çizikler oluşturur ve cila sürüldüğünde bu çizikler çok belirgin hale gelir. Lif yönünde zımparalama pürüzsüz bir yüzey sağlar."),
        ("Ahşap tutkallamada işkence ne kadar süre sıkılı kalmalıdır?",
         "PVA bazlı standart marangoz tutkallarında işkenceler en az 30-60 dakika sıkılı kalmalı, tam kürlenme ve nihai yük dayanımı için 24 saat beklenmelidir."),
        ("Bıçak bileme açısı iskarpela ve rendelerde kaç derece olmalıdır?",
         "Genel marangozluk iskarpela ve rende tığlarında birincil taşlama açısı 25 derece, mikro bileme (kılavuz) açısı ise 30 derece olmalıdır."),
        ("Ahşapta budaklı kısımlar nasıl işlenmelidir?",
         "Budaklar ana liften çok daha serttir ve lif akışını saptırır. Bu bölgelerde kesici aletler çok keskin olmalı, talaş derinliği azaltılmalı ve rendeleme budağın çevresindeki lif akışı yönünde yapılmalıdır.")
    ]

    for q, a in scenarios:
        dataset.append({
            "instruction": q,
            "input": "",
            "output": a
        })

    # Multiply with stylistic variations to reach the desired target count
    templates = [
        ("Usta marangoz olarak açıkla: {q}", "{a}"),
        ("Ahşap atölyesinde şu durumla karşılaşıldı: {q}", "Teknik Çözüm: {a}"),
        ("Marangozluk tekniği sorusu: {q}", "{a}")
    ]

    base_items = list(dataset)
    for q_tpl, a_tpl in templates:
        for item in base_items:
            new_q = q_tpl.format(q=item["instruction"])
            new_a = a_tpl.format(a=item["output"])
            dataset.append({
                "instruction": new_q,
                "input": item.get("input", ""),
                "output": new_a
            })
            if len(dataset) >= target_count:
                break
        if len(dataset) >= target_count:
            break

    random.seed(42)
    random.shuffle(dataset)

    out_dir = 'data/pedagogy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'carpenter_specialization_dataset.jsonl')

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print(" MARANGOZLUK ALAN UZMANLIĞI VERİ SETİ TAMAMLANDI!")
    print("=" * 60)
    print(f"Toplam Uzmanlık Örnek Sayısı: {len(dataset):,}")
    print(f"Kayıt Yolu: {out_path}")

if __name__ == '__main__':
    generate_carpenter_dataset()
