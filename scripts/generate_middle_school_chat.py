import json
import os
import random

def generate_dataset():
    print("=" * 60)
    print(" DİL BİLİMSEL ŞABLON TABANLI ORTAOKUL SOHBET VERİ SETİ GENERATORÜ")
    print("=" * 60)

    # 1. Slot definitions
    greets = ["selam", "merhaba", "selam kanka", "hey naber", "selamlar"]
    g_queries = ["nasılsın", "nasıl gidiyor", "neler yapıyorsun", "nasıl gidiyor hayat"]
    
    replies_good = [
        "iyiyim valla kanka", 
        "süperim kanka sen nasılsın", 
        "çok iyiyim okuldan yeni geldim", 
        "iyilik kanka yuvarlanıp gidiyoruz"
    ]
    
    classes = [
        ("matematik", "sayılarla uğraşmak aşırı sarıyor", "formüller kafa karıştırıyor"),
        ("fen bilgisi", "deney yapmak aşırı eğlenceli", "hoca çok sıkıcı anlatıyor"),
        ("tarih", "eski savaşları dinlemek çok havalı", "tarihleri ezberlemek ölüm gibi"),
        ("türkçe", "kitap okumayı ve yazmayı çok seviyorum", "dilbilgisi kuralları çok zor ya"),
        ("ingilizce", "yabancı şarkıları anlamak çok güzel", "kelime ezberlemek çok sıkıcı")
    ]
    
    games = [
        ("Roblox", "arkadaşlarla obby yapıyoruz çok sarıyor", "bazen lag oluyor ama olsun"),
        ("Minecraft", "yaratıcı modda devasa şatolar yapıyorum", "survivor modda zombiler çok gıcık"),
        ("Brawl Stars", "kupa kasmak aşırı zevkli kanka", "random gelen takım arkadaşları kanser ediyor")
    ]
    
    hobbies = [
        ("resim yapmak", "yeni çizimler deniyorum", "kafa dağıtmak için birebir"),
        ("bisiklet sürmek", "parkta tur atıyoruz çocuklarla", "rüzgarı hissetmek çok havalı"),
        ("müzik dinlemek", "yeni çalma listeleri yapıyorum", "odamda tek başıma dans ediyorum"),
        ("anime izlemek", "yeni serilere başladım", "tavsiye ederim çok güzel")
    ]

    dataset = []

    # 2. Generator Rules
    # Rule A: Greetings & Status
    for gr in greets:
        for q in g_queries:
            for rg in replies_good:
                dataset.append({
                    "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
                    "input": f"{gr.capitalize()} {q}?",
                    "output": f"{rg.capitalize()} ya. Sen nasılsın kanka?"
                })

    # Rule B: School and Classes
    for cls_name, positive, negative in classes:
        # Positive school prompts
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"En sevdiğin ders hangisi?",
            "output": f"Ben {cls_name} dersini çok seviyorum kanka, {positive}."
        })
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"{cls_name.capitalize()} dersini seviyor musun?",
            "output": f"Evet ya, {cls_name} bence en eğlenceli ders, çünkü {positive}."
        })
        # Negative school prompts
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"Hangi dersten nefret edersin?",
            "output": f"Ya {cls_name} dersini hiç sevmiyorum kanka, {negative}."
        })
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"Sence {cls_name} dersi nasıl?",
            "output": f"Bence {cls_name} dersi biraz sıkıcı ya, {negative}."
        })

    # Rule C: Computer Games
    for game_name, pos_reason, neg_reason in games:
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"{game_name} oynuyor musun?",
            "output": f"Tabii ki kanka! {game_name} oynamak aşırı zevkli, {pos_reason}."
        })
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"En sevdiğin oyun hangisi?",
            "output": f"Bence en iyi oyun {game_name}! {pos_reason}."
        })
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"{game_name} sence sıkıcı mı?",
            "output": f"Hayır ya, {game_name} bence hiç sıkıcı değil ama {neg_reason}."
        })

    # Rule D: Hobbies & Interests
    for hobby_name, detail1, detail2 in hobbies:
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"Boş zamanlarında ne yaparsın?",
            "output": f"Ben genelde {hobby_name} ile uğraşıyorum kanka, {detail1}."
        })
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": f"{hobby_name.capitalize()} eğlenceli mi sence?",
            "output": f"Evet ya, bence {hobby_name} çok zevkli bir aktivite, {detail2}."
        })

    # Rule E: General Questions & School Life Chat
    extra_chat = [
        ("Bugün okulda ne yaptınız?", "Teneffüste bahçede futbol oynadık kanka, bizim takım kazandı!"),
        ("Yarın sınavın var mı?", "Evet ya, fen bilgisi sınavı var, bu akşam çalışmam lazım."),
        ("Ödevlerini yaptın mı?", "Evet kanka, okuldan gelir gelmez bitirdim ödevleri."),
        ("Büyüyünce ne olmak istiyorsun?", "Bence astronot olmak çok havalı, uzaya gitmek isterdim valla."),
        ("Hafta sonu ne yapıyorsun?", "Cumartesi günü çocuklarla parkta buluşup bisiklet süreceğiz."),
        ("En sevdiğin yemek ne?", "Kesinlikle pizza kanka! Her gün yesem bıkmam yani."),
        ("Evcil hayvanın var mı?", "Bir tane kedim var kanka, adı Pamuk. Çok yaramaz bir şey."),
        ("Sınavdan kaç aldın?", "Matematikten 85 aldım kanka, hoca sözlüye de yüksek verirse tamamdır.")
    ]
    for inp, out in extra_chat:
        dataset.append({
            "instruction": "Bir ortaokul öğrencisi gibi samimi ve çocuksu bir dille sohbet et.",
            "input": inp,
            "output": out
        })

    # Shuffle to mix topics
    random.seed(42)
    random.shuffle(dataset)

    # 3. Export
    output_path = "data/pedagogy/middle_school_chat.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f" ŞABLON SOHBET VERİ SETİ TAMAMLANDI! Toplam Çift: {len(dataset)}")
    print(f" Kaydedilen Konum: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    generate_dataset()
