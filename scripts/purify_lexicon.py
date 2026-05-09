import os

def purify_lexicon(input_path, output_path):
    """
    Kristal-Vektörel mimarisi için leksikonu (roots.tsv) arındırır.
    Türetilmiş formları (morfotaktik tarafından halledilmesi gerekenler) siler.
    """
    
    # Korunması gereken (leksikalleşmiş veya atomik) istisnalar
    whitelist = {
        "ekmek", "çakmak", "yemek", "ahmak", "parmak", "basamak", "kaymak", 
        "tokmak", "ırmak", "kaynak", "topmak", "kasnak", "kaynak", "konak",
        "durak", "ayak", "kulak", "bardak", "yaprak", "toprak", "mızrak",
        "anne", "meme", "kademe", "kelime", "dövme", "yüzme", "yürütme", # Bazı isimleşmiş formlar
        "acı", "bacı", "keçeci", "hancı", # -CI ile biten ama atomik gibi duranlar
    }

    # Silinecek ek kalıpları (Suffix patterns to remove if at the end of a lemma)
    blacklist_suffixes = [
        "mak", "mek", # Mastarlar
        "leşme", "laşma", "leştirme", "laştırma", # Karmaşık türetimler
        "lenme", "lanma", "lendirme", "landırma",
        "ici", "ıcı", "ucu", "ücü", # Sıfat-fiil/İsim yapım
        "lik", "lık", "luk", "lük", # -lIk eki (Graph tarafından halledilecek)
        "li", "lı", "lu", "lü",    # -lI eki
        "siz", "sız", "suz", "süz", # -sIz eki
        "ci", "cı", "cu", "cü",     # -CI eki
    ]

    new_roots = {}
    purified_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        # Header'ı geç
        header = f_in.readline()
        
        for line in f_in:
            total_count += 1
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            lemma = parts[0]
            pos = parts[1]
            attr = parts[2] if len(parts) > 2 else "-"
            
            # Whitelist kontrolü
            if lemma in whitelist:
                new_roots[(lemma, pos)] = attr
                continue
            
            # Blacklist kontrolü
            should_remove = False
            for suffix in blacklist_suffixes:
                if lemma.endswith(suffix):
                    # Mastarlar için daha esnek olalım (ak-mak, ye-mek gibi kısa kökleri kaçırmayalım)
                    min_len = len(suffix) + 2
                    if suffix in ["mak", "mek"]:
                        min_len = len(suffix) + 1 # En az 1 harf kalsın (örn. ye-mek)
                    
                    if len(lemma) >= min_len:
                        # Özel Durum: Eğer mastar siliyorsak, kökü VERB olarak ekleyelim
                        if suffix in ["mak", "mek"]:
                            stem = lemma[:-len(suffix)]
                            # Kökü VERB olarak sakla (eğer zaten daha saf bir hali yoksa)
                            if (stem, "VERB") not in new_roots:
                                new_roots[(stem, "VERB")] = attr
                        
                        should_remove = True
                        break
            
            if should_remove:
                purified_count += 1
                continue 
            
            new_roots[(lemma, pos)] = attr

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(header)
        for (lemma, pos), attr in sorted(new_roots.items()):
            f_out.write(f"{lemma}\t{pos}\t{attr}\n")

    print(f"Leksikon Arındırma Tamamlandı:")
    print(f"  Toplam Kayıt: {total_count}")
    print(f"  Silinen (Türetilmiş): {purified_count}")
    print(f"  Yeni Leksikon: {total_count - purified_count} saf kök.")

if __name__ == "__main__":
    purify_lexicon('data/lexicon/roots.tsv', 'data/lexicon/roots_purified.tsv')
    # Yedek alıp yer değiştirme
    os.rename('data/lexicon/roots.tsv', 'data/lexicon/roots_original.tsv')
    os.rename('data/lexicon/roots_purified.tsv', 'data/lexicon/roots.tsv')
