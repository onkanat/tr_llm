# Skill 2: Morfotaktik Graf Yapısı (Morphotactics Graph) - Faz 2

Bu doküman, Türkçe Kristal-Vektörel Mimarisi'nin dilbilgisi kurallarını ve morfemlerin ardışık birleşim sırasını düzenleyen **Morfotaktik Durum Makinesini (Finite State Machine / Directed Acyclic Graph)** detaylandırır.

---

## 🏛️ Sınıf Yapısı ve Mimari Tasarım

Morfotaktik sistemi, her durumun (State) bir dilbilgisi aşamasını temsil ettiği yönlü bir graf (Directed Graph) şeklinde modellenmiştir (`src/compiler/morphotactics.py`).

### 1. `State` Sınıfı
Sistemdeki tüm geçerli morfotaktik durumları tanımlar:
*   **`START`**: Giriş durumu (`S_START`).
*   **Kök Durumları:** `VERB_ROOT`, `NOUN_ROOT`, `ADJ_ROOT`, `ADV_ROOT`.
*   **Fiil Çatı Durumları:** `VERB_VOICE_CAUSATIVE`, `VERB_VOICE_PASSIVE`.
*   **Yeterlilik Durumu:** `VERB_POTENTIAL` (Yeterlilik).
*   **Çekim Durumları (Fiil):** `VERB_NEGATION` (Olumsuzluk), `VERB_POST_TENSE` (Kip/Zaman), `VERB_POST_COPULA` (Ek-fiil/Birleşik Zaman), `VERB_POST_PERSON` (Şahıs).
*   **Çekim Durumları (İsim):** `NOUN_POST_PLURAL` (Çoğul), `NOUN_POST_POSSESSIVE` (İyelik), `NOUN_POST_POSSESSIVE_3` (3. Tekil İyelik), `NOUN_POST_CASE` (Hâl).
*   **Bitiş Durumu:** `STOP` (Terminal son durum).

### 2. `Transition` Sınıfı
Durumlar arasındaki kenarları (Edges) temsil eder:
*   **`to_state`** (`str`): Geçiş yapılacak hedef durum.
*   **`affix_id`** (`str`): Ekin sözlükteki anlamsal kimliği (örn. `TENSE_FUT`, `REL_ki`).
*   **`affix_template`** (`str`): Ekin fonetik şablonu (örn. `(y)AcAk`, `ki`).
*   **`attributes`** (`str`): Ekin sonraki ses olaylarını etkileyen niteliği (örn. `VOICING`).

### 3. `MorphotacticsGraph` Sınıfı
*   **`add_transition(from_state, to_state, affix_id, affix_template, attributes)`**: Geçiş kuralı ekler.
*   **`mark_terminal(state)`**: Durumu yasal bir bitiş durumu olarak işaretler.
*   **`get_valid_transitions(current_state)`**: O anki durumdan gidilebilecek tüm yasal geçişleri döner.
*   **`is_terminal(state)`**: Durumun kelime bitişi için uygun olup olmadığını doğrular.

---

## 🗺️ Varsayılan Graf Yapısı (`build_default_graph`)

### 1. Fiil Yolu (Verb Path)
*   **Çatı (Voice):** Fiil kökünden Ettirgen (`VOICE_CAUS_t` / `VOICE_CAUS_DIr`) veya Edilgen (`VOICE_PASS_Il`) çatı durumlarına geçilir.
*   **Yeterlilik:** Fiilden Yeterliliğe `(y)Abil` ile geçilip tekrar `VERB_ROOT` durumuna dönülebilir; eksi-yeterlilik durumunda `(y)AmA` şablonu ile doğrudan `VERB_NEGATION` durumuna dallanır.
*   **Zaman/Kip:** Fiil, Çatı ve Olumsuzluk durumlarından sonra 9 farklı zaman kipi (`TENSE_FUT`, `TENSE_PROG`, `TENSE_PAST`, `TENSE_AORIST` vb.) ile `VERB_POST_TENSE` durumuna geçilir.
*   **Birleşik Zaman (Copula) ve Şahıs:** Kip durumundan sonra ek-fiiller (`COPULA_PAST` `-DI`, `COPULA_EVIDENTIAL` `-mIş`, `COPULA_COND` `-sA`) ve ardından Şahıs ekleri (`PERSON_1SG`, `PERSON_2SG` vb.) eklenerek kelime sonlandırılır.

### 2. İsim Yolu (Noun Path)
*   İsim kökünden önce Çoğul (`PLURAL` = `lAr`), ardından İyelik (`POSS_1SG` = `(I)m`, `POSS_3SG` = `(s)I` vb.) ve son olarak Hâl ekleri (`CASE_LOC`, `CASE_ABL`, `CASE_ACC`, `CASE_DAT` vb.) eklenerek ilerlenir.
*   **3. Tekil İyelik Tamponu:** `POSS_3SG` ve `POSS_3PL` eklerinden sonra gelen hâl ekleri arasına zamir n'si girdiği için bu yol `NOUN_POST_POSSESSIVE_3` durumundan `nDA`, `nDAn`, `nI` gibi özel tamponlu hâl eklerine yönlendirilir.
*   **İlgi Eki (`-ki` / `REL_ki`):**
    İsim çekiminde bulunma/tamlayan hâlinden veya doğrudan kökten sıfat türetimini sağlayan `-ki` eki sisteme entegre edilmiştir:
    $$\text{NOUN\_POST\_CASE} \xrightarrow{\text{REL\_ki: } ki} \text{ADJ\_ROOT}$$
    $$\text{NOUN\_ROOT} \xrightarrow{\text{REL\_ki: } ki} \text{ADJ\_ROOT}$$
    (Örn: *ev-de-ki* $\rightarrow$ sıfatlaşarak yeni çekim veya türetimlere açılır).
*   **İsimlerde Ek-fiil:** İsim köklü veya isim çekimli durumdaki sözcükler `COPULA_AORIST` (`DIr`) veya geçmiş zaman copulalarıyla fiil çekim yoluna (`VERB_POST_COPULA`) bağlanabilir.

### 3. Türetim Yolları (Derivations)
*   **Fiilden İsim (Fiilimsi):** Fiil çekim aşamalarından isim köküne geçiş sağlar. İsim fiiller (`INF_mAk` = `mAk`, `INF_mA` = `mA`) ve sıfat fiiller (`PART_An` = `(y)An`, `PART_DIk` = `DIk`) bu geçişi gerçekleştirir.
*   **Zarf-Fiiller:** Fiil tabanını doğrudan `STOP` durumuna bağlar (örn. `GERUND_ArAk` = `(y)ArAk`, `GERUND_InCA` = `(y)IncA`).
*   **İsimden İsim/Sıfat:** İsim çekim yollarından tekrar isim/sıfat köküne döner (örn. `DERIV_lIk` = `lIk`, `DERIV_CI` = `CI`).
*   **İsimden Fiil:** İsim veya sıfat kökünü fiil köküne bağlar (`DERIV_lA` = `lA`, `DERIV_lAş` = `lAş` vb.).
