(Faz 2)
Sorumluluk: Gramer kural sistemini (G) ve ardışık birleşim operatörünü (Σ) yönlü bir graf (State Machine) olarak modellemek
.
Bileşenler:
Durumlar: S_START, S_VERB_ROOT, S_VERB_POST_TENSE, S_STOP vb. terminal ve ara düğümler
.
Geçişler (Edges): Hangi durumdan hangi ekin (TENSE_FUT vb.) eklenebileceğini gösteren kurallar
.
Kontrol: Verilen bir dizilimin (Örn: ROOT_VERB → TENSE → PERSON) yasal olup olmadığını is_terminal() ve get_valid_transitions() ile doğrular
.
