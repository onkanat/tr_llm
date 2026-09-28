# İLAN-2 — T-0157 K5 pyflakes-clause netleştirmesi (koşum-ÖNCESİ revizyon)

**Damga (BETİKTEN):** `2026-09-28T07:51:50Z`
**Meşruiyet:** İLAN-1 (`t0157_gateway_okunabilirlik_ilan_2026-09-28.md`, damga
`2026-09-28T07:48:56Z`) K5 maddesinde "bağımsız pyflakes TEMİZ" ifadesi
**formülasyon-eksik** yazılmış: `src/rag/epistemic_agent.py`'de
`clean_query_tags` atandı-ama-kullanılmadı bulgusu **onarım-ÖNCESİ mevcuttur**
(`git show HEAD:src/rag/epistemic_agent.py` satır 292'de teyitli; T-0155
kapanış-notunda da "kapsam-dışı" kayıtlı). Koşum HENÜZ BAŞLAMADI —
İLAN-formülasyon-kusuru koşum-öncesi revizyon kalıbı (T-0151/T-0149).

## Tek madde — K5'in operasyonel tanımı (kapılar DEĞİŞMEZ; sertleşir)

K5 pyflakes kapısı **beyaz-listeli fail-closed** olarak ölçülür:

- **Beyaz-liste (İLAN-2 ile sabit):** yalnızca
  `local variable 'clean_query_tags' is assigned to but never used`
  (onarım-öncesi mevcut; bu turun onarım-yüzeyi DEĞİL).
- pyflakes çıktısında **beyaz-liste dışında HERHANGİ bir bulgu** → K5
  **DÜŞER** (yeni bulgu sessiz geçmez — İLAN-1'in "TEMİZ" ifadesinden daha
  SERT kapı).
- K5'in diğer bileşenleri (py_compile + AST sabit/bastırma/capitalize
  kontrolleri) İLAN-1'de sabitlendikleri gibidir.

Diğer tüm kapılar (K1, K2, K3, K4, K6), hedef-değerler, DOKUNULMAZLAR ve
ölçüm-cihazı beyanları İLAN-1'de **SABİT** kalır — yumuşatma YOK.