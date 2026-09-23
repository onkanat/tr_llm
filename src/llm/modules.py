"""YETENEK MODÜLLERİ (LoRA) — taban model BİR KEZ eğitilir, yetenekler modül olarak eklenir.

TASARIM SÖZLEŞMESİ (üç madde, hepsi ölçülebilir):

1. **Taban checkpoint'i DEĞİŞMEZ.** `LoRAKatmani` bir `nn.Linear`'dır: eski katmanın
   parametre adlarını (`weight`, `bias`) ve şeklini AYNEN taşır, üstüne yalnız
   `lora_A`/`lora_B` ekler. Bu yüzden `blocks.0.attn.q_proj.weight` anahtarı hem taban
   checkpoint'inde hem modüllü modelde AYNIdır ⇒ `chat_prompt.py`, `test_model.py`,
   `scripts/evaluate_*.py` gibi TÜM mevcut yükleyiciler taban checkpoint'i
   değiştirilmeden yüklemeye devam eder. Modül takmak için hiçbir değerlendirme betiği
   kırılmaz; modülü kullanan taraf `modul_yukle(...)` çağırır.

2. **Takıldığı anda çıktı TABANLA ÖZDEŞ.** `lora_B` SIFIR başlar ⇒ `delta = 0` ⇒
   `forward` tam olarak taban `Linear`'ın çıktısını verir. Bu, modülün "bedava pozitif
   kontrolü"dür: eşitlik bozulursa mekanizma yanlıştır, eğitim değil.

3. **Sessiz modül YOK.** Hedef deseni hiçbir katmana uymazsa `modul_ekle` DURUR; modül
   dosyası taban digest'ini taşır ve uyuşmazsa `modul_yukle` DURUR; modül durumu taban
   ağırlığı TAŞIYAMAZ (taşırsa zaten modül değil, gizli bir tam checkpoint'tir).

Katman seçimi: VARSAYILAN her blokta `attn.{q,k,v,out}_proj` + `mlp.0` + `mlp.2` = 6 katman ⇒
6 blok × 6 = **36 katman**. `ln*` (LayerNorm) dışarıda bırakılır.

`embedding`/`lm_head` artık **hedeflenebilir** (22 Eyl 2026). Varsayılan listeye girmezler ama
`--hedefler` ile eklenebilirler; ölçüldü ki **varsayılan 36 katman yeterli değil**:
`lm_head` hedefe eklenince ROUGE 0,0191 → **0,0696** (biçim kazanıldı, içerik değil).
`KristalEmbedding`'in KENDİSİ hedeflenemez — imza maskesi (kristal yapı) orada yaşar ve
sarılırsa baypas edilirdi; hedef içteki `embedding.embedding`'dir.

Parametre bütçesi: `r=16` için 36 katman × ort. 2.304 = **1.327.104** parametre ≈ %1,43
of 93.022.292 (ölçüldü).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MODUL_SURUM = 1

# Varsayılan hedef desenleri: isim SON EKİ olarak eşleşir ("blocks.0.attn.q_proj" ↔ "attn.q_proj").
VARSAYILAN_HEDEFLER: Tuple[str, ...] = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.out_proj",
    "mlp.0",
    "mlp.2",
)


@dataclass(frozen=True)
class ModulSpec:
    """Bir yetenek modülünün tarifi. Modül dosyası bu tarifi YANINDA taşır.

    Tarif dosyada saklanır çünkü modül ağırlıkları tek başına anlamsızdır: aynı ağırlıklar
    farklı `r`/`hedefler` ile farklı şekillere oturur. Yükleme, tarifi dosyadan okur.
    """

    ad: str
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    hedefler: Tuple[str, ...] = VARSAYILAN_HEDEFLER

    def olcek(self) -> float:
        """LoRA ölçeği (alpha/r). Sabit: tarifin parçası, öğrenilmez."""
        if self.r <= 0:
            raise ValueError(f"ModulSpec.r pozitif olmali: {self.r}")
        return self.alpha / self.r

    def sozluk(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hedefler"] = list(self.hedefler)
        d["olcek"] = self.olcek()
        return d

    @staticmethod
    def sozlukten(d: Dict[str, Any]) -> "ModulSpec":
        return ModulSpec(
            ad=str(d["ad"]),
            r=int(d["r"]),
            alpha=int(d["alpha"]),
            dropout=float(d.get("dropout", 0.0)),
            hedefler=tuple(d.get("hedefler", VARSAYILAN_HEDEFLER)),
        )


class LoRAKatmani(nn.Linear):
    """Taban `nn.Linear`'ın yerine geçen düşük-rütbeli delta katmanı.

    `forward(x) = W x + b + olcek * (x A^T B^T)`; `B` sıfır başlar ⇒ ikinci terim TAM
    sıfırdır ve çıktı tabanla bit-özdeş olur.
    """

    def __init__(self, temel: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        if isinstance(temel, LoRAKatmani):
            raise TypeError("LoRAKatmani LoRAKatmani'na sarilamaz (idempotent takma yok)")
        if not isinstance(temel, nn.Linear):
            raise TypeError(f"temel nn.Linear olmali, alindi: {type(temel).__name__}")
        if r <= 0:
            raise ValueError(f"r pozitif olmali: {r}")
        # CİHAZ/TİP DEVRALINIR (ölçüldü, 21 Eyl 2026): `nn.Linear` yeni parametreleri
        # VARSAYILAN olarak CPU/fp32 doğurur. Model zaten MPS'te ise takılan katman CPU'da
        # kalır ve ilk forward `RuntimeError: Tensor for argument weight is on cpu but
        # expected on mps` ile patlar. Eğitim koşumunda sıra (önce tak, sonra `.to(device)`)
        # bunu GİZLİYORDU; gerçek değerlendirme yolu ise tabanı yükleyip cihaza taşır ve
        # SONRA modülü yükler ⇒ kusur orada zorunlu olarak ateşler. Bu yüzden devralma
        # burada, İNŞA GEREĞİ yapılır: sıra ne olursa olsun katman tabanla aynı cihazda olur.
        super().__init__(temel.in_features, temel.out_features, bias=temel.bias is not None,
                         device=temel.weight.device, dtype=temel.weight.dtype)
        # AGIRLIK DEVRALINIR (kopyalanir): parametre ADLARI ve sekilleri tabanla ayni kalir.
        with torch.no_grad():
            self.weight.copy_(temel.weight)
            if self.bias is not None and temel.bias is not None:
                self.bias.copy_(temel.bias)

        self.r = int(r)
        self.alpha = int(alpha)
        self.olcek = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        _d, _t = temel.weight.device, temel.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, temel.in_features, device=_d, dtype=_t))
        self.lora_B = nn.Parameter(torch.zeros(temel.out_features, r, device=_d, dtype=_t))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B KASITLI olarak sifir: takma aninda delta = 0 (sozlesme maddesi 2).

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Yalnız modülün ürettiği katkı (taban çıktısı hariç). Sıfır-kontrolünde 0 olmalı."""
        h = self.lora_dropout(x) @ self.lora_A.t()
        return (h @ self.lora_B.t()) * self.olcek

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias) + self.delta(x)


class LoRAEmbedding(nn.Embedding):
    """Taban `nn.Embedding`'in yerine geçen düşük-rütbeli delta katmanı.

    `E = W + olcek * (B @ A)`; `B` sıfır başlar ⇒ delta TAM sıfır ve çıktı tabanla
    bit-özdeş olur (sözleşme maddesi 2, `LoRAKatmani` ile aynı güvence).

    NEDEN AYRI SINIF (ölçüldü, 22 Eyl 2026): `modul_ekle` yalnız `nn.Linear` sarıyordu, ve
    `embedding` hedefi **sessizce düşüyordu** — başka hedefler eşleştiği için "hiçbir katman
    uymadı" kapısı ateşlenmiyordu ⇒ koşum, ölçülmek istenen değişkeni HİÇ içermeden
    tamamlanıyordu ve sonuç "değişken etkisiz" diye **sahte bir null** olurdu.

    `KristalEmbedding`'in KENDİSİ sarılmaz: imza maskesi (kristal yapı) onun `forward`'ında
    yaşar ve sarılsaydı **baypas edilirdi**. Hedef, içteki `embedding.embedding`'dir ⇒ mask
    delta'nın ÜSTÜNDE, tabandaki yerde uygulanmaya devam eder.
    """

    def __init__(self, temel: nn.Embedding, r: int, alpha: int, dropout: float = 0.0):
        if isinstance(temel, LoRAEmbedding):
            raise TypeError("LoRAEmbedding LoRAEmbedding'e sarilamaz (idempotent takma yok)")
        if not isinstance(temel, nn.Embedding):
            raise TypeError(f"temel nn.Embedding olmali, alindi: {type(temel).__name__}")
        if r <= 0:
            raise ValueError(f"r pozitif olmali: {r}")
        # CİHAZ/TİP DEVRALINIR — `LoRAKatmani`'da ölçülen gerekçenin aynısı (MPS'te CPU
        # doğan katman ilk forward'da patlar; sıra ne olursa olsun aynı cihazda olmalı).
        super().__init__(temel.num_embeddings, temel.embedding_dim,
                         padding_idx=temel.padding_idx, max_norm=temel.max_norm,
                         norm_type=temel.norm_type, scale_grad_by_freq=temel.scale_grad_by_freq,
                         sparse=temel.sparse, _weight=temel.weight.detach().clone())

        self.r = int(r)
        self.alpha = int(alpha)
        self.olcek = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        _d, _t = temel.weight.device, temel.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, temel.num_embeddings, device=_d, dtype=_t))
        self.lora_B = nn.Parameter(torch.zeros(temel.embedding_dim, r, device=_d, dtype=_t))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B KASITLI olarak sifir: takma aninda delta = 0.

    @property
    def in_features(self) -> int:
        """`nn.Linear` adları: parametre formülü ve kayıt yolu iki sınıfta AYNI kalsın."""
        return int(self.num_embeddings)

    @property
    def out_features(self) -> int:
        return int(self.embedding_dim)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """Yalnız modülün ürettiği katkı (taban çıktısı hariç). Sıfır-kontrolünde 0 olmalı."""
        ax = nn.functional.embedding(x, self.lora_A.t().contiguous())
        return (self.lora_dropout(ax) @ self.lora_B.t()) * self.olcek

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        taban = nn.functional.embedding(x, self.weight, self.padding_idx, self.max_norm,
                                        self.norm_type, self.scale_grad_by_freq, self.sparse)
        return taban + self.delta(x)


LORA_TIPLERI: Tuple[type, ...] = (LoRAKatmani, LoRAEmbedding)


def _hedef_mi(ad: str, hedefler: Sequence[str]) -> bool:
    return any(ad == h or ad.endswith("." + h) for h in hedefler)


def modul_ekle(model: nn.Module, spec: ModulSpec) -> List[str]:
    """`spec.hedefler`e uyan her `nn.Linear`/`nn.Embedding`'i LoRA ile değiştirir.

    Dönen liste takılan katmanların TAM adlarıdır. İKİ KAPI, ikisi de fail-closed:

      · Hiçbir katman uymazsa DURUR. Sessizce boş liste dönmek, "modül eğitildi ama
        0 katman takılıydı" faciasını üretirdi (kayıp düşmez, kimse fark etmez).
      · **Her hedef** en az bir katmana uymalıdır, yoksa DURUR. Ölçüldü (22 Eyl 2026):
        `embedding` hedefi `nn.Linear` olmadığı için sessizce düşüyordu; diğerleri
        eşleştiği için ilk kapı ateşlenmiyordu ⇒ koşum, ölçülmek istenen değişkeni HİÇ
        içermeden tamamlanıyordu ve sonuç "değişken etkisiz" diye SAHTE bir null olurdu.
    """
    hedefli: List[str] = []
    for ad, mod in model.named_modules():
        if isinstance(mod, LORA_TIPLERI):
            if _hedef_mi(ad, spec.hedefler):
                raise ValueError(f"'{ad}' katmanina modul ZATEN takili; cift takma delta'yi ikiler")
            continue
        if isinstance(mod, (nn.Linear, nn.Embedding)) and _hedef_mi(ad, spec.hedefler):
            hedefli.append(ad)

    if not hedefli:
        raise ValueError(
            f"Modul hedefleri HICBIR katmana uymadi: {list(spec.hedefler)}. "
            f"Modelde bulunan sarmalanabilir katmanlar: "
            f"{[a for a, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))][:8]}…"
        )

    # İKİNCİ KAPI — KISMİ uyum (ölçüldü, 22 Eyl 2026): `embedding` hedefi `nn.Linear`
    # olmadığı için sessizce düşüyordu; diğer hedefler eşleştiği için yukarıdaki kapı
    # ateşlenmiyordu ⇒ koşum, ölçülmek istenen DEĞİŞKENİ HİÇ İÇERMEDEN tamamlanıyor ve
    # sonuç "değişken etkisiz" diye SAHTE bir null olarak yazılıyordu.
    uyan = {h for h in spec.hedefler if any(_hedef_mi(ad, (h,)) for ad in hedefli)}
    uymayan = [h for h in spec.hedefler if h not in uyan]
    if uymayan:
        raise ValueError(
            f"Modul hedefleri KISMEN uydu: {uymayan} hicbir katmana uymadi "
            f"(uyanlar: {sorted(uyan)}). Sessizce yok saymak yerine DURUYORUM: o hedef "
            f"olculmeden kosum 'degisken etkisiz' diye SAHTE bir null uretirdi."
        )

    for ad in hedefli:
        ebeveyn_ad, _, son = ad.rpartition(".")
        ebeveyn = model.get_submodule(ebeveyn_ad) if ebeveyn_ad else model
        eski = ebeveyn[int(son)] if son.isdigit() else getattr(ebeveyn, son)
        yeni = (LoRAKatmani(eski, spec.r, spec.alpha, spec.dropout)
                if isinstance(eski, nn.Linear)
                else LoRAEmbedding(eski, spec.r, spec.alpha, spec.dropout))
        if son.isdigit():
            ebeveyn[int(son)] = yeni
        else:
            setattr(ebeveyn, son, yeni)
    return hedefli


def modul_katmanlari(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(ad, m) for ad, m in model.named_modules() if isinstance(m, LORA_TIPLERI)]


def tabani_dondur(model: nn.Module) -> int:
    """Taban parametrelerini dondurur, YALNIZ modül parametrelerini eğitilebilir bırakır.

    Döner: eğitilebilir **ELEMAN** sayısı (`numel` toplamı) — tensör SAYISI değil. Birim
    uyuşmazlığı gerçek bir tuzağa yol açtı (ölçüldü, T-0098 sonrası): tensör sayısı 24 iken
    eleman sayısı 9.216'dır ve `beklenen_parametre_sayisi` eleman döndürür; iki sayı
    karşılaştırılınca kapı yanlış yere kapanır. Tek birim: **eleman**.

    Çağrı bir modül takılmadan yapılırsa DURAR: "taban donuk" sanıp bütün modeli eğitmek,
    modül işinin sessizce iptalidir.
    """
    katmanlar = modul_katmanlari(model)
    if not katmanlar:
        raise RuntimeError("tabani_dondur: modelde hic modul yok (once modul_ekle cagrilmali)")

    for p in model.parameters():
        p.requires_grad_(False)
    for _, m in katmanlar:
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)

    kotu = [ad for ad, p in model.named_parameters()
            if p.requires_grad and ".lora_A" not in ad and ".lora_B" not in ad]
    if kotu:
        raise RuntimeError(f"taban parametresi egitilebilir kaldi (sozlesme ihlali): {kotu[:5]}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def egitilebilir_parametreler(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def beklenen_parametre_sayisi(model: nn.Module) -> int:
    """Takılı modüllerin ürettiği parametre sayısı (formül, ölçülmüş sayı değil).

    Her katman `r*(in_features + out_features)` parametre ekler. Eğitim koşumu bu değeri
    gerçek eğitilebilir sayıyla karşılaştırır: tutmazsa mekanizma ile niyet ayrışmıştır.
    """
    return sum(m.r * (m.in_features + m.out_features) for _, m in modul_katmanlari(model))


def modul_durumu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """YALNIZ modül tensörlerini döner (taban ağırlıkları hariç).

    Taban anahtarı sızarsa DURAR: modül dosyası tabanı taşırsa "5 MB'lık yetenek" iddiası
    yalandır ve dosya gizli bir tam checkpoint'e dönüşür.
    """
    durum: Dict[str, torch.Tensor] = {}
    for ad, t in model.state_dict().items():
        if ad.endswith(".lora_A") or ad.endswith(".lora_B"):
            durum[ad] = t
    if not durum:
        raise RuntimeError("Modul durumu BOS: hic lora_A/lora_B anahtari yok")
    return durum


def sha256_dosya(yol: str) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def modul_kaydet(
    model: nn.Module,
    spec: ModulSpec,
    yol: str,
    taban_yol: str,
    ek: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Modülü (taban digest'i + tarif + yalnız lora tensörleri) atomik olarak kaydeder."""
    os.makedirs(os.path.dirname(os.path.abspath(yol)), exist_ok=True)
    katmanlar = modul_katmanlari(model)
    if not katmanlar:
        raise RuntimeError("modul_kaydet: modelde hic modul yok")

    durum = modul_durumu(model)
    taban = taban_yol if os.path.exists(taban_yol) else None
    if taban is None:
        raise RuntimeError(f"modul_kaydet: taban checkpoint yok: '{taban_yol}' (digest capalanamaz)")

    meta: Dict[str, Any] = {
        "surum": MODUL_SURUM,
        "spec": spec.sozluk(),
        "base_sha256": sha256_dosya(taban_yol),
        "base_yol": os.path.basename(taban_yol),
        "taban_parametre": sum(p.numel() for p in model.parameters()),
        "modul_parametre": sum(t.numel() for t in durum.values()),
        "katman_sayisi": len(katmanlar),
        "katmanlar": [ad for ad, _ in katmanlar],
        "vocab_size": int(model.lm_head.out_features),
    }
    if ek:
        meta.update(ek)

    gecici = yol + ".tmp"
    torch.save({"meta": meta, "modul": durum}, gecici)
    os.replace(gecici, yol)
    return meta


def modul_yukle(
    model: nn.Module,
    yol: str,
    taban_yol: str,
    dogrula_digest: bool = True,
) -> Dict[str, Any]:
    """Modülü `model`e takar ve ağırlıklarını yükler.

    SIRA ÖNEMLİ: taban checkpoint'i modele yüklendikten SONRA çağrılmalıdır; çünkü takma
    anındaki `weight` kopyası o anda modelde ne varsa onu alır.

    İki kapı: (a) taban digest'i modülün çapasıyla uyuşmalı, (b) modülün her anahtarı
    modelde karşılık bulmalı. İkisi de sessiz geçilemez.
    """
    paket = torch.load(yol, map_location="cpu", weights_only=False)
    if not isinstance(paket, dict) or "meta" not in paket or "modul" not in paket:
        raise RuntimeError(f"Modul dosyasi bozuk: '{yol}' (meta/modul alani yok)")
    meta = paket["meta"]
    durum = paket["modul"]

    if dogrula_digest:
        beklenen = meta.get("base_sha256")
        if not beklenen:
            raise RuntimeError(f"Modul dosyasinda base_sha256 YOK: '{yol}' => capa dogrulanamaz")
        gercek = sha256_dosya(taban_yol)
        if gercek != beklenen:
            raise RuntimeError(
                f"TABAN UYUSMUYOR: modul '{yol}' base_sha256={beklenen[:16]}… bekliyor, "
                f"ama '{taban_yol}' digest'i {gercek[:16]}…. Modul BASKA bir tabana aittir."
            )

    spec = ModulSpec.sozlukten(meta["spec"])
    beklenen_vocab = int(meta.get("vocab_size", -1))
    gercek_vocab = int(model.lm_head.out_features)
    if beklenen_vocab != gercek_vocab:
        raise RuntimeError(
            f"SOZLUK BOYUTU UYUSMUYOR: modul vocab_size={beklenen_vocab}, model={gercek_vocab}"
        )

    takilan = modul_ekle(model, spec)
    kullanilmayan = model.load_state_dict(durum, strict=False)
    if kullanilmayan.unexpected_keys:
        raise RuntimeError(
            f"Modul anahtarlari modelde KARSILIK BULMADI: {kullanilmayan.unexpected_keys[:5]} "
            f"(takilan katmanlar: {len(takilan)})"
        )
    return meta


def modul_ozeti(yol: str) -> Dict[str, Any]:
    """Modül dosyasının meta'sını (ağırlıkları yüklemeden) okur."""
    paket = torch.load(yol, map_location="cpu", weights_only=False)
    return dict(paket["meta"])
