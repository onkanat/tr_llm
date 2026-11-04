#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.1 — QDRANT_URL PARAMETRELEŞMESİ (TDD: RED→GREEN)
====================================================
Sözleşme (ana plan P2.1):
1. Varsayılan KORUNUR: URL arg'ı da QDRANT_URL env'i de yokken gateway
   create_default localhost:6333'a bağlanır (birebir eski davranış);
   VectorMemory host'suz çağrı :memory: kalır.
2. QDRANT_URL env'i VEYA açık url arg'ı verilince uzak adrese
   (ör. 192.168.1.9:6333) bağlanır.
3. URL çözümü BAĞLANTI KURULMADAN ÖNCE sınanabilir — mock/monkeypatch
   ile (canlı sunucu gerekmez; K6 canlı-dokunulmazlık korunur).
"""

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

VOCAB_PATH = "data/rebuild/vocab_anka_r1_33114.json"
LEXICON_PATH = "data/lexicon/roots.tsv"


# ---------------------------------------------------------------------------
# 1) parse_qdrant_url — saf URL→(host, port) çözümü (bağlantı YOK)
# ---------------------------------------------------------------------------

def test_parse_full_url_with_scheme():
    from src.rag.vector_memory import parse_qdrant_url
    assert parse_qdrant_url("http://192.168.1.9:6333") == ("192.168.1.9", 6333)


def test_parse_host_port_without_scheme():
    from src.rag.vector_memory import parse_qdrant_url
    assert parse_qdrant_url("192.168.1.9:6333") == ("192.168.1.9", 6333)


def test_parse_default_port_is_6333():
    from src.rag.vector_memory import parse_qdrant_url
    assert parse_qdrant_url("http://192.168.1.9") == ("192.168.1.9", 6333)
    assert parse_qdrant_url("192.168.1.9") == ("192.168.1.9", 6333)


def test_parse_strips_whitespace():
    from src.rag.vector_memory import parse_qdrant_url
    assert parse_qdrant_url("  http://192.168.1.9:6333 \n") == ("192.168.1.9", 6333)


def test_parse_invalid_raises_value_error(monkeypatch):
    # Standart modül import'u — kod çalıştırma (eval) DEĞİL
    from src.rag.vector_memory import parse_qdrant_url
    monkeypatch.delenv("QDRANT_URL", raising=False)
    for bad in ["", "   ", "http://", ":6333", "192.168.1.9:notanumber"]:
        with pytest.raises(ValueError):
            parse_qdrant_url(bad)


# ---------------------------------------------------------------------------
# 2) resolve_qdrant_url — açık arg > QDRANT_URL env; ikisi de yoksa None
# ---------------------------------------------------------------------------

def test_resolve_env_used_when_no_arg(monkeypatch):
    from src.rag.vector_memory import resolve_qdrant_url
    monkeypatch.setenv("QDRANT_URL", "http://192.168.1.9:6333")
    assert resolve_qdrant_url() == ("192.168.1.9", 6333)


def test_resolve_explicit_arg_beats_env(monkeypatch):
    from src.rag.vector_memory import resolve_qdrant_url
    monkeypatch.setenv("QDRANT_URL", "http://10.0.0.1:6333")
    assert resolve_qdrant_url("http://192.168.1.9:6334") == ("192.168.1.9", 6334)


def test_resolve_none_when_unset_and_no_arg(monkeypatch):
    from src.rag.vector_memory import resolve_qdrant_url
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert resolve_qdrant_url() is None


# ---------------------------------------------------------------------------
# 3) VectorMemory url arg'ı — sahte QdrantClient ile bağlantı-KAYDI
# ---------------------------------------------------------------------------

class _FakeQdrantClient:
    """Bağlantı kwargs'ını KAYDEDEN sahte istemci; koleksiyon yokmuş gibi davranır."""
    calls = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append({"args": args, "kwargs": kwargs})
        self._exists = False

    def get_collections(self):
        return ["fake"]

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, **kwargs):
        self._exists = True

    def count(self, collection_name):
        return SimpleNamespace(count=0)

    def close(self):
        pass


def test_vector_memory_url_param_connects_remote(monkeypatch):
    """url='http://192.168.1.9:6333' → QdrantClient(host='192.168.1.9', port=6333)."""
    import src.rag.vector_memory as vm_mod
    _FakeQdrantClient.calls = []
    monkeypatch.setattr(vm_mod, "QdrantClient", _FakeQdrantClient)
    vm = vm_mod.VectorMemory(
        collection_name="test_url_coll", vector_size=768,
        url="http://192.168.1.9:6333",
    )
    remote_calls = [c for c in _FakeQdrantClient.calls if "host" in c["kwargs"]]
    assert len(remote_calls) == 1, f"beklenen 1 remote bağlantı: {_FakeQdrantClient.calls}"
    assert remote_calls[0]["kwargs"]["host"] == "192.168.1.9"
    assert remote_calls[0]["kwargs"]["port"] == 6333
    assert vm.storage_type == "remote (192.168.1.9:6333)"
    vm.close()


def test_vector_memory_url_overrides_host_port(monkeypatch):
    """url verilirse host/port arg'larını EZER (tek doğru-adres sözleşmesi)."""
    import src.rag.vector_memory as vm_mod
    _FakeQdrantClient.calls = []
    monkeypatch.setattr(vm_mod, "QdrantClient", _FakeQdrantClient)
    vm = vm_mod.VectorMemory(
        collection_name="test_url_override_coll", vector_size=768,
        host="127.0.0.1", port=1, url="http://192.168.1.9:6333",
    )
    remote_calls = [c for c in _FakeQdrantClient.calls if "host" in c["kwargs"]]
    assert remote_calls[0]["kwargs"]["host"] == "192.168.1.9"
    assert remote_calls[0]["kwargs"]["port"] == 6333
    vm.close()


def test_vector_memory_env_not_read_implicitly(monkeypatch):
    """VectorMemory env'i AÇIK arg olmaksızın OKUMAZ: host'suz çağrı :memory: kalır
    (QDRANT_URL env'i set olsa bile) — eski in-memory varsayılanı bozulmaz;
    env-okuma YALNIZ gateway/başlatıcı katmanındadır."""
    monkeypatch.setenv("QDRANT_URL", "http://192.168.1.9:6333")
    from src.rag.vector_memory import VectorMemory
    vm = VectorMemory(collection_name="test_env_implicit_coll", vector_size=768)
    assert vm.is_in_memory is True
    assert vm.storage_type == "in-memory (:memory:)"
    vm.close()


def test_vector_memory_default_without_url_still_memory():
    """Varsayılan KORUNUR: url'siz/host'suz çağrı :memory: — mevcut test-ortam sözleşmesi."""
    from src.rag.vector_memory import VectorMemory
    vm = VectorMemory(collection_name="test_default_coll", vector_size=768)
    assert vm.is_in_memory is True
    vm.close()


def test_vector_memory_timeout_param(monkeypatch):
    """P2.1: uzak istemci zaman aşımı PARAMETRELİ (varsayılan 2.0 KORUNUR);
    ekim/başlatıcı LAN için daha uzun süre verebilir (ölçüldü: create_collection
    sunucuda başarılı olsa da yanıt 2.0s'yi aşabiliyor)."""
    import src.rag.vector_memory as vm_mod
    _FakeQdrantClient.calls = []
    monkeypatch.setattr(vm_mod, "QdrantClient", _FakeQdrantClient)

    vm1 = vm_mod.VectorMemory(
        collection_name="test_timeout_coll", vector_size=768,
        url="http://192.168.1.9:6333", timeout=30.0,
    )
    remote = [c for c in _FakeQdrantClient.calls if "host" in c["kwargs"]]
    assert remote[0]["kwargs"]["timeout"] == 30.0
    vm1.close()  # paylaşılan-istemci önbelleğinden çıkar (test izolasyonu)

    _FakeQdrantClient.calls = []
    vm2 = vm_mod.VectorMemory(
        collection_name="test_timeout_default_coll", vector_size=768,
        url="http://192.168.1.9:6333",
    )
    remote = [c for c in _FakeQdrantClient.calls if "host" in c["kwargs"]]
    assert remote[0]["kwargs"]["timeout"] == 2.0, "varsayılan 2.0 birebir korunmalı"
    vm2.close()


# ---------------------------------------------------------------------------
# 4) AgentGateway.create_default — QDRANT_URL çözümü (bağlantı öncesi kanıt)
# ---------------------------------------------------------------------------

def _make_fake_kristal_lm():
    class _FakeKristalLM:
        def __init__(self, **kwargs):
            pass

        def load_state_dict(self, sd, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        def to(self, dev):
            return self

        def eval(self):
            return self

    return _FakeKristalLM


def _run_create_default(monkeypatch, tmp_path, qdrant_url=None, env_url=None):
    """create_default'ı ağır-yükleri (model/router/istekci) dillere-dolare
    without bağlantı kurarak koşar; VectorMemory çağrı kwargs'ını kaydeder."""
    import src.gateway.agent_gateway as ag
    import scripts.train_step_demo as tsd
    import scripts.train_step_b1_5_rigorous as b15
    import src.llm.prompt_contract as pc

    vm_calls = []

    class _RecordingVM:
        def __init__(self, **kwargs):
            vm_calls.append(dict(kwargs))
            self.collection_name = kwargs.get("collection_name")

        def get_document_count(self):
            return 0

        def close(self):
            pass

    class _DummyAgent:
        def __init__(self, **kwargs):
            pass

    class _DummyCuriosity:
        def __init__(self, **kwargs):
            pass

    fake_model = tmp_path / "fake_model.pt"
    fake_model.write_bytes(b"x")
    fake_router = tmp_path / "fake_router.pt"
    fake_router.write_bytes(b"x")

    # Vocab satır sayısıyla sahte checkpoint'i UYUMLU yap (T-0087 kapısı)
    from src.llm.tokenizer import Vocabulary
    v = Vocabulary()
    v.load(VOCAB_PATH)
    fake_sd = {"embedding.embedding.weight": torch.zeros(len(v.stoi), 4)}

    monkeypatch.setattr(ag, "VectorMemory", _RecordingVM)
    monkeypatch.setattr(ag, "EpistemicCuriosityAgent", _DummyAgent)
    monkeypatch.setattr(ag, "CuriosityEngine", _DummyCuriosity)
    monkeypatch.setattr(tsd, "KristalLM", _make_fake_kristal_lm())
    monkeypatch.setattr(b15, "compute_sha256", lambda p: "fake-sha")
    monkeypatch.setattr(pc, "resize_state_dict", lambda model, sd: sd)
    monkeypatch.setattr(ag.torch, "load", lambda *a, **k: fake_sd)

    if env_url is not None:
        monkeypatch.setenv("QDRANT_URL", env_url)
    else:
        monkeypatch.delenv("QDRANT_URL", raising=False)

    gw = ag.AgentGateway.create_default(
        model_path=str(fake_model),
        vocab_path=VOCAB_PATH,
        lexicon_path=LEXICON_PATH,
        router_state_path=str(fake_router),
        expert_model_paths=None,
        device="cpu",
        qdrant_url=qdrant_url,
    )
    return gw, vm_calls


def test_create_default_env_qdrant_url_routes_remote(tmp_path, monkeypatch):
    """QDRANT_URL env'i → anka_bellek + simulasyon_bellek uzak adrese bağlanır."""
    gw, vm_calls = _run_create_default(
        monkeypatch, tmp_path, env_url="http://192.168.1.9:6333"
    )
    assert len(vm_calls) == 2, vm_calls
    for c in vm_calls:
        assert c["host"] == "192.168.1.9", c
        assert c["port"] == 6333, c
    assert gw._qdrant_host == "192.168.1.9"
    assert gw._qdrant_port == 6333


def test_create_default_explicit_arg_beats_env(tmp_path, monkeypatch):
    gw, vm_calls = _run_create_default(
        monkeypatch, tmp_path,
        qdrant_url="http://192.168.1.9:6444", env_url="http://10.0.0.1:6333",
    )
    for c in vm_calls:
        assert c["host"] == "192.168.1.9" and c["port"] == 6444, c
    assert (gw._qdrant_host, gw._qdrant_port) == ("192.168.1.9", 6444)


def test_create_default_localhost_6333_preserved(tmp_path, monkeypatch):
    """Varsayılan KORUNUR: env/arg yok → localhost:6333 (birebir eski sabit)."""
    gw, vm_calls = _run_create_default(monkeypatch, tmp_path)
    assert len(vm_calls) == 2
    for c in vm_calls:
        assert c["host"] == "localhost", c
        assert c["port"] == 6333, c
    assert (gw._qdrant_host, gw._qdrant_port) == ("localhost", 6333)


# ---------------------------------------------------------------------------
# 5) inject_reasoning_trace — hardcoded localhost:6333 yerine çözülen hedef
# ---------------------------------------------------------------------------

class _RecordingVM:
    calls = []
    batches = []

    def __init__(self, **kwargs):
        type(self).calls.append(dict(kwargs))
        self.collection_name = kwargs.get("collection_name")

    def add_documents_batch(self, **kwargs):
        type(self).batches.append(dict(kwargs))

    def get_document_count(self):
        return 0

    def close(self):
        pass


class _FakeTok:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return "soru cevap"


def _light_gateway(monkeypatch, **gw_kwargs):
    import src.gateway.agent_gateway as ag
    _RecordingVM.calls = []
    _RecordingVM.batches = []
    monkeypatch.setattr(ag, "VectorMemory", _RecordingVM)
    return ag.AgentGateway(
        model=None,
        tokenizer=_FakeTok(),
        memory=_RecordingVM(collection_name="anka_bellek"),
        general_memory=_RecordingVM(collection_name="simulasyon_bellek"),
        **gw_kwargs,
    )


def test_inject_reasoning_trace_routes_to_resolved_target(monkeypatch):
    """Gateway hedefi 192.168.1.9:6333 ise muhakeme_bellek DE oraya bağlanır
    (eski localhost:6333 sabiti param'le çözülür)."""
    gw = _light_gateway(monkeypatch, qdrant_host="192.168.1.9", qdrant_port=6333)
    res = gw.inject_reasoning_trace("Soru?", "Düşünce...", "Cevap.")
    muh_calls = [c for c in _RecordingVM.calls if c.get("collection_name") == "muhakeme_bellek"]
    assert len(muh_calls) == 1, _RecordingVM.calls
    assert muh_calls[0]["host"] == "192.168.1.9"
    assert muh_calls[0]["port"] == 6333
    assert res["status"] == "success" and res["collection"] == "muhakeme_bellek"


def test_inject_reasoning_trace_default_localhost_preserved(monkeypatch):
    """Varsayılan KORUNUR: param yoksa localhost:6333 (eski davranış birebir)."""
    gw = _light_gateway(monkeypatch)
    gw.inject_reasoning_trace("Soru?", "Düşünce...", "Cevap.")
    muh_calls = [c for c in _RecordingVM.calls if c.get("collection_name") == "muhakeme_bellek"]
    assert muh_calls[0]["host"] == "localhost"
    assert muh_calls[0]["port"] == 6333


# ---------------------------------------------------------------------------
# 6) Ekim betiği sözleşmesi — kayıt-kimlikli idempotent upsert (P2.1)
#    cot_vault.jsonl 91 kaydının HER BİRİ ayrı süpervizyon olayıdır (91 ayrı
#    timestamp; içerik 22 benzersizdir — tek sorgu 70x yinelenir). Nokta kimliği
#    TAM KAYITTAN türer → 91 nokta; tekrar koşumda ÇOĞALTMAZ.
# ---------------------------------------------------------------------------

def test_add_documents_batch_explicit_point_ids(monkeypatch):
    """point_ids verilen id'ler upsert'te BİREBİR kullanılır; tekrar koşum
    aynı id'yi GÜNCELER (çiftleme YOK)."""
    from qdrant_client.http import models
    import src.rag.vector_memory as vm_mod
    vm = vm_mod.VectorMemory(collection_name="test_pids_coll", vector_size=768)
    upserted_ids = []
    gercek_upsert = vm.client.upsert

    def kayit_upsert(collection_name, points, **kwargs):
        upserted_ids.extend(p.id for p in points)
        return gercek_upsert(collection_name=collection_name, points=points, **kwargs)

    monkeypatch.setattr(vm.client, "upsert", kayit_upsert)
    sparse = models.SparseVector(indices=[1], values=[1.0])
    for _ in range(2):  # betiğin İKİ kez koşulması
        vm.add_documents_batch(
            texts=["metin-a"],
            dense_vectors=[[0.1] * 768],
            sparse_vectors=[sparse],
            metadatas=[{"a": 1}],
            upsert=True,
            point_ids=[12345],
        )
    assert upserted_ids == [12345, 12345], upserted_ids
    assert vm.get_document_count() == 1, "aynı id iki kez yazıldı → TEK nokta kalmalı"
    vm.close()


def test_add_documents_batch_point_ids_length_mismatch():
    from qdrant_client.http import models
    import src.rag.vector_memory as vm_mod
    vm = vm_mod.VectorMemory(collection_name="test_pids_mismatch_coll", vector_size=768)
    sparse = models.SparseVector(indices=[1], values=[1.0])
    with pytest.raises(ValueError):
        vm.add_documents_batch(
            texts=["metin-a", "metin-b"],
            dense_vectors=[[0.1] * 768, [0.2] * 768],
            sparse_vectors=[sparse, sparse],
            metadatas=[{}, {}],
            upsert=True,
            point_ids=[111],
        )
    vm.close()


def test_seed_nokta_kimligi_tam_kayittan_turer():
    """Kimlik TAM KAYITTAN (tüm alanlar, kanonik) türer: aynı kayıt → aynı id
    (idempotent); farklı timestamp → FARKLI id (91 ayrı olay ayrı nokta olur)."""
    from scripts.seed_muhakeme_bellek import nokta_kimligi
    base = {"query": "Soru?", "thought_trace": "Düşünce", "final_answer": "Cevap."}
    k1 = nokta_kimligi(dict(base, timestamp="t1", source="Gemini"))
    assert k1 == nokta_kimligi(dict(base, timestamp="t1", source="Gemini"))
    assert k1 != nokta_kimligi(dict(base, timestamp="t2", source="Gemini"))
    assert 0 <= k1 < 2 ** 63


def test_seed_doc_text_gateway_bicimiyle_ayni():
    """doc_text, gateway inject_reasoning_trace biçimiyle BİREBİR aynı."""
    from scripts.seed_muhakeme_bellek import doc_text_olustur
    rec = {"query": "Açık istiare nedir?", "thought_trace": " Teşbih türüdür. ",
           "final_answer": "Kendisine benzetilen anılır."}
    beklenen = ("Soru: Açık istiare nedir?\n\n[Akıl Yürütme]:\nTeşbih türüdür.\n\n"
                "[Çözüm]: Kendisine benzetilen anılır.")
    assert doc_text_olustur(rec) == beklenen