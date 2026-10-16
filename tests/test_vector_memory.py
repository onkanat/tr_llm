#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Birim Testleri: Vektörel Bellek & Fail-Closed Sözleşmesi (test_vector_memory.py)
================================================================================
Uzak bağlantı hatalarında fail-closed davranışı (sessiz düşüşün engellenmesi),
önbellek kirliliği koruması, dolu koleksiyon yıkım koruması, boyut uyuşmazlığı
koruması, açık in-memory yapılandırması ve build_from_disk fail-closed sözleşmesini
doğrular.
"""

import os
import tempfile
import pytest
from qdrant_client.http import models

from src.rag.vector_memory import VectorMemory
from src.rag.rag_pipeline import build_from_disk


def test_remote_host_unreachable_raises_runtime_error():
    """Uzak sunucuya bağlanılamadığında sessizce local'e düşülmez, RuntimeError fırlatılır."""
    # 127.0.0.1:65534 kapalı porttur; hızlıca connection refused verir
    with pytest.raises(RuntimeError) as exc_info:
        VectorMemory(
            collection_name="test_fail_coll",
            vector_size=768,
            host="127.0.0.1",
            port=65534,
            allow_fallback=False
        )

    err_msg = str(exc_info.value)
    assert "[VEKTOR_BELLEK_HATA]" in err_msg
    assert "remote (127.0.0.1:65534)" in err_msg
    assert "sessiz-local-fallback kapalıdır" in err_msg


def test_failed_connection_does_not_poison_shared_cache():
    """Başarısız bağlantı denemeleri _shared_clients önbelleğini kirletmez."""
    initial_keys = set(VectorMemory._shared_clients.keys())
    bad_key = "remote:127.0.0.1:65534"

    with pytest.raises(RuntimeError):
        VectorMemory(
            collection_name="test_cache_coll",
            vector_size=768,
            host="127.0.0.1",
            port=65534,
            allow_fallback=False
        )

    assert bad_key not in VectorMemory._shared_clients
    assert set(VectorMemory._shared_clients.keys()) == initial_keys


def test_recreate_dolu_collection_without_confirmation_raises_error():
    """Dolu koleksiyonun confirm_destroy=False ile yıkılması engellenir (fail-closed)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm = VectorMemory("test_dolu_coll", 768, storage_path=tmp_dir)

        # 1 belge ekle
        dense = [0.1] * 768
        sparse = models.SparseVector(indices=[1], values=[1.0])
        vm.add_document("Önemli ahşap bilgi kartı.", dense, sparse, {"crystal_tags": "ahşap"})

        assert vm.get_document_count() == 1

        # confirm_destroy=False ile recreate çağrısı hata vermeli
        with pytest.raises(RuntimeError) as exc_info:
            vm.recreate_collection(768, confirm_destroy=False)

        err_msg = str(exc_info.value)
        assert "[VEKTOR_BELLEK_HATA]" in err_msg
        assert "DOLU" in err_msg
        assert "confirm_destroy=True gerekir" in err_msg

        # Belge silinmemiş olmalı
        assert vm.get_document_count() == 1
        vm.close()


def test_recreate_dolu_collection_with_confirmation_succeeds():
    """confirm_destroy=True ile dolu koleksiyon açık onayla yıkılıp yeniden oluşturulabilir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm = VectorMemory("test_confirm_coll", 768, storage_path=tmp_dir)

        dense = [0.1] * 768
        sparse = models.SparseVector(indices=[1], values=[1.0])
        vm.add_document("Yıkılacak geçici belge.", dense, sparse, {"crystal_tags": "gecici"})
        assert vm.get_document_count() == 1

        # confirm_destroy=True ile temizlik
        vm.recreate_collection(768, confirm_destroy=True)
        assert vm.get_document_count() == 0
        vm.close()


def test_dimension_mismatch_on_dolu_collection_fails_closed():
    """Boyut uyuşmazlığı olan dolu koleksiyon açılırken sessiz silme veya yutma yapılmaz, fail-closed durulur."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm_768 = VectorMemory("dim_test_coll", 768, storage_path=tmp_dir)
        dense = [0.1] * 768
        sparse = models.SparseVector(indices=[1], values=[1.0])
        vm_768.add_document("768 boyutlu kanonik doküman.", dense, sparse, {"crystal_tags": "kanonik"})
        assert vm_768.get_document_count() == 1
        vm_768.close()
        del vm_768

        # Aynı depolama yolundaki dolu koleksiyon 384 boyutuyla açılmaya çalışıldığında
        # RuntimeError fırlatılmalı (hata yutulmamalı)
        with pytest.raises(RuntimeError) as exc_info:
            VectorMemory("dim_test_coll", 384, storage_path=tmp_dir)

        err_msg = str(exc_info.value)
        assert "[VEKTOR_BELLEK_HATA]" in err_msg
        assert "DOLU" in err_msg

        # Koleksiyon verisi sağlam kalmalı
        vm_check = VectorMemory("dim_test_coll", 768, storage_path=tmp_dir)
        assert vm_check.get_document_count() == 1
        vm_check.close()


def test_in_memory_explicit_configuration():
    """Açık in-memory yapılandırması ve allow_fallback davranışları doğrulanır."""
    # 1. location=":memory:" açık beyanı
    vm1 = VectorMemory("test_mem1", 768, location=":memory:")
    assert vm1.is_in_memory is True
    assert vm1.storage_type == "in-memory (:memory:)"
    vm1.close()

    # 2. Host ve storage_path yoksa repo-içi yazımı engellemek için in-memory açılır
    vm2 = VectorMemory("test_mem2", 768)
    assert vm2.is_in_memory is True
    assert vm2.storage_type == "in-memory (:memory:)"
    vm2.close()

    # 3. allow_fallback=True ile uzak sunucu yokluğunda sessizce geçici belleğe düşülür
    vm3 = VectorMemory(
        "test_mem3",
        768,
        host="127.0.0.1",
        port=65534,
        allow_fallback=True
    )
    assert vm3.is_in_memory is True
    assert vm3.storage_type == "in-memory (:memory:)"
    vm3.close()


def test_hybrid_recall_delegation_and_filtering():
    """VectorMemory.hybrid_recall delegasyonu düşünce filtrelerini ve kök eşleşmesini uygular."""
    vm = VectorMemory("test_recall_coll", 64, location=":memory:")

    dense = [0.2] * 64
    sparse = models.SparseVector(indices=[1, 2], values=[0.8, 0.9])

    # 1. Saf düşünce metni (elenmeli)
    vm.add_document(
        text="<think>We should analyze carpentry joints step by step.</think>",
        dense_vector=dense,
        sparse_vector=sparse,
        metadata={"crystal_tags": "marangozluk ahşap"}
    )

    # 2. Temiz bilgi metni (öne çıkmalı)
    vm.add_document(
        text="Kırlangıç kuyruğu zıvana ahşap mobilya yapımında yüksek çekme direnci sağlar.",
        dense_vector=dense,
        sparse_vector=sparse,
        metadata={"crystal_tags": "kırlangıç kuyruk zıvana ahşap"}
    )

    results = vm.hybrid_recall(
        dense_query=dense,
        sparse_query=sparse,
        top_k=3,
        query_tags="<ENT> kırlangıç zıvana </ENT>"
    )

    # Saf düşünce elendiği için sadece temiz metin dönmeli
    assert len(results) == 1
    assert "Kırlangıç kuyruğu" in results[0]["text"]
    assert results[0]["has_root_match"] is True
    assert results[0]["score"] > 0.0
    vm.close()


def test_rag_pipeline_build_from_disk_fail_closed():
    """build_from_disk uzak sunucu yokluğunda fail-closed RuntimeError fırlatır; allow_fallback=True ile geçici bellek açar."""
    vocab_path = "data/rebuild/vocab_anka_r1_33114.json"
    model_path = "data/anka_a1r.pt"

    # 1. allow_fallback=False (varsayılan): Uzak sunucu yoksa RuntimeError fırlatılır
    with pytest.raises(RuntimeError) as exc_info:
        build_from_disk(
            vocab_path=vocab_path,
            model_path=model_path,
            host="127.0.0.1",
            port=65534,
            allow_fallback=False
        )
    assert "[VEKTOR_BELLEK_HATA]" in str(exc_info.value)

    # 2. allow_fallback=True: Geçici bellek (:memory:) ile başlatılır ve örnek belgeler yüklenir
    pipeline = build_from_disk(
        vocab_path=vocab_path,
        model_path=model_path,
        host="127.0.0.1",
        port=65534,
        allow_fallback=True
    )
    assert pipeline is not None
    assert pipeline.memory.is_in_memory is True
    assert pipeline.memory.get_document_count() >= 3
    pipeline.memory.close()
