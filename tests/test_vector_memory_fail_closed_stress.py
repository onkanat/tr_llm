#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite: VectorMemory Fail-Closed Contract (Milestone M3)
==============================================================================
Empirical Challenger: challenger_m3_1

Aggressively stress-tests the fail-closed behavior of VectorMemory and RAG pipeline:
1. Unreachable host & unresolvable hostname fail-closed rejection.
2. Fallback toggle strictness (only fallback when allow_fallback=True is explicit).
3. Cache poisoning prevention after failed connections.
4. Non-empty collection destruction protection (confirm_destroy requirement).
5. Dimension mismatch fail-closed behavior and data preservation.
6. Empty collection dimension adaptation.
7. Invalid vector dimension write & query rejection.
8. Invalid / unwriteable storage paths fail-closed behavior.
9. Upsert deduplication vs monotonic ID generation.
10. AgentGateway invalid target-collection validation.
"""

import os
import tempfile
import pytest
from qdrant_client.http import models

from src.rag.vector_memory import VectorMemory
from src.rag.rag_pipeline import build_from_disk
from src.gateway.agent_gateway import AgentGateway


# ==============================================================================
# 1. Unreachable Host & Remote Connection Stress Tests
# ==============================================================================

def test_stress_unreachable_host_port_refused():
    """Unreachable closed port immediately raises fail-closed RuntimeError."""
    bad_key = "remote:127.0.0.1:65534"
    with pytest.raises(RuntimeError) as exc_info:
        VectorMemory(
            collection_name="stress_unreachable",
            vector_size=768,
            host="127.0.0.1",
            port=65534,
            allow_fallback=False
        )

    err = str(exc_info.value)
    assert "[VEKTOR_BELLEK_HATA]" in err
    assert "remote (127.0.0.1:65534)" in err
    assert "sessiz-local-fallback kapalıdır" in err
    assert bad_key not in VectorMemory._shared_clients


def test_stress_unresolvable_hostname_fail_closed():
    """Unresolvable invalid hostname immediately raises fail-closed RuntimeError."""
    bad_host = "invalid.unresolvable.domain.local"
    bad_key = f"remote:{bad_host}:6333"

    with pytest.raises(RuntimeError) as exc_info:
        VectorMemory(
            collection_name="stress_bad_host",
            vector_size=768,
            host=bad_host,
            port=6333,
            allow_fallback=False
        )

    err = str(exc_info.value)
    assert "[VEKTOR_BELLEK_HATA]" in err
    assert f"remote ({bad_host}:6333)" in err
    assert "sessiz-local-fallback kapalıdır" in err
    assert bad_key not in VectorMemory._shared_clients


def test_stress_unreachable_location_url_fail_closed():
    """Unreachable HTTP location URL fails closed unless allow_fallback=True."""
    bad_location = "http://127.0.0.1:65534"

    # 1. allow_fallback=False raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        VectorMemory(
            collection_name="stress_loc_fail",
            vector_size=768,
            location=bad_location,
            allow_fallback=False
        )

    err = str(exc_info.value)
    assert "[VEKTOR_BELLEK_HATA]" in err
    assert f"remote location ({bad_location})" in err
    assert "sessiz-local-fallback kapalıdır" in err

    # 2. allow_fallback=True falls back to :memory:
    vm_fallback = VectorMemory(
        collection_name="stress_loc_fallback",
        vector_size=768,
        location=bad_location,
        allow_fallback=True
    )
    assert vm_fallback.is_in_memory is True
    assert vm_fallback.storage_type == "in-memory (:memory:)"
    vm_fallback.close()


def test_stress_host_fallback_toggle_behavior():
    """Fallback strictly requires allow_fallback=True; default (False) raises."""
    # 1. Default allow_fallback is False
    with pytest.raises(RuntimeError):
        VectorMemory(
            collection_name="stress_default_fallback",
            vector_size=768,
            host="127.0.0.1",
            port=65534
        )

    # 2. Explicit allow_fallback=True succeeds in :memory:
    vm = VectorMemory(
        collection_name="stress_explicit_fallback",
        vector_size=768,
        host="127.0.0.1",
        port=65534,
        allow_fallback=True
    )
    assert vm.is_in_memory is True
    assert vm.storage_type == "in-memory (:memory:)"
    vm.close()


# ==============================================================================
# 2. RAG Pipeline build_from_disk Stress Tests
# ==============================================================================

def test_stress_build_from_disk_unreachable_host_fail_closed():
    """build_from_disk with unreachable host raises RuntimeError by default."""
    vocab_path = "data/rebuild/vocab_anka_r1_33114.json"
    model_path = "data/anka_a1r.pt"

    # Default allow_fallback must be False and raise
    with pytest.raises(RuntimeError) as exc_info:
        build_from_disk(
            vocab_path=vocab_path,
            model_path=model_path,
            host="127.0.0.1",
            port=65534
        )
    assert "[VEKTOR_BELLEK_HATA]" in str(exc_info.value)


def test_stress_build_from_disk_unresolvable_host_fail_closed():
    """build_from_disk with unresolvable host raises RuntimeError when allow_fallback=False."""
    vocab_path = "data/rebuild/vocab_anka_r1_33114.json"
    model_path = "data/anka_a1r.pt"

    with pytest.raises(RuntimeError) as exc_info:
        build_from_disk(
            vocab_path=vocab_path,
            model_path=model_path,
            host="invalid.unresolvable.domain.local",
            port=6333,
            allow_fallback=False
        )
    assert "[VEKTOR_BELLEK_HATA]" in str(exc_info.value)


def test_stress_build_from_disk_explicit_fallback_succeeds():
    """build_from_disk with allow_fallback=True initializes in-memory and loads sample documents."""
    vocab_path = "data/rebuild/vocab_anka_r1_33114.json"
    model_path = "data/anka_a1r.pt"

    pipeline = build_from_disk(
        vocab_path=vocab_path,
        model_path=model_path,
        host="127.0.0.1",
        port=65534,
        allow_fallback=True
    )
    try:
        assert pipeline is not None
        assert pipeline.memory.is_in_memory is True
        assert pipeline.memory.get_document_count() >= 3
    finally:
        pipeline.memory.close()


# ==============================================================================
# 3. Collection Deletion & Destruction Protection Stress Tests
# ==============================================================================

def test_stress_recreate_non_empty_collection_destruction_guard():
    """Non-empty collection recreation strictly requires confirm_destroy=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm = VectorMemory("stress_destroy_guard", 768, storage_path=tmp_dir)

        # Populate with 3 documents
        dense = [0.1] * 768
        sparse = models.SparseVector(indices=[1, 2], values=[0.5, 0.8])
        for idx in range(3):
            vm.add_document(f"Kritik doküman {idx}", dense, sparse, {"idx": idx})

        assert vm.get_document_count() == 3

        # Attempt recreate with confirm_destroy=False -> RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            vm.recreate_collection(768, confirm_destroy=False)
        assert "[VEKTOR_BELLEK_HATA]" in str(exc_info.value)
        assert "DOLU (3 nokta)" in str(exc_info.value)
        assert "confirm_destroy=True gerekir" in str(exc_info.value)

        # Attempt recreate without confirm_destroy parameter (default False) -> RuntimeError
        with pytest.raises(RuntimeError) as exc_info2:
            vm.recreate_collection(768)
        assert "confirm_destroy=True gerekir" in str(exc_info2.value)

        # Verify documents were not wiped or partially deleted
        assert vm.get_document_count() == 3
        docs = vm.list_documents(limit=10)
        assert len(docs) == 3

        # Recreate with confirm_destroy=True succeeds and empties collection
        vm.recreate_collection(768, confirm_destroy=True)
        assert vm.get_document_count() == 0
        vm.close()


def test_stress_recreate_empty_collection_permitted():
    """Empty collection recreate without confirmation is permitted (clean probe path)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm = VectorMemory("stress_empty_recreate", 768, storage_path=tmp_dir)
        assert vm.get_document_count() == 0

        # Should not raise
        vm.recreate_collection(768, confirm_destroy=False)
        assert vm.get_document_count() == 0
        vm.close()


# ==============================================================================
# 4. Dimension Mismatch Stress Tests
# ==============================================================================

def test_stress_dimension_mismatch_non_empty_collection_fails_closed():
    """Opening non-empty collection with mismatched vector_size fails closed and preserves data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 768-dim collection and insert document
        vm_orig = VectorMemory("stress_dim_guard", 768, storage_path=tmp_dir)
        dense = [0.42] * 768
        sparse = models.SparseVector(indices=[10], values=[0.95])
        vm_orig.add_document("Paha biçilmez bilgi", dense, sparse, {"security": "high"})
        assert vm_orig.get_document_count() == 1
        vm_orig.close()

        # Attempt to open same storage path with 384 dimensions
        with pytest.raises(RuntimeError) as exc_info:
            VectorMemory("stress_dim_guard", 384, storage_path=tmp_dir)

        err = str(exc_info.value)
        assert "[VEKTOR_BELLEK_HATA]" in err
        assert "DOLU" in err
        assert "confirm_destroy=True gerekir" in err

        # Attempt to open same storage path with 1024 dimensions
        with pytest.raises(RuntimeError) as exc_info2:
            VectorMemory("stress_dim_guard", 1024, storage_path=tmp_dir)
        assert "DOLU" in str(exc_info2.value)

        # Reopen with correct 768 dimensions and verify data integrity
        vm_restore = VectorMemory("stress_dim_guard", 768, storage_path=tmp_dir)
        assert vm_restore.get_document_count() == 1
        docs = vm_restore.list_documents()
        assert len(docs) == 1
        assert docs[0]["payload"]["text"] == "Paha biçilmez bilgi"
        assert docs[0]["payload"]["security"] == "high"
        vm_restore.close()


def test_stress_dimension_mismatch_empty_collection_adapts():
    """Opening empty collection with new dimension safely recreates with new vector_size."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vm_empty = VectorMemory("stress_empty_adapt", 768, storage_path=tmp_dir)
        assert vm_empty.get_document_count() == 0
        vm_empty.close()

        # Reopen with 384 dimensions (empty collection should adapt without error)
        vm_adapted = VectorMemory("stress_empty_adapt", 384, storage_path=tmp_dir)
        assert vm_adapted.get_document_count() == 0
        coll_info = vm_adapted.client.get_collection("stress_empty_adapt")
        assert coll_info.config.params.vectors["dense"].size == 384

        # Verify insertion of 384-dim vector succeeds
        dense_384 = [0.1] * 384
        sparse = models.SparseVector(indices=[1], values=[1.0])
        vm_adapted.add_document("384 boyutlu yeni veri", dense_384, sparse)
        assert vm_adapted.get_document_count() == 1
        vm_adapted.close()


# ==============================================================================
# 5. Invalid Operations & Edge Cases
# ==============================================================================

def test_stress_invalid_vector_dimension_write_rejected():
    """Inserting vector with dimension mismatch is immediately rejected with ValueError."""
    vm = VectorMemory("stress_dim_write", 768, location=":memory:")
    dense_bad = [0.1] * 256  # 256 != 768
    sparse = models.SparseVector(indices=[1], values=[1.0])

    with pytest.raises(ValueError) as exc_info:
        vm.add_document("Hatalı boyutlu veri", dense_bad, sparse)

    assert "could not broadcast" in str(exc_info.value)
    # The corrupted entry has no valid vector and cannot be retrieved by semantic search
    res = vm.dense_recall([0.1] * 768)
    assert len(res) == 0
    vm.close()


def test_stress_dense_recall_dimension_mismatch_rejected():
    """Querying dense_recall with dimension mismatch is immediately rejected."""
    vm = VectorMemory("stress_dim_query", 768, location=":memory:")
    dense_good = [0.1] * 768
    sparse = models.SparseVector(indices=[1], values=[1.0])
    vm.add_document("Geçerli veri", dense_good, sparse)
    assert vm.get_document_count() == 1

    dense_query_bad = [0.1] * 256  # 256 != 768
    with pytest.raises(ValueError):
        vm.dense_recall(dense_query_bad)

    vm.close()


def test_stress_invalid_storage_path_unwriteable():
    """storage_path pointing to an invalid file path fails closed when allow_fallback=False."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        # File path cannot be used as Qdrant sqlite/storage directory
        with pytest.raises(RuntimeError) as exc_info:
            VectorMemory("stress_bad_path", 768, storage_path=tmp_file.name, allow_fallback=False)

        err = str(exc_info.value)
        assert "[VEKTOR_BELLEK_HATA]" in err
        assert "açılamadı; sessiz-fallback kapalıdır" in err

        # allow_fallback=True falls back to :memory:
        vm_fb = VectorMemory("stress_bad_path", 768, storage_path=tmp_file.name, allow_fallback=True)
        assert vm_fb.is_in_memory is True
        assert vm_fb.storage_type == "in-memory (:memory:)"
        vm_fb.close()


def test_stress_upsert_deduplication_vs_monotonic():
    """upsert=True computes deterministic point-id preventing duplicates; upsert=False is monotonic."""
    vm = VectorMemory("stress_upsert_test", 768, location=":memory:")
    dense = [0.2] * 768
    sparse = models.SparseVector(indices=[1], values=[1.0])

    # 1. upsert=True: Repeated insertion of identical text replaces existing point
    vm.add_documents_batch(["Tekrarlı Metin"], [dense], [sparse], [{"ver": 1}], upsert=True)
    assert vm.get_document_count() == 1

    vm.add_documents_batch(["Tekrarlı Metin"], [dense], [sparse], [{"ver": 2}], upsert=True)
    assert vm.get_document_count() == 1
    docs = vm.list_documents()
    assert docs[0]["payload"]["ver"] == 2

    # 2. upsert=False: Monotonic ID creates separate points
    vm.add_documents_batch(["Monoton Metin"], [dense], [sparse], [{"ver": 1}], upsert=False)
    assert vm.get_document_count() == 2
    vm.add_documents_batch(["Monoton Metin"], [dense], [sparse], [{"ver": 2}], upsert=False)
    assert vm.get_document_count() == 3

    vm.close()


def test_stress_delete_collection_non_existent():
    """delete_collection returns False on non-existent collection without crashing."""
    vm = VectorMemory("stress_del_coll", 768, location=":memory:")
    res = vm.delete_collection("tamamen_olmayan_koleksiyon_xyz")
    assert res is False
    vm.close()


def test_stress_gateway_invalid_target_collection_rejected():
    """AgentGateway._resolve_target_memory strictly raises ValueError on unknown collections."""
    mem1 = VectorMemory("simulasyon_bellek", 768, location=":memory:")
    mem2 = VectorMemory("anka_bellek", 768, location=":memory:")
    gateway = AgentGateway(memory=mem1, general_memory=mem2)
    try:
        with pytest.raises(ValueError) as exc_info:
            gateway._resolve_target_memory("olmayan_sahte_koleksiyon")

        err = str(exc_info.value)
        assert "[GATEWAY_HATA]" in err
        assert "bilinmeyen target-collection 'olmayan_sahte_koleksiyon'" in err
    finally:
        mem1.close()
        mem2.close()
