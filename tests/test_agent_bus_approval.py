import pytest
import os
import json
from scripts.agent_bus_mcp import AgentBus

def test_awaiting_approval_cannot_be_claimed(tmp_path):
    repo_root = str(tmp_path)
    bus = AgentBus(repo_root)
    bus.ensure_directories()

    # 1. Görevi awaiting_approval statüsünde oluştur
    res = bus.post_task(
        title="Onay Bekleyen Görev",
        spec="Bu görev operatör onayı beklemektedir.",
        status="awaiting_approval"
    )
    task_id = res["id"]

    task = bus.get_task(task_id)
    assert task["status"] == "awaiting_approval"

    # 2. Claim etmeyi dene -> reddedilmeli
    claim_res = bus.claim_task(task_id, owner="antigravity")
    assert claim_res["ok"] is False
    assert "awaiting_approval" in claim_res["conflicts"][0]

    # Görev hâlâ awaiting_approval olmalı
    task_after = bus.get_task(task_id)
    assert task_after["status"] == "awaiting_approval"
    assert task_after["claimed_by"] is None

    # 3. Operatör onayı simülasyonu (post_task güncellemesi ile open yap)
    bus.post_task(
        title="Onay Bekleyen Görev",
        spec="Bu görev operatör tarafından onaylanmıştır.",
        task_id=task_id,
        status="open"
    )

    task_approved = bus.get_task(task_id)
    assert task_approved["status"] == "open"

    # 4. Artık devralınabilmeli
    claim_res2 = bus.claim_task(task_id, owner="antigravity")
    assert claim_res2["ok"] is True
    assert claim_res2["conflicts"] == []

    task_claimed = bus.get_task(task_id)
    assert task_claimed["status"] == "claimed"
    assert task_claimed["claimed_by"] == "antigravity"


def test_approval_flip_requires_claude_channel(tmp_path):
    """T-0171 (V1 onarımı): awaiting_approval → open geçişi yalnız danışman
    (claude) kanalından; antigravity'den geçiş ValueError ile DUR (fail-closed)."""
    repo_root = str(tmp_path)
    bus = AgentBus(repo_root)
    bus.ensure_directories()

    res = bus.post_task(
        title="Kısıtlı Onay Görevi",
        spec="Onay-geçişi yetki-kapsamı testi.",
        status="awaiting_approval"
    )
    task_id = res["id"]

    # 1. antigravity'den kendi-kendine onay denemesi -> ValueError
    with pytest.raises(ValueError, match="Onay geçişi reddedildi"):
        bus.post_task(
            title="Kısıtlı Onay Görevi",
            spec="Kendi-kendine onay denemesi.",
            task_id=task_id,
            status="open",
            from_agent="antigravity"
        )

    # Görev hâlâ awaiting_approval (durum-sabit)
    task_after = bus.get_task(task_id)
    assert task_after["status"] == "awaiting_approval"
    assert task_after["claimed_by"] is None

    # 2. danışman (claude) kanalından onay -> open (pozitif-kontrol)
    bus.post_task(
        title="Kısıtlı Onay Görevi",
        spec="Operatör onayı claude kanalından verildi.",
        task_id=task_id,
        status="open",
        from_agent="claude"
    )
    assert bus.get_task(task_id)["status"] == "open"
