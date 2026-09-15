import pytest
import os
import json
import sys

from scripts.agent_bus_mcp import AgentBus

def test_report_result_without_narrative_log_backward_compatible(tmp_path):
    repo_root = str(tmp_path)
    bus = AgentBus(repo_root)
    bus.ensure_directories()

    task_id = "T-9901"
    # Create task file so report_result can update task status
    task_file = os.path.join(bus.tasks_dir, f"{task_id}.json")
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({"id": task_id, "status": "claimed", "claimed_by": "antigravity"}, f)

    res = bus.report_result(
        task_id=task_id,
        status="done",
        summary="Eski sema ile tamamlandi",
        evidence=["adim 1", "adim 2"],
        changed_files=["foo.py"]
    )
    assert res == {"ok": True}

    result_path = os.path.join(bus.results_dir, f"{task_id}.json")
    assert os.path.exists(result_path)
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == task_id
    assert data["status"] == "done"
    assert data["summary"] == "Eski sema ile tamamlandi"
    assert data["evidence"] == ["adim 1", "adim 2"]
    assert data["changed_files"] == ["foo.py"]
    assert "narrative_log" not in data, "narrative_log verilmediginde sonucta yer almamali (geriye uyumluluk)"


def test_report_result_with_valid_narrative_log(tmp_path):
    repo_root = str(tmp_path)
    bus = AgentBus(repo_root)
    bus.ensure_directories()

    task_id = "T-9902"
    task_file = os.path.join(bus.tasks_dir, f"{task_id}.json")
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({"id": task_id, "status": "claimed", "claimed_by": "antigravity"}, f)

    narrative = {
        "path": ".agent-bus/notes/T-9902.md",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ozet": "T-9902 gercek zamanli yurutum notu"
    }

    res = bus.report_result(
        task_id=task_id,
        status="done",
        summary="Yeni sema ile tamamlandi",
        evidence=["adim 1"],
        changed_files=["bar.py"],
        narrative_log=narrative
    )
    assert res == {"ok": True}

    result_path = os.path.join(bus.results_dir, f"{task_id}.json")
    assert os.path.exists(result_path)
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task_id"] == task_id
    assert "narrative_log" in data
    assert data["narrative_log"]["path"] == ".agent-bus/notes/T-9902.md"
    assert data["narrative_log"]["sha256"] == narrative["sha256"]
    assert data["narrative_log"]["ozet"] == narrative["ozet"]


def test_report_result_negative_rejects_escaping_paths(tmp_path):
    repo_root = str(tmp_path)
    bus = AgentBus(repo_root)
    bus.ensure_directories()

    task_id = "T-9903"
    task_file = os.path.join(bus.tasks_dir, f"{task_id}.json")
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({"id": task_id, "status": "claimed", "claimed_by": "antigravity"}, f)

    result_path = os.path.join(bus.results_dir, f"{task_id}.json")

    # 1. Ust dizin kurali ihlali (../x)
    with pytest.raises(ValueError, match="Yol güvenlik ihlali"):
        bus.report_result(
            task_id=task_id,
            status="done",
            summary="Kacak yol denemesi",
            narrative_log={
                "path": "../escaping_note.md",
                "sha256": "1234567890abcdef",
                "ozet": "gecersiz yol"
            }
        )
    assert not os.path.exists(result_path), "Hata durumunda diske hicbir dosya yazilmamali"

    # 2. Mutlak yol ihlali (/tmp/x)
    with pytest.raises(ValueError, match="Yol güvenlik ihlali"):
        bus.report_result(
            task_id=task_id,
            status="done",
            summary="Mutlak yol denemesi",
            narrative_log={
                "path": "/tmp/escaping_note.md",
                "sha256": "1234567890abcdef",
                "ozet": "mutlak yol"
            }
        )
    assert not os.path.exists(result_path), "Hata durumunda diske hicbir dosya yazilmamali"
