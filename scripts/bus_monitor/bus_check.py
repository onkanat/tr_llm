#!/usr/bin/env python3
"""
scripts/bus_monitor/bus_check.py

agent-bus kuyruk ve gelen kutusu denetleyicisi.
Cron ile periyodik olarak çalıştırılır (LLM token tüketmez).
Yalnızca antigravity için okunmamış yeni mesaj veya açık görev olduğunda
rc=10 döner ve özet bilgi basar (böylece agent tetiklenebilir).
Yeni bir şey yoksa sessizce rc=0 ile çıkar.
"""

import sys
import os
import glob
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INBOX_DIR = os.path.join(REPO_ROOT, ".agent-bus", "state", "inbox", "antigravity")
TASKS_DIR = os.path.join(REPO_ROOT, ".agent-bus", "state", "tasks")

def check_bus() -> int:
    unread_messages = []
    if os.path.exists(INBOX_DIR):
        for f in glob.glob(os.path.join(INBOX_DIR, "*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    if not d.get("read", False):
                        unread_messages.append((os.path.basename(f), d.get("subject", ""), d.get("from", "")))
            except Exception:
                pass

    open_tasks = []
    if os.path.exists(TASKS_DIR):
        for f in glob.glob(os.path.join(TASKS_DIR, "*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    status = d.get("status")
                    target = d.get("to")
                    task_id = d.get("id")
                    if status == "open" and (target in ["antigravity", None, ""]):
                        open_tasks.append((task_id, d.get("title", "")))
            except Exception:
                pass

    if unread_messages or open_tasks:
        print(f"[YENİ VERİ TESPİT EDİLDİ]")
        if unread_messages:
            print(f"  * Okunmamış Mesaj Sayısı: {len(unread_messages)}")
            for fname, subj, sender in unread_messages[:3]:
                print(f"    - {sender}: {subj}")
        if open_tasks:
            print(f"  * Açık Görev Sayısı: {len(open_tasks)}")
            for tid, title in open_tasks:
                print(f"    - {tid}: {title}")
        return 10  # Yeni olay var: tetikleme gerekli

    return 0  # Yeni olay yok: sessiz

if __name__ == "__main__":
    sys.exit(check_bus())
