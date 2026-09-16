#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/agent_bus_hook.py
=========================
Antigravity yaşam döngüsü kancası (Lifecycle Hook).
PreInvocation aşamasında çalışır:
- agent-bus kuyruğunu yoklar.
- Eğer 'antigravity' için açık görev varsa, ajanın önüne otomatik yönlendirme
  mesajı (ephemeralMessage) enjekte eder. Böylece kullanıcının elle '/agent-bus'
  yazmasına gerek kalmaz.
"""

import sys
import json
import os

def main():
    # Antigravity hook stdin girdisi (JSON)
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)

    try:
        from scripts.agent_bus_mcp import AgentBus
        bus = AgentBus(repo_root)
        open_tasks = bus.list_tasks(status="open")

        # antigravity veya herkese açık görevleri süz
        actionable_tasks = [
            t for t in open_tasks
            if (not t.get("to") or t.get("to") == "antigravity")
            and (not t.get("claimed_by") or t.get("claimed_by") == "antigravity")
        ]

        # Ayrıca gelen kutusunda okunmamış mesaj var mı kontrol et
        unread_msgs = bus.read_inbox(who="antigravity", unread_only=True)

        injected = []
        if actionable_tasks:
            next_task = min(actionable_tasks, key=lambda x: str(x.get("id", "")))
            msg = (
                f"🚨 [AGENT-BUS UYARISI]: Bekleyen {len(actionable_tasks)} açık görev var! "
                f"Sıradaki görev: {next_task.get('id')} — '{next_task.get('title')}'. "
                f"Lütfen yürütücü çevrimi adımlarını işlet (bus_claim_task -> bus_acquire_lease -> uygula -> bus_report_result)."
            )
            injected.append({"ephemeralMessage": msg})
        elif unread_msgs:
            last_msg = unread_msgs[-1]
            msg = (
                f"📬 [AGENT-BUS BİLDİRİMİ]: Gelen kutunda {len(unread_msgs)} okunmamış mesaj var. "
                f"Son mesaj: '{last_msg.get('subject')}' (Kimden: {last_msg.get('from')}). "
                f"bus_inbox(who='antigravity') ile mesajı oku."
            )
            injected.append({"ephemeralMessage": msg})

        output = {
            "injectSteps": injected
        }
    except Exception as e:
        # Kanca hatası ajanı kilitlememeli
        output = {"injectSteps": []}

    sys.stdout.write(json.dumps(output, ensure_ascii=False))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
