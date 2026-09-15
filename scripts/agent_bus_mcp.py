#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
agent-bus — Ajanlar Arası Koordinasyon Protokolü (MCP Stdio Sunucusu)
=====================================================================
Bu sunucu; aynı çalışma alanında görev yapan danışman (Claude Code) ve yürütücü
(Antigravity) ajanların birbirlerinin üzerine yazmasını yapısal olarak imkânsız
kılan koordinasyon veri yolunu (agent-bus) uygular.

Standartlar (SPEC.md & CLAUDE.md):
- Yalnızca Python standart kütüphanesi (sıfır harici bağımlılık)
- Satır-sınırlı JSON-RPC 2.0 stdio taşıması (initialize, tools/list, tools/call)
- 10 MCP aracı:
    1. bus_post_task
    2. bus_list_tasks
    3. bus_claim_task
    4. bus_acquire_lease
    5. bus_release_lease
    6. bus_lease_status
    7. bus_report_result
    8. bus_send
    9. bus_inbox
   10. bus_frozen_list
- Üç değişmez kural:
    1. Kiralama olmadan yazma yok
    2. Donmuş artefakt (frozen.json) koruması: writes[] bildirimi VE effective_scope: 'dir'
       zorunludur; kapsam GERÇEKLİKTEN (dosya sisteminden) türetilir, '/' şekliyle veya
       'scope=dir' beyanıyla genişletilemez.
    3. Kiraların TTL denetimi ve devralma
- Tip ipuçları zorunlu
- Saf fonksiyonlar, atomik yazma (<dosya>.tmp -> os.replace)
- Yol güvenliği (.. ve mutlak yol engeli; adlar ve task_id regex denetimi)
- 11 adımlı kabul kriteri ve 36 hücreli çapraz çarpım matris testi:
  python3 scripts/agent_bus_mcp.py --selftest
"""

import os
import sys
import json
import re
import fnmatch
import argparse
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set


# =====================================================================
# YOL VE DEĞER DOĞRULAMA YARDIMCILARI (PATH & VALIDATION UTILITIES)
# =====================================================================

AGENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
TASK_ID_PATTERN = re.compile(r"^T-\d{4}$")


def validate_agent_name(name: str, field_name: str = "ajan") -> str:
    """
    Ajan adını doğrular: ^[a-z0-9][a-z0-9_-]{0,63}$
    Uymayan değerler ValueError fırlatır (yol kaçışını yapısal engeller).
    """
    if not name or not isinstance(name, str) or not AGENT_NAME_PATTERN.match(name):
        raise ValueError(
            f"Geçersiz {field_name} adı: '{name}'. "
            f"Değer '^[a-z0-9][a-z0-9_-]{{0,63}}$' kuralına uymalıdır."
        )
    return name


def validate_task_id(task_id: str, field_name: str = "task_id") -> str:
    r"""
    Görev kimliğini doğrular: ^T-\d{4}$
    Uymayan değerler ValueError fırlatır (yol kaçışını yapısal engeller).
    """
    if not task_id or not isinstance(task_id, str) or not TASK_ID_PATTERN.match(task_id):
        raise ValueError(
            f"Geçersiz {field_name}: '{task_id}'. "
            f"Değer '^T-\\d{{4}}$' kuralına uymalıdır (örn: 'T-0001')."
        )
    return task_id


def sanitize_rel_path(path: str) -> str:
    """
    Yol güvenliği doğrulaması yapar ve normalize edilmiş göreceli yol döner.
    Kök dışına çıkan ('..') ve mutlak ('/...') yolları reddeder.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Yol boş olamaz ve metin tipinde olmalıdır.")

    clean = path.replace("\\", "/").strip()
    if clean.startswith("/") or os.path.isabs(clean):
        raise ValueError(f"Yol güvenlik ihlali: Mutlak yol reddedildi: '{path}'")

    parts = clean.split("/")
    if ".." in parts:
        raise ValueError(f"Yol güvenlik ihlali: Üst dizin ('..') geçişi reddedildi: '{path}'")

    norm = os.path.normpath(clean).replace("\\", "/")
    if norm == "." or norm.startswith("..") or norm.startswith("/"):
        raise ValueError(f"Yol güvenlik ihlali: Geçersiz göreceli yol: '{path}'")

    # Dizin ise sondaki slash'ı koru
    if clean.endswith("/") and not norm.endswith("/"):
        norm += "/"

    return norm


def path_to_slug(rel_path: str) -> str:
    """
    Yolu dosya adına (slug) dönüştürür: '/' -> '__' ve '.' -> '_'.
    Örnek: 'data/realistic_rag/val.jsonl' -> 'data__realistic_rag__val_jsonl'
    """
    clean = rel_path.strip("/")
    return clean.replace("/", "__").replace(".", "_")


def now_utc_iso() -> str:
    """ISO 8601 UTC formatında geçerli zaman damgasını üretir."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_datetime(dt_str: str) -> Optional[datetime]:
    """ISO 8601 zaman dizgisini datetime nesnesine çevirir."""
    if not dt_str or not isinstance(dt_str, str):
        return None
    try:
        clean = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        return None


def atomic_write_json(file_path: str, data: Any) -> None:
    """
    Atomik JSON yazma: Önce <dosya>.tmp yazar, ardından os.replace ile taşır.
    Böylece yarım/bozuk dosya gözlenmesi imkânsız kılınır.
    """
    dir_name = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = f"{file_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise e


def resolve_repo_root(cli_root: Optional[str] = None) -> str:
    """
    Kök dizin çözümlemesi:
    Öncelik sırası: --repo-root (CLI) > AGENT_BUS_ROOT (Ortam değişkeni) > os.getcwd()
    Kökte '.agent-bus/' dizini bulunmazsa stderr'e uyarı loglar.
    """
    if cli_root and cli_root.strip():
        chosen_root = cli_root.strip()
    elif os.environ.get("AGENT_BUS_ROOT", "").strip():
        chosen_root = os.environ["AGENT_BUS_ROOT"].strip()
    else:
        chosen_root = os.getcwd()

    abs_root = os.path.abspath(chosen_root)
    bus_dir = os.path.join(abs_root, ".agent-bus")
    if not os.path.exists(bus_dir) or not os.path.isdir(bus_dir):
        sys.stderr.write(
            f"[agent-bus UYARI] Çözümlenen kök dizinde '.agent-bus/' bulunamadı: '{abs_root}'\n"
        )
        sys.stderr.flush()

    return abs_root


def extract_message_order(fname: str) -> int:
    """
    Dosya adındaki çakışma numarasını sayısal olarak çıkarır:
    <time>-<sender>.json -> 0
    <time>-<sender>-1.json -> 1
    <time>-<sender>-2.json -> 2
    Bu fonksiyon aynı saniye içindeki mesajların kronolojik sıralanmasını sağlar.
    """
    base = fname[:-5] if fname.endswith(".json") else fname
    m = re.search(r"-(\d+)$", base)
    if m:
        return int(m.group(1))
    return 0


# =====================================================================
# AGENT-BUS ÇEKİRDEK YÖNETİCİSİ (CORE BUS MANAGER)
# =====================================================================

class AgentBus:
    """
    Koordinasyon veri yolunun tüm dizin hiyerarşisini, değişmez kurallarını
    ve atomik durum yönetimini sağlayan saf kontrol sınıfı.
    """
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.bus_dir = os.path.join(self.repo_root, ".agent-bus")
        self.frozen_file = os.path.join(self.bus_dir, "frozen.json")
        self.state_dir = os.path.join(self.bus_dir, "state")
        self.tasks_dir = os.path.join(self.state_dir, "tasks")
        self.leases_dir = os.path.join(self.state_dir, "leases")
        self.results_dir = os.path.join(self.state_dir, "results")
        self.inbox_dir = os.path.join(self.state_dir, "inbox")
        self.log_dir = os.path.join(self.bus_dir, "log")
        self.events_file = os.path.join(self.log_dir, "events.jsonl")

    def ensure_directories(self) -> None:
        """Gerekli state ve log alt dizinlerini oluşturur."""
        for d in (self.tasks_dir, self.leases_dir, self.results_dir, self.inbox_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)

    def log_event(self, event_type: str, actor: str, payload: Dict[str, Any]) -> None:
        """Append-only denetim kaydı ekler (log/events.jsonl)."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            record = {
                "timestamp": now_utc_iso(),
                "event": event_type,
                "actor": actor,
                "payload": payload
            }
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            sys.stderr.write(f"[agent-bus LOG ERROR] Olay kaydedilemedi: {e}\n")

    def load_frozen_patterns(self) -> List[str]:
        """frozen.json'dan değiştirilemez yol desenlerini okur."""
        if not os.path.exists(self.frozen_file):
            return []
        try:
            with open(self.frozen_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("patterns", [])
        except Exception as e:
            sys.stderr.write(f"[agent-bus ERROR] frozen.json okunamadı: {e}\n")
            return []

    def is_path_frozen(self, rel_path: str, frozen_patterns: List[str]) -> bool:
        """Belirtilen göreceli yolun donmuş bir desene uyup uymadığını kontrol eder."""
        norm_p = rel_path.strip("/")
        for pat in frozen_patterns:
            norm_pat = pat.strip()
            if "**" in norm_pat:
                prefix = norm_pat.split("**")[0].rstrip("/")
                if norm_p == prefix or norm_p.startswith(prefix + "/"):
                    return True
            elif fnmatch.fnmatch(norm_p, norm_pat):
                return True
        return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Görev nesnesini okur (state/tasks/ veya .agent-bus/tasks/)."""
        if not task_id or not isinstance(task_id, str) or not TASK_ID_PATTERN.match(task_id):
            return None
        candidates = [
            os.path.join(self.tasks_dir, f"{task_id}.json"),
            os.path.join(self.bus_dir, "tasks", f"{task_id}.json")
        ]
        for c in candidates:
            if os.path.exists(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    sys.stderr.write(f"[agent-bus ERROR] Görev dosyası okunamadı ({c}): {e}\n")
        return None

    def get_task_file_path(self, task_id: str) -> str:
        """Görev dosyasının birincil yazma yolunu döner."""
        validate_task_id(task_id, "task_id")
        legacy = os.path.join(self.bus_dir, "tasks", f"{task_id}.json")
        if os.path.exists(legacy):
            return legacy
        return os.path.join(self.tasks_dir, f"{task_id}.json")

    # -----------------------------------------------------------------
    # ARAÇ 1: bus_post_task
    # -----------------------------------------------------------------
    def post_task(
        self,
        title: str,
        spec: str,
        writes: Optional[List[str]] = None,
        acceptance: Optional[List[str]] = None,
        to: Optional[str] = None,
        from_agent: str = "claude",
        ttl_minutes: int = 120
    ) -> Dict[str, Any]:
        self.ensure_directories()
        validate_agent_name(from_agent, "from_agent")
        if to is not None and to != "":
            validate_agent_name(to, "to")

        clean_writes = [sanitize_rel_path(w) for w in (writes or [])]
        clean_acceptance = [str(a) for a in (acceptance or [])]

        # Sonraki ID'yi belirle (T-XXXX)
        max_num = 0
        search_dirs = [self.tasks_dir, os.path.join(self.bus_dir, "tasks")]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.startswith("T-") and fname.endswith(".json"):
                        part = fname[2:-5]
                        if part.isdigit():
                            max_num = max(max_num, int(part))

        next_id = f"T-{max_num + 1:04d}"
        task_data = {
            "id": next_id,
            "from": from_agent,
            "to": to,
            "created": now_utc_iso(),
            "title": title,
            "spec": spec,
            "writes": clean_writes,
            "acceptance": clean_acceptance,
            "status": "open",
            "claimed_by": None,
            "claimed_at": None,
            "ttl_minutes": int(ttl_minutes)
        }

        task_path = os.path.join(self.tasks_dir, f"{next_id}.json")
        atomic_write_json(task_path, task_data)
        self.log_event("task_posted", from_agent, {"id": next_id, "title": title})
        return {"id": next_id}

    # -----------------------------------------------------------------
    # ARAÇ 2: bus_list_tasks
    # -----------------------------------------------------------------
    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        self.ensure_directories()
        tasks_map: Dict[str, Dict[str, Any]] = {}
        search_dirs = [self.tasks_dir, os.path.join(self.bus_dir, "tasks")]

        for sdir in search_dirs:
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.startswith("T-") and fname.endswith(".json"):
                        fpath = os.path.join(sdir, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                t = json.load(f)
                                tid = t.get("id", fname[:-5])
                                if tid not in tasks_map:
                                    tasks_map[tid] = t
                        except Exception as e:
                            sys.stderr.write(f"[agent-bus ERROR] Görev okuma hatası: {e}\n")

        results = list(tasks_map.values())
        if status:
            results = [t for t in results if t.get("status") == status]

        results.sort(key=lambda x: str(x.get("id", "")))
        return results

    # -----------------------------------------------------------------
    # ARAÇ 3: bus_claim_task
    # -----------------------------------------------------------------
    def claim_task(self, task_id: str, owner: str) -> Dict[str, Any]:
        self.ensure_directories()
        validate_task_id(task_id, "id")
        validate_agent_name(owner, "owner")

        task = self.get_task(task_id)
        if not task:
            return {"ok": False, "conflicts": [f"Görev bulunamadı: '{task_id}'"]}

        cur_status = task.get("status", "open")
        claimed_by = task.get("claimed_by")
        claimed_at_str = task.get("claimed_at")
        ttl_min = int(task.get("ttl_minutes", 120))

        # Tersine dönüş yok: 'done' tekrar açılamaz
        if cur_status == "done":
            return {"ok": False, "conflicts": [f"Görev '{task_id}' tamamlanmış (done) durumdadır, tekrar devralınamaz."]}

        now_dt = datetime.now(timezone.utc)
        is_expired = False
        if claimed_at_str:
            claimed_dt = parse_iso_datetime(claimed_at_str)
            if claimed_dt and now_dt > (claimed_dt + timedelta(minutes=ttl_min)):
                is_expired = True

        if cur_status == "claimed" and not is_expired and claimed_by != owner:
            return {"ok": False, "conflicts": [f"Görev '{task_id}' şu anda '{claimed_by}' tarafından kiralanmıştır (claimed)."]}

        # Görevi sahiplen
        task["status"] = "claimed"
        task["claimed_by"] = owner
        task["claimed_at"] = now_utc_iso()

        target_file = self.get_task_file_path(task_id)
        atomic_write_json(target_file, task)

        event_type = "task_claim_takeover" if is_expired else "task_claimed"
        self.log_event(event_type, owner, {"id": task_id, "expired_previous_owner": claimed_by if is_expired else None})
        return {"ok": True, "conflicts": []}

    # -----------------------------------------------------------------
    # ARAÇ 4: bus_acquire_lease
    # -----------------------------------------------------------------
    def acquire_lease(
        self,
        paths: List[str],
        task_id: str,
        owner: str,
        ttl_minutes: int = 120,
        scope: Optional[str] = None
    ) -> Dict[str, Any]:
        self.ensure_directories()
        validate_task_id(task_id, "task_id")
        validate_agent_name(owner, "owner")

        if scope is not None and scope not in ("file", "dir"):
            return {
                "ok": False,
                "conflicts": [{
                    "path": "",
                    "reason": f"Geçersiz kiralama kapsamı (scope): '{scope}'. 'file' veya 'dir' olmalıdır."
                }]
            }

        if not paths:
            return {"ok": False, "conflicts": [{"path": "", "reason": "Kiralama için en az bir yol belirtilmelidir."}]}

        # 1. Yolları doğrula ve normalize et; effective_scope'u GERÇEKLİKTEN türet
        # KURAL (T-0004):
        #   if os.path.exists(abs_p):
        #       fs_scope = "dir" if os.path.isdir(abs_p) else "file"  # var olan hedef: dosya sistemi kazanır
        #   else:
        #       fs_scope = "dir" if norm_p.endswith("/") else "file"  # var olmayan yol: '/' niyet beyanıdır
        # DÜZELTME 2: Yol kimliğini tekilleştir (var olan dosya için sondaki '/' kaldırılır).
        validated_paths: List[Tuple[str, str]] = []
        conflicts: List[Dict[str, Any]] = []

        try:
            for p in paths:
                norm_p = sanitize_rel_path(p)
                abs_p = os.path.join(self.repo_root, norm_p)

                # DÜZELTME 1: Kapsamı GERÇEKLİKTEN türet, şekilden değil.
                clean_abs = abs_p.rstrip("/")
                if os.path.exists(clean_abs):
                    fs_scope = "dir" if os.path.isdir(clean_abs) else "file"
                else:
                    fs_scope = "dir" if norm_p.endswith("/") else "file"

                # DÜZELTME 2: Yol kimliğini tekilleştir (canonical path)
                if fs_scope == "file":
                    norm_p = norm_p.rstrip("/")
                else:
                    if not norm_p.endswith("/"):
                        norm_p += "/"

                # Çağıranın verdiği scope yalnızca DARALTABİLİR, genişletemez:
                if scope == "dir" and fs_scope == "file":
                    conflicts.append({
                        "path": norm_p,
                        "reason": f"Kapsam genişletme reddedildi: '{norm_p}' dosya sisteminde bir dosyadır (file); 'scope=dir' ile genişletilemez. Kapsam dosya sistemine bağlıdır."
                    })
                    continue

                if scope == "file":
                    effective_scope = "file"  # Dizini dosya gibi daraltma kabul edilir
                else:
                    effective_scope = fs_scope

                validated_paths.append((norm_p, effective_scope))
        except ValueError as ve:
            return {"ok": False, "conflicts": [{"path": str(paths), "reason": str(ve)}]}

        if conflicts:
            return {"ok": False, "conflicts": conflicts}

        # 2. Donmuş artefakt denetimi (Kural 2: (a) writes içinde olmalı, (b) effective_scope: 'dir' olmalı)
        frozen_patterns = self.load_frozen_patterns()
        task = self.get_task(task_id)
        task_writes = [sanitize_rel_path(w) for w in (task.get("writes", []) if task else [])]

        for norm_p, effective_scope in validated_paths:
            if self.is_path_frozen(norm_p, frozen_patterns):
                # Şart (a): Görevin writes listesinde bildirilmiş olmalı
                if not task:
                    conflicts.append({
                        "path": norm_p,
                        "reason": f"Yol '{norm_p}' donmuş (frozen.json) statüsündedir; geçerli bir task_id olmadan kiralanamaz."
                    })
                    continue

                is_covered = False
                clean_p = norm_p.strip("/")
                for w in task_writes:
                    clean_w = w.strip("/")
                    if clean_p == clean_w or clean_p.startswith(clean_w + "/"):
                        is_covered = True
                        break
                    if "**" in clean_w:
                        prefix = clean_w.split("**")[0].rstrip("/")
                        if clean_p == prefix or clean_p.startswith(prefix + "/"):
                            is_covered = True
                            break
                    elif fnmatch.fnmatch(clean_p, clean_w):
                        is_covered = True
                        break

                if not is_covered:
                    conflicts.append({
                        "path": norm_p,
                        "reason": f"Yol '{norm_p}' donmuş (frozen.json) statüsündedir ancak '{task_id}' görevinin 'writes' listesinde bildirilmemiştir: {task_writes}"
                    })
                    continue

                # Şart (b): Donmuş yola dosya kapsamlı kiralama verilemez; effective_scope 'dir' olmalıdır (Kural 2b)
                if effective_scope != "dir":
                    conflicts.append({
                        "path": norm_p,
                        "reason": f"Yol '{norm_p}' donmuş (frozen.json) statüsündedir; donmuş bir yola tek dosya kiralanamaz (kiralama kapsamı 'dir' olmalıdır). Lütfen ilgili dizini kiralayınız."
                    })
                    continue

        # 3. Mevcut aktif kiralamalarla çakışma denetimi
        now_dt = datetime.now(timezone.utc)
        existing_leases = self.list_leases()

        for norm_p, effective_scope in validated_paths:
            clean_p = norm_p.strip("/")
            for ex in existing_leases:
                ex_path = ex.get("path", "").strip("/")
                ex_owner = ex.get("owner")
                ex_expires = parse_iso_datetime(ex.get("expires", ""))

                # Süresi dolmuş kiralama çakışma üretmez (devralınabilir)
                if ex_expires and now_dt > ex_expires:
                    continue

                # Aynı sahibe aitse çakışma yok (yenileme/uzatma)
                if ex_owner == owner:
                    continue

                # Farklı sahip -> yol örtüşmesi kontrolü
                ex_scope = ex.get("scope", "file")
                is_overlap = False

                if clean_p == ex_path:
                    is_overlap = True
                elif ex_scope == "dir" and clean_p.startswith(ex_path + "/"):
                    is_overlap = True
                elif effective_scope == "dir" and ex_path.startswith(clean_p + "/"):
                    is_overlap = True

                if is_overlap:
                    conflicts.append({
                        "path": norm_p,
                        "conflicting_lease": ex,
                        "reason": f"Aktif kiralama '{ex_owner}' tarafından tutulmaktadır (bitiş: {ex.get('expires')})."
                    })

        # DEĞİŞMEZ KURAL: Çakışma varsa KISMİ BAŞARI YOKTUR! Diske HİÇBİR dosya yazılmaz.
        if conflicts:
            return {"ok": False, "conflicts": conflicts}

        # 4. Çakışma yok -> Tüm kiralamaları atomik olarak diske yaz
        acquired_iso = now_utc_iso()
        expires_dt = now_dt + timedelta(minutes=int(ttl_minutes))
        expires_iso = expires_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        for norm_p, effective_scope in validated_paths:
            slug = path_to_slug(norm_p)
            lease_file = os.path.join(self.leases_dir, f"{slug}.json")
            lease_data = {
                "path": norm_p,
                "scope": effective_scope,
                "task_id": task_id,
                "owner": owner,
                "pid": str(os.getpid()),
                "acquired": acquired_iso,
                "expires": expires_iso
            }
            atomic_write_json(lease_file, lease_data)
            self.log_event("lease_acquired", owner, lease_data)

        return {"ok": True, "conflicts": []}

    # -----------------------------------------------------------------
    # ARAÇ 5: bus_release_lease
    # -----------------------------------------------------------------
    def release_lease(self, paths: List[str], task_id: str) -> Dict[str, Any]:
        self.ensure_directories()
        validate_task_id(task_id, "task_id")
        released_paths = []
        for p in paths:
            try:
                norm_p = sanitize_rel_path(p)
                abs_p = os.path.join(self.repo_root, norm_p)
                clean_abs = abs_p.rstrip("/")
                if os.path.exists(clean_abs):
                    is_dir = os.path.isdir(clean_abs)
                else:
                    is_dir = norm_p.endswith("/")
                if is_dir:
                    if not norm_p.endswith("/"):
                        norm_p += "/"
                else:
                    norm_p = norm_p.rstrip("/")
            except ValueError:
                continue
            slug = path_to_slug(norm_p)
            lease_file = os.path.join(self.leases_dir, f"{slug}.json")
            if os.path.exists(lease_file):
                try:
                    with open(lease_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # task_id uyumu gözetilir
                    if data.get("task_id") == task_id or not task_id:
                        os.remove(lease_file)
                        released_paths.append(norm_p)
                        self.log_event("lease_released", data.get("owner", "unknown"), {"path": norm_p, "task_id": task_id})
                except Exception as e:
                    sys.stderr.write(f"[agent-bus ERROR] Kiralama silinemedi ({lease_file}): {e}\n")

        return {"ok": True, "released": released_paths}

    # -----------------------------------------------------------------
    # ARAÇ 6: bus_lease_status
    # -----------------------------------------------------------------
    def list_leases(self, paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self.ensure_directories()
        leases = []
        if os.path.exists(self.leases_dir):
            for fname in os.listdir(self.leases_dir):
                if fname.endswith(".json") and not fname.endswith(".tmp"):
                    fpath = os.path.join(self.leases_dir, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            leases.append(data)
                    except Exception as e:
                        sys.stderr.write(f"[agent-bus ERROR] Kiralama dosyası okunamadı ({fname}): {e}\n")

        if paths:
            filter_set = set()
            filter_clean = set()
            for p in paths:
                try:
                    norm_p = sanitize_rel_path(p)
                    abs_p = os.path.join(self.repo_root, norm_p)
                    clean_abs = abs_p.rstrip("/")
                    if os.path.exists(clean_abs):
                        is_dir = os.path.isdir(clean_abs)
                    else:
                        is_dir = norm_p.endswith("/")
                    if is_dir:
                        if not norm_p.endswith("/"):
                            norm_p += "/"
                    else:
                        norm_p = norm_p.rstrip("/")
                    filter_set.add(norm_p)
                    filter_clean.add(norm_p.strip("/"))
                except ValueError:
                    pass
            leases = [
                l for l in leases
                if l.get("path", "") in filter_set or l.get("path", "").strip("/") in filter_clean
            ]

        leases.sort(key=lambda x: str(x.get("path", "")))
        return leases

    # -----------------------------------------------------------------
    # ARAÇ 7: bus_report_result
    # -----------------------------------------------------------------
    def report_result(
        self,
        task_id: str,
        status: str,
        summary: str,
        evidence: Optional[List[str]] = None,
        changed_files: Optional[List[str]] = None,
        narrative_log: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.ensure_directories()
        validate_task_id(task_id, "task_id")
        if status not in ("done", "blocked"):
            raise ValueError(f"Geçersiz görev sonucu statüsü: '{status}'. 'done' veya 'blocked' olmalıdır.")

        clean_narrative = None
        if narrative_log is not None:
            if not isinstance(narrative_log, dict):
                raise ValueError("narrative_log bir sözlük (dict) olmalıdır.")
            if "path" not in narrative_log or "sha256" not in narrative_log or "ozet" not in narrative_log:
                raise ValueError("narrative_log 'path', 'sha256' ve 'ozet' alanlarını içermelidir.")
            clean_path = sanitize_rel_path(str(narrative_log["path"]))
            clean_narrative = {
                "path": clean_path,
                "sha256": str(narrative_log["sha256"]),
                "ozet": str(narrative_log["ozet"])
            }

        result_data: Dict[str, Any] = {
            "task_id": task_id,
            "status": status,
            "finished": now_utc_iso(),
            "summary": summary,
            "evidence": evidence or [],
            "changed_files": [sanitize_rel_path(f) for f in (changed_files or [])]
        }
        if clean_narrative is not None:
            result_data["narrative_log"] = clean_narrative

        res_path = os.path.join(self.results_dir, f"{task_id}.json")
        atomic_write_json(res_path, result_data)

        # Görev dosyasındaki durumu da güncelle
        task = self.get_task(task_id)
        if task:
            task["status"] = status
            target_task_file = self.get_task_file_path(task_id)
            atomic_write_json(target_task_file, task)

        self.log_event("result_reported", task.get("claimed_by", "unknown") if task else "unknown", result_data)
        return {"ok": True}

    # -----------------------------------------------------------------
    # ARAÇ 8: bus_send
    # -----------------------------------------------------------------
    def send_message(
        self,
        to: str,
        subject: str,
        content: str,
        from_agent: str = "antigravity"
    ) -> Dict[str, Any]:
        self.ensure_directories()
        validate_agent_name(to, "to")
        validate_agent_name(from_agent, "from_agent")

        recipient_dir = os.path.join(self.inbox_dir, to)
        os.makedirs(recipient_dir, exist_ok=True)

        sent_iso = now_utc_iso()
        safe_time = sent_iso.replace(":", "-")
        base_fname = f"{safe_time}-{from_agent}.json"
        base_path = os.path.join(recipient_dir, base_fname)

        # SPEC.md: Aynı saniyede birden fazla mesaj durumunda çakışmayı önle (<ISO8601>-<sender>-<n>.json)
        if os.path.exists(base_path):
            n = 1
            while True:
                cand_fname = f"{safe_time}-{from_agent}-{n}.json"
                cand_path = os.path.join(recipient_dir, cand_fname)
                if not os.path.exists(cand_path):
                    fname = cand_fname
                    msg_path = cand_path
                    break
                n += 1
        else:
            fname = base_fname
            msg_path = base_path

        msg_data = {
            "from": from_agent,
            "to": to,
            "sent": sent_iso,
            "subject": subject,
            "content": content,
            "read": False
        }
        atomic_write_json(msg_path, msg_data)
        self.log_event("message_sent", from_agent, {"to": to, "subject": subject, "file": fname})
        return {"ok": True, "file": fname}

    # -----------------------------------------------------------------
    # ARAÇ 9: bus_inbox
    # -----------------------------------------------------------------
    def read_inbox(self, who: str, unread_only: bool = False) -> List[Dict[str, Any]]:
        self.ensure_directories()
        validate_agent_name(who, "who")

        recipient_dir = os.path.join(self.inbox_dir, who)
        if not os.path.exists(recipient_dir):
            return []

        messages = []
        for fname in os.listdir(recipient_dir):
            if fname.endswith(".json") and not fname.endswith(".tmp"):
                fpath = os.path.join(recipient_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["_filename"] = fname
                        if unread_only and data.get("read") is True:
                            continue
                        messages.append(data)
                except Exception as e:
                    sys.stderr.write(f"[agent-bus ERROR] Mesaj okuma hatası ({fname}): {e}\n")

        # Sıralama: sent damgası, ardından dosya adındaki sayısal çakışma indeksi (kronolojik düzen)
        messages.sort(key=lambda x: (
            str(x.get("sent", "")),
            extract_message_order(str(x.get("_filename", ""))),
            str(x.get("_filename", ""))
        ))
        return messages

    # -----------------------------------------------------------------
    # ARAÇ 10: bus_frozen_list
    # -----------------------------------------------------------------
    def frozen_list(self) -> List[str]:
        return self.load_frozen_patterns()


# =====================================================================
# MCP PROTOKOL TAŞIYICISI (JSON-RPC 2.0 OVER STDIO)
# =====================================================================

TOOL_DEFINITIONS = [
    {
        "name": "bus_post_task",
        "description": "Yeni bir ajan koordinasyon görevi oluşturur (state/tasks/T-XXXX.json).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Görevin kısa başlığı"},
                "spec": {"type": "string", "description": "Görevin tam teknik şartnamesi"},
                "writes": {"type": "array", "items": {"type": "string"}, "description": "Görevin değiştireceği dosya ve dizinler"},
                "acceptance": {"type": "array", "items": {"type": "string"}, "description": "Kabul kriteri doğrulama adımları"},
                "to": {"type": "string", "description": "Hedef ajan (antigravity veya claude)"},
                "from_agent": {"type": "string", "description": "Görevi açan ajan (varsayılan: claude)"},
                "ttl_minutes": {"type": "integer", "description": "Görevin yaşam süresi (varsayılan: 120)"}
            },
            "required": ["title", "spec"]
        }
    },
    {
        "name": "bus_list_tasks",
        "description": "Mevcut görevleri listeler, istenirse duruma göre filtreler.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["open", "claimed", "done", "blocked"], "description": "Filtrelenecek durum"}
            }
        }
    },
    {
        "name": "bus_claim_task",
        "description": "Açık bir görevi sahiplenir (claimed) ve TTL sayacını başlatır.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Görev ID (örn: T-0001)"},
                "owner": {"type": "string", "description": "Görevi devralan ajan (antigravity veya claude)"}
            },
            "required": ["id", "owner"]
        }
    },
    {
        "name": "bus_acquire_lease",
        "description": "Bir veya daha fazla dosya/dizin üzerine atomik kiralama alır. Kiralama kapsamı dosya sisteminden türetilir; donmuş yollarda tek dosya kiralanamaz, dizin kiralanmalıdır.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Kiralanacak göreceli dosya/dizin yolları"},
                "task_id": {"type": "string", "description": "Kiralamanın bağlı olduğu görev ID (örn: T-0001)"},
                "owner": {"type": "string", "description": "Kiralama sahibi ajan"},
                "ttl_minutes": {"type": "integer", "description": "Kiralama süresi (dakika, varsayılan: 120)"},
                "scope": {"type": "string", "enum": ["file", "dir"], "description": "İsteğe bağlı kiralama kapsamı (yalnızca daraltma amaçlı). Kapsam dosya sisteminden türetilir; dosyalara 'dir' verilemez, donmuş yollar için dizin yolu kiralanmalıdır."}
            },
            "required": ["paths", "task_id", "owner"]
        }
    },
    {
        "name": "bus_release_lease",
        "description": "Daha önce alınmış kiralamaları serbest bırakır ve kiralama dosyalarını siler.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Serbest bırakılacak dosya yolları"},
                "task_id": {"type": "string", "description": "Görevin ID'si"}
            },
            "required": ["paths", "task_id"]
        }
    },
    {
        "name": "bus_lease_status",
        "description": "Aktif kiralamaların listesini döner.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Filtrelenecek yollar"}
            }
        }
    },
    {
        "name": "bus_report_result",
        "description": "Görevin sonucunu raporlar (state/results/T-XXXX.json) ve görev durumunu günceller.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Tamamlanan görev ID"},
                "status": {"type": "string", "enum": ["done", "blocked"], "description": "Nihai durum"},
                "summary": {"type": "string", "description": "Yapılan işin özeti"},
                "evidence": {"type": "array", "items": {"type": "string"}, "description": "Kanıtlar ve komut çıktıları"},
                "changed_files": {"type": "array", "items": {"type": "string"}, "description": "Değiştirilen dosya yolları"},
                "narrative_log": {
                    "type": "object",
                    "description": "Opsiyonel yürütücü anlatı günlüğü (.agent-bus/notes/T-XXXX.md) işaretçisi",
                    "properties": {
                        "path": {"type": "string", "description": "Not dosyasının göreceli yolu (.agent-bus/notes/T-XXXX.md)"},
                        "sha256": {"type": "string", "description": "Not dosyasının SHA-256 özeti"},
                        "ozet": {"type": "string", "description": "Yürütümün kısa özeti"}
                    },
                    "required": ["path", "sha256", "ozet"]
                }
            },
            "required": ["task_id", "status", "summary"]
        }
    },
    {
        "name": "bus_send",
        "description": "Diğer ajana asenkron mesaj gönderir (state/inbox/<alıcı>/).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Hedef ajan (antigravity veya claude)"},
                "subject": {"type": "string", "description": "Mesaj konusu"},
                "content": {"type": "string", "description": "Mesaj içeriği"},
                "from_agent": {"type": "string", "description": "Gönderen ajan"}
            },
            "required": ["to", "subject", "content"]
        }
    },
    {
        "name": "bus_inbox",
        "description": "Ajana gelen gelen kutusu mesajlarını listeler.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "who": {"type": "string", "description": "Mesajları okunacak ajan adı"},
                "unread_only": {"type": "boolean", "description": "Yalnızca okunmamış mesajları getir"}
            },
            "required": ["who"]
        }
    },
    {
        "name": "bus_frozen_list",
        "description": "frozen.json dosyasındaki değiştirilemez yol desenlerini döner.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def validate_tool_arguments(name: str, args: Dict[str, Any]) -> None:
    """Aracın girdi argümanlarını şemadaki 'required' listesine göre denetler."""
    for tool_def in TOOL_DEFINITIONS:
        if tool_def["name"] == name:
            schema = tool_def.get("inputSchema", {})
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for field in required:
                if field not in args or args[field] is None:
                    field_info = props.get(field, {})
                    field_type = field_info.get("type", "değer")
                    field_desc = field_info.get("description", "")
                    desc_str = f" ({field_desc})" if field_desc else ""
                    raise ValueError(f"Eksik zorunlu parametre: '{field}'. Beklenen biçim: {field_type}{desc_str}")
            break


def execute_tool_call(bus: AgentBus, name: str, args: Dict[str, Any]) -> Any:
    """MCP tool çağrısını ilgili AgentBus metoduna yönlendirir."""
    if not isinstance(args, dict):
        args = {}
    validate_tool_arguments(name, args)

    if name == "bus_post_task":
        return bus.post_task(
            title=args["title"],
            spec=args["spec"],
            writes=args.get("writes"),
            acceptance=args.get("acceptance"),
            to=args.get("to"),
            from_agent=args.get("from_agent", "claude"),
            ttl_minutes=args.get("ttl_minutes", 120)
        )
    elif name == "bus_list_tasks":
        return bus.list_tasks(status=args.get("status"))
    elif name == "bus_claim_task":
        return bus.claim_task(task_id=args["id"], owner=args["owner"])
    elif name == "bus_acquire_lease":
        return bus.acquire_lease(
            paths=args["paths"],
            task_id=args["task_id"],
            owner=args["owner"],
            ttl_minutes=args.get("ttl_minutes", 120),
            scope=args.get("scope")
        )
    elif name == "bus_release_lease":
        return bus.release_lease(paths=args["paths"], task_id=args["task_id"])
    elif name == "bus_lease_status":
        return bus.list_leases(paths=args.get("paths"))
    elif name == "bus_report_result":
        return bus.report_result(
            task_id=args["task_id"],
            status=args["status"],
            summary=args["summary"],
            evidence=args.get("evidence"),
            changed_files=args.get("changed_files"),
            narrative_log=args.get("narrative_log")
        )
    elif name == "bus_send":
        return bus.send_message(
            to=args["to"],
            subject=args["subject"],
            content=args["content"],
            from_agent=args.get("from_agent", "antigravity")
        )
    elif name == "bus_inbox":
        return bus.read_inbox(who=args["who"], unread_only=args.get("unread_only", False))
    elif name == "bus_frozen_list":
        return bus.frozen_list()
    else:
        raise ValueError(f"Bilinmeyen MCP aracı: '{name}'")


def handle_jsonrpc(bus: AgentBus, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Tek bir JSON-RPC 2.0 isteğini işler."""
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agent-bus", "version": "1.0.0"}
            }
        }
    elif method == "notifications/initialized":
        return None
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS}
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}
        try:
            output = execute_tool_call(bus, tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(output, ensure_ascii=False, indent=2)}
                    ]
                }
            }
        except KeyError as ke:
            missing = ke.args[0] if ke.args else "bilinmeyen"
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Hata: Eksik zorunlu parametre: '{missing}'. Lütfen araca ait gerekli tüm alanları belirtiniz."}]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Hata: {str(e)}"}]
                }
            }
    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"}
            }
        return None


def run_stdio_server(repo_root: str) -> None:
    """Standart girdi/çıktı (stdio) üzerinden JSON-RPC 2.0 dinler."""
    bus = AgentBus(repo_root)
    sys.stderr.write(f"[agent-bus] MCP Stdio Sunucusu başlatıldı (kök: {repo_root})\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line_str = line.strip()
        if not line_str:
            continue
        try:
            req = json.loads(line_str)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_jsonrpc(bus, req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


# =====================================================================
# KABUL KRİTERİ VE ÇAPRAZ ÇARPIM TESTLERİ (--selftest)
# =====================================================================

def evaluate_matrix_cell_expectation(
    root_dir: str,
    target_rel_path: str,
    scope_arg: Optional[str],
    is_frozen: bool
) -> Tuple[bool, bool, Optional[str]]:
    """
    Çapraz çarpım hücresinin kuramsal beklenen sonucunu dosya sistemi gerçekliğinden türetir:
    (expected_ok, expected_lease_written, expected_scope)
    """
    abs_p = os.path.join(root_dir, target_rel_path.rstrip("/"))
    if os.path.exists(abs_p):
        fs_scope = "dir" if os.path.isdir(abs_p) else "file"
    else:
        fs_scope = "dir" if target_rel_path.endswith("/") else "file"

    # 2. Kapsam genişletme kuralı:
    if scope_arg == "dir" and fs_scope == "file":
        return False, False, None

    # 3. effective_scope belirle:
    if scope_arg == "file":
        eff_scope = "file"
    else:
        eff_scope = fs_scope

    # 4. Donmuş artefakt kuralı (Kural 2b):
    if is_frozen:
        if eff_scope != "dir":
            return False, False, None
        return True, True, "dir"

    # 5. Donmamış hedef:
    return True, True, eff_scope


def run_selftest(real_repo_root: Optional[str] = None) -> bool:
    """
    SPEC.md ve T-0001..T-0004 kabul kriterlerini harfiyen test eder (11 Adım):
    Adım 1: bus_post_task -> Görev açılır (T-XXXX)
    Adım 2: bus_acquire_lease -> Donmuş dizine dir-scope kiralama kabul edilir (ok: true)
    Adım 3: Donmuş OLMAYAN yolda çift kiralama -> Farklı owner ile ok: false, conflicts yalnızca 'Aktif kiralama'
    Adım 4: bus_report_result -> results/T-XXXX.json yazılır (ok: true)
    Adım 5: bus_release_lease -> kiralama silinir (ok: true)
    Adım 6: Kusur 4 Doğrulaması: Gerçek .agent-bus/frozen.json desen sayısı == 10
    Adım 7: Kusur 1 Doğrulaması: Yol kaçışı engelleme (to='../k', task_id='../../k' reddedildi)
    Adım 8: Kusur 3 Doğrulaması: Aynı saniyede iki mesaj -> gelen kutusunda kronolojik düzen (M1, M2)
    Adım 9: T-0003 & T-0004 -> Donmuş DOSYAYA ve dosya+'/' kiralama reddedildi (üç scope değerinde de)
    Adım 10: T-0003 -> Donmuş dizin kiralandı, altındaki dosyayı kapsadı, events.jsonl denetlendi
    Adım 11: T-0004 Çapraz Çarpım Matrisi -> 36 Hücrenin tamamı tablo-güdümlü çalıştırıldı ve doğrulandı
    """
    print("=================================================================", file=sys.stderr)
    print("   agent-bus MCP: 11 ADIMLI KABUL KRİTERİ VE MATRİS TESTİ        ", file=sys.stderr)
    print("=================================================================", file=sys.stderr)

    if real_repo_root is None:
        real_repo_root = resolve_repo_root(None)

    # İzolasyon için geçici dizinde bağımsız repo ortamı kur (git çalışma alanını kirletmez)
    with tempfile.TemporaryDirectory() as temp_root:
        # Mock repo yapısı oluştur
        bus_dir = os.path.join(temp_root, ".agent-bus")
        os.makedirs(bus_dir, exist_ok=True)
        data_dir = os.path.join(temp_root, "data", "realistic_rag")
        os.makedirs(data_dir, exist_ok=True)
        val_file = os.path.join(data_dir, "val.jsonl")
        with open(val_file, "w", encoding="utf-8") as f:
            f.write('{"sample": "test"}\n')

        # Uzantısız dosya fikstürleri (T-0006 Düzeltme 2)
        val_noext = os.path.join(data_dir, "UZANTISIZ")
        with open(val_noext, "w", encoding="utf-8") as f:
            f.write("no extension content\n")

        # Mock scratch dizini oluştur
        scratch_dir = os.path.join(temp_root, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        scratch_file = os.path.join(scratch_dir, "x.txt")
        with open(scratch_file, "w", encoding="utf-8") as f:
            f.write("test scratch\n")

        scratch_noext = os.path.join(scratch_dir, "NOEXT")
        with open(scratch_noext, "w", encoding="utf-8") as f:
            f.write("scratch no extension content\n")

        # Matris testi için alt dizinler oluştur
        os.makedirs(os.path.join(data_dir, "subdir"), exist_ok=True)
        os.makedirs(os.path.join(scratch_dir, "mydir"), exist_ok=True)

        # frozen.json dosyasını taklit et
        frozen_data = {
            "version": 1,
            "patterns": ["data/realistic_rag/**", "scripts/*.py"]
        }
        with open(os.path.join(bus_dir, "frozen.json"), "w", encoding="utf-8") as f:
            json.dump(frozen_data, f)

        bus = AgentBus(temp_root)

        # -------------------------------------------------------------
        # ADIM 1: bus_post_task
        # -------------------------------------------------------------
        task_res = bus.post_task(
            title="Val Koruması Test Görevi",
            spec="val.jsonl dosyasının kiralama korumasını test et",
            writes=["data/realistic_rag/", "scratch/"],
            acceptance=["selftest pass"],
            to="antigravity",
            from_agent="claude"
        )
        task_id = task_res.get("id")
        step1_pass = bool(task_id and TASK_ID_PATTERN.match(task_id))
        print(f"[{'GEÇTİ' if step1_pass else 'KALDI'}] Adım 1: bus_post_task -> Görev açıldı (ID: {task_id})", file=sys.stderr)
        if not step1_pass:
            return False

        # -------------------------------------------------------------
        # ADIM 2: bus_acquire_lease (Donmuş dizine dir-scope kiralama: ok: true)
        # -------------------------------------------------------------
        acq_res1 = bus.acquire_lease(
            paths=["data/realistic_rag/"],
            task_id=task_id,
            owner="antigravity",
            scope="dir"
        )
        step2_pass = (acq_res1.get("ok") is True and len(acq_res1.get("conflicts", [])) == 0)
        slug = path_to_slug("data/realistic_rag/")
        lease_file_path = os.path.join(bus.leases_dir, f"{slug}.json")
        step2_pass = step2_pass and os.path.exists(lease_file_path)
        print(f"[{'GEÇTİ' if step2_pass else 'KALDI'}] Adım 2: bus_acquire_lease -> Donmuş dizine dir-scope kiralama kabul edildi (ok: true)", file=sys.stderr)
        if not step2_pass:
            sys.stderr.write(f"  Detay: {acq_res1}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 3: Donmuş OLMAYAN yolda çift kiralama (ok: false, conflicts yalnızca 'Aktif kiralama')
        # -------------------------------------------------------------
        # İlki ok: true
        acq_scratch1 = bus.acquire_lease(
            paths=["scratch/x.txt"],
            task_id=task_id,
            owner="antigravity"
        )
        if not acq_scratch1.get("ok"):
            print(f"[KALDI] Adım 3: scratch/x.txt ilk kiralama başarısız oldu: {acq_scratch1}", file=sys.stderr)
            return False

        files_before = set(os.listdir(bus.leases_dir))
        # İkincisi farklı owner ile -> ok: false
        acq_scratch2 = bus.acquire_lease(
            paths=["scratch/x.txt"],
            task_id=task_id,
            owner="claude"
        )
        files_after = set(os.listdir(bus.leases_dir))

        step3_ok_false = (acq_scratch2.get("ok") is False)
        confs = acq_scratch2.get("conflicts", [])
        step3_has_conf = (len(confs) == 1)
        conf_reason = confs[0].get("reason", "") if confs else ""
        step3_active_lease = ("Aktif kiralama" in conf_reason)
        step3_not_frozen = ("donmuş" not in conf_reason.lower() and "frozen" not in conf_reason.lower())
        step3_no_files = (files_before == files_after)

        step3_pass = step3_ok_false and step3_has_conf and step3_active_lease and step3_not_frozen and step3_no_files
        print(f"[{'GEÇTİ' if step3_pass else 'KALDI'}] Adım 3: Donmuş olmayan yolda çift kiralama -> ok: false, yalnızca 'Aktif kiralama' çakışması", file=sys.stderr)
        if not step3_pass:
            sys.stderr.write(f"  Detay: ok_false={step3_ok_false}, conflicts={confs}, active_lease={step3_active_lease}, not_frozen={step3_not_frozen}, no_files={step3_no_files}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 4: bus_report_result -> results/T-XXXX.json
        # -------------------------------------------------------------
        rep_res = bus.report_result(
            task_id=task_id,
            status="done",
            summary="Selftest başarıyla icra edildi.",
            evidence=["Adım 1-3 doğrulandı"],
            changed_files=["scripts/agent_bus_mcp.py"]
        )
        result_file = os.path.join(bus.results_dir, f"{task_id}.json")
        step4_pass = (rep_res.get("ok") is True and os.path.exists(result_file))
        print(f"[{'GEÇTİ' if step4_pass else 'KALDI'}] Adım 4: bus_report_result -> results/{task_id}.json yazıldı (ok: true)", file=sys.stderr)
        if not step4_pass:
            return False

        # -------------------------------------------------------------
        # ADIM 5: bus_release_lease -> kiralama silinir
        # -------------------------------------------------------------
        rel_res = bus.release_lease(paths=["data/realistic_rag/", "scratch/x.txt"], task_id=task_id)
        scratch_slug = path_to_slug("scratch/x.txt")
        scratch_lease_path = os.path.join(bus.leases_dir, f"{scratch_slug}.json")
        step5_pass = (rel_res.get("ok") is True and not os.path.exists(lease_file_path) and not os.path.exists(scratch_lease_path))
        print(f"[{'GEÇTİ' if step5_pass else 'KALDI'}] Adım 5: bus_release_lease -> kiralamalar silindi (ok: true)", file=sys.stderr)
        if not step5_pass:
            return False

        # -------------------------------------------------------------
        # ADIM 6: Kusur 4 Doğrulaması: Gerçek .agent-bus/frozen.json desen sayısı == 10
        # -------------------------------------------------------------
        real_bus = AgentBus(real_repo_root)
        real_patterns = real_bus.load_frozen_patterns()
        step6_pass = (len(real_patterns) == 10)
        print(f"[{'GEÇTİ' if step6_pass else 'KALDI'}] Adım 6: Gerçek .agent-bus/frozen.json -> {len(real_patterns)} desen yüklendi (beklenen: 10)", file=sys.stderr)
        if not step6_pass:
            sys.stderr.write(f"  Detay: Yüklenen desenler ({len(real_patterns)}): {real_patterns}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 7: Kusur 1 Doğrulaması: Yol kaçışı engelleme (to='../k', task_id='../../k' reddedildi)
        # -------------------------------------------------------------
        escape_to_blocked = False
        try:
            bus.send_message(to="../k", subject="kaçış", content="test")
        except ValueError:
            escape_to_blocked = True

        escape_task_blocked = False
        try:
            bus.report_result(task_id="../../k", status="done", summary="kaçış")
        except ValueError:
            escape_task_blocked = True

        # Kök dışına hiçbir dosya/dizin yazılmadığını doğrula
        leaked_inbox_dir = os.path.join(bus.state_dir, "k")
        leaked_res_file = os.path.join(bus.bus_dir, "k.json")
        no_leaks = (not os.path.exists(leaked_inbox_dir)) and (not os.path.exists(leaked_res_file))

        # Meşru çağrılar çalışıyor mu?
        legit_send_ok = False
        try:
            res_legit = bus.send_message(to="antigravity", subject="meşru", content="selam", from_agent="claude")
            legit_send_ok = res_legit.get("ok") is True
        except Exception:
            legit_send_ok = False

        step7_pass = escape_to_blocked and escape_task_blocked and no_leaks and legit_send_ok
        print(f"[{'GEÇTİ' if step7_pass else 'KALDI'}] Adım 7: Kusur 1 -> Yol kaçışı engellendi (to='../k', task_id='../../k' reddedildi, meşru çağrılar korundu)", file=sys.stderr)
        if not step7_pass:
            sys.stderr.write(f"  Detay: escape_to={escape_to_blocked}, escape_task={escape_task_blocked}, no_leaks={no_leaks}, legit={legit_send_ok}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 8: Kusur 3 Doğrulaması: Aynı saniyede iki mesaj -> kronolojik düzen
        # -------------------------------------------------------------
        # İki mesajı peş peşe gönder (aynı saniyede çakışma)
        m1 = bus.send_message(to="claude", subject="Mesaj 1", content="Birinci", from_agent="antigravity")
        m2 = bus.send_message(to="claude", subject="Mesaj 2", content="İkinci", from_agent="antigravity")

        inbox_msgs = bus.read_inbox(who="claude")
        subjects = [m.get("subject") for m in inbox_msgs]
        step8_two_msgs = (len(inbox_msgs) == 2)
        step8_has_both = ("Mesaj 1" in subjects and "Mesaj 2" in subjects)
        step8_distinct_files = (m1.get("file") != m2.get("file"))
        step8_order = (subjects == ["Mesaj 1", "Mesaj 2"])

        step8_pass = step8_two_msgs and step8_has_both and step8_distinct_files and step8_order
        print(f"[{'GEÇTİ' if step8_pass else 'KALDI'}] Adım 8: Kusur 3 -> Aynı saniyede iki mesaj gelen kutusunda kronolojik korundu (M1, sonra M2)", file=sys.stderr)
        if not step8_pass:
            sys.stderr.write(f"  Detay: m1={m1}, m2={m2}, inbox_count={len(inbox_msgs)}, subjects={subjects}, order_ok={step8_order}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 9: T-0003 & T-0004 -> Donmuş DOSYA ve dosya+'/' reddedildi
        # -------------------------------------------------------------
        t2_res = bus.post_task(
            title="Kural 2 Doğrulama Görevi",
            spec="Donmuş dosya kiralama testi",
            writes=["data/realistic_rag/val.jsonl"],
            to="antigravity",
            from_agent="claude"
        )
        t2_id = t2_res["id"]
        val_slug = path_to_slug("data/realistic_rag/val.jsonl")
        val_lease_file = os.path.join(bus.leases_dir, f"{val_slug}.json")
        leases_before = set(os.listdir(bus.leases_dir))

        # 6 Varyantın hepsi reddedilmeli: val.jsonl ve val.jsonl/ x (None, file, dir)
        all_file_attempts_rejected = True
        for test_path in ["data/realistic_rag/val.jsonl", "data/realistic_rag/val.jsonl/"]:
            for test_scope in [None, "file", "dir"]:
                res_attempt = bus.acquire_lease(
                    paths=[test_path],
                    task_id=t2_id,
                    owner="antigravity",
                    scope=test_scope
                )
                if res_attempt.get("ok") is not False:
                    all_file_attempts_rejected = False
                    sys.stderr.write(f"  [HATA] Donmuş dosya çağrısı hatalı şekilde kabul edildi: path={test_path}, scope={test_scope}\n")

        leases_after = set(os.listdir(bus.leases_dir))
        no_lease_created = (leases_before == leases_after) and (not os.path.exists(val_lease_file))

        step9_pass = all_file_attempts_rejected and no_lease_created
        print(f"[{'GEÇTİ' if step9_pass else 'KALDI'}] Adım 9: T-0003 & T-0004 -> Donmuş dosyaya ve dosya+'/' kiralama reddedildi (6 kombinasyonun hepsi, diskte dosya oluşmadı)", file=sys.stderr)
        if not step9_pass:
            return False

        # -------------------------------------------------------------
        # ADIM 10: T-0003 -> Donmuş dizin kiralandı, alt dosyayı kapsadı, events.jsonl denetlendi
        # -------------------------------------------------------------
        t3_res = bus.post_task(
            title="Dizin Kiralama Görevi",
            spec="data/realistic_rag/ dizin kiralama testi",
            writes=["data/realistic_rag/"],
            to="antigravity",
            from_agent="claude"
        )
        t3_id = t3_res["id"]

        # 1. Donmuş dizini kirala (meşru kullanım) -> KABUL EDİLMELİ
        acq_dir = bus.acquire_lease(
            paths=["data/realistic_rag/"],
            task_id=t3_id,
            owner="antigravity",
            scope="dir"
        )
        dir_accepted = (acq_dir.get("ok") is True and len(acq_dir.get("conflicts", [])) == 0)

        # 2. Başka bir ajan (claude) data/realistic_rag/val.jsonl kiralamaya çalışsın -> Dizin kiralaması örtüşmesiyle REDDEDİLMELİ
        t4_res = bus.post_task(
            title="Val Kiralama Denemesi",
            spec="Çakışma denemesi",
            writes=["data/realistic_rag/val.jsonl"],
            to="claude",
            from_agent="antigravity"
        )
        t4_id = t4_res["id"]
        acq_overlap = bus.acquire_lease(
            paths=["data/realistic_rag/val.jsonl"],
            task_id=t4_id,
            owner="claude"
        )
        overlap_rejected = (acq_overlap.get("ok") is False)
        overlap_confs = acq_overlap.get("conflicts", [])
        overlap_reason = overlap_confs[0].get("reason", "") if overlap_confs else ""
        overlap_covered = ("Aktif kiralama" in overlap_reason or "donmuş" in overlap_reason)

        # Temizlik
        bus.release_lease(paths=["data/realistic_rag/"], task_id=t3_id)

        step10_pass = dir_accepted and overlap_rejected and overlap_covered
        print(f"[{'GEÇTİ' if step10_pass else 'KALDI'}] Adım 10: T-0003 -> Donmuş dizin kiralandı, alt dosyayı kapsadı, kiralama serbest bırakıldı", file=sys.stderr)
        if not step10_pass:
            sys.stderr.write(f"  Detay: dir_accepted={dir_accepted}, overlap_rejected={overlap_rejected}\n")
            return False

        # -------------------------------------------------------------
        # ADIM 11: T-0004 & T-0006 Çapraz Çarpım Matrisi (48 Hücre)
        # -------------------------------------------------------------
        # Eksenler:
        # 1. Yol şekli (8): file_plain, file_slash, noext_plain, noext_slash, dir_plain, dir_slash, nonexist_plain, nonexist_slash
        # 2. Scope argümanı (3): None, "file", "dir"
        # 3. Donmuşluk (2): True (data/realistic_rag/..), False (scratch/..)
        shapes = [
            ("file_plain", "val.jsonl", "x.txt"),
            ("file_slash", "val.jsonl/", "x.txt/"),
            ("noext_plain", "UZANTISIZ", "NOEXT"),
            ("noext_slash", "UZANTISIZ/", "NOEXT/"),
            ("dir_plain", "subdir", "mydir"),
            ("dir_slash", "subdir/", "mydir/"),
            ("nonexist_plain", "nonexist_file.txt", "nonexist_file.txt"),
            ("nonexist_slash", "nonexist_dir/", "nonexist_dir/")
        ]
        scopes = [None, "file", "dir"]
        frozens = [True, False]

        matrix_task = bus.post_task(
            title="Matris Test Görevi",
            spec="48 hücreli çapraz çarpım testi",
            writes=["data/realistic_rag/", "scratch/"],
            to="antigravity",
            from_agent="claude"
        )
        matrix_task_id = matrix_task["id"]

        matrix_passed_count = 0
        total_cells = len(shapes) * len(scopes) * len(frozens)

        for shape_name, rel_frozen, rel_unfrozen in shapes:
            for scope_arg in scopes:
                for is_frozen in frozens:
                    target_rel = f"data/realistic_rag/{rel_frozen}" if is_frozen else f"scratch/{rel_unfrozen}"
                    # DÜZELTME 2: Beklentiyi şekil etiketinden değil dosya sistemi gerçekliğinden türet
                    exp_ok, exp_lease, exp_scope = evaluate_matrix_cell_expectation(temp_root, target_rel, scope_arg, is_frozen)

                    # Kiralamayı dene
                    cell_res = bus.acquire_lease(
                        paths=[target_rel],
                        task_id=matrix_task_id,
                        owner="antigravity",
                        scope=scope_arg
                    )
                    cell_ok = (cell_res.get("ok") is True)
                    cell_lease_files = [f for f in os.listdir(bus.leases_dir) if f.endswith(".json") and not f.endswith(".tmp")]
                    has_lease_file = (len(cell_lease_files) > 0)

                    cell_success = (cell_ok == exp_ok) and (has_lease_file == exp_lease)

                    if exp_lease:
                        # Kiralama içeriğini ve normalize edilmiş yol kimliğini doğrula
                        lease_path = os.path.join(bus.leases_dir, cell_lease_files[0])
                        with open(lease_path, "r", encoding="utf-8") as lf:
                            ldata = json.load(lf)
                        scope_match = (ldata.get("scope") == exp_scope)

                        # Normalize kimlik denetimi (Düzeltme 2): Hedef dosya ise '/' ile BİTMEMELİ, dizin ise '/' ile BİTMELİ
                        clean_target = target_rel.rstrip("/")
                        abs_target = os.path.join(temp_root, clean_target)
                        if os.path.exists(abs_target):
                            is_dir_target = os.path.isdir(abs_target)
                        else:
                            is_dir_target = target_rel.endswith("/")

                        if is_dir_target:
                            norm_ok = ldata.get("path", "").endswith("/")
                        else:
                            norm_ok = not ldata.get("path", "").endswith("/")
                        cell_success = cell_success and scope_match and norm_ok

                        # Sonraki hücre için kiralamayı temizle
                        bus.release_lease([target_rel], task_id=matrix_task_id)
                        remaining = [f for f in os.listdir(bus.leases_dir) if f.endswith(".json") and not f.endswith(".tmp")]
                        cell_success = cell_success and (len(remaining) == 0)

                    if cell_success:
                        matrix_passed_count += 1
                    else:
                        sys.stderr.write(
                            f"  [MATRİS HÜCRE HATASI] shape={shape_name}, scope={scope_arg}, frozen={is_frozen} -> "
                            f"beklenen_ok={exp_ok}, alınan_ok={cell_ok}, beklenen_lease={exp_lease}, alınan_lease={has_lease_file}\n"
                        )

        # DÜZELTME 1: Denetim Kütüğü (events.jsonl) Değişmez Kural Denetimi:
        # Uzantı beyaz listesi YOKTUR; dosya sistemi kontrolü (os.path.isfile) kullanılır.
        # events.jsonl içinde dosya olan herhangi bir hedefe 'scope: dir' yazan tek bir kayıt bile olmamalıdır!
        # Bozuk kayıtlar sessizce atlanmaz (except Exception: pass kaldırıldı).
        events_audit_clean = True
        if os.path.exists(bus.events_file):
            with open(bus.events_file, "r", encoding="utf-8") as ef:
                for line in ef:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        payload = ev.get("payload", {})
                        p = payload.get("path", "")
                        sc = payload.get("scope")
                        if sc == "dir" and p:
                            clean_p = p.rstrip("/")
                            abs_audit_p = os.path.join(temp_root, clean_p)
                            if os.path.exists(abs_audit_p) and os.path.isfile(abs_audit_p):
                                events_audit_clean = False
                                sys.stderr.write(f"  [DENETİM İHLALİ] events.jsonl içinde dosyaya 'scope: dir' yazılmış: {line.strip()}\n")
                    except Exception as e:
                        events_audit_clean = False
                        sys.stderr.write(f"  [DENETİM HATASI] events.jsonl satırı çözümlenemedi: {line.strip()} -> {e}\n")

        # DÜZELTME 3 DOĞRULAMASI: Eksik argüman ham KeyError metni döndürmemeli
        missing_arg_req = {
            "jsonrpc": "2.0",
            "id": 999,
            "method": "tools/call",
            "params": {
                "name": "bus_inbox",
                "arguments": {"recipient": "claude"}  # 'who' eksik
            }
        }
        res_missing = handle_jsonrpc(bus, missing_arg_req)
        is_err = res_missing.get("result", {}).get("isError") is True
        err_text = res_missing.get("result", {}).get("content", [{}])[0].get("text", "")
        clean_err = is_err and ("Eksik zorunlu parametre: 'who'" in err_text) and ("KeyError" not in err_text) and (err_text != "Hata: 'who'")
        if not clean_err:
            sys.stderr.write(f"  [DÜZELTME 3 HATASI] Eksik argüman beklenen temiz hatayı üretmedi: {res_missing}\n")
            return False

        step11_pass = (matrix_passed_count == total_cells) and events_audit_clean and clean_err
        print(f"[{'GEÇTİ' if step11_pass else 'KALDI'}] Adım 11: T-0004 & T-0006 -> 48 Hücreli Çapraz Çarpım Matrisi tamamlandı ({matrix_passed_count}/{total_cells} hücre geçti, events.jsonl denetimi temiz, Düzeltme 3 doğrulandı)", file=sys.stderr)
        if not step11_pass:
            return False

        # -------------------------------------------------------------
        # EK KONTROL: Kök Çözümleme Önceliği (CLI > AGENT_BUS_ROOT > cwd)
        # -------------------------------------------------------------
        orig_env = os.environ.get("AGENT_BUS_ROOT")
        try:
            # 1. CLI argümanı önceliği
            assert resolve_repo_root(temp_root) == os.path.abspath(temp_root)
            # 2. Ortam değişkeni önceliği
            os.environ["AGENT_BUS_ROOT"] = temp_root
            assert resolve_repo_root(None) == os.path.abspath(temp_root)
            # 3. os.getcwd() varsayılanı
            del os.environ["AGENT_BUS_ROOT"]
            assert resolve_repo_root(None) == os.path.abspath(os.getcwd())
        finally:
            if orig_env is not None:
                os.environ["AGENT_BUS_ROOT"] = orig_env
            elif "AGENT_BUS_ROOT" in os.environ:
                del os.environ["AGENT_BUS_ROOT"]

    print("=================================================================", file=sys.stderr)
    print("   NİHAİ SONUÇ: 11 ADIMIN HEPSİ BAŞARIYLA GEÇTİ (PASSED)        ", file=sys.stderr)
    print("=================================================================", file=sys.stderr)
    return True


# =====================================================================
# ANA GİRİŞ NOKTASI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="agent-bus MCP Koordinasyon Sunucusu")
    parser.add_argument("--selftest", action="store_true", help="11 adımlık kabul kriteri ve çapraz çarpım matris testini koşturur")
    parser.add_argument("--repo-root", type=str, default=None, help="Hedef git çalışma alanı kökü (öncelik: CLI > AGENT_BUS_ROOT > cwd)")
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)

    if args.selftest:
        success = run_selftest(repo_root)
        sys.exit(0 if success else 1)
    else:
        run_stdio_server(repo_root)


if __name__ == "__main__":
    main()
