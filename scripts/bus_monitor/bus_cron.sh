#!/usr/bin/env bash
# scripts/bus_monitor/bus_cron.sh
# agent-bus hafif (sıfır LLM tokenli) kontrolcü.
# Cron tarafından her 1 veya 2 dakikada bir çağrılır.
# Yalnızca yeni mesaj veya açık görev tespit edilirse agy üzerinden /agent-bus komutunu çalıştırır.

set -euo pipefail

REPO_DIR="/Users/hakankilicaslan/Git/tr_llm"
PYTHON_BIN="${REPO_DIR}/venv/bin/python"
CHECK_SCRIPT="${REPO_DIR}/scripts/bus_monitor/bus_check.py"
AGY_BIN="/Users/hakankilicaslan/.local/bin/agy"
LOCK_FILE="/tmp/bus_cron.lock"

# Tekil çalışma kilidi (zaten bir işlem sürüyorsa üst üste binmesin)
if [ -f "$LOCK_FILE" ]; then
    PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        exit 0
    fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

# 1. Sıfır tokenli Python dosya kontrolü
set +e
"$PYTHON_BIN" "$CHECK_SCRIPT"
STATUS=$?
set -e

# Eğer durum 10 ise (yeni mesaj veya açık görev var)
if [ $STATUS -eq 10 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Yeni veri bulundu! Antigravity /agent-bus tetikleniyor..."
    cd "$REPO_DIR"
    # agy'yi non-interactive /agent-bus ile çalıştır
    "$AGY_BIN" --print "/agent-bus" --dangerously-skip-permissions >> "${REPO_DIR}/scripts/bus_monitor/bus_cron.log" 2>&1
else
    # 0 token tüketildi, sessiz çıkış
    exit 0
fi
