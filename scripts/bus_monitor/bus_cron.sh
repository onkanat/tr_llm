#!/usr/bin/env bash
# scripts/bus_monitor/bus_cron.sh
# agent-bus hafif (sıfır LLM tokenli) kontrolcü.
# Cron tarafından her 1 veya 2 dakikada bir çağrılır.
# Yalnızca yeni mesaj veya açık görev tespit edilirse agy üzerinden /agent-bus komutunu çalıştırır.
#
# T-0175 onarımı (ölçüm: bus_cron.log — agy --print 5 dk "turn in progress"
# zaman-aşımı, 26–27 Eyl):
#   1. agy üst-sınır: AGY_TIMEOUT_SEC=240 (perl-alarm; exec sonrası SIGALRM
#      süreci öldürür — gtimeout bağımlılığı yok).
#   2. Soğuma-kapısı: bus_check imzası (okunmamış sayısı + açık görev id'leri)
#      değişmediyse ve son tetiklemeden bu yana COOLDOWN_SEC'ten az süre geçtiyse
#      tetikleme ATLANIR — tek tıklaklık mesajı zinciri sonsuza dek döndürmez.
#      İmza .son_tetik'e tetiklemeden ÖNCE yazılır (askıda koşum da kapıyı kurar).

set -euo pipefail

REPO_DIR="/Users/hakankilicaslan/Git/tr_llm"
PYTHON_BIN="${REPO_DIR}/venv/bin/python"
CHECK_SCRIPT="${REPO_DIR}/scripts/bus_monitor/bus_check.py"
AGY_BIN="${AGY_BIN_OVERRIDE:-/Users/hakankilicaslan/.local/bin/agy}"
LOCK_FILE="/tmp/bus_cron.lock"
STATE_FILE="${REPO_DIR}/scripts/bus_monitor/.son_tetik"
AGY_TIMEOUT_SEC=240
COOLDOWN_SEC=1800

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
OUT=$("$PYTHON_BIN" "$CHECK_SCRIPT")
STATUS=$?
set -e

# Eğer durum 10 ise (yeni mesaj veya açık görev var)
if [ $STATUS -eq 10 ]; then
    # 2. Soğuma-kapısı: imza değişmediyse ve soğuma süresi dolmadıysa atlama
    SIG=$(printf '%s\n' "$OUT" | sed -n 's/^\[IMZA\] //p' | head -1)
    [ -z "$SIG" ] && SIG="imza-yok"
    NOW=$(date +%s)
    if [ -f "$STATE_FILE" ]; then
        LAST=$(cat "$STATE_FILE" 2>/dev/null || echo "|0")
        LAST_SIG="${LAST%|*}"
        LAST_TS="${LAST#*|}"
        ELAPSED=$((NOW - LAST_TS))
        if [ "$LAST_SIG" = "$SIG" ] && [ "$ELAPSED" -lt "$COOLDOWN_SEC" ]; then
            exit 0
        fi
    fi
    printf '%s|%s' "$SIG" "$NOW" > "$STATE_FILE"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Yeni veri bulundu! Antigravity /agent-bus tetikleniyor (üst-sınır ${AGY_TIMEOUT_SEC}s; imza: $SIG)..."
    cd "$REPO_DIR"
    # agy'yi non-interactive /agent-bus ile çalıştır; perl-alarm üst-sınır
    set +e
    perl -e 'alarm shift @ARGV; exec @ARGV' "$AGY_TIMEOUT_SEC" \
        "$AGY_BIN" --print "/agent-bus" --dangerously-skip-permissions \
        >> "${REPO_DIR}/scripts/bus_monitor/bus_cron.log" 2>&1
    AGY_RC=$?
    set -e
    if [ $AGY_RC -eq 142 ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] agy üst-sınır (${AGY_TIMEOUT_SEC}s) ile kesildi (rc=142) — soğuma-kapısı ${COOLDOWN_SEC}s için kuruldu." >> "${REPO_DIR}/scripts/bus_monitor/bus_cron.log"
    elif [ $AGY_RC -ne 0 ]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] agy rc=${AGY_RC} ile döndü — soğuma-kapısı ${COOLDOWN_SEC}s için kuruldu." >> "${REPO_DIR}/scripts/bus_monitor/bus_cron.log"
    fi
else
    # 0 token tüketildi, sessiz çıkış
    exit 0
fi
