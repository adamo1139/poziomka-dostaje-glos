#!/usr/bin/env bash
set -euo pipefail
# Cały asystent na jednej karcie (RTX 5080 16 GB): SGLang (Poziomka FP8) -> TTS -> STT -> bot Pipecat.
# Kolejność ma znaczenie: każda usługa startuje dopiero, gdy poprzednia odpowiada na /health.
#   bash uruchom.sh        # potem http://localhost:7860/client
#   bash zatrzymaj.sh
cd "$(dirname "$0")"
mkdir -p logi
: > logi/pids

start() {  # nazwa, port (0 = bez /health), polecenie...
    local name=$1 port=$2; shift 2
    nohup "$@" > "logi/${name}.log" 2>&1 &
    local pid=$!
    echo "${pid} ${name}" >> logi/pids
    [[ "${port}" == 0 ]] && return
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:${port}/health" > /dev/null && { echo "${name}: gotowe (port ${port})"; return; }
        kill -0 "${pid}" 2> /dev/null || { echo "${name}: proces padł, zobacz logi/${name}.log" >&2; exit 1; }
        sleep 3
    done
    echo "${name}: brak odpowiedzi na /health po 10 min" >&2; exit 1
}

start llm 8000 env MODEL_PATH=modele/Poziomka-Instruct-2026-10-20-FP8 QUANT= EAGER=1 CONTEXT=2048 MEM=0.90 MAX_TOKENS=16384 bash start_llm.sh
start tts 8002 .venv-tts/bin/python tts_voxcpm2/server.py --port 8002 --voices glosy --default-voice poziomka
start stt 8001 .venv-stt/bin/python stt_canary/server.py --port 8001
start bot 0 .venv-bot/bin/python bot/bot.py -t webrtc --host "${BOT_HOST:-localhost}" --port "${BOT_PORT:-7860}"
for _ in $(seq 1 40); do
    curl -sf -o /dev/null "http://localhost:${BOT_PORT:-7860}/client/" && break
    sleep 1
done
echo "bot: http://localhost:${BOT_PORT:-7860}/client"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
