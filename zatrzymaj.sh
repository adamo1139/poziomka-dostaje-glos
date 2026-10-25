#!/usr/bin/env bash
# Zatrzymuje procesy zapisane przez uruchom.sh (po PID, w odwrotnej kolejności).
cd "$(dirname "$0")"
[[ -f logi/pids ]] || { echo "brak logi/pids"; exit 0; }
tac logi/pids | while read -r pid name; do
    # start_llm.sh robi exec, więc PID to sam serwer SGLang; jego procesy potomne zamykają się razem z nim.
    if kill "${pid}" 2> /dev/null; then echo "zatrzymano ${name} (${pid})"; fi
done
rm -f logi/pids
