#!/usr/bin/env bash
set -euo pipefail
# Tworzy środowiska uv dla usług asystenta, z wersjami, na których wszystko było testowane
# (Linux x86_64, RTX 5080, sterownik NVIDIA 590.48, uv 0.7.19, Python 3.11).
#   bash setup_venvs.sh                 # tts stt llm bot
#   bash setup_venvs.sh quant           # tylko środowisko do kwantyzuj_fp8.py
# Każda usługa ma osobne środowisko, bo ich zależności się wykluczają (np. sglang przypina
# torch 2.14 i transformers 5.19, voxcpm działa na torch 2.8, llm-compressor wymaga transformers <= 5.17).
cd "$(dirname "$0")"
PY="${PY:-3.11}"
CU128=https://download.pytorch.org/whl/cu128
CU130=https://download.pytorch.org/whl/cu130

make_venv() {  # nazwa, argumenty uv pip install...
    local dir=".venv-$1"; shift
    [[ -d "${dir}" ]] && { echo "${dir} już istnieje, pomijam"; return; }
    echo "== ${dir}"
    uv venv "${dir}" --python "${PY}" -q
    VIRTUAL_ENV="${dir}" uv pip install -q "$@"
}

tts() {  # VoxCPM2 (TTS) — torch 2.8 cu128
    make_venv tts "torch==2.8.0" "torchaudio==2.8.0" "torchcodec==0.7.0" "voxcpm==2.0.3" "transformers==5.20.2" \
        "fastapi==0.162.0" "uvicorn[standard]==0.54.0" "python-multipart==0.0.32" "httpx==0.28.1" "soundfile==0.14.0" \
        --extra-index-url "${CU128}" --index-strategy unsafe-best-match
}
stt() {  # Canary-1b-v2 przez transformers (bez NeMo) — torch 2.8 cu128
    make_venv stt "torch==2.8.0" "torchaudio==2.8.0" "transformers==5.20.2" "librosa==0.11.0" "soundfile==0.14.0" \
        "jiwer==4.0.0" "pyarrow==26.0.0" "fastapi==0.162.0" "uvicorn[standard]==0.54.0" "python-multipart==0.0.32" \
        "httpx==0.28.1" --extra-index-url "${CU128}" --index-strategy unsafe-best-match
}
llm() {  # SGLang — przypina torch 2.14.1 i flashinfer cu13; cuda-tile jest w wersji rc, stąd --prerelease
    make_venv llm "sglang==0.5.22" --prerelease=allow
}
bot() {  # Pipecat: WebRTC + prebuilt UI (runner), Silero VAD, klienci OpenAI
    make_venv bot "pipecat-ai[webrtc,runner,silero,openai]==1.14.0" "httpx==0.28.1"
}
quant() {  # llm-compressor do kwantyzuj_fp8.py — torch 2.13 cu130
    make_venv quant "llmcompressor==0.14.0.1" "torch==2.13.0" --extra-index-url "${CU130}" --index-strategy unsafe-best-match
}

command -v uv > /dev/null || { echo "Brak uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
for target in "${@:-tts stt llm bot}"; do
    for t in ${target}; do
        declare -F "${t}" > /dev/null || { echo "Nieznane środowisko: ${t} (tts stt llm bot quant)" >&2; exit 1; }
        "${t}"
    done
done
echo "Gotowe. Start: bash uruchom.sh"
