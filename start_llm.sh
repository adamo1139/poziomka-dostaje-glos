#!/usr/bin/env bash
set -euo pipefail
# Poziomka-Instruct na SGLang, endpoint OpenAI pod http://127.0.0.1:${PORT}/v1, model "poziomka".
# Jedna karta z STT i TTS (RTX 5080, 16 GB): checkpoint FP8 z kwantyzuj_fp8.py (~4 GB wag).
# Kwantyzacja w locie (QUANT=fp8) trzyma tu 11 GB, więc się nie mieści.
#   MODEL_PATH=modele/Poziomka-Instruct-2026-10-20-FP8 QUANT= EAGER=1 CONTEXT=2048 bash start_llm.sh
# Attention przez Triton i sampling w PyTorch: FlashInfer kompiluje kernele w locie, a nvcc z venvu
# (13.4) nie pasuje do nagłówków cudart (13.0). Na osobnym GPU w bf16: MODEL_PATH= QUANT= MEM=0.85.
# Bot musi wysyłać chat_template_kwargs {"enable_thinking": false}; bez tego szablon v13 nie
# wstawia bloku <think> i model sam decyduje, czy rozumować.

cd "$(dirname "$0")"
MODEL="${MODEL:-cpral/Poziomka-Instruct-2026-10-20}"
REVISION="${REVISION:-c74ec04fbd2279653d8a2b383f6b529607180890}"
QUANT="${QUANT-fp8}"
MEM="${MEM:-0.38}"
PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-8192}"
# EAGER=1: bez grafów CUDA (szybszy start, wolniejsze dekodowanie).
if [[ "${EAGER:-0}" == 1 ]]; then GRAPH_ARGS=(--disable-cuda-graph); else GRAPH_ARGS=(--cuda-graph-max-bs-decode 4); fi
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
# Kernele JIT SGLang (np. kwantyzacja FP8) kompiluje nvcc z CUDA_HOME; bierzemy nvcc CUDA 13 z venvu
# (sm_120f), bo systemowe CUDA_HOME bywa nieaktualne. DeepGEMM nie ma kerneli dla sm_120.
export CUDA_HOME="${PWD}/.venv-llm/lib/python3.11/site-packages/nvidia/cu13"
export PATH="${CUDA_HOME}/bin:${PATH}"
# Linker JIT szuka ${CUDA_HOME}/lib64/libcudart.so; wheel ma tylko lib/libcudart.so.13.
if [[ ! -e "${CUDA_HOME}/lib64/libcudart.so" ]]; then
    mkdir -p "${CUDA_HOME}/lib64" && ln -s ../lib/libcudart.so.13 "${CUDA_HOME}/lib64/libcudart.so"
fi
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

# MODEL_PATH: lokalny checkpoint (np. modele/Poziomka-Instruct-2026-10-20-FP8 z kwantyzuj_fp8.py,
# wtedy QUANT= bo kwantyzacja jest w config.json); inaczej snapshot MODEL@REVISION z HF.
path="${MODEL_PATH:-$(.venv-llm/bin/hf download "${MODEL}" --revision "${REVISION}" --quiet)}"
exec .venv-llm/bin/python -m sglang.launch_server --model-path "${path}" --served-model-name poziomka \
    --trust-remote-code ${QUANT:+--quantization "${QUANT}"} --mem-fraction-static "${MEM}" \
    --context-length "${CONTEXT}" --max-running-requests 4 "${GRAPH_ARGS[@]}" \
    --attention-backend "${ATTENTION:-triton}" --sampling-backend "${SAMPLING:-pytorch}" \
    --host 127.0.0.1 --port "${PORT}"
