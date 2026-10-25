

# Poziomka dostaje głos



https://github.com/user-attachments/assets/b5dec5d6-139a-4b66-988f-b3d19f94938c

*Demo (94 s): rozmowa z Poziomką w przeglądarce, wszystko na jednej RTX 5080.*

Polski asystent głosowy na własnym GPU: Silero VAD v6 → Canary-1b-v2 (STT) → Poziomka-Instruct (SGLang, FP8) → VoxCPM2 (TTS), połączone przez [Pipecat](https://github.com/pipecat-ai/pipecat). Całość mieści się na jednej karcie 16 GB.

## Wymagania

- Linux x86_64, karta NVIDIA z ≥ 16 GB VRAM (testowane: RTX 5080, sterownik 590.48; SGLang używa CUDA 13, więc sterownik musi ją obsługiwać).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), `ffmpeg` (biblioteki dla torchcodec), `curl`.
- Około 30 GB na środowiska i około 15 GB na modele w cache Hugging Face.

## Instalacja

```
git clone https://github.com/adamo1139/poziomka-dostaje-glos && cd poziomka-dostaje-glos
bash setup_venvs.sh          # .venv-tts .venv-stt .venv-llm .venv-bot (~1 min z ciepłym cache uv)
```

Każda usługa ma osobne środowisko, bo ich zależności się wykluczają:

| środowisko | do czego | kluczowe wersje |
|---|---|---|
| `.venv-llm` | SGLang z Poziomką | sglang 0.5.22 (torch 2.14.1, CUDA 13) |
| `.venv-tts` | `tts_voxcpm2/server.py` | voxcpm 2.0.3, torch 2.8.0 cu128 |
| `.venv-stt` | `stt_canary/server.py` | transformers 5.20.2 (Canary bez NeMo), torch 2.8.0 cu128 |
| `.venv-bot` | `bot/bot.py` | pipecat-ai 1.14.0 (webrtc, runner, silero, openai) |
| `.venv-quant` | `kwantyzuj_fp8.py` (opcjonalnie) | llmcompressor 0.14.0.1, torch 2.13.0 cu130 — `bash setup_venvs.sh quant` |

Modele pobierają się same przy pierwszym starcie: VoxCPM2 i Canary-1b-v2 z Hugging Face, a Poziomka FP8 z
[cpral/Poziomka-Instruct-2026-10-20-FP8](https://huggingface.co/cpral/Poziomka-Instruct-2026-10-20-FP8)
(albo z lokalnego `modele/Poziomka-Instruct-2026-10-20-FP8`, jeśli zrobisz go sam przez `kwantyzuj_fp8.py`).

## Uruchomienie

```
bash uruchom.sh      # LLM -> TTS -> STT -> bot, potem http://localhost:7860/client
bash zatrzymaj.sh
```

Mikrofon działa w przeglądarce tylko przez `localhost` albo HTTPS. Domyślnie mikrofon jest wyciszany, gdy Poziomka mówi
(bez echa przy głośnikach); `BOT_PRZERYWANIE=1 bash uruchom.sh` pozwala jej przerywać — do rozmowy w słuchawkach.

Decyzje, pomiary i znane problemy: [PLAN.md](PLAN.md).
