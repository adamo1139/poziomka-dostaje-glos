# Poziomka dostaje głos — plan

Cel: asystent głosowy po polsku z Poziomką jako LLM, w całości na własnych GPU (rigga, 8 GPU,
do 24 GB VRAM na usługę). Najpierw lokalne demo w przeglądarce.

## Decyzje (2026-10-25)

| warstwa | wybór | dlaczego |
|---|---|---|
| orkiestracja | **Pipecat** (`pipecat-ai`, BSD-2) | przerywanie, streaming zdań do TTS, Silero i gotowe UI WebRTC w pakiecie; własne STT/TTS to małe adaptery |
| VAD | **Silero VAD v6** (wbudowany w Pipecat, CPU) | v6 jest w paczce Pipecat (CHANGELOG 0.0.86) |
| koniec tury | Silero + próg ciszy (`SpeechTimeoutUserTurnStopStrategy`) | żaden semantyczny detektor końca tury nie ma potwierdzonego polskiego (LiveKit turn-detector: 14 języków bez pl; Smart Turn v3: brak informacji) |
| STT | **nvidia/canary-1b-v2** (NeMo, CC-BY-4.0) | 25 języków z polskim; model offline, transkrybuje całą wypowiedź po końcu mowy |
| LLM | **Poziomka na SGLang**, `/v1/chat/completions`, streaming | Pipecat: `OpenAILLMService(base_url=...)` bez zmian w kodzie |
| TTS | **openbmb/VoxCPM2** (2B, Apache-2.0, 48 kHz) | polski, wbudowany `generate_streaming`, RTF ~0,30 na RTX 4090 (~0,13 z Nano-vLLM) według karty |

Odrzucone:
- **Unmute** — Kyutai STT/TTS tylko en/fr; podmiana wymaga emulacji protokołu moshi-server, a prompty i frontend są tylko en/fr.
- **HF speech-to-speech** — wewnętrzne audio 16 kHz (traci 48 kHz z VoxCPM2), angielski tokenizer zdań.
- **Dograh** — Pipecat plus ciężka platforma (Postgres, Redis, MinIO) z telemetrią włączoną domyślnie; wrócić do niego przy telefonii.
- **TEN Framework** — licencja z zakazem użycia na urządzeniach końcowych.
- **Chatterbox Multilingual** (0,5B) — mniejszy, ale bez podanego RTF i streamingu w karcie modelu.

## Architektura

```
przeglądarka (WebRTC, Pipecat prebuilt UI /client)
   │ audio 16 kHz in / 48 kHz out
   ▼
bot Pipecat (CPU): transport → Silero VAD v6 → STT → agregator → LLM → TTS → transport
                                               │           │         │
                    GPU A: usługa STT (Canary) ┘           │         │
                    GPU B: SGLang (Poziomka) ──────────────┘         │
                    GPU C: usługa TTS (VoxCPM2) ─────────────────────┘
```

Canary i VoxCPM2 działają jako **osobne usługi HTTP** z API w stylu OpenAI, każda we własnym
środowisku i na własnym GPU:
- NeMo, VoxCPM i Pipecat nie muszą współistnieć w jednym środowisku Pythona;
- każda usługa ma własny proces (bez wspólnego GIL) i własne GPU;
- te same usługi pasują do HF speech-to-speech i Dograh (provider „Speaches”), więc wybór frameworka da się zmienić.

### Usługa STT — `stt_canary/`
- `POST /v1/audio/transcriptions` (multipart: plik WAV 16 kHz mono, `language=pl`) → `{"text": ...}`.
- Canary przez transformers (`CanaryForConditionalGeneration`, bez NeMo):
  `processor.apply_transcription_request(audio, source_language="pl", target_language="pl")`; model ładowany raz przy starcie, bf16.
- Po stronie Pipecat: podklasa `SegmentedSTTService` (szablon: `services/whisper/stt.py`), która wysyła odcinek mowy z VAD do usługi.

### Usługa TTS — `tts_voxcpm2/`
- `POST /v1/audio/speech` (`input`, `voice`, `response_format=pcm`) → strumień PCM s16le 48 kHz (chunked).
- `generate_streaming` z VoxCPM2; głos z pliku referencyjnego (klonowanie) albo domyślny.
- Po stronie Pipecat: podklasa `TTSService` z `run_tts` jako generatorem (szablon: `services/pocket_tts/tts.py`), `audio_out_sample_rate=48000`.

### LLM
- SGLang z wybranym checkpointem Poziomki, port lokalny, `OpenAILLMService(base_url="http://<host>:<port>/v1", model=...)`.
- Thinking wyłączony (szablon v13 ma tryb rozumowania; `<think>` to sekundy ciszy).
- Prompt systemowy pod mowę: krótkie odpowiedzi, bez markdown, list i emoji, liczby w formie do przeczytania.
- Dzielenie odpowiedzi na zdania: agregator Pipecat z językiem `pl` (domyślnie `en`).

### GPU (szacunki do potwierdzenia pomiarem)

| usługa | VRAM |
|---|---|
| SGLang + Poziomka bf16 | ~8,3 GB wag + KV cache (limit do 24 GB) |
| Canary-1b-v2 | ~4 GB fp32, mniej w bf16 |
| VoxCPM2 | ~8 GB |
| Silero VAD, bot Pipecat | CPU |

Run 22 zajmuje teraz wszystkie 8 GPU (PP8). Pomiary i prototyp po jego końcu albo na karcie,
na której zostaje ≥ 10–12 GB wolnego.

## Kroki

1. **Pomiary modeli (bez Pipecat).**
   - Canary: WER na kilkudziesięciu polskich nagraniach i czas transkrypcji wypowiedzi 2–10 s.
   - VoxCPM2: na ~20 typowych krótkich odpowiedziach Poziomki czas do pierwszej próbki audio, RTF i VRAM; ze standardowym backendem i z Nano-vLLM.
   - Gotowe, gdy są liczby w `wyniki/` i odsłuchane próbki.
2. **Usługi STT i TTS** (`stt_canary/`, `tts_voxcpm2/`) z endpointami jak wyżej i testem curl.
3. **Bot Pipecat** (`bot/`): SmallWebRTC + prebuilt UI, Silero v6, adaptery STT/TTS, Poziomka przez SGLang.
   - Pomiar opóźnień na każdym etapie: koniec mowy → transkrypcja → pierwszy token LLM → pierwsze audio.
4. **Koniec tury i przerywanie.**
   - Strojenie `stop_secs` i progów Silero na polskiej mowie.
   - Test barge-in (echo cancellation przeglądarki).
   - Osobno sprawdzić Smart Turn v3 na polskim.
5. **Styl mówiony Poziomki.** Najpierw prompt; jeśli nie wystarczy, mały SFT „na głos” w pipeline'ie z Ling-V2 (run 21/22).
6. **Dalej:** wybór i klonowanie głosu, frontend, ewentualnie telefonia (LiveKit / Dograh).

## Wyniki kroku 1 (2026-10-25, RTX 5080, testy/ → wyniki/)

| | VoxCPM2 (`.venv-tts`, torch.compile, streaming) | Canary-1b-v2 (`.venv-stt`, transformers 5.20.2, bf16, batch 1) |
|---|---|---|
| szybkość | pierwsze audio mediana 0,045 s; RTF mediana 0,46; fragmenty po 160 ms | wypowiedź ~3,6 s: 0,13 s; ~9,6 s: 0,22 s (p90 0,31 s) |
| jakość | odsłuch: bardzo dobry i naturalny | WER FLEURS pl test (150 nagrań) 7,9 %; krótkie odpowiedzi z VoxCPM2 12,2 % |
| VRAM | 5,7 GB | 2,0 GB |

- Canary działa w transformers (`CanaryForConditionalGeneration`) bez NeMo; wystarczy `model.safetensors` (3,9 GB).
- WER zawyża normalizacja: Canary zapisuje liczby cyframi („12 stopni”), referencje mają słowa.
- Bez nagrania referencyjnego VoxCPM2 daje inny głos w każdym zdaniu — w asystencie potrzebny stały `reference_wav`.

## Wyniki kroku 2 (usługi, RTX 5080, obie na jednym GPU, ~9 GB VRAM)

```
.venv-stt/bin/python stt_canary/server.py --port 8001 --device cuda:0
.venv-tts/bin/python tts_voxcpm2/server.py --port 8002 --device cuda:0 --voices glosy
.venv-stt/bin/python testy/test_uslug.py
```

- TTS przez HTTP: pierwszy bajt ~0,10 s, PCM s16le 48 kHz, nagłówek `X-Sample-Rate`.
- Przerwanie: rozłączenie klienta zatrzymuje syntezę po bieżącym fragmencie; następne żądanie dostaje pierwszy bajt po 0,2 s.
- STT przez HTTP: 0,15–0,16 s na zdanie 3–4 s (pierwsze żądanie 0,5 s — rozgrzewka przepróbkowania).
- Kółko TTS → STT: 2 z 3 zdań bez różnic; „czternastej” → „czternaście”.

## Wszystko na jednej karcie (RTX 5080 16 GB, limit mocy 250 W) — 2026-10-25

| usługa | VRAM | uruchomienie |
|---|---|---|
| SGLang + Poziomka-Instruct FP8 (eager, kontekst 2048) | ~6,1 GB (wagi 3,97 + KV 1,44) | `MODEL_PATH=modele/Poziomka-Instruct-2026-10-20-FP8 QUANT= EAGER=1 CONTEXT=2048 bash start_llm.sh` |
| VoxCPM2 | ~5,9 GB | `.venv-tts/bin/python tts_voxcpm2/server.py --port 8002` |
| Canary-1b-v2 | ~2,2 GB | `.venv-stt/bin/python stt_canary/server.py --port 8001` |
| razem | **15,0 / 16,3 GB** | kolejność: LLM, TTS, STT |

Kółko (`testy/test_kolko.py`, od końca wypowiedzi do pierwszego dźwięku, bez czekania VAD na koniec
tury): **0,55–1,01 s**. STT 0,11–0,13 s, pierwszy token +0,05 s, potem czekanie na koniec pierwszego
zdania (największa część, ~44 tok/s w eager), pierwszy bajt TTS +0,1 s. Maks. pobór mocy 155 W.

Ustalenia:
- Kwantyzacja FP8 w locie (`--quantization fp8`) trzyma 11,2 GB; checkpoint FP8 offline
  (`.venv-quant/bin/python kwantyzuj_fp8.py`, llm-compressor FP8_DYNAMIC, bez kalibracji) ma 4,0 GB.
- Kod Poziomki z HF jest pod transformers 4.x; `modele/zrodlo/` to łatka pod 5.x (import fx, RoPE
  `default`, `_tied_weights_keys`). W transformers 5 bufor `inv_freq` zostaje niezainicjalizowany,
  więc **forward przez HF transformers 5 jest błędny** — SGLang liczy RoPE sam i działa poprawnie.
- SGLang na sm_120: `CUDA_HOME` na nvcc z venvu, dowiązanie `lib64/libcudart.so`, bez DeepGEMM,
  attention Triton, sampling PyTorch (FlashInfer JIT: nvcc 13.4 nie pasuje do nagłówków cudart 13.0).
- Styl: mimo promptu Poziomka odpowiada długo, z listami i markdownem; 17 × 3 = „74” (bf16 nieporównane).

## Krok 3: bot Pipecat (`bot/bot.py`, `.venv-bot`, pipecat-ai 1.14.0) — 2026-10-25

```
bash uruchom.sh      # LLM -> TTS -> STT -> bot, potem http://localhost:7860/client
bash zatrzymaj.sh
.venv-bot/bin/python testy/test_bota_webrtc.py --pytanie "..."   # test bez przeglądarki (aiortc)
```

- Gotowe klienty Pipecat z `base_url`: `OpenAISTTService` (Canary, `Language.PL`), `OpenAILLMService`
  (`extra_body.chat_template_kwargs.enable_thinking=false`), `OpenAITTSService` (48 kHz, `strip_markdown`).
- Głos stały: `glosy/poziomka.wav` (próbka VoxCPM2 z kroku 1); serwer TTS mapuje nazwy głosów OpenAI
  na `--default-voice`.
- Koniec tury: Silero VAD (`stop_secs=0.3`) + `SpeechTimeoutUserTurnStopStrategy(0.6 s)`.
- Test WebRTC: powitanie + poprawne odpowiedzi; na rozgrzanym serwerze STT TTFB 0,39 s, LLM TTFB 0,07 s,
  TTS TTFB 0,47 s (więcej niż 0,1 s bezpośrednio — klient Pipecat buforuje porcje audio).

## Otwarte pytania
- Który checkpoint Poziomki: wynik run 22 czy obecny najlepszy instruct.
- Głos: własny (klonowany z nagrania) czy gotowy.
- Czy Smart Turn v3 działa na polskim (do testu w kroku 4).
- Normalizacja liczb między Canary (cyfry) a Poziomką/TTS (słowa) — czy przeszkadza LLM-owi.
