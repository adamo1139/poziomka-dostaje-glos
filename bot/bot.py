#!/usr/bin/env python3
"""Poziomka — asystent głosowy (Pipecat): Silero VAD v6 -> Canary (STT) -> Poziomka (SGLang) -> VoxCPM2 (TTS).

  .venv-bot/bin/python bot/bot.py -t webrtc            # UI: http://localhost:7860/client

Usługi muszą już działać (adresy w zmiennych środowiskowych):
  POZIOMKA_LLM_URL  http://127.0.0.1:8000/v1   start_llm.sh (SGLang, model "poziomka")
  POZIOMKA_STT_URL  http://127.0.0.1:8001/v1   stt_canary/server.py
  POZIOMKA_TTS_URL  http://127.0.0.1:8002/v1   tts_voxcpm2/server.py (głos domyślny: glosy/poziomka.wav)

Wszystkie trzy mówią API OpenAI, więc bot używa gotowych klientów Pipecat (OpenAISTTService,
OpenAILLMService, OpenAITTSService) z base_url. Koniec tury: Silero + przerwa w mowie
(SpeechTimeoutUserTurnStopStrategy); semantyczne detektory końca tury nie mają potwierdzonego polskiego.

Echo: przy głośnikach echo cancellation przeglądarki nie wystarcza i Canary transkrybuje głos Poziomki
jako wypowiedź użytkownika, co przerywa jej odpowiedź. Dlatego domyślnie mikrofon jest wyciszony, gdy
bot mówi (bez przerywania w pół zdania). BOT_PRZERYWANIE=1 włącza przerywanie — do rozmowy w słuchawkach.
"""
import os

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.transforms.strip_markdown import strip_markdown
from pipecat.workers.runner import WorkerRunner

LLM_URL = os.environ.get("POZIOMKA_LLM_URL", "http://127.0.0.1:8000/v1")
STT_URL = os.environ.get("POZIOMKA_STT_URL", "http://127.0.0.1:8001/v1")
TTS_URL = os.environ.get("POZIOMKA_TTS_URL", "http://127.0.0.1:8002/v1")
TTS_SAMPLE_RATE = 48000  # VoxCPM2
PRZERYWANIE = os.environ.get("BOT_PRZERYWANIE", "0") == "1"

SYSTEM = (
    "Jesteś Poziomką, polską asystentką głosową. Twoje odpowiedzi są czytane na głos, więc mów "
    "naturalnie i krótko: zwykle jedno lub dwa zdania. Nie używaj list, punktów, nagłówków, "
    "pogrubień, emoji ani wzorów. Liczby zapisuj tak, jak się je wymawia. Jeśli czegoś nie "
    "usłyszałaś albo nie wiesz, powiedz to wprost."
)
POWITANIE = "Cześć, tu Poziomka. W czym mogę pomóc?"

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = OpenAISTTService(
        api_key="brak", base_url=STT_URL,
        settings=OpenAISTTService.Settings(model="canary-1b-v2", language=Language.PL),
    )
    llm = OpenAILLMService(
        api_key="brak", base_url=LLM_URL,
        settings=OpenAILLMService.Settings(
            model="poziomka", system_instruction=SYSTEM, temperature=0.7, max_tokens=200,
            # Szablon v13: bez enable_thinking model sam decyduje o bloku <think> (sekundy ciszy).
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )
    tts = OpenAITTSService(
        api_key="brak", base_url=TTS_URL, sample_rate=TTS_SAMPLE_RATE,
        # "alloy" to tylko nazwa wymagana przez klienta; serwer podstawia głos domyślny (poziomka).
        settings=OpenAITTSService.Settings(model="voxcpm2", voice="alloy", language=Language.PL),
        text_transforms=[("*", strip_markdown)],  # Poziomka i tak bywa skłonna do **pogrubień** i list
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)),
            user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)]),
            user_mute_strategies=[] if PRZERYWANIE else [AlwaysUserMuteStrategy()],
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_out_sample_rate=TTS_SAMPLE_RATE, enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Klient połączony (przerywanie: {'tak' if PRZERYWANIE else 'nie, mikrofon wyciszany gdy bot mówi'})")
        await worker.queue_frames([TTSSpeakFrame(POWITANIE)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Klient rozłączony")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
