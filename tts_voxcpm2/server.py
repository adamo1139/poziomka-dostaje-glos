#!/usr/bin/env python3
"""Usługa TTS: VoxCPM2 ze strumieniowaniem, API w stylu OpenAI.

  .venv-tts/bin/python tts_voxcpm2/server.py --port 8002 --device cuda:0 --voices glosy/

POST /v1/audio/speech  JSON {"input": "...", "voice": "nazwa", "response_format": "pcm"|"wav"}
  pcm -> strumień PCM s16le mono 48 kHz (chunked, fragmenty po 160 ms, nagłówek X-Sample-Rate)
  wav -> cały plik WAV po zakończeniu syntezy
GET  /health, GET /v1/voices

Głos: plik <voices>/<voice>.wav jako reference_wav_path (klonowanie). Żądanie bez głosu albo z nazwą
głosu OpenAI (alloy, nova, ...; klient Pipecat OpenAITTSService zawsze ją wysyła) dostaje
--default-voice. Bez żadnego głosu VoxCPM2 losuje nowy głos przy każdym żądaniu. Synteza jedna naraz na GPU; rozłączenie klienta (np. przerwanie
wypowiedzi) zamyka generator i zatrzymuje syntezę.
"""
import argparse
import asyncio
import io
import os
import threading

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from voxcpm import VoxCPM


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    response_format: str = "pcm"
    model: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10


def to_s16le(chunk):
    return (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()


OPENAI_VOICES = {"alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage",
                 "shimmer", "verse"}


def build_app(device, voices_dir, compile_model, default_voice):
    tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=compile_model, device=device)
    sr = tts.tts_model.sample_rate
    lock = threading.Lock()
    app = FastAPI(title="tts-voxcpm2")

    def voices():
        if not voices_dir or not os.path.isdir(voices_dir):
            return {}
        return {os.path.splitext(f)[0]: os.path.join(voices_dir, f)
                for f in sorted(os.listdir(voices_dir)) if f.lower().endswith((".wav", ".flac"))}

    def produce(req, reference, put, stop):
        """Wątek syntezy: fragmenty do kolejki, stop sprawdzany po każdym fragmencie (160 ms audio)."""
        with lock:
            stream = tts.generate_streaming(text=req.input, reference_wav_path=reference,
                                            cfg_value=req.cfg_value, inference_timesteps=req.inference_timesteps)
            try:
                for chunk in stream:
                    if stop.is_set():
                        break
                    put(chunk)
            except Exception as e:
                put(e)
            finally:
                stream.close()
                put(None)

    async def chunks(req, reference):
        loop = asyncio.get_running_loop()
        queue, stop = asyncio.Queue(), threading.Event()
        put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)
        threading.Thread(target=produce, args=(req, reference, put, stop), daemon=True).start()
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()  # klient rozłączony albo koniec: wątek kończy po bieżącym fragmencie

    for _ in tts.generate_streaming(text="Rozgrzewka."):  # torch.compile przy starcie
        pass

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "openbmb/VoxCPM2", "device": device, "sample_rate": sr,
                "default_voice": default_voice}

    @app.get("/v1/voices")
    def list_voices():
        return {"voices": list(voices())}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        if not req.input.strip():
            raise HTTPException(400, "Pusty tekst")
        name = default_voice if req.voice in (None, "") or req.voice in OPENAI_VOICES else req.voice
        reference = voices().get(name) if name else None
        if name and reference is None:
            raise HTTPException(404, f"Nie ma głosu {name!r}; dostępne: {list(voices())}")
        headers = {"X-Sample-Rate": str(sr)}
        if req.response_format == "pcm":
            async def body():
                async for chunk in chunks(req, reference):
                    yield to_s16le(chunk)
            return StreamingResponse(body(), media_type="audio/pcm", headers=headers)
        if req.response_format == "wav":
            buf = io.BytesIO()
            audio = np.concatenate([c async for c in chunks(req, reference)])
            sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            return Response(buf.getvalue(), media_type="audio/wav", headers=headers)
        raise HTTPException(400, "response_format musi być pcm albo wav")

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--voices", default="glosy", help="katalog z <nazwa>.wav do klonowania głosu")
    ap.add_argument("--default-voice", default="poziomka", help="głos dla żądań bez głosu; pusty = losowy")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    a = ap.parse_args()
    uvicorn.run(build_app(a.device, a.voices, not a.no_compile, a.default_voice or None), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
