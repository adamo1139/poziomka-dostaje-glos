#!/usr/bin/env python3
"""Usługa STT: Canary-1b-v2 (transformers, bez NeMo) z API w stylu OpenAI.

  .venv-stt/bin/python stt_canary/server.py --port 8001 --device cuda:0

POST /v1/audio/transcriptions  (multipart: file=<audio>, language=pl, response_format=json|text)
  -> {"text": "..."}  (+ "duration" i "processing_s")
GET  /health

Plik audio w dowolnym formacie czytanym przez soundfile (WAV, FLAC, OGG); wielokanałowe audio
jest uśredniane do mono i przepróbkowane do 16 kHz. Jedna transkrypcja naraz na GPU.
"""
import argparse
import asyncio
import io
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from transformers import AutoProcessor, CanaryForConditionalGeneration

SR = 16000


def build_app(model_id, device, dtype):
    processor = AutoProcessor.from_pretrained(model_id)
    model = CanaryForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    lock = asyncio.Lock()
    app = FastAPI(title="stt-canary")

    @torch.inference_mode()
    def transcribe(audio, language):
        inputs = processor.apply_transcription_request(audio, source_language=language, target_language=language,
                                                       sampling_rate=SR, return_tensors="pt")
        out = model.generate(**inputs.to(device, dtype=model.dtype), max_new_tokens=448)
        return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    transcribe(np.zeros(SR, dtype=np.float32), "pl")  # rozgrzewka

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id, "device": device}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), language: str = Form("pl"),
                             response_format: str = Form("json"), model_name: str = Form("", alias="model")):
        try:
            audio, sr = sf.read(io.BytesIO(await file.read()), dtype="float32", always_2d=True)
        except Exception as e:
            raise HTTPException(400, f"Nie da się odczytać audio: {e}")
        audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if len(audio) < SR // 10:
            return {"text": "", "duration": len(audio) / SR, "processing_s": 0.0}
        async with lock:
            start = time.time()
            text = await asyncio.to_thread(transcribe, audio, language)
            elapsed = time.time() - start
        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text, "duration": round(len(audio) / SR, 3), "processing_s": round(elapsed, 3)}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/canary-1b-v2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    uvicorn.run(build_app(a.model, a.device, getattr(torch, a.dtype)), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
