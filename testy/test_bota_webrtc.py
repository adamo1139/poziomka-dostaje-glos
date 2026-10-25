#!/usr/bin/env python3
"""Test bota bez przeglądarki: klient WebRTC (aiortc) mówi nagraniem pytania i nagrywa odpowiedź.

  .venv-bot/bin/python testy/test_bota_webrtc.py [--pytanie "..."]

Wejście: 5 s ciszy (powitanie bota), pytanie z TTS, 12 s ciszy. Odpowiedź bota trafia do
wyniki/bot/odpowiedz.wav i jest transkrybowana usługą STT.
"""
import argparse
import asyncio
import io
import os

import httpx
import numpy as np
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder

BOT, TTS, STT = "http://localhost:7860", "http://127.0.0.1:8002", "http://127.0.0.1:8001"
OUT = "wyniki/bot"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytanie", default="Jaka jest stolica Francji?")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{TTS}/v1/audio/speech", json={"input": a.pytanie, "response_format": "wav"})
        speech, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    silence = lambda s: np.zeros(int(s * sr), dtype=np.float32)
    sf.write(f"{OUT}/wejscie.wav", np.concatenate([silence(5), speech, silence(12)]), sr)

    pc = RTCPeerConnection()
    pc.addTransceiver("audio", direction="sendrecv")
    player = MediaPlayer(f"{OUT}/wejscie.wav")
    pc.getTransceivers()[0].sender.replaceTrack(player.audio)
    recorder = MediaRecorder(f"{OUT}/odpowiedz.wav")

    @pc.on("track")
    def on_track(track):
        recorder.addTrack(track)

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BOT}/api/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"})
        r.raise_for_status()
        answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    await recorder.start()
    await asyncio.sleep(5 + len(speech) / sr + 12)
    await recorder.stop()
    await pc.close()

    audio, rsr = sf.read(f"{OUT}/odpowiedz.wav", dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    print(f"nagrano {len(audio) / rsr:.1f} s odpowiedzi bota, RMS {float(np.sqrt((audio ** 2).mean())):.4f}")
    async with httpx.AsyncClient(timeout=60) as client:
        buf = io.BytesIO(); sf.write(buf, audio, rsr, format="WAV", subtype="PCM_16")
        r = await client.post(f"{STT}/v1/audio/transcriptions", files={"file": ("o.wav", buf.getvalue(), "audio/wav")},
                              data={"language": "pl"})
        print("bot powiedział (STT):", r.json()["text"])


if __name__ == "__main__":
    asyncio.run(main())
