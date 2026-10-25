#!/usr/bin/env python3
"""Test usług STT i TTS przez HTTP: czas do pierwszego bajtu, przerwanie strumienia, kółko TTS -> STT.

  .venv-stt/bin/python testy/test_uslug.py --tts http://127.0.0.1:8002 --stt http://127.0.0.1:8001
"""
import argparse
import io
import time

import httpx
import numpy as np
import soundfile as sf

ZDANIA = ["Dzień dobry! W czym mogę pomóc?",
          "Pociąg do Krakowa odjeżdża z peronu trzeciego o czternastej piętnaście.",
          "Pamiętaj, żeby wziąć parasol, bo po południu może padać."]


def speak(client, tts, text, stop_after=None):
    """Zwraca (audio float32, sample rate, czas do pierwszego bajtu, czas całości)."""
    start, first, data = time.time(), None, bytearray()
    with client.stream("POST", f"{tts}/v1/audio/speech", json={"input": text, "response_format": "pcm"}) as r:
        r.raise_for_status()
        sr = int(r.headers["X-Sample-Rate"])
        for part in r.iter_bytes():
            first = first or time.time() - start
            data += part
            if stop_after and len(data) >= stop_after * sr * 2:
                break  # zamknięcie połączenia w połowie = przerwanie wypowiedzi
    audio = np.frombuffer(bytes(data[: len(data) // 2 * 2]), dtype="<i2").astype(np.float32) / 32767
    return audio, sr, first, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default="http://127.0.0.1:8002")
    ap.add_argument("--stt", default="http://127.0.0.1:8001")
    a = ap.parse_args()
    with httpx.Client(timeout=60) as client:
        print("health:", client.get(f"{a.tts}/health").json(), client.get(f"{a.stt}/health").json())
        for text in ZDANIA:
            audio, sr, first, total = speak(client, a.tts, text)
            buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
            t = time.time()
            res = client.post(f"{a.stt}/v1/audio/transcriptions", files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                              data={"language": "pl"}).json()
            print(f"TTS pierwszy bajt {first:.3f} s, całość {total:.2f} s, audio {len(audio) / sr:.2f} s | "
                  f"STT {time.time() - t:.3f} s (model {res['processing_s']} s)\n   tekst: {text}\n   STT:   {res['text']}")

        long = "To jest długie zdanie, które przerwiemy w połowie, żeby sprawdzić, czy usługa od razu zwalnia kartę graficzną dla następnego żądania."
        _, sr, first, total = speak(client, a.tts, long, stop_after=0.5)
        print(f"przerwanie po 0,5 s audio: rozłączono po {total:.2f} s")
        _, _, first, _ = speak(client, a.tts, ZDANIA[0])
        print(f"następne żądanie po przerwaniu: pierwszy bajt {first:.3f} s (ma być < 0,5 s, czyli GPU zwolnione)")


if __name__ == "__main__":
    main()
