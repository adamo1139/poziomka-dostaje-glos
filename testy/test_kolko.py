#!/usr/bin/env python3
"""Pełne kółko bez Pipecat: pytanie (TTS) -> STT -> Poziomka (stream) -> zdania do TTS (stream).

  .venv-stt/bin/python testy/test_kolko.py

Mierzy czas od końca wypowiedzi użytkownika do pierwszego dźwięku odpowiedzi, z podziałem na STT,
pierwszy token LLM, pierwsze zdanie i pierwszy bajt TTS. Wszystkie trzy usługi działają naraz,
więc LLM generuje dalsze zdania, gdy TTS czyta pierwsze.
"""
import io
import json
import re
import threading
import time

import httpx
import numpy as np
import soundfile as sf

LLM, STT, TTS = "http://127.0.0.1:8000", "http://127.0.0.1:8001", "http://127.0.0.1:8002"
SYSTEM = ("Jesteś Poziomką, asystentką głosową. Odpowiadasz po polsku, krótko, jednym lub dwoma zdaniami, "
          "bez list, punktów i formatowania.")
PYTANIA = ["Jaka będzie najlepsza pora na spacer w październiku?", "Opowiedz mi krótko, kim była Maria Skłodowska-Curie.",
           "Co mogę ugotować z ziemniaków i jajek?"]
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def tts_pcm(client, text):
    with client.stream("POST", f"{TTS}/v1/audio/speech", json={"input": text, "response_format": "pcm"}) as r:
        sr, data = int(r.headers["X-Sample-Rate"]), bytearray()
        for part in r.iter_bytes():
            data += part
    return np.frombuffer(bytes(data), dtype="<i2").astype(np.float32) / 32767, sr


def main():
    with httpx.Client(timeout=120) as client:
        for q in PYTANIA:
            question, sr = tts_pcm(client, q)  # "nagranie" użytkownika
            buf = io.BytesIO(); sf.write(buf, question, sr, format="WAV", subtype="PCM_16")

            t0 = time.time()  # koniec wypowiedzi użytkownika
            heard = client.post(f"{STT}/v1/audio/transcriptions", files={"file": ("q.wav", buf.getvalue(), "audio/wav")},
                                data={"language": "pl"}).json()["text"]
            t_stt = time.time() - t0

            body = {"model": "poziomka", "stream": True, "max_tokens": 150, "temperature": 0.7,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": heard}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            marks, answer, pending, threads = {}, "", "", []

            def speak(sentence, first):
                with client.stream("POST", f"{TTS}/v1/audio/speech", json={"input": sentence, "response_format": "pcm"}) as r:
                    for _ in r.iter_bytes():
                        if first and "first_audio" not in marks:
                            marks["first_audio"] = time.time() - t0

            with client.stream("POST", f"{LLM}/v1/chat/completions", json=body) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line.endswith("[DONE]"):
                        continue
                    delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                    if delta and "first_token" not in marks:
                        marks["first_token"] = time.time() - t0
                    answer += delta; pending += delta
                    parts = SENTENCE_END.split(pending)
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            if "first_sentence" not in marks:
                                marks["first_sentence"] = time.time() - t0
                            th = threading.Thread(target=speak, args=(sentence.strip(), not threads)); th.start(); threads.append(th)
                    pending = parts[-1]
            if pending.strip():
                marks.setdefault("first_sentence", time.time() - t0)
                th = threading.Thread(target=speak, args=(pending.strip(), not threads)); th.start(); threads.append(th)
            for th in threads:
                th.join()
            print(f"STT {t_stt:.2f} s | pierwszy token {marks['first_token']:.2f} s | pierwsze zdanie "
                  f"{marks['first_sentence']:.2f} s | PIERWSZY DŹWIĘK {marks['first_audio']:.2f} s\n"
                  f"  usłyszane: {heard}\n  odpowiedź: {answer.strip()!r}")


if __name__ == "__main__":
    main()
