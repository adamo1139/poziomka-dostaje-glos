#!/usr/bin/env python3
"""Canary-1b-v2 (transformers) po polsku: WER i czas transkrypcji jednej wypowiedzi.

  .venv-stt/bin/python testy/stt_canary.py [--n 150] [--dtype bfloat16]

Dwa zestawy, każda wypowiedź osobno (batch 1, jak w asystencie po końcu tury):
  fleurs  — prawdziwa mowa: google/fleurs pl_pl test (dane/fleurs_pl_test.parquet), pierwsze --n,
            porównanie z polem `transcription` (małe litery, bez interpunkcji);
  tts     — krótkie odpowiedzi z wyniki/tts (VoxCPM2, 48 kHz -> 16 kHz) z ich tekstem.
WER liczony po normalizacji obu stron: małe litery, bez interpunkcji, pojedyncze spacje.
"""
import argparse
import io
import json
import os
import re
import time

import jiwer
import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import AutoProcessor, CanaryForConditionalGeneration

MODEL = "nvidia/canary-1b-v2"
SR = 16000


def norm(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="ile nagrań FLEURS")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--out", default="wyniki/stt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = CanaryForConditionalGeneration.from_pretrained(MODEL, dtype=getattr(torch, a.dtype)).to("cuda").eval()
    load_s = time.time() - t
    print(f"model załadowany w {load_s:.1f} s", flush=True)

    @torch.inference_mode()
    def transcribe(audio):
        torch.cuda.synchronize()
        start = time.time()
        inputs = processor.apply_transcription_request(audio, source_language="pl", target_language="pl",
                                                       sampling_rate=SR, return_tensors="pt")
        inputs = inputs.to("cuda", dtype=model.dtype)
        out = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        torch.cuda.synchronize()
        return text, time.time() - start

    sets = {}
    rows = pq.read_table("dane/fleurs_pl_test.parquet").slice(0, a.n).to_pylist()
    fleurs = []
    for r in rows:
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        fleurs.append((r["id"], librosa.resample(audio, orig_sr=sr, target_sr=SR) if sr != SR else audio,
                       r["transcription"]))
    sets["fleurs"] = fleurs
    tts = json.load(open("wyniki/tts/wyniki.json", encoding="utf-8"))["rows"]
    sets["tts"] = [(f"{r['i']:02d}", librosa.load(f"wyniki/tts/{r['i']:02d}.wav", sr=SR)[0], r["text"]) for r in tts]

    transcribe(sets["tts"][0][1])  # rozgrzewka
    torch.cuda.reset_peak_memory_stats()

    summary = {"model": MODEL, "dtype": a.dtype, "gpu": torch.cuda.get_device_name(), "load_s": round(load_s, 1)}
    for name, items in sets.items():
        out_rows, lat, durs = [], [], []
        for uid, audio, ref in items:
            hyp, dt = transcribe(audio)
            dur = len(audio) / SR
            lat.append(dt); durs.append(dur)
            out_rows.append({"id": uid, "audio_s": round(dur, 2), "latency_s": round(dt, 3),
                             "ref": ref, "hyp": hyp, "wer": round(jiwer.wer(norm(ref), norm(hyp)), 3)})
        wer = jiwer.wer([norm(r["ref"]) for r in out_rows], [norm(r["hyp"]) for r in out_rows])
        summary[name] = {"n": len(out_rows), "wer": round(100 * wer, 2),
                         "audio_s_median": round(float(np.median(durs)), 2),
                         "latency_s": {"median": round(float(np.median(lat)), 3), "p90": round(float(np.percentile(lat, 90)), 3),
                                       "max": round(max(lat), 3)},
                         "rtf_median": round(float(np.median(np.array(lat) / np.array(durs))), 4)}
        with open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=1)
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
        for r in sorted(out_rows, key=lambda r: -r["wer"])[:3]:
            print(f"   WER {r['wer']:.2f}\n     ref: {r['ref']}\n     hyp: {r['hyp']}")
    summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    with open(os.path.join(a.out, "wyniki.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
