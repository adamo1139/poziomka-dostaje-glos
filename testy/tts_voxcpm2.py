#!/usr/bin/env python3
"""VoxCPM2 na polskich odpowiedziach asystenta: czas do pierwszego audio, RTF, VRAM.

  .venv-tts/bin/python testy/tts_voxcpm2.py [--reference glos.wav] [--out wyniki/tts]

Tryb streaming (generate_streaming), bez odszumiacza. Pierwsze zdanie to rozgrzewka
(torch.compile), liczone osobno. Wynik: wyniki/tts/wyniki.json + jeden WAV 48 kHz na zdanie.
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM

ZDANIA = [
    "Dzień dobry! W czym mogę pomóc?",
    "Jutro w Warszawie będzie pochmurno, około dwunastu stopni.",
    "Jasne, ustawiam przypomnienie na siódmą trzydzieści rano.",
    "Nie jestem pewna, ale wydaje mi się, że to było w tysiąc dziewięćset osiemdziesiątym dziewiątym roku.",
    "Pociąg do Krakowa odjeżdża z peronu trzeciego o czternastej piętnaście.",
    "Przepraszam, nie usłyszałam. Możesz powtórzyć?",
    "Najprościej będzie ugotować makaron, a w tym czasie podsmażyć cebulę z czosnkiem.",
    "Szczebrzeszyn słynie z chrząszcza, który brzmi w trzcinie.",
    "To zależy od tego, ile masz czasu. Jeśli godzinę, polecam krótki spacer nad Wisłą.",
    "Twoje spotkanie z panią Agnieszką zaczyna się za piętnaście minut.",
    "Tak.",
    "Mam dla ciebie trzy propozycje: kino, teatr albo koncert jazzowy w Starym Mieście.",
    "Według moich danych liczba ludności Polski to około trzydziestu siedmiu milionów.",
    "Oczywiście, zaraz sprawdzę rozkład autobusów.",
    "Fotosynteza to proces, w którym rośliny zamieniają światło słoneczne na energię chemiczną.",
    "Hmm, dobre pytanie. Daj mi chwilę do namysłu.",
    "Wysłałam wiadomość do Tomka z informacją, że się spóźnisz.",
    "Gżegżółka i źdźbło to słowa, które sprawiają kłopot nawet Polakom.",
    "Na razie to wszystko. Miłego dnia!",
    "Pamiętaj, żeby wziąć parasol, bo po południu może padać.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", help="WAV do klonowania głosu (bez niego głos jest losowy)")
    ap.add_argument("--out", default="wyniki/tts")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=not a.no_compile)
    load_s = time.time() - t
    sr = model.tts_model.sample_rate
    print(f"model załadowany w {load_s:.1f} s, sample rate {sr}", flush=True)

    def synth(text):
        torch.cuda.synchronize()
        start = time.time()
        first, chunks = None, []
        for chunk in model.generate_streaming(text=text, reference_wav_path=a.reference):
            if first is None:
                first = time.time() - start
            chunks.append(chunk)
        total = time.time() - start
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, first, total, len(chunks)

    t = time.time()
    synth("To jest zdanie na rozgrzewkę.")
    warmup_s = time.time() - t
    print(f"rozgrzewka {warmup_s:.1f} s", flush=True)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for i, text in enumerate(ZDANIA):
        audio, first, total, n = synth(text)
        dur = len(audio) / sr
        sf.write(os.path.join(a.out, f"{i:02d}.wav"), audio, sr)
        rows.append({"i": i, "text": text, "first_audio_s": round(first, 3), "total_s": round(total, 3),
                     "audio_s": round(dur, 2), "rtf": round(total / dur, 3) if dur else None, "chunks": n})
        print(f"{i:02d} pierwsze audio {first:.2f} s, całość {total:.2f} s, audio {dur:.1f} s, "
              f"RTF {total / dur:.2f}  | {text[:50]}", flush=True)

    firsts = [r["first_audio_s"] for r in rows]
    rtfs = [r["rtf"] for r in rows if r["rtf"]]
    summary = {
        "model": "openbmb/VoxCPM2", "gpu": torch.cuda.get_device_name(), "sample_rate": sr,
        "compile": not a.no_compile, "reference": a.reference,
        "load_s": round(load_s, 1), "warmup_s": round(warmup_s, 1),
        "first_audio_s": {"median": round(float(np.median(firsts)), 3), "max": round(max(firsts), 3)},
        "rtf": {"median": round(float(np.median(rtfs)), 3), "max": round(max(rtfs), 3)},
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "rows": rows,
    }
    with open(os.path.join(a.out, "wyniki.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
