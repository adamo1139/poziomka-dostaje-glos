#!/usr/bin/env python3
"""Poziomka-Instruct -> checkpoint FP8 (llm-compressor FP8_DYNAMIC, bez danych kalibracyjnych).

  .venv-quant/bin/python kwantyzuj_fp8.py [--out modele/Poziomka-Instruct-2026-10-20-FP8]

FP8_DYNAMIC: wagi FP8 per kanał (statycznie), aktywacje FP8 per token (dynamicznie przy inferencji).
Kwantyzowane są wszystkie nn.Linear poza lm_head: eksperci, eksperci współdzieleni, attention
(query_key_value, dense) i gęsty MLP pierwszej warstwy. Router MoE (BailingMoeV2Gate) to nie
nn.Linear, więc zostaje w bf16 razem z expert_bias. Wynik w formacie compressed-tensors, czytany
natywnie przez SGLang (--quantization nie jest wtedy potrzebne).
"""
import argparse
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, REVISION = "cpral/Poziomka-Instruct-2026-10-20", "c74ec04fbd2279653d8a2b383f6b529607180890"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="modele/Poziomka-Instruct-2026-10-20-FP8")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} już istnieje")

    # Kod modelu z repo jest pisany pod transformers 4.x; modele/zrodlo to dowiązania do snapshotu
    # z poprawionym modeling_bailing_moe_v2.py (import is_torch_fx_available, RoPE "default",
    # _tied_weights_keys jako dict).
    # W transformers 5 bufor inv_freq zostaje niezainicjalizowany (model na meta), więc forward przez
    # HF jest tu niepoprawny; kwantyzacja FP8_DYNAMIC nie robi forwardu, a inv_freq nie trafia do
    # checkpointu, więc wynik nie jest tym dotknięty.
    src = "modele/zrodlo"
    snapshot = snapshot_download(MODEL, revision=REVISION)
    for name in os.listdir(snapshot):  # dowiązania do snapshotu; poprawiony modeling_*.py jest w repo
        link = os.path.join(src, name)
        if name != "modeling_bailing_moe_v2.py" and not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), link)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe, trust_remote_code_model=True)

    model.save_pretrained(a.out, save_compressed=True)
    tokenizer.save_pretrained(a.out)
    # Kod modelu i szablon czatu z oryginału (save_pretrained ich nie kopiuje).
    for name in ("configuration_bailing_moe_v2.py", "modeling_bailing_moe_v2.py", "chat_template.jinja",
                 "generation_config.json", "special_tokens_map.json"):
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(a.out, name))
    with open(os.path.join(a.out, "ZRODLO.txt"), "w") as f:
        f.write(f"{MODEL}@{REVISION}\nllm-compressor FP8_DYNAMIC, targets=Linear, ignore=lm_head\n")
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
