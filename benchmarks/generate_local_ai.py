#!/usr/bin/env python3
"""Generate in-domain AI-written model risk documents with local open-weight LLMs.

These complement the public ChatGPT/LLaMA/Cohere datasets with text that is
*about model risk* (validation findings, model documentation, monitoring
reports ...), produced by model families other than the one used for the
committed Claude-written examples.

Usage::

    pip install '.[lm]'
    python benchmarks/generate_local_ai.py --models Qwen/Qwen2.5-1.5B-Instruct --n 30

Outputs ``benchmarks/ai_mrm/<model-slug>/<nnn>.md`` with the prompt recorded
in a leading HTML comment.
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DOC_TYPES = [
    "the conceptual soundness section of an independent model validation report",
    "the executive summary of a model validation report, including findings and their severity",
    "the data quality and representativeness assessment section of a validation report",
    "the outcomes analysis and back-testing section of a validation report",
    "the model limitations and compensating controls section of model development documentation",
    "the methodology section of model development documentation",
    "a quarterly ongoing performance monitoring report",
    "an annual model review memo to the model risk committee",
    "a validation finding write-up with remediation actions and due dates",
    "a section of a bank's model risk management policy covering model tiering and validation frequency",
    "a gap analysis of the bank's model risk framework against PRA SS1/23",
    "an explanation of how the bank complies with SR 11-7 effective challenge requirements",
    "the sensitivity analysis and benchmarking section of a validation report",
    "the model change and re-validation assessment for a material model change",
]
# Genre-matched prompts: AI text in the same genres as the human reference corpus
# (regulatory guidance, academic papers, audit findings, comment letters ...), so the
# detector cannot learn "genre" as a proxy for authorship.
GENRE_DOC_TYPES = [
    "a section of supervisory guidance from a banking regulator setting out expectations for",
    "the introduction of an academic finance paper studying model risk in",
    "the literature review section of an academic paper on the validation of",
    "a finding from an Inspector General audit report on a regulator's oversight of",
    "a comment letter from a bank responding to a regulator's consultation on the governance of",
    "a section of a supervisory report summarising findings from a review of banks' practices for",
    "a section of a central bank staff working paper evaluating the performance of",
    "the model description section of a regulator's published methodology document for",
]
MODELS = [
    "a retail mortgage PD scorecard (logistic regression) used for IRB capital",
    "an IFRS 9 expected credit loss model with macroeconomic scenarios",
    "a wholesale LGD model based on workout recoveries",
    "a CCAR/DFAST pre-provision net revenue (PPNR) regression model",
    "a historical-simulation Value-at-Risk model for the trading book",
    "a Heston stochastic-volatility model used to price equity exotics",
    "a SABR model for interest rate swaption volatility",
    "a gradient-boosted machine learning credit card application scorecard",
    "an anti-money-laundering transaction monitoring model",
    "a CVA/XVA exposure simulation engine",
    "an interest rate risk in the banking book (IRRBB) deposit behaviour model",
    "a vendor commercial real estate default model",
    "an EAD/CCF model for revolving credit facilities",
    "a fraud detection neural network",
]
STYLES = [
    "",
    "Write in a formal, professional tone.",
    "Be concise and specific, and include realistic quantitative results (test statistics, thresholds, sample sizes, dates).",
    "Write as an experienced model validator would, in plain language.",
    "Use headings and bullet points where helpful.",
    "Make it sound like it was written by a human practitioner, not an AI.",
]


def build_prompts(n: int, seed: int, prompt_set: str = "mrm"):
    rng = random.Random(seed)
    if prompt_set == "genre":
        combos = [(f"{d} {m}", "", st) for d, m, st in itertools.product(GENRE_DOC_TYPES, MODELS, STYLES[:4])]
        rng.shuffle(combos)
        for doc, _, style in combos[:n]:
            yield f"Write {doc}. Aim for about {rng.choice([300, 400, 500, 600])} words. {style}".strip()
        return
    combos = list(itertools.product(DOC_TYPES, MODELS, STYLES))
    rng.shuffle(combos)
    for doc, model, style in combos[:n]:
        words = rng.choice([300, 400, 500, 600])
        yield f"Write {doc} for {model}. Aim for about {words} words. {style}".strip()


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.split("/")[-1].lower()).strip("-")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"])
    ap.add_argument("--n", type=int, default=30, help="documents per model")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--prompt-set", choices=["mrm", "genre"], default="mrm")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    for mi, name in enumerate(args.models):
        out_dir = ROOT / "ai_mrm" / slug(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
        prefix = "g" if args.prompt_set == "genre" else ""
        for i, prompt in enumerate(build_prompts(args.n, args.seed + mi, args.prompt_set)):
            path = out_dir / f"{prefix}{i:03d}.md"
            if path.exists():
                continue
            t0 = time.time()
            msgs = [{"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
            ids = enc["input_ids"]
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.8, top_p=0.95,
                    repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
            path.write_text(f"<!-- model: {name}; prompt: {prompt.replace('--', '-')} -->\n{text}\n", encoding="utf-8")
            print(f"{name} {i:03d} {len(text.split())} words {time.time() - t0:.0f}s", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
