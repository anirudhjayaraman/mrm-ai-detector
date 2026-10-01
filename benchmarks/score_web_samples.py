#!/usr/bin/env python3
"""Score the 8 web-detector samples with this package and tabulate against GPTZero / ZeroGPT / Sapling.

The web detectors were run by hand (results in ``results/web_detectors.json``);
the source documents of these samples are pinned to the held-out split, so the
comparison is out-of-sample for this package as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from mrm_ai_detector import Detector, ScoringModel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lm", action="store_true", help="also score with the language-model-augmented model")
    args = ap.parse_args(argv)

    web = json.loads((ROOT / "results" / "web_detectors.json").read_text())
    samples = sorted((ROOT / "web_samples").glob("*.txt"))
    detectors = {"mrm-detector": Detector(ScoringModel.default(neural=False), bootstrap=0)}
    if args.lm:
        detectors["mrm-detector+lm"] = Detector(ScoringModel.default(neural=True), use_neural=True, bootstrap=0)

    ours = {name: {} for name in detectors}
    for path in samples:
        text = path.read_text(encoding="utf-8")
        for name, det in detectors.items():
            r = det.analyze_text(text, source=path.name)
            ours[name][path.stem] = {"ai": round(100 * r.probability, 1), "verdict": r.verdict, "flagged": r.probability >= det.model.ai_threshold}

    cols = ["gptzero_basic", "zerogpt", "sapling"] + list(detectors)
    header = "| Sample | Truth | " + " | ".join(cols) + " |"
    lines = [header, "|" + "---|" * (len(cols) + 2)]
    for path in samples:
        key = path.stem
        truth = "AI" if key.startswith("a") else "Human"
        cells = []
        for c in cols:
            v = (web.get(c) or ours.get(c, {})).get(key, {}) if c in web else ours[c].get(key, {})
            cells.append(f"{v.get('ai', '–')}%" if v else "–")
        lines.append(f"| `{key}` | {truth} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (ROOT / "results" / "web_comparison.md").write_text(
        "AI likelihood (%) reported by each detector. Human samples are pre-2023 US federal texts; "
        "AI samples a1–a3 are Claude-written, a4 is Qwen2.5-1.5B.\n\n" + table + "\n"
    )
    (ROOT / "results" / "web_comparison_ours.json").write_text(json.dumps(ours, indent=2))
    print(table)
    for name, det in detectors.items():
        print(f"{name}: threshold {det.model.ai_threshold:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
