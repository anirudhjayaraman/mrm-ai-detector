"""Command-line interface: ``mrm-detect`` / ``python -m mrm_ai_detector``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .detector import Detector, DocumentResult
from .features import REGISTRY, extract
from .features.neural import NEURAL_FEATURES, NeuralUnavailable
from .ingest import IngestError, iter_documents, read_document
from .report import render_html, render_json, render_markdown, render_text
from .scoring import ScoringModel
from .text import build_unit

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_FLAGGED = 0, 1, 2, 3
logger = logging.getLogger("mrm_ai_detector")


def _load_model(args: argparse.Namespace) -> Optional[ScoringModel]:
    return ScoringModel.load(args.model) if getattr(args, "model", None) else None


def _write(text: str, output: Optional[str]) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
        logger.info("wrote %s", output)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ---------------------------------------------------------------------------
def cmd_scan(args: argparse.Namespace) -> int:
    model = _load_model(args)
    if model is not None and args.threshold is not None:
        model.ai_threshold = args.threshold
    detector = Detector(model, use_neural=args.lm, segment_words=args.segment_words, bootstrap=args.bootstrap)
    if model is None and args.threshold is not None:
        detector.model.ai_threshold = args.threshold

    results: List[DocumentResult] = []
    errors = 0
    if args.paths == ["-"]:
        results.append(detector.analyze_text(sys.stdin.read(), source="<stdin>"))
    else:
        for path in iter_documents(args.paths, recursive=not args.no_recursive):
            try:
                results.append(detector.analyze_file(path))
            except IngestError as exc:
                errors += 1
                logger.error("%s", exc)
    if not results:
        logger.error("no readable documents found")
        return EXIT_ERROR

    fmt = args.format
    if fmt == "json":
        out = render_json(results, include_text=args.include_text)
    elif fmt == "html":
        out = render_html(results)
    elif fmt == "markdown":
        out = render_markdown(results)
    else:
        out = render_text(results, color=sys.stdout.isatty() and not args.output and not args.no_color, detail=not args.summary)
    _write(out, args.output)

    if errors:
        return EXIT_ERROR
    if args.fail_on_ai and any(r.flagged for r in results):
        return EXIT_FLAGGED
    return EXIT_OK


def cmd_features(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.path == "-" else read_document(args.path)
    unit = build_unit(text)
    values = extract(unit)
    if args.lm:
        from .features.neural import NeuralScorer

        values.update(NeuralScorer().score(unit.text))
    rows = {k: {"value": v, "description": REGISTRY[k].description if k in REGISTRY else NEURAL_FEATURES.get(k, "")} for k, v in values.items()}
    _write(json.dumps({"words": unit.word_count, "sentences": len(unit.sentences), "features": rows}, indent=2), args.output)
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace) -> int:
    from .baseline import compare_to_baseline, render_baseline

    outputs = []
    for target in iter_documents(args.target):
        result = compare_to_baseline(args.reference, read_document(target), str(target))
        outputs.append(result)
    if args.format == "json":
        _write(json.dumps([r.to_dict() for r in outputs], indent=2), args.output)
    else:
        _write("\n\n".join(render_baseline(r) for r in outputs), args.output)
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from .features.neural import NeuralScorer
    from .train import load_rows, rows_from_paths, save_rows, train

    if args.dataset:
        rows = []
        for ds in args.dataset:
            rows.extend(load_rows(Path(ds)))
    else:
        if not (args.human and args.ai):
            logger.error("provide --human and --ai directories, or --dataset JSONL files")
            return EXIT_USAGE
        neural = NeuralScorer() if args.lm else None
        rows = rows_from_paths(args.human, 0, neural=neural) + rows_from_paths(args.ai, 1, neural=neural)
        if args.save_dataset:
            save_rows(rows, Path(args.save_dataset))
    features = None
    if not args.lm:
        features = [f for f in REGISTRY]
    model, cv = train(rows, folds=args.folds, name=args.name, description=args.description, l2=args.l2, features=features)
    model.save(args.output)
    print(json.dumps({"output": args.output, "segments": len(rows), "cv": cv}, indent=2, default=str))
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .train import classification_metrics

    detector = Detector(_load_model(args), use_neural=args.lm, bootstrap=0)
    probs, labels, rows = [], [], []
    for label, paths in ((0, args.human), (1, args.ai)):
        for path in iter_documents(paths):
            r = detector.analyze_file(path)
            probs.append(r.probability)
            labels.append(label)
            rows.append({"source": r.source, "label": label, "probability": round(r.probability, 4), "verdict": r.verdict})
    metrics = classification_metrics(probs, labels, threshold=detector.model.ai_threshold)
    _write(json.dumps({"model": detector.model.name, "metrics": metrics, "documents": rows}, indent=2), args.output)
    return EXIT_OK


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mrm-detect",
        description="Detect AI-generated content in model risk documentation (validation reports, model documentation, policies).",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0, help="increase log verbosity")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="score documents (.txt .md .docx .pdf .html) or directories")
    s.add_argument("paths", nargs="+", help="files or directories; '-' reads stdin")
    s.add_argument("-f", "--format", choices=["text", "json", "markdown", "html"], default="text")
    s.add_argument("-o", "--output", help="write report to this file instead of stdout")
    s.add_argument("--model", help="path to a calibrated model JSON (default: shipped model)")
    s.add_argument("--lm", action="store_true", help="add language-model features (needs the [lm] extra)")
    s.add_argument("--threshold", type=float, help="AI probability threshold for flagging (default from model, 0.70)")
    s.add_argument("--segment-words", type=int, default=180, help="target words per scored segment")
    s.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for the confidence interval (0 = off)")
    s.add_argument("--fail-on-ai", action="store_true", help=f"exit with code {EXIT_FLAGGED} if any document is flagged (for CI gates)")
    s.add_argument("--summary", action="store_true", help="text format: only print the summary table")
    s.add_argument("--include-text", action="store_true", help="json format: include segment text")
    s.add_argument("--no-recursive", action="store_true", help="do not descend into sub-directories")
    s.add_argument("--no-color", action="store_true")
    s.set_defaults(func=cmd_scan)

    f = sub.add_parser("features", help="dump raw feature values for one document")
    f.add_argument("path")
    f.add_argument("--lm", action="store_true")
    f.add_argument("-o", "--output")
    f.set_defaults(func=cmd_features)

    b = sub.add_parser("baseline", help="compare documents with an author's own earlier writing (style-shift test)")
    b.add_argument("target", nargs="+", help="document(s) to test")
    b.add_argument("--reference", nargs="+", required=True, help="the same author's earlier documents (ideally pre-2023)")
    b.add_argument("-f", "--format", choices=["text", "json"], default="text")
    b.add_argument("-o", "--output")
    b.set_defaults(func=cmd_baseline)

    t = sub.add_parser("train", help="calibrate a model on your own labelled human / AI documents")
    t.add_argument("--human", nargs="+", help="files/directories of human-written documents")
    t.add_argument("--ai", nargs="+", help="files/directories of AI-generated documents")
    t.add_argument("--dataset", nargs="+", help="pre-extracted feature JSONL file(s) instead of --human/--ai")
    t.add_argument("--save-dataset", help="cache extracted features as JSONL")
    t.add_argument("-o", "--output", required=True, help="where to write the model JSON")
    t.add_argument("--name", default="custom")
    t.add_argument("--description", default="")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--l2", type=float, default=2.0)
    t.add_argument("--lm", action="store_true", help="include language-model features")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="measure a model on labelled documents")
    e.add_argument("--human", nargs="+", required=True)
    e.add_argument("--ai", nargs="+", required=True)
    e.add_argument("--model")
    e.add_argument("--lm", action="store_true")
    e.add_argument("-o", "--output")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except (IngestError, NeuralUnavailable, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
