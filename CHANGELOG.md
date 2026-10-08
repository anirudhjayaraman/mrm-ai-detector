# Changelog

## 1.0.0 (2026-10-08)

First public release.

- Zero-dependency detector for AI-generated text in model risk documentation:
  domain-evidence, LLM-marker, stylometric and information-theoretic features
  combined in a calibrated logistic model, with per-segment scores, bootstrap
  confidence intervals and mixed-authorship detection.
- Optional `[lm]` ensemble: Binoculars and Fast-DetectGPT (zero-shot) plus the
  HC3, Fakespot and Desklib classifiers.
- Inputs: `.txt .md .rst .docx .pdf .html .eml .msg` and images. OCR is built
  in on macOS (Apple Vision); RapidOCR elsewhere.
- `mrm-detect baseline` compares a document with the author's own earlier
  writing.
- `mrm-detect train` / `evaluate` calibrate on an organisation's own documents.
- Reproducible benchmark against nine open-source detectors and three free web
  detectors (see `benchmarks/`).
