# mrm-ai-detector

**Detect AI-generated text in model risk documentation**: validation reports,
model development documentation, audit issues, MRM policies, monitoring packs
and review e-mails.

* **Zero-install core.** Pure Python standard library, 3.9+. Clone and run.
* **Built for MRM text.** Features include evidence density (test statistics,
  data windows, model IDs, regulatory cross-references) and *unsupported
  evaluative claims* ("the model is robust and well-calibrated" with nothing
  behind it), alongside LLM-style markers and stylometry.
* **Calibrated and benchmarked on real MRM literature.** Human-written
  documents (Fed SR 11-7, PRA SS3/18 and CP6/22, ECB TRIM, Fed and FHFA OIG
  model-validation reviews, GAO model reviews, Fed DFAST model documentation,
  q-fin papers) against AI-written MRM text from several model families. See
  [benchmarks/](benchmarks/README.md).
* **Optional ensemble with open-source detectors** (`--lm`). Binoculars and
  Fast-DetectGPT (zero-shot), plus the HC3, Fakespot (2025) and Desklib (2025)
  classifiers. On their own, the 2025 classifiers flag a third or more of human
  regulatory text in our benchmark. Inside the calibrated ensemble their
  false positives are kept in check.
* **Explainable output.** Per-segment scores, the top contributing signals, a
  bootstrap confidence interval, and *mixed-authorship* detection that flags
  which sections look AI-written.
* **Reads what colleagues actually send.** `.docx`, `.pdf`, `.md`, `.txt`,
  `.html`, e-mails (`.eml`, Outlook `.msg`; quoted thread history is stripped),
  and photos or screenshots via OCR (built in on macOS).

> **Read [Responsible use](#responsible-use) before applying this to colleagues' work.**
> Every AI-text detector, this one included, produces false positives. A score
> is a reason to look more closely, never evidence on its own.

---

## Quick start

```bash
git clone <this repo> && cd mrm-ai-detector
python -m mrm_ai_detector scan examples/
```

No installation is needed for `.txt`, `.md`, `.docx`, `.html` and `.eml`, or,
**on a Mac, for photos and screenshots** (`.heic`, `.jpg`, `.png` …). These use
Apple's built-in Live Text OCR engine:

```bash
python3 -m mrm_ai_detector scan ~/Downloads/IMG_1234.HEIC
```

```
Document                                              Words  AI prob  AI share  Verdict
----------------------------------------------------------------------------------------------------
examples/ai_generated_methodology_qwen.md               561      90%       65%  Likely AI-generated
examples/ai_generated_validation_summary_claude.md      364      11%        0%  Likely human-written
examples/human_sr11-7_validation_excerpt.txt            723      52%        0%  Inconclusive
examples/mixed_validation_report.md                    1284      26%        0%  Likely human-written
```

The examples were chosen to show the limits honestly. Text from a small open
model is caught, but the Claude-written summary passes as human, and so does
the report with an inserted Claude-written section. SR 11-7, a human text,
sits in the inconclusive band. See [Benchmark summary](#benchmark-summary).

Install the command-line tool and optional extras:

```bash
pip install .            # adds the `mrm-detect` command (no dependencies)
pip install ".[pdf]"     # PDF support (pypdf)
pip install ".[email]"   # Outlook .msg support (olefile)
pip install ".[ocr]"     # photos/screenshots on Windows/Linux (RapidOCR); macOS needs nothing
pip install ".[lm]"      # language-model ensemble (torch + transformers; ~3.5 GB of models downloaded on first use)
pip install ".[all]"
```

## Usage

```bash
mrm-detect scan report.docx                       # detailed text report
mrm-detect scan reports/ --summary                # one line per document
mrm-detect scan reports/ -f html -o report.html   # self-contained HTML report with highlighted segments
mrm-detect scan issue.msg -f json                 # machine-readable
mrm-detect scan report.pdf --lm                   # add Binoculars / Fast-DetectGPT / HC3 / Fakespot / Desklib features
cat draft.txt | mrm-detect scan -                 # stdin
mrm-detect scan submissions/ --fail-on-ai         # exit code 3 if anything is flagged (CI / workflow gate)
mrm-detect features report.docx                   # raw feature values, for analysts
```

Python API:

```python
from mrm_ai_detector import Detector

result = Detector().analyze_file("validation_report.docx")
print(result.verdict, f"{result.probability:.0%}", f"(90% CI {result.ci_low:.0%}–{result.ci_high:.0%})")
for seg in result.segments:
    if seg.flagged:
        print(seg.section, f"{seg.probability:.0%}", [s.feature for s in seg.signals])
```

### Verdicts

| Verdict | Meaning |
|---|---|
| Likely AI-generated | Document probability ≥ the model's AI threshold (calibrated for ≤ 2% false positives on human MRM documents) |
| Mixed: partially AI-generated | Not flagged overall, but ≥ 15% of the text sits in segments that individually exceed the threshold |
| Likely human-written | Probability ≤ 30% and no material flagged sections |
| Inconclusive | Anything in between |
| Insufficient text | Fewer than 150 words of prose. Detection is unreliable on short text |

## How it works

1. **Ingest and normalise.** Text is extracted, PDF line breaks reflowed,
   running headers and footers removed, and e-mail threads cut to the newest
   message. Prose inside table cells is kept, since audit issues are often
   written in tables.
2. **Segment.** The text is split into roughly 180-word chunks. Headings are
   treated as soft boundaries, so heading-dense validation reports still give
   scorable chunks.
3. **Features** (`mrm-detect features` lists them all):
   * *Domain evidence.* Numeric density, metric statements with values (Gini
     0.61, PSI = 0.08), specific references (model IDs, dates, SR 11-7,
     "Section 4.2"), the ratio of unsupported evaluative claims, governance
     boilerplate, hedging.
   * *LLM markers.* Weighted lexicon ("delve", "it is important to note",
     "plays a pivotal role" …) with MRM vocabulary such as "robust" and
     "comprehensive" down-weighted. Also stock sentence openers, ", ensuring
     …" participle tails, "not only … but", and triplets.
   * *Stylometry.* Sentence-length burstiness, lexical diversity, word length,
     function words, punctuation habits.
   * *Information theory.* Windowed compression ratio, character-trigram
     entropy, phrase repetition.
   * *Optional language-model features* (`--lm`). Binoculars (Hans et al.,
     ICML 2024) and Fast-DetectGPT (Bao et al., ICLR 2024) with a
     Qwen2.5-0.5B observer/performer pair, plus the logits of three
     open-source classifiers: HC3 RoBERTa (Guo et al., 2023), Fakespot RoBERTa
     (2025) and Desklib DeBERTa-v3 (2025).
4. **Score.** Features are standardised against the calibration corpus and
   combined by an L2-regularised logistic model. Logits on short texts are
   shrunk towards 50% ("no evidence"). A sentence bootstrap gives a 90%
   confidence interval.
5. **Explain.** Every score comes with the features that pushed it up or down.

## Benchmark summary

Held-out documents never used in training are split by document, and every
human document predates ChatGPT. Full tables are in
[`benchmarks/results/`](benchmarks/results). Reproduce them with the
commands in [benchmarks/README.md](benchmarks/README.md).

### Head-to-head on 331 held-out documents (102 in-domain MRM documents)

| Detector | In-domain doc AUC | In-domain TPR @ 5% FPR | All-doc AUC | Doc TPR / FPR at its own threshold |
|---|---:|---:|---:|---|
| **mrm-detector `--lm`** (this package, ensemble) | **0.962** | **0.891** | **0.985** | 0.906 / **0.000** |
| Fakespot RoBERTa (2025) | 0.929 | 0.848 | 0.965 | 0.947 / 0.163 |
| Desklib DeBERTa-v3 (2025) | 0.924 | 0.870 | 0.961 | 0.930 / 0.144 |
| **mrm-detector** (stdlib only, zero install) | 0.899 | 0.783 | 0.933 | 0.515 / 0.006 |
| HC3 ChatGPT RoBERTa (2023) | 0.755 | 0.217 | 0.923 | 0.620 / 0.019 |
| RADAR (2023) | 0.641 | 0.130 | 0.825 | 0.655 / 0.169 |
| Binoculars (Qwen2.5-0.5B pair) | 0.505 | 0.109 | 0.787 | n/a |
| Fast-DetectGPT (Qwen2.5-0.5B) | 0.408 | 0.087 | 0.741 | n/a |
| OpenAI GPT-2 output detector (2019) | 0.407 | 0.022 | 0.801 | 0.550 / 0.037 |

Fakespot and Desklib, used on their own, flag a third or more of human
regulatory, supervisory and academic MRM documents (Fakespot: half of the
academic papers and comment letters). They are, however, the only standalone
detectors that recognised some Claude-written MRM text (56% and 44%). Inside the
ensemble, false positives were zero across every human document family.

**Detection rate by generator (ensemble, held-out documents):** Qwen2.5 100%,
SmolLM2 100%, Phi-3.5 100%, gpt-3.5-turbo / ChatGPT 100%, Cohere 100%,
Flan-T5 87%, **text-davinci-003 20%, Claude 33%**.

### Free web detectors on 8 short samples (held out for this package too)

Four human samples are pre-2023 US federal texts (SR 11-7, Fed OIG, GAO, FHFA
OIG). Three AI samples are Claude-written MRM text and one is Qwen2.5-written.

| Detector | AI samples caught | Human samples falsely flagged |
|---|---|---|
| GPTZero (free tier) | 4 / 4 | 1 / 4 (FHFA OIG 2020 report: 100% AI) |
| mrm-detector `--lm` | 2 / 4 | 1 / 4 (SR 11-7 excerpt: 63%) |
| ZeroGPT | 1 / 4 | 0 / 4 |
| mrm-detector (stdlib) | 1 / 4 | 0 / 4 |
| Sapling | 1 / 4 | 2 / 4 (SR 11-7: 95.5%, FHFA OIG: 71.9%) |

Per-sample numbers are in [`benchmarks/results/web_comparison.md`](benchmarks/results/web_comparison.md).

### What this means in practice

* Text from small and mid-sized open models, and from ChatGPT-3.5-era models,
  is detected reliably, with a very low false-positive rate on genuine MRM
  documents.
* **Polished frontier-model text (Claude-class) is the hard case.** Only about
  a third is caught on a full document. On short excerpts, GPTZero, a commercial
  detector trained on current models, did better. But it means sending
  documents to a third party, and it falsely flagged a 2020 FHFA OIG report.
* For colleague reviews, combine the document score with the
  [author-baseline comparison](#comparing-against-an-authors-own-earlier-work-recommended-for-colleague-reviews)
  and with a substantive review of the evidence (numbers, tests,
  references). The domain-evidence signals in every report support that review.

## Comparing against an author's own earlier work (recommended for colleague reviews)

Generic detectors judge everyone against one population, so writers whose
natural style is formal, template-driven or non-native get flagged more often.
The **baseline** mode asks a fairer question: *is this document unusual for
this person?*

```bash
mrm-detect baseline new_issue_writeup.docx --reference colleague_2019_2022_docs/
```

```text
# illustrative output
Style distance: 1.84  (author's own documents: median 0.71, 90th pct 1.02)
Empirical p-value: 0.08   Shift towards AI-typical features: +0.62
Unusual for this author, and the shift is towards LLM-typical writing. Worth a closer look.
Largest shifts (z vs author's own writing):
  +2.91 AI-ward  ai_marker_rate ...
```

Use at least 5 reference documents, ideally 10 or more, of the same genre
(issue write-ups against issue write-ups) from before 2023. A low p-value with
an AI-ward shift is a reason to talk to the author, not a conclusion.

## Calibrating on your organisation's documents

The shipped calibration uses public documents. Your organisation's house style
will differ, so **calibrate locally before relying on scores**:

1. **Human set.** Collect documents written before December 2022: validation
   reports, audit issues, e-mails. Their authorship isn't in doubt.
2. **AI set.** Ask your approved LLM(s) to write comparable documents (same
   templates, same models), including prompts like "make it sound like a
   human validator wrote it".
3. Train and evaluate. Everything stays on your machine:

```bash
mrm-detect train --human corpus/human --ai corpus/ai -o our_model.json --save-dataset features.jsonl
mrm-detect evaluate --human holdout/human --ai holdout/ai --model our_model.json
mrm-detect scan new_reports/ --model our_model.json
```

`train` reports document-grouped cross-validated AUC and TPR at 1% and 5%
FPR. Choose `--threshold` from your own false-positive tolerance.

## Responsible use

* **False positives are unavoidable.** In our tests the free web detectors
  rated a 2011 Federal Reserve guidance text 95.5% AI (Sapling) and a 2020
  FHFA OIG report 100% AI (GPTZero). This tool is calibrated to keep human
  MRM documents below a 2% false-positive rate on the benchmark, but your
  documents aren't the benchmark.
* **Non-native English writers are flagged more often** by perplexity- and
  style-based detectors (Liang et al., 2023, *GPT detectors are biased against
  non-native English writers*). Formulaic, template-driven writing such as
  audit-issue templates also looks "low-perplexity". Weigh scores
  accordingly.
* **AI assistance isn't misconduct.** Translation, grammar fixes and
  structured drafting may be permitted under your firm's AI policy. Ask what
  your policy actually prohibits, and whether the *content* (evidence, numbers,
  conclusions) is sound. The domain-evidence signals in the report help with
  that question whoever wrote the text.
* **Use scores for triage, never as the sole basis** for any decision about a
  person. Keep a human in the loop, and give authors the opportunity to
  explain.
* **Confidentiality.** Everything runs locally. Nothing is sent anywhere,
  including with `--lm`, whose models are downloaded once from Hugging Face
  and run on your machine. Don't paste internal documents into web detectors.
* **OCR input is noisy.** Scan the original document rather than a photo when
  you can. Reports from OCR'd input carry a warning.

## Troubleshooting

* **`--lm` is slow the first time.** About 3.5 GB of models are downloaded
  once, then cached. Scoring runs at about 1–2 s per 180-word segment on a
  laptop.
* **Model download hangs.** Set `HF_HUB_DISABLE_XET=1`. Behind a corporate
  proxy, pre-download the models listed in
  `mrm_ai_detector/features/neural.py` and set `HF_HUB_OFFLINE=1`.
* **Memory.** `--lm` needs about 4 GB of RAM. Use
  `MRM_DETECTOR_CLASSIFIERS=hc3,fakespot` to drop the large Desklib model,
  then recalibrate.
* **Scanned PDFs.** Text-less PDFs raise an error; convert pages to images
  and scan those (OCR), or run OCR first.

## Project layout

```
mrm_ai_detector/      the package (stdlib core; neural/OCR/PDF features optional)
  text.py             normalisation, sentence splitting, segmentation
  ingest.py           .txt .md .docx .pdf .html .eml .msg, images (OCR)
  features/           domain, markers, stylometry, entropy, neural (optional)
  scoring.py          standardised logistic scoring model (JSON)
  detector.py         orchestration, bootstrap CI, verdicts
  train.py            calibration: IRLS ridge logistic, grouped CV, metrics
  report.py           text / JSON / Markdown / HTML reports
  models/             shipped calibrated models
benchmarks/           corpus manifest, fetch/generate/calibrate/compare scripts, results
tests/                pytest suite
examples/             public-domain human text, AI-written and mixed examples
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check mrm_ai_detector tests
```

## Licence and attribution

MIT. Method attributions: Binoculars (Hans et al., 2024, BSD-3), Fast-DetectGPT
(Bao et al., 2024, MIT), HC3 / chatgpt-detector-roberta (Guo et al., 2023),
fakespot-ai/roberta-base-ai-text-detection-v1 (Apache-2.0),
desklib/ai-text-detector-v1.01 (MIT), M4
(Wang et al., 2024), RapidOCR (Apache-2.0). Human corpus texts are downloaded
from their publishers at benchmark time and aren't redistributed. The examples
use US federal government works, which are in the public domain.
