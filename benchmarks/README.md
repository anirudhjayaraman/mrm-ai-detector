# Benchmarks and calibration

Everything needed to reproduce the shipped calibration and the head-to-head
comparison with other detectors lives here.

```bash
pip install -e ".[all]"                       # pypdf, torch, transformers, OCR extras
python benchmarks/fetch_corpus.py             # download + normalise the corpus (~5 min)
python benchmarks/generate_local_ai.py        # optional: regenerate local open-model AI text
python benchmarks/calibrate.py --lm           # fit default_model.json and default_model_lm.json
python benchmarks/compare_detectors.py        # head-to-head vs open-source detectors
python benchmarks/build_mixed.py              # optional: regenerate mixed-authorship documents (needs mlx-lm)
python benchmarks/eval_mixed.py --lm          # AI-share accuracy on mixed documents, APT-Eval, MixSet
```

Downloaded text, feature caches and third-party detector scores go to
`benchmarks/data/` (git-ignored). Results are written to `benchmarks/results/`.

## Corpus design

The corpus is built so the detector can't win by learning shortcuts like
"PDF vs markdown", "long vs short" or "regulator vs validation report".

### Human-written (label 0)

Every in-domain human document is dated **before 30 November 2022**, when ChatGPT
was released, so its human authorship isn't in doubt.

| Family | Sources |
|---|---|
| `regulatory_guidance` | Fed SR 11-7, SR 15-18; FHFA AB 2013-07; PRA SS3/18, CP6/22; ECB Guide to internal models (2019); EBA GL/2017/16 and ML-for-IRB discussion paper; BoE/FCA ML survey |
| `model_documentation` | Federal Reserve DFAST supervisory stress-test methodology (2020, 2021) |
| `supervisory_findings` | ECB TRIM project report; Fed OIG 2015-SR-B-018 (stress-test model validation) and 2022-SR-B-016; FHFA OIG EVL-2020-001, EVL-2016-001 |
| `validation_report` | GAO/AIMD-97-145 (review of OMB's credit subsidy model), GAO-16-269, GAO-13-400R, BoE IEO forecast evaluation (2015) |
| `comment_letter` | Public comment letters on the 2021 interagency AI/ML RFI (docket OP-1743) |
| `academic` | q-fin papers on model risk, backtesting, XVA and explainable credit scoring (arXiv, 2017–2022); BoE staff working papers |
| `business_email` | Enron e-mail corpus (2000–01), substantive risk/trading/business messages only, newest message only. This is the e-mail-genre false-positive control |
| `hc3_finance`, `m4_*` | Human halves of the HC3 finance and M4 (PeerRead reviews, arXiv abstracts) datasets |

### AI-generated (label 1)

| Family | Generator | Notes |
|---|---|---|
| `mrm_claude` | Claude (Anthropic) | 28 documents: validation report sections, findings, policies, monitoring reports, model documentation, plus *genre-matched* texts (regulatory guidance, academic sections, Inspector General findings, comment letters). Several prompts ask the model to "sound human" |
| `mrm_qwen2-5-1-5b-instruct`, `mrm_smollm2-1-7b-instruct`, `mrm_phi-3-5-mini-instruct` | Local open-weight LLMs | `generate_local_ai.py`, MRM prompts × model types × styles, plus a genre-matched prompt set |
| `hc3_finance` | gpt-3.5-turbo | HC3 finance answers |
| `m4_peer_review`, `m4_arxiv_abstract` | gpt-3.5-turbo, text-davinci-003, LLaMA, Cohere | M4 dataset |

### Leakage and confound controls

* **Document-level split.** Train and test are split by document, stratified by
  family. Segments of one document never appear on both sides, and
  cross-validation folds are grouped by document as well.
* **Segment cap.** At most 40 segments per document, so long regulatory PDFs
  don't dominate.
* **Deduplication.** Identical texts are hashed and removed; M4 reuses the same
  human reviews across files.
* **Format-sensitive features excluded.** Paragraph-length variation,
  bullet/bold/colon rates and question rate depend on PDF extraction and genre
  rather than authorship, so they're left out of the shipped calibration.
* **Length-robust features.** Entropy and repetition features are computed over
  fixed windows.
* **In-domain weighting.** MRM documents carry 3× the weight of the generic
  HC3/M4 datasets.
* **Threshold choice.** The AI threshold is set on *training* documents to keep
  the human-document false-positive rate at or below 2%.

## Mixed-authorship documents

Real AI use in model documentation rarely produces a wholly AI-written
document. More often an author drafts, then asks an assistant to polish a
paragraph, rewrite a section, turn notes into prose, continue the text, or
write the conclusions. `build_mixed.py` reproduces those workflows on
~1,500-word windows of the pre-2022 human documents, keeping a label on every
word:

| Operation | What the assistant is asked | Counted as |
|---|---|---|
| polish | light edit for grammar and flow, keep everything | polish (reported separately, not in the AI share) |
| rewrite | rewrite the paragraph, keeping every fact | AI |
| notes | paragraph → terse notes (discarded) → new paragraph from the notes | AI |
| continue | write the next paragraph from the preceding text only | AI |
| summary | append a "key findings" section for the window | AI |

Each window gets a target AI share (0–100%). Human paragraphs are scrubbed of
PDF debris (running headers, footnote digits) so AI-written paragraphs can't be
told apart by their cleanliness, and two sources whose extraction split many
words (FHFA AB 2013-07, the EBA ML discussion paper) are excluded for the same
reason. Mixed documents inherit the train/test split of their source document.

Generators: Qwen2.5-7B, Llama-3.1-8B and Gemma-2-9B run locally (4-bit, MLX,
`data/mixed/`, git-ignored), and 10 Claude-written documents built from US
federal public-domain sources (`mixed_claude/`, committed). Any OpenAI model can
be added with your own key; the script prints a token and cost estimate and
asks before spending anything (about 37 documents and under $1 at GPT-4o
prices with `--per-doc 2`):

```bash
export OPENAI_API_KEY=...   # in your own shell; read from the environment, never stored
python benchmarks/build_mixed.py --generators openai:gpt-4o --per-doc 2
```

`eval_mixed.py` reports, on documents from held-out sources: the error of the
predicted AI share, segment-level detection by operation and generator, and
flag rates on two external benchmarks, APT-Eval (Saha & Feizi, ACL 2025:
human text polished by GPT-4o, DeepSeek-V3 and Llama models at controlled
levels) and MixSet (Gao et al., NAACL 2024). Both are downloaded at run time
and not redistributed. Results: `results/mixed_*.md` (`_v1` = models before
recalibration, `_v2` = shipped models).

`finetune_classifier.py` fine-tunes a RoBERTa classifier on in-domain
paragraphs (training-split sources only), for experiments with GPT- or
Claude-written training data. Its output isn't shipped with the package.

## Known limitations of this benchmark

* Frontier-model text is scarce: 28 synthetic and 10 mixed Claude documents,
  and no in-domain GPT-4o/5 text unless you generate it with `build_mixed.py`.
  On grounded frontier-model text, no detector we tested (including a 7B
  Binoculars pair) separated AI from human writing; see the main README. Measure performance on your own documents (see the main
  README, "Calibrating on your organisation's documents").
* The Binoculars/Fast-DetectGPT observer is Qwen2.5-0.5B. Text generated by
  Qwen-family models is therefore an easier case for those two features;
  results are reported per generator so this is visible.
* Human in-domain documents are public regulatory, supervisory and academic
  texts. Internal bank documents (validation reports, audit issues) aren't
  public. The Enron e-mails and OIG/GAO findings are the closest public proxies.
* Web detectors were tested by hand on 8 samples (`web_samples/`, all
  public-domain or AI-generated text). That's indicative only.
