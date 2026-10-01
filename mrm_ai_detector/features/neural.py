"""Optional language-model features (``pip install .[lm]``).

These wrap established open-source detectors so they can be combined with the
handcrafted, domain-aware features:

Zero-shot (no training data):

* **Binoculars**: Hans et al., "Spotting LLMs With Binoculars: Zero-Shot
  Detection of Machine-Generated Text" (ICML 2024),
  https://github.com/ahans30/Binoculars (BSD-3-Clause). The performer
  model's log-perplexity divided by the observer/performer cross-perplexity.
  *Lower* means more machine-like.
* **Fast-DetectGPT**: Bao et al., "Fast-DetectGPT: Efficient Zero-Shot
  Detection of Machine-Generated Text via Conditional Probability Curvature"
  (ICLR 2024), https://github.com/baoguangsheng/fast-detect-gpt (MIT). The
  analytic sampling discrepancy. *Higher* means more machine-like.

Supervised classifiers (each contributes the logit of its AI probability):

* ``hc3``: Hello-SimpleAI/chatgpt-detector-roberta (Guo et al., 2023)
* ``fakespot``: fakespot-ai/roberta-base-ai-text-detection-v1 (Mozilla/Fakespot, Apache-2.0, 2025)
* ``desklib``: desklib/ai-text-detector-v1.01 (DeBERTa-v3-large, MIT, 2025)

In the benchmark (``benchmarks/results``), the 2025 classifiers were the only
open-source detectors that recognised some frontier-model MRM text, but on
their own they flagged 33–50% of human regulatory and academic documents.
Inside the calibrated ensemble their false positives are kept in check by the
domain features.

The default observer/performer pair is Qwen2.5-0.5B base/instruct, which share
a tokenizer and run on a laptop CPU or Apple GPU. Override it with
``MRM_DETECTOR_OBSERVER`` / ``MRM_DETECTOR_PERFORMER``; any change of model
requires recalibration (``benchmarks/calibrate.py --lm``).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# name -> (Hugging Face repo, how to read the AI probability)
CLASSIFIERS: Dict[str, Tuple[str, str]] = {
    "hc3": ("Hello-SimpleAI/chatgpt-detector-roberta", "label:chatgpt"),
    "fakespot": ("fakespot-ai/roberta-base-ai-text-detection-v1", "label:ai"),
    "desklib": ("desklib/ai-text-detector-v1.01", "desklib"),
    # Not used by default (weak on MRM text in the benchmark) but available:
    "openai": ("openai-community/roberta-base-openai-detector", "label:fake"),
    "radar": ("TrustSafeAI/RADAR-Vicuna-7B", "index:0"),
}
DEFAULT_CLASSIFIERS: Tuple[str, ...] = ("hc3", "fakespot", "desklib")

NEURAL_FEATURES: Dict[str, str] = {
    "binoculars": "Binoculars score (perplexity / cross-perplexity); lower = more AI-like",
    "fast_detectgpt": "Fast-DetectGPT analytic sampling discrepancy; higher = more AI-like",
    "lm_log_ppl": "Observer-model log-perplexity; lower = more predictable",
    "hc3_logit": "HC3 ChatGPT-detector RoBERTa, logit of AI probability",
    "fakespot_logit": "Fakespot AI-text RoBERTa (2025), logit of AI probability",
    "desklib_logit": "Desklib AI-text DeBERTa-v3 (2025), logit of AI probability",
}
NEURAL_DIRECTIONS = {
    "binoculars": "-", "fast_detectgpt": "+", "lm_log_ppl": "-",
    "hc3_logit": "+", "fakespot_logit": "+", "desklib_logit": "+",
}


class NeuralUnavailable(RuntimeError):
    """Raised when the optional ``[lm]`` dependencies are not installed."""


def _env_classifiers() -> Tuple[str, ...]:
    raw = os.environ.get("MRM_DETECTOR_CLASSIFIERS")
    return tuple(c.strip() for c in raw.split(",") if c.strip()) if raw else DEFAULT_CLASSIFIERS


@dataclass
class NeuralConfig:
    observer: str = os.environ.get("MRM_DETECTOR_OBSERVER", "Qwen/Qwen2.5-0.5B")
    performer: str = os.environ.get("MRM_DETECTOR_PERFORMER", "Qwen/Qwen2.5-0.5B-Instruct")
    classifiers: Sequence[str] = field(default_factory=_env_classifiers)
    max_tokens: int = 512
    device: Optional[str] = os.environ.get("MRM_DETECTOR_DEVICE")
    use_zero_shot: bool = True


def pick_device(requested: Optional[str] = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_logit(p: float, eps: float = 1e-4) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def load_classifier(repo: str, mode: str, device: str) -> Callable[[str], float]:
    """Load a Hugging Face AI-text classifier; return ``text -> P(AI)`` (chunked, length-weighted)."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel

    tok = AutoTokenizer.from_pretrained(repo)
    if mode == "desklib":
        # Architecture published on the model card: DeBERTa-v3 encoder, mean pooling, linear head.
        class DesklibAIDetectionModel(PreTrainedModel):
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = torch.nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids, attention_mask=None):
                hidden = self.model(input_ids, attention_mask=attention_mask)[0]
                mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return self.classifier(pooled)

        model = DesklibAIDetectionModel.from_pretrained(repo).to(device).eval()

        def prob(enc) -> float:
            return torch.sigmoid(model(enc["input_ids"], enc["attention_mask"])).float().item()

    else:
        model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
        if mode.startswith("index:"):
            idx = int(mode.split(":")[1])
        else:
            want = mode.split(":")[1]
            labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
            idx = next((i for name, i in labels.items() if want in name), None)
            if idx is None:
                raise ValueError(f"{repo}: no label matching {want!r} in {labels}")

        def prob(enc) -> float:
            return torch.softmax(model(**enc).logits.float(), dim=-1)[0, idx].item()

    def score(text: str) -> float:
        vals: List[float] = []
        weights: List[int] = []
        for chunk in _chunk_words(text, 300):
            enc = tok(chunk, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                vals.append(prob(enc))
            weights.append(max(1, len(chunk.split())))
        return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

    return score


class NeuralScorer:
    """Lazily-loaded wrapper that turns text into neural feature values."""

    def __init__(self, config: Optional[NeuralConfig] = None) -> None:
        self.config = config or NeuralConfig()
        self._loaded = False
        self._classifiers: Dict[str, Callable[[str], float]] = {}

    # -- loading ---------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise NeuralUnavailable(
                "Language-model features need optional dependencies: pip install 'mrm-ai-detector[lm]'"
            ) from exc

        self._torch = torch
        cfg = self.config
        self.device = pick_device(cfg.device)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        logger.info("Loading neural detectors on %s", self.device)
        if cfg.use_zero_shot:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.observer)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.observer = AutoModelForCausalLM.from_pretrained(cfg.observer, dtype=dtype).to(self.device).eval()
            self.performer = AutoModelForCausalLM.from_pretrained(cfg.performer, dtype=dtype).to(self.device).eval()
        for name in cfg.classifiers:
            if name not in CLASSIFIERS:
                raise ValueError(f"unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}")
            repo, mode = CLASSIFIERS[name]
            self._classifiers[name] = load_classifier(repo, mode, self.device)
        self._loaded = True

    # -- scoring ---------------------------------------------------------
    def score(self, text: str) -> Dict[str, Optional[float]]:
        """Return neural feature values for ``text`` (None for too-short text)."""
        self._load()
        out: Dict[str, Optional[float]] = {k: None for k in NEURAL_FEATURES}
        if len(text.split()) < 25:
            return out
        if self.config.use_zero_shot:
            out.update(self._zero_shot(text))
        for name, fn in self._classifiers.items():
            out[f"{name}_logit"] = to_logit(fn(text))
        return out

    def _zero_shot(self, text: str) -> Dict[str, Optional[float]]:
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_tokens).to(self.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 16:
            return {}
        with torch.no_grad():
            obs_logits = self.observer(**enc).logits[:, :-1].float()
            perf_logits = self.performer(**enc).logits[:, :-1].float()
        targets = ids[:, 1:]
        vocab = min(obs_logits.shape[-1], perf_logits.shape[-1])
        obs_logits, perf_logits = obs_logits[..., :vocab], perf_logits[..., :vocab]

        perf_lp = torch.log_softmax(perf_logits, dim=-1)
        obs_lp = torch.log_softmax(obs_logits, dim=-1)
        # Binoculars: performer log-ppl / cross-entropy(observer probs, performer log-probs)
        ppl = -perf_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        x_ppl = -(obs_lp.exp() * perf_lp).sum(-1).mean()
        binoculars = (ppl / x_ppl).item()

        # Fast-DetectGPT (white-box, sampling model == scoring model == observer)
        ll = obs_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        probs = obs_lp.exp()
        mean_ref = (probs * obs_lp).sum(-1)
        var_ref = (probs * obs_lp.square()).sum(-1) - mean_ref.square()
        discrepancy = ((ll.sum() - mean_ref.sum()) / var_ref.sum().clamp_min(1e-6).sqrt()).item()
        # Normalise by sqrt(length) so values are comparable across text sizes.
        discrepancy /= max(1.0, float(ll.shape[1])) ** 0.5

        return {"binoculars": binoculars, "fast_detectgpt": discrepancy, "lm_log_ppl": (-ll.mean()).item()}


def _chunk_words(text: str, size: int) -> List[str]:
    words = text.split()
    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
    if len(chunks) > 1 and len(chunks[-1].split()) < size // 3:
        tail = chunks.pop()
        chunks[-1] += " " + tail
    return chunks


def neural_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True
