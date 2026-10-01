"""mrm-ai-detector: AI-generated content detection for model risk documentation.

Quick start::

    from mrm_ai_detector import Detector

    result = Detector().analyze_file("validation_report.docx")
    print(result.verdict, f"{result.probability:.0%}")

Scores are probabilistic indicators intended to prioritise human review.
They are not proof of authorship and must not be used as the sole basis for
any decision about a person.
"""

__version__ = "1.0.0"

from .detector import Detector, DocumentResult, SegmentResult  # noqa: E402
from .scoring import ScoringModel  # noqa: E402

__all__ = ["Detector", "DocumentResult", "SegmentResult", "ScoringModel", "__version__"]
