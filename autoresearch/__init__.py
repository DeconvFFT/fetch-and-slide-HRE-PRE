"""Safe, config-driven autonomous experiments for the native FetchSlide runtime."""

from .config import CandidateConfig, config_from_mapping, validate_overrides
from .metrics import EvaluationMetrics, score_metrics

__all__ = [
    "CandidateConfig",
    "EvaluationMetrics",
    "config_from_mapping",
    "score_metrics",
    "validate_overrides",
]
