from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationMetrics:
    success_rate: float
    mean_final_distance: float
    mean_return: float
    episodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EvaluationMetrics":
        return cls(
            success_rate=float(values["success_rate"]),
            mean_final_distance=float(values["mean_final_distance"]),
            mean_return=float(values["mean_return"]),
            episodes=int(values["episodes"]),
        )


def score_metrics(metrics: EvaluationMetrics, skill: str = "slide", score_config: str = "{}") -> float:
    """Lower is better. Skill-aware scoring so agents optimize the metric that matters
    for the task. score_config (JSON string) lets agents override the weights, e.g.
    for push where distance matters more, or rotation where yaw error matters.

    Defaults (skill-aware):
      - slide: 10*(1-success) + distance  (success dominates, distance for push)
      - pick:  10*(1-success) + distance  (success dominates)
      - rotate: 10*(1-success) + yaw_error
      - spin:   10*(1-success) + rate_error
    """
    import json as _json

    try:
        cfg = _json.loads(score_config) if score_config else {}
    except _json.JSONDecodeError:
        cfg = {}
    success_weight = float(cfg.get("success_weight", 10.0))
    distance_weight = float(cfg.get("distance_weight", 1.0))
    yaw_weight = float(cfg.get("yaw_weight", 1.0))
    rate_weight = float(cfg.get("rate_weight", 1.0))
    base = success_weight * (1.0 - metrics.success_rate)
    if skill == "rotate":
        # mean_final_distance encodes position + yaw error for rotate.
        return base + yaw_weight * metrics.mean_final_distance
    if skill == "spin":
        return base + rate_weight * metrics.mean_final_distance
    return base + distance_weight * metrics.mean_final_distance
