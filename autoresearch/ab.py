"""Paired A/B trials for validating novel contributions (H-PER, CRI).

Both arms start from the same checkpoint and train with the same seed and
episode budget; only the knob under test differs. Lower score is better
(score = 10*(1-success) + mean_final_distance).

Usage:
    python3 -m autoresearch.ab --init-checkpoint PATH --episodes N \
        --key hper --on true --off false [--hidden-dim 256] [--out DIR]

Prints a JSON summary: per-arm metrics plus which arm won.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import config_from_mapping
from .metrics import EvaluationMetrics, score_metrics
from .trainer import train_and_evaluate


def _arm(base: dict, key: str, value: bool, init: Path, out: Path, hidden_dim: int) -> dict:
    cfg = config_from_mapping({**base, key: value, "hidden_dim": hidden_dim})
    result = train_and_evaluate(cfg, out, init)
    metrics = EvaluationMetrics.from_mapping(result["metrics"])
    return {
        "config": cfg.to_dict(),
        "metrics": metrics.to_dict(),
        "score": score_metrics(metrics, cfg.skill, cfg.score_config),
        "steps": result["steps"],
        "warm_start_fallback": result["warm_start_fallback"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--key", required=True, choices=["hper", "rehearse_critic"])
    parser.add_argument("--on", dest="on", type=json.loads, required=True)
    parser.add_argument("--off", dest="off", type=json.loads, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("/tmp/ab"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    base = {
        "seed": args.seed,
        "train_episodes": args.episodes,
        "eval_episodes": 5,
        "device": args.device,
    }
    on_dir = args.out / "on"
    off_dir = args.out / "off"
    on = _arm(base, args.key, args.on, args.init_checkpoint, on_dir, args.hidden_dim)
    off = _arm(base, args.key, args.off, args.init_checkpoint, off_dir, args.hidden_dim)

    winner = "on" if on["score"] < off["score"] else "off"
    summary = {"key": args.key, "on": on, "off": off, "winner": winner}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
