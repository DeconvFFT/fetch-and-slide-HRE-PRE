from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .config import config_from_mapping
from .trainer import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated FetchSlide experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        config = config_from_mapping(json.loads(args.config.read_text()))
        train_and_evaluate(config, args.output, args.init_checkpoint)
    except Exception as exc:  # worker failure is an experiment result, not a runner crash
        (args.output / "error.json").write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n"
        )
        print(f"experiment failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
