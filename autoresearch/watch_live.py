"""Tail the live run.log of the most recently started trial in a run directory.

Usage:
    python3 -m autoresearch.watch_live [RUN_DIR]

Defaults to the newest run under ~/Documents/autoresearch-runs.
Prints the runner's trace events and the worker's live progress lines.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .paths import runs_dir


def _newest_run(root: Path) -> Path:
    runs = sorted((e for e in root.iterdir() if e.is_dir() and not e.name.startswith(".")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"no runs under {root}")
    return runs[0]


def _tail(path: Path, seen: int) -> tuple[list[str], int]:
    if not path.is_file():
        return [], seen
    with path.open("rb") as handle:
        handle.seek(seen)
        data = handle.read().decode("utf-8", errors="replace")
    return data.splitlines(), seen + len(data.encode("utf-8"))


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else runs_dir()
    run_dir = _newest_run(root)
    print(f"watching {run_dir}", flush=True)
    trace_seen = 0
    log_seen: dict[str, int] = {}  # per-trial-log byte offset, keyed by log path
    while True:
        # print runner trace events as they land (only new ones)
        trace = run_dir / "trace.jsonl"
        if trace.is_file():
            with trace.open("rb") as handle:
                handle.seek(trace_seen)
                raw = handle.read().decode("utf-8", errors="replace")
            trace_seen += len(raw.encode("utf-8"))
            for line in raw.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") in ("trial_started", "trial_finished", "baseline_completed"):
                    print(f"[trace] {event['event']} trial={event.get('trial')} status={event.get('status')} score={event.get('best_score')}", flush=True)
        # find newest trial dir and tail its run.log (per-log offset so switching
        # to a new trial's log doesn't skip past its start)
        trials = sorted((e for e in run_dir.iterdir() if e.is_dir() and e.name.startswith("trial-")), key=lambda p: p.name)
        if trials:
            log = trials[-1] / "run.log"
            key = str(log)
            seen = log_seen.get(key, 0)
            lines, seen = _tail(log, seen)
            log_seen[key] = seen
            for line in lines:
                print(f"[trial {trials[-1].name}] {line}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
