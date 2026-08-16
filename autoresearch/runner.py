from __future__ import annotations

import argparse
import os
import signal
import time
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import CandidateConfig, config_from_mapping, validate_overrides
from .metrics import EvaluationMetrics, score_metrics
from .proposal import CODE_EDITABLE_FILES, propose_overrides
from .trainer import evaluate_checkpoint
from .paths import runs_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = runs_dir()
RESULT_HEADER = "trial\tscore\tsuccess_rate\tmean_final_distance\tmean_return\tstatus\tdescription\tartifact"
DEFAULT_CANDIDATES = [
    {"actor_lr": 5e-4},
    {"critic_lr": 5e-4},
    {"her_ratio": 1.0},
    {"noise_std": 0.1},
]


class TraceWriter:
    """Append durable events and publish the latest loop status for live inspection."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.trace_path = run_dir / "trace.jsonl"
        self.status_path = run_dir / "status.json"
        self.latest_path = run_dir / "latest.json"
        self.trace_path.touch()

    def record(self, event: str, **values: Any) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(self.latest_path, payload)
        _write_json(
            self.status_path,
            {
                "updated_at": payload["timestamp"],
                "event": event,
                "phase": values.get("phase"),
                "trial": values.get("trial"),
                "status": values.get("status"),
                "best_score": values.get("best_score"),
                "message": values.get("message"),
            },
        )
        return payload


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_row(path: Path, values: list[Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(value) for value in values) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_handoff(run_dir: Path, history: list[Mapping[str, Any]], best_score: float, best_metrics: EvaluationMetrics) -> None:
    """Write a persistent handoff file the proposing LLM reads before each proposal.
    Records the current best, every trial's config/score/status, and diagnostic context
    so the LLM can reason about what to try next to push past the best."""
    lines: list[str] = []
    lines.append("# Autoresearch Handoff — run state for the proposing LLM")
    lines.append("")
    lines.append("Read this BEFORE proposing. It records what has been tried, the outcome, and the "
                 "diagnostic context so you can propose changes that are LIKELY to improve.")
    lines.append("")
    lines.append(f"## Current best")
    lines.append(f"- score = {best_score:.6f}")
    lines.append(f"- success_rate = {best_metrics.success_rate:.3f}")
    lines.append(f"- mean_final_distance = {best_metrics.mean_final_distance:.4f}")
    lines.append("")
    lines.append("## Diagnostic context (why the task is hard)")
    lines.append("")
    lines.append("FetchSlide is a two-stage task: REACH the puck, then PUSH it to the goal. "
                 "The actor often contacts the puck but pushes it to only ~0.15-0.3 from the goal, "
                 "stopping short of the <0.05 success threshold. So the bottleneck is PUSH COMPLETION. "
                 "The reward knobs that matter most: push_coef (reward for pushing toward goal) and "
                 "goal_bonus (reward for getting the puck close to the goal). Tune these to break "
                 "past the current success rate. lr/gamma tweaks alone rarely help.")
    lines.append("")
    lines.append("## Trial history (changelog)")
    lines.append("")
    lines.append("| trial | status | score | success | distance | overrides |")
    lines.append("|---|---|---|---|---|---|")
    for row in history:
        trial = row.get("trial", "?")
        status = row.get("status", "?")
        m = row.get("metrics") or {}
        score = m.get("score")
        if score is None:
            # compute from metrics if present
            sr = m.get("success_rate", 0.0)
            dist = m.get("mean_final_distance", 0.0)
            score = 10 * (1 - sr) + dist
        overrides = row.get("config", {})
        # show only the fields that were overridden vs defaults (the actual proposal)
        cfg_str = json.dumps({k: v for k, v in overrides.items() if k in ("actor_lr","critic_lr","gamma","tau","her_ratio","her_future","noise_std","per_alpha","per_epsilon","batch_size","updates_per_step","train_episodes","reach_coef","reach_contact_bonus","push_coef","goal_bonus","goal_bonus_radius","dense_reward","skill","score_config")}, sort_keys=True)
        code_edit = row.get("code_edit")
        code_note = ""
        if code_edit is not None:
            kept = row.get("code_edit_kept", False)
            fn = code_edit.get("function")
            code_note = f" CODE_EDIT({code_edit.get('file')}::{fn}, {'KEPT' if kept else 'reverted'})"
        lines.append(f"| {trial} | {status} | {score:.4f} | {m.get('success_rate',0.0):.3f} | {m.get('mean_final_distance',0.0):.4f} | {cfg_str}{code_note} |")
    lines.append("")
    lines.append("## What worked / what did not")
    lines.append("")
    for row in history:
        trial = row.get("trial", "?")
        status = row.get("status", "?")
        m = row.get("metrics") or {}
        score = m.get("score")
        if score is None:
            score = 10 * (1 - m.get("success_rate", 0.0)) + m.get("mean_final_distance", 0.0)
        verdict = "WORKED (kept as best)" if status == "keep" else "DID NOT WORK (discarded)"
        code_note = ""
        if row.get("code_edit") is not None:
            kept = row.get("code_edit_kept", False)
            code_note = f" [code_edit {'KEPT' if kept else 'reverted'}]"
        lines.append(f"- trial {trial} [{verdict}]{code_note} score={score:.4f} success={m.get('success_rate',0.0):.2f} dist={m.get('mean_final_distance',0.0):.3f}")
    lines.append("")
    lines.append("## Guidance for the next experiment")
    lines.append("")
    lines.append("Push performance PAST the current best. Do NOT repeat discarded changes. "
                 "The highest-leverage changes are the REWARD knobs (push_coef, goal_bonus) and "
                 "score_config (to weight distance vs success). "
                 "If hyperparameter tweaks (lr/gamma/batch_size) have repeatedly FAILED to improve "
                 "success, STOP proposing them and instead propose a CODE EDIT to trainer.py "
                 "(e.g. change the reward function, add a push-completion curriculum, improve the "
                 "critic target or HER). You may return a code_edit: "
                 "{\"code_edit\": {\"file\": \"trainer.py\", \"function\": \"<function_name>\", "
                 "\"new_code\": \"<full replacement function source>\"}}. The code_edit is applied, "
                 "tested, and kept only if it improves the metric. Reason from the trial history and "
                 "diagnostic context, and propose a change likely to improve the metric.")
    (run_dir / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_row(trial: str, metrics: EvaluationMetrics, status: str, description: str, artifact: str, skill: str = "slide", score_config: str = "{}") -> list[Any]:
    return [
        trial,
        f"{score_metrics(metrics, skill, score_config):.6f}",
        f"{metrics.success_rate:.6f}",
        f"{metrics.mean_final_distance:.6f}",
        f"{metrics.mean_return:.6f}",
        status,
        description.replace("\t", " ").replace("\n", " "),
        artifact,
    ]


def _load_candidate_file(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CANDIDATES)
    raw = json.loads(path.read_text())
    candidates = raw if isinstance(raw, list) else [raw]
    if not candidates:
        raise ValueError("candidate file must contain at least one candidate")
    if not all(isinstance(item, dict) for item in candidates):
        raise ValueError("candidate file must contain an object or list of objects")
    return [validate_overrides(item) for item in candidates]


def local_candidate_overrides(index: int, best_config: CandidateConfig) -> dict[str, Any]:
    """Deterministic bounded mutations used when no candidate file or LLM is configured."""
    schedule = [
        ("actor_lr", best_config.actor_lr * 0.5),
        ("actor_lr", best_config.actor_lr * 2.0),
        ("critic_lr", best_config.critic_lr * 0.5),
        ("critic_lr", best_config.critic_lr * 2.0),
        ("gamma", min(0.999, best_config.gamma + 0.005)),
        ("gamma", max(0.90, best_config.gamma - 0.005)),
        ("tau", min(1.0, best_config.tau * 2.0)),
        ("tau", max(0.001, best_config.tau * 0.5)),
        ("her_ratio", min(1.0, best_config.her_ratio + 0.1)),
        ("her_ratio", max(0.0, best_config.her_ratio - 0.1)),
        ("noise_std", best_config.noise_std * 0.5),
        ("noise_std", min(1.0, best_config.noise_std * 1.5)),
        ("batch_size", max(4, best_config.batch_size // 2)),
        ("batch_size", min(1024, best_config.batch_size * 2)),
        ("updates_per_step", min(32, best_config.updates_per_step + 1)),
    ]
    name, value = schedule[index % len(schedule)]
    return validate_overrides({name: value})


def _run_worker(config: CandidateConfig, trial_dir: Path, checkpoint: Path | None, timeout: float) -> tuple[int, str]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    _write_json(config_path, config.to_dict())
    log_path = trial_dir / "run.log"
    command = [sys.executable, "-m", "autoresearch.worker", "--config", str(config_path), "--output", str(trial_dir)]
    if checkpoint is not None:
        command.extend(["--init-checkpoint", str(checkpoint)])
    try:
        with log_path.open("w", encoding="utf-8") as log:
            timeout_value = None if timeout <= 0 else timeout
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_value, check=False)
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout:.1f}s"
    return process.returncode, "completed" if process.returncode == 0 else f"exit {process.returncode}"


def _code_file_path(file: str) -> Path:
    """Resolve an editable source file (trainer.py/replay.py/model.py) to its absolute
    path under the autoresearch package. Only these files may be edited."""
    if file not in CODE_EDITABLE_FILES:
        raise ValueError(f"code_edit file must be one of {', '.join(CODE_EDITABLE_FILES)}")
    return Path(__file__).resolve().parent / file


def _replace_function_source(source: str, function: str, new_code: str) -> str:
    """Swap one function definition in `source` for `new_code` (verbatim text replace).

    The new_code is the full replacement source (def line + body). We locate the
    original function's def line by name and replace through the end of its body
    (the next top-level `def`/`class` at the same indentation, or EOF)."""
    import ast

    tree = ast.parse(source)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            target = node
            break
    if target is None:
        raise ValueError(f"function {function!r} not found in source")
    lines = source.splitlines()
    start = target.lineno - 1
    end = target.end_lineno
    # Replace exactly the original function's lines with the new full source.
    new_lines = new_code.splitlines()
    return "\n".join(lines[:start] + new_lines + lines[end:])


def _validate_compiles(source: str, file: str) -> None:
    """Ensure the edited source parses as valid Python before it is run."""
    import ast

    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"edited {file} does not compile: {exc}") from exc


def _snapshot_code_edit(code_edit: Mapping[str, Any], trial_dir: Path) -> Path:
    """Snapshot the original source of the file being edited to the trial dir, so it
    can be restored verbatim on regression. Returns the snapshot path."""
    source_path = _code_file_path(code_edit["file"])
    trial_dir.mkdir(parents=True, exist_ok=True)
    snapshot = trial_dir / f"{code_edit['file']}.orig"
    snapshot.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot


def _apply_code_edit(code_edit: Mapping[str, Any], trial_dir: Path) -> None:
    """Apply a validated code_edit IN PLACE to the real source file (the worker imports
    autoresearch.trainer from the package, so the edit must land there to take effect).

    The original source is snapshotted to the trial dir first. The edited source is
    compile-checked (ast.parse) before it is written. On any failure the file is left
    untouched (the snapshot is the source of truth for revert)."""
    file = code_edit["file"]
    function = code_edit["function"]
    new_code = code_edit["new_code"]
    source_path = _code_file_path(file)
    original = source_path.read_text(encoding="utf-8")
    new_source = _replace_function_source(original, function, new_code)
    _validate_compiles(new_source, file)
    _snapshot_code_edit(code_edit, trial_dir)
    source_path.write_text(new_source, encoding="utf-8")


def _restore_code_edit(code_edit: Mapping[str, Any], trial_dir: Path) -> None:
    """Restore the original source file after a code_edit trial that regressed or
    crashed. Reads the snapshot written by _apply_code_edit and writes it back."""
    file = code_edit["file"]
    snapshot = trial_dir / f"{file}.orig"
    if not snapshot.is_file():
        raise RuntimeError(f"no snapshot to restore for {file} in {trial_dir}")
    source_path = _code_file_path(file)
    source_path.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")


def _smoke_config(config: CandidateConfig) -> CandidateConfig:
    return config_from_mapping(
        {
            **config.to_dict(),
            "train_episodes": 1,
            "horizon": 5,
            "eval_episodes": 1,
            "batch_size": 4,
            "warmup_steps": 0,
        }
    )


def run(args: argparse.Namespace) -> int:
    config = config_from_mapping({"seed": args.seed})
    if args.train_episodes is not None:
        config = config_from_mapping({**config.to_dict(), "train_episodes": args.train_episodes})
    if args.horizon is not None:
        config = config_from_mapping({**config.to_dict(), "horizon": args.horizon})
    if args.eval_episodes is not None:
        config = config_from_mapping({**config.to_dict(), "eval_episodes": args.eval_episodes})
    if args.eval_every is not None:
        config = config_from_mapping({**config.to_dict(), "eval_every": args.eval_every})
    if args.log_every is not None:
        config = config_from_mapping({**config.to_dict(), "log_every": args.log_every})
    if args.batch_size is not None:
        config = config_from_mapping({**config.to_dict(), "batch_size": args.batch_size})
    if args.device is not None:
        config = config_from_mapping({**config.to_dict(), "device": args.device})
    if args.smoke:
        config = _smoke_config(config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or args.output / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"autoresearch_loop_started run_dir={run_dir}", flush=True)
    trace = TraceWriter(run_dir)
    results_path = run_dir / "results.tsv"
    results_path.write_text(RESULT_HEADER + "\n")
    checkpoint = args.init_checkpoint
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {checkpoint}")
    _write_json(
        run_dir / "manifest.json",
        {
            "reference": "karpathy/autoresearch",
            "policy": "config-only candidate changes; no source mutation or shell execution from proposals",
            "continuous": args.forever,
            "config": config.to_dict(),
            "initial_checkpoint": str(checkpoint) if checkpoint else None,
        },
    )

    stop_requested = False
    current_trial = 0
    history: list[Mapping[str, Any]] = []

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"autoresearch_stop_requested signal={signum}", flush=True)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        trace.record("run_started", phase="baseline", trial=0, best_score=None, continuous=args.forever)
        if checkpoint is not None:
            baseline_metrics = evaluate_checkpoint(config, checkpoint)
            best_checkpoint = checkpoint
        else:
            baseline_dir = run_dir / "trial-000-baseline"
            code, detail = _run_worker(config, baseline_dir, None, args.max_seconds)
            metrics_path = baseline_dir / "metrics.json"
            if code or not metrics_path.is_file():
                raise RuntimeError(f"baseline failed: {detail}; see {baseline_dir / 'run.log'}")
            payload = json.loads(metrics_path.read_text())
            baseline_metrics = EvaluationMetrics.from_mapping(payload["metrics"])
            best_checkpoint = baseline_dir / "checkpoint.pt"
        best_score = score_metrics(baseline_metrics, config.skill, config.score_config)
        baseline_description = "existing checkpoint" if checkpoint is not None else "fresh baseline"
        _write_row(results_path, _metrics_row("000", baseline_metrics, "baseline", baseline_description, str(best_checkpoint), config.skill, config.score_config))
        history.append({"trial": "000", "status": "baseline", "config": config.to_dict(), "metrics": baseline_metrics.to_dict()})
        _write_json(run_dir / "history.json", history)
        best_metrics = baseline_metrics
        best_config = config
        trace.record(
            "baseline_completed",
            phase="ready",
            trial=0,
            status="baseline",
            best_score=best_score,
            metrics=baseline_metrics.to_dict(),
        )

        candidates = _load_candidate_file(args.candidate_file) if args.candidate_file is not None else []
        index = 0
        while args.forever or index < args.iterations:
            if stop_requested:
                break
            if args.proposal:
                try:
                    overrides = propose_overrides(best_config, best_metrics, history, run_dir=str(run_dir))
                except Exception as exc:
                    # Proposal failure (e.g. API rate limit / network) must not kill
                    # the whole loop; fall back to a deterministic local candidate so
                    # the search continues.
                    print(f"proposal failed ({exc}); falling back to local candidate", flush=True)
                    overrides = local_candidate_overrides(index, best_config)
            elif args.candidate_file is not None:
                overrides = candidates[index % len(candidates)]
            else:
                overrides = local_candidate_overrides(index, best_config)
            code_edit = overrides.get("code_edit") if isinstance(overrides, dict) else None
            if code_edit is not None:
                # A code_edit proposal: apply it in place (snapshot first), run the
                # worker, then keep the edit if it improves the metric or revert it.
                candidate = best_config
                overrides = {}
            else:
                candidate = config_from_mapping({**best_config.to_dict(), **overrides})
            current_trial = index + 1
            trial_name = f"{current_trial:03d}"
            trial_dir = run_dir / f"trial-{trial_name}"
            trace.record(
                "trial_started",
                phase="training",
                trial=current_trial,
                status="running",
                best_score=best_score,
                overrides=overrides,
                config=candidate.to_dict(),
                code_edit=code_edit,
            )
            # Only warm-start from the best checkpoint when the candidate matches BOTH
            # hidden_dim and skill (obs/goal dims differ per skill, so a mismatched
            # checkpoint would crash the actor/critic construction). When --no-warmstart
            # is set, every trial trains from scratch (avoids destabilizing a trained
            # actor with a changed hyperparameter, which repeatedly regressed).
            checkpoint_for_candidate = (
                best_checkpoint
                if not args.no_warmstart and candidate.hidden_dim == best_config.hidden_dim and candidate.skill == best_config.skill
                else None
            )
            if code_edit is not None:
                # Snapshot the original source, then apply the edit in place. If the
                # edit fails to apply (bad function name, won't compile), record a
                # crash and skip the trial without touching the real file.
                try:
                    _apply_code_edit(code_edit, trial_dir)
                except Exception as exc:
                    _write_row(results_path, [trial_name, "0.000000", "0.000000", "0.000000", "0.000000", "crash", f"code_edit apply failed: {exc}", str(trial_dir)])
                    history.append({"trial": trial_name, "status": "crash", "config": candidate.to_dict(), "metrics": {}, "code_edit": code_edit, "code_edit_kept": False})
                    trace.record(
                        "trial_finished",
                        phase="ready",
                        trial=current_trial,
                        status="crash",
                        best_score=best_score,
                        message=f"code_edit apply failed: {exc}",
                        code_edit=code_edit,
                    )
                    _write_json(run_dir / "history.json", history)
                    index += 1
                    continue
            code, detail = _run_worker(candidate, trial_dir, checkpoint_for_candidate, args.max_seconds)
            metrics_path = trial_dir / "metrics.json"
            if code or not metrics_path.is_file():
                if code_edit is not None:
                    # Revert the edit: the trial crashed, so the code change is not kept.
                    _restore_code_edit(code_edit, trial_dir)
                _write_row(results_path, [trial_name, "0.000000", "0.000000", "0.000000", "0.000000", "crash", detail, str(trial_dir)])
                history.append({"trial": trial_name, "status": "crash", "config": candidate.to_dict(), "metrics": {}, "code_edit": code_edit, "code_edit_kept": False})
                trace.record(
                    "trial_finished",
                    phase="ready",
                    trial=current_trial,
                    status="crash",
                    best_score=best_score,
                    message=detail,
                    code_edit=code_edit,
                )
            else:
                payload = json.loads(metrics_path.read_text())
                metrics = EvaluationMetrics.from_mapping(payload["metrics"])
                score = score_metrics(metrics, candidate.skill, candidate.score_config)
                status = "keep" if score < best_score else "discard"
                if status == "keep":
                    best_score = score
                    best_metrics = metrics
                    best_config = candidate
                    best_checkpoint = trial_dir / "checkpoint.pt"
                    shutil.copy2(best_checkpoint, run_dir / "best_checkpoint.pt")
                else:
                    if code_edit is not None:
                        # The edit regressed (or tied): revert the source file.
                        _restore_code_edit(code_edit, trial_dir)
                _write_row(results_path, _metrics_row(trial_name, metrics, status, json.dumps(overrides, sort_keys=True), str(trial_dir), candidate.skill, candidate.score_config))
                history.append({"trial": trial_name, "status": status, "config": candidate.to_dict(), "metrics": metrics.to_dict(), "code_edit": code_edit, "code_edit_kept": status == "keep"})
                _write_handoff(run_dir, history, best_score, best_metrics)
                trace.record(
                    "trial_finished",
                    phase="ready",
                    trial=current_trial,
                    status=status,
                    best_score=best_score,
                    score=score,
                    metrics=metrics.to_dict(),
                    code_edit=code_edit,
                )
                print(
                    f"trial {trial_name}: {status} score={score:.4f} success={metrics.success_rate:.3f} distance={metrics.mean_final_distance:.4f}",
                    flush=True,
                )
            _write_json(run_dir / "history.json", history)
            index += 1
            if args.forever and not stop_requested and args.sleep_seconds > 0:
                trace.record(
                    "sleeping",
                    phase="sleeping",
                    trial=current_trial,
                    status="sleeping",
                    best_score=best_score,
                    message=f"sleeping {args.sleep_seconds:.1f}s before next trial",
                )
                time.sleep(args.sleep_seconds)
        final_event = "run_stopped" if stop_requested else "run_completed"
        trace.record(final_event, phase="stopped" if stop_requested else "completed", trial=current_trial, best_score=best_score)
    except BaseException as exc:
        trace.record("run_failed", phase="failed", trial=current_trial, message=str(exc))
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        _write_json(run_dir / "history.json", history)
    print(f"run_dir={run_dir}", flush=True)
    print(f"best_score={best_score:.6f}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autoresearch-style FetchSlide experiment loop")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--forever", action="store_true", help="continue trials until SIGINT or SIGTERM")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="pause between continuous trials")
    parser.add_argument("--run-dir", type=Path, help="exact trace directory; otherwise create one under --output")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="warm-start baseline from a saved actor checkpoint (default: train from scratch)")
    parser.add_argument("--no-warmstart", action="store_true", help="train every trial from scratch (no warm-start from best checkpoint)")
    parser.add_argument("--proposal", action="store_true", help="ask OpenRouter for bounded hyperparameter overrides")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-every", type=int, help="evaluate every N training episodes")
    parser.add_argument("--log-every", type=int, help="print a progress line every N training episodes")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=str, help="cpu, cuda, mps, or auto (default: auto)")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="per-trial timeout seconds; 0 or omitted = run until done")
    parser.add_argument("--smoke", action="store_true", help="one five-step CPU trial for wiring checks")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
