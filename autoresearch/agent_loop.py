"""Karpathy-style single-agent autonomous research loop.

The LLM is the autonomous researcher. It reads ``program.md`` and the current
``trainer.py`` source, proposes a change (a ``config`` override and/or a
``code_edit`` to one of the editable training files), the driver applies it,
commits it to the ``autoresearch/<tag>`` branch, runs a bounded training job,
reads the score, and keeps (advances) or discards (``git reset``) the change.

This is an ADDITIONAL entry point. It does not touch ``runner.py``/``worker.py``/
``proposal.py`` — those remain the config-driven orchestrator path.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .config import CandidateConfig, validate_overrides
from .metrics import EvaluationMetrics
from .proposal import _validate_code_edit, propose_overrides
from .runner import _replace_function_source

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = os.getenv("BASETEN_MODEL_ID", "deepseek/deepseek-v4-flash-0731")
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# Editable training files (mirrors proposal.CODE_EDITABLE_FILES).
EDITABLE_FILES = ("trainer.py", "replay.py", "model.py")

# Fixed, bounded training budget so trials are comparable (like karpathy's 5 min).
# eval_every/log_every are set high so only the final eval runs — keeps wall-clock
# bounded and trials comparable.
BUDGET = {
    "train_episodes": 30,
    "horizon": 50,
    "eval_episodes": 3,
    "batch_size": 64,
    "warmup_steps": 10,
    "updates_per_step": 1,
    "eval_every": 100_000,
    "log_every": 100_000,
    "device": "auto",
}
# Clamp budget-critical fields so the LLM cannot blow up wall-clock time.
BUDGET_CLAMPS = {"train_episodes": (1, 60), "eval_episodes": (1, 5), "horizon": (1, 50)}


# ---------------------------------------------------------------------------
# LLM plumbing (OpenRouter, urllib — same pattern as proposal.py)
# ---------------------------------------------------------------------------
def _api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("BASETEN_API_KEY")


# ---------------------------------------------------------------------------
# Config / code-edit application
# ---------------------------------------------------------------------------
def build_config(
    overrides: Mapping[str, Any],
    *,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(budget or BUDGET)
    cfg.update(overrides)
    for name, (low, high) in BUDGET_CLAMPS.items():
        if name in cfg:
            cfg[name] = int(max(low, min(high, cfg[name])))
    return cfg


def _apply_edit(package_dir: Path, code_edit: Mapping[str, Any]) -> None:
    """Replace one function in an editable training file, compile-checked, in place."""
    file = code_edit["file"]
    function = code_edit["function"]
    new_code = code_edit["new_code"]
    source_path = package_dir / file
    original = source_path.read_text(encoding="utf-8")
    new_source = _replace_function_source(original, function, new_code)
    try:
        ast.parse(new_source)
    except SyntaxError as exc:
        raise ValueError(f"edited {file} does not compile: {exc}") from exc
    source_path.write_text(new_source, encoding="utf-8")


# ---------------------------------------------------------------------------
# Git + results helpers
# ---------------------------------------------------------------------------
def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def _ensure_branch(repo_root: Path, branch: str) -> None:
    existing = _git(repo_root, "branch", "--list", branch).stdout.strip()
    if existing:
        # Resume on an existing branch (supports restart-on-failure): check it out
        # and return the current best score so the loop continues from there.
        _git(repo_root, "checkout", branch)
        return
    _git(repo_root, "checkout", "-b", branch)


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            rows.append(
                {"commit": parts[0], "score": parts[1], "status": parts[2], "description": parts[3]}
            )
    return rows


def _append_result(path: Path, commit: str, score: float | None, status: str, description: str) -> None:
    score_str = f"{score:.6f}" if score is not None else "0.000000"
    description = description.replace("\t", " ").replace("\n", " ")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{commit}\t{score_str}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------
def _metrics_from_mapping(values: Mapping[str, Any]) -> EvaluationMetrics:
    """Build EvaluationMetrics from a metrics.json 'metrics' sub-dict."""
    return EvaluationMetrics(
        success_rate=float(values.get("success_rate", 0.0)),
        mean_final_distance=float(values.get("mean_final_distance", 1.0)),
        mean_return=float(values.get("mean_return", 0.0)),
        episodes=int(values.get("episodes", 0)),
    )


def _metrics_from_trial(trial_dir: Path) -> EvaluationMetrics:
    """Read the best trial's metrics.json and return its EvaluationMetrics."""
    try:
        data = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics = _metrics_from_mapping(data.get("metrics", data))
        # contact_rate is stored at the TOP level of metrics.json (not in the
        # 'metrics' sub-dict), so merge it in.
        if "contact_rate" in data:
            metrics = EvaluationMetrics(
                success_rate=metrics.success_rate,
                mean_final_distance=metrics.mean_final_distance,
                mean_return=metrics.mean_return,
                episodes=metrics.episodes,
                contact_rate=float(data["contact_rate"]),
            )
        return metrics
    except (OSError, ValueError, json.JSONDecodeError):
        return EvaluationMetrics(success_rate=0.0, mean_final_distance=1.0, mean_return=0.0, episodes=0)


def _best_trial_context(repo_root: Path, history: list[dict[str, str]]) -> tuple[CandidateConfig, EvaluationMetrics]:
    """Recover the best kept trial's config + metrics for the proposer context."""
    best_row = None
    best_score = float("inf")
    for row in history:
        if row.get("status") != "keep":
            continue
        try:
            s = float(row["score"])
        except ValueError:
            continue
        if s < best_score:
            best_score = s
            best_row = row
    if best_row is None:
        return CandidateConfig(), EvaluationMetrics(success_rate=0.0, mean_final_distance=1.0, mean_return=0.0, episodes=0)
    # Find the trial dir whose metrics.json score matches the best kept score.
    for trial_dir in sorted((repo_root / "trials").glob("iter*")):
        try:
            data = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if abs(float(data.get("score", float("inf"))) - best_score) < 1e-6:
            cfg = data.get("config", {})
            config = CandidateConfig(**{k: v for k, v in cfg.items() if k in CandidateConfig.__dataclass_fields__})
            return config, _metrics_from_trial(trial_dir)
    return CandidateConfig(), EvaluationMetrics(success_rate=0.0, mean_final_distance=1.0, mean_return=0.0, episodes=0)


def _run_trial(
    config_json: Path,
    trial_dir: Path,
    *,
    repo_root: Path,
    package_dir: Path,
) -> float | None:
    """Run one bounded training job via the worker subprocess. Returns the score,
    or None if the run crashed (no metrics.json)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "run.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "autoresearch.worker",
        "--config",
        str(config_json),
        "--output",
        str(trial_dir),
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    metrics_path = trial_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(metrics["score"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------
def _ask_for_change(
    program: str,
    trainer_src: str,
    history: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    current_config: CandidateConfig,
    best_metrics: EvaluationMetrics,
) -> dict[str, Any]:
    """Propose the next experiment via the two-model proposal system: a PRO strategist
    diagnoses WHY the current best stalls and plans the direction, a flash PROPOSER +
    CRITIC + CODE-EDITOR turn that into an incremental config override or a code_edit.
    Maps the multi-agent output onto the loop's {config, code_edit, description} shape."""
    # Retry transient LLM-proposal failures (bad JSON, malformed score_config,
    # invalid overrides) so the loop never stops on a flaky proposal — the charter
    # says NEVER STOP.
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            result = propose_overrides(
                current_config,
                best_metrics,
                history,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
            break
        except (ValueError, RuntimeError) as exc:
            last_err = exc
            print(f"[agent_loop] proposal failed (attempt {attempt+1}/4): {exc}", flush=True)
    else:
        raise RuntimeError(f"proposal failed after 4 attempts: {last_err}")
    if "code_edit" in result:
        return {
            "config": {},
            "code_edit": result["code_edit"],
            "description": f"code_edit {result['code_edit'].get('function', '')}",
        }
    # result is a dict of validated config overrides.
    return {"config": result, "code_edit": None, "description": f"config {list(result)}"}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def run_loop(
    *,
    tag: str,
    iterations: int | None = None,
    api_key: str | None = None,
    model: str = MODEL,
    timeout: float = 120.0,
    repo_root: Path = REPO_ROOT,
    package_dir: Path = PACKAGE_DIR,
    budget: Mapping[str, Any] | None = None,
) -> int:
    api_key = api_key or _api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for agent_loop")
    branch = f"autoresearch/{tag}"
    results_path = repo_root / "results.tsv"

    # --- Setup: fresh branch + baseline commit + results header ---
    _ensure_branch(repo_root, branch)
    _git(repo_root, "add", "--", "autoresearch/trainer.py", "autoresearch/replay.py",
         "autoresearch/model.py", "autoresearch/config.py", "program.md")
    _git(repo_root, "commit", "-m", f"{branch}: baseline", "--allow-empty")
    if not results_path.is_file():
        results_path.write_text("commit\tscore\tstatus\tdescription\n", encoding="utf-8")

    # --- Baseline run (no edits) ---
    # On a fresh branch, establish the baseline. On resume (existing branch),
    # reuse the recorded best score so we don't re-run the baseline.
    history = _read_results(results_path)
    if history:
        best_score = max((float(r["score"]) for r in history if r["status"] == "keep"), default=None)
        if best_score is None:
            best_score = 0.0
        print(f"[agent_loop] resuming branch {branch} with best score={best_score:.4f}", flush=True)
        # Recover the best kept trial's config + metrics for the proposer context.
        best_config, best_metrics = _best_trial_context(repo_root, history)
    else:
        baseline_cfg = build_config({}, budget=budget)
        baseline_dir = repo_root / "trials" / "baseline"
        config_json = baseline_dir / "config.json"
        config_json.parent.mkdir(parents=True, exist_ok=True)
        config_json.write_text(json.dumps(baseline_cfg, indent=2) + "\n", encoding="utf-8")
        baseline_score = _run_trial(config_json, baseline_dir, repo_root=repo_root, package_dir=package_dir)
        baseline_commit = _git(repo_root, "rev-parse", "--short", "HEAD").stdout.strip()
        if baseline_score is None:
            baseline_score = 0.0
            baseline_status = "crash"
        else:
            baseline_status = "keep"
        _append_result(results_path, baseline_commit, baseline_score, baseline_status, "baseline")
        best_score = baseline_score
        best_config = CandidateConfig(**{k: v for k, v in baseline_cfg.items() if k in CandidateConfig.__dataclass_fields__})
        best_metrics = _metrics_from_trial(baseline_dir)
        print(f"[agent_loop] baseline score={baseline_score:.4f} status={baseline_status}", flush=True)

    # --- Experiment loop ---
    index = 0
    while iterations is None or index < iterations:
        index += 1
        program = (repo_root / "program.md").read_text(encoding="utf-8")
        trainer_src = (package_dir / "trainer.py").read_text(encoding="utf-8")
        history = _read_results(results_path)
        proposal = _ask_for_change(
            program, trainer_src, history, api_key=api_key, model=model, timeout=timeout,
            current_config=best_config, best_metrics=best_metrics,
        )
        description = proposal["description"] or f"iteration {index}"

        # Apply the change.
        cfg = build_config(proposal["config"], budget=budget)
        if proposal["code_edit"] is not None:
            _apply_edit(package_dir, proposal["code_edit"])

        # Commit the change (so git can revert on regression).
        _git(repo_root, "add", "--", "autoresearch/trainer.py", "autoresearch/replay.py",
             "autoresearch/model.py", "autoresearch/config.py")
        _git(repo_root, "commit", "-m", f"{branch}: {description}", "--allow-empty")
        commit = _git(repo_root, "rev-parse", "--short", "HEAD").stdout.strip()

        # Run the experiment.
        trial_dir = repo_root / "trials" / f"iter{index:04d}"
        config_json = trial_dir / "config.json"
        config_json.parent.mkdir(parents=True, exist_ok=True)
        config_json.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        score = _run_trial(config_json, trial_dir, repo_root=repo_root, package_dir=package_dir)

        # Keep / discard.
        if score is None:
            status = "crash"
            _git(repo_root, "reset", "--hard", "HEAD~1")
        elif score < best_score:
            status = "keep"
            best_score = score
            best_config = CandidateConfig(**{k: v for k, v in cfg.items() if k in CandidateConfig.__dataclass_fields__})
            best_metrics = _metrics_from_trial(trial_dir)
        else:
            status = "discard"
            _git(repo_root, "reset", "--hard", "HEAD~1")

        _append_result(results_path, commit, score, status, description)
        print(
            f"[agent_loop] iter={index} score={score if score is None else round(score, 4)} "
            f"status={status} desc={description}",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Karpathy-style autonomous research loop")
    parser.add_argument("--tag", required=True, help="run tag, e.g. aug16 (branch autoresearch/<tag>)")
    parser.add_argument("--iterations", type=int, default=None, help="max iterations (default: run forever)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args(argv)
    return run_loop(
        tag=args.tag,
        iterations=args.iterations,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
