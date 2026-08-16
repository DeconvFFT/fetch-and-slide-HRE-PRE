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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import validate_overrides
from .proposal import _validate_code_edit
from .runner import _replace_function_source

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = os.getenv("BASETEN_MODEL_ID", "deepseek/deepseek-v4-pro-0813")
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
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
def _endpoint() -> str:
    return os.getenv("OPENROUTER_ENDPOINT") or DEFAULT_ENDPOINT


def _api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("BASETEN_API_KEY")


def _content(response: Mapping[str, Any]) -> str:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("OpenRouter response is missing an assistant message") from exc
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    raise ValueError("OpenRouter returned no text proposal")


def _call_llm(
    system: str,
    user: dict[str, Any],
    *,
    api_key: str,
    model: str,
    timeout: float = 120.0,
    max_tokens: int = 8192,
) -> str:
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "reasoning": {"exclude": True},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ],
    }
    request = Request(
        _endpoint(),
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
    return _content(payload)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("proposal must be a JSON object")
    return parsed


def _parse_proposal(text: str) -> dict[str, Any]:
    """Parse + validate the LLM's proposal into {config, code_edit, description}."""
    raw = _parse_json(text)
    config = raw.get("config", {})
    code_edit = raw.get("code_edit")
    description = str(raw.get("description", "")).strip()
    if config:
        config = validate_overrides(config)
    if code_edit is not None:
        code_edit = _validate_code_edit(code_edit)
    if not config and code_edit is None:
        raise ValueError("proposal must set config and/or code_edit")
    return {"config": config, "code_edit": code_edit, "description": description}


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
        raise RuntimeError(f"branch {branch} already exists — pick a fresh tag")
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
) -> dict[str, Any]:
    system = (
        "You are an autonomous RL researcher running a Karpathy-style autoresearch loop. "
        "You directly edit the training file to improve the score. "
        "score = 10*(1 - success_rate) + mean_final_distance (LOWER is better). "
        "Return a JSON object with exactly these keys:\n"
        '- "description": a short text description of the experiment.\n'
        '- "config": optional object of hyperparameter overrides (e.g. {"push_coef": 8.0}).\n'
        '- "code_edit": optional {"file", "function", "new_code"} replacing ONE function in '
        "trainer.py/replay.py/model.py (new_code is the full replacement source).\n"
        "Make ONE small change at a time. Prefer editing the reward function or a single "
        "hyperparameter. Do NOT propose edits to runner.py/worker.py/proposal.py/"
        "agent_loop.py/program.md. You must set config and/or code_edit."
    )
    user = {
        "program": program,
        "trainer.py": trainer_src,
        "history": history[-20:],
        "instructions": "Propose the next experiment as JSON.",
    }
    text = _call_llm(system, user, api_key=api_key, model=model, timeout=timeout)
    return _parse_proposal(text)


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
    print(f"[agent_loop] baseline score={baseline_score:.4f} status={baseline_status}", flush=True)

    # --- Experiment loop ---
    index = 0
    while iterations is None or index < iterations:
        index += 1
        program = (repo_root / "program.md").read_text(encoding="utf-8")
        trainer_src = (package_dir / "trainer.py").read_text(encoding="utf-8")
        history = _read_results(results_path)
        proposal = _ask_for_change(
            program, trainer_src, history, api_key=api_key, model=model, timeout=timeout
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
