from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import CandidateConfig, validate_overrides
from .metrics import EvaluationMetrics


# Strict pydantic schema for a hyperparameter proposal. Every field is optional
# (agents may propose only a few knobs), but each must be a valid number in range.
# This guarantees complete, well-formed outputs before they are passed between agents.
class Proposal(BaseModel):
    actor_lr: Optional[float] = Field(default=None, gt=0.0)
    actor_l2: Optional[float] = Field(default=None, ge=0.0)
    critic_lr: Optional[float] = Field(default=None, gt=0.0)
    gamma: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    tau: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    her_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    noise_std: Optional[float] = Field(default=None, ge=0.0)
    per_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    per_epsilon: Optional[float] = Field(default=None, ge=0.0)
    her_future: Optional[int] = Field(default=None, ge=1, le=16)
    batch_size: Optional[int] = Field(default=None, ge=4, le=1024)
    updates_per_step: Optional[int] = Field(default=None, ge=1, le=32)
    train_episodes: Optional[int] = Field(default=None, ge=1, le=100000)
    reach_coef: Optional[float] = Field(default=None, ge=0.0)
    reach_contact_bonus: Optional[float] = Field(default=None, ge=0.0)
    push_coef: Optional[float] = Field(default=None, ge=0.0)
    goal_bonus: Optional[float] = Field(default=None, ge=0.0)
    goal_bonus_radius: Optional[float] = Field(default=None, gt=0.0)
    dense_reward: Optional[bool] = None
    skill: Optional[str] = Field(default=None)
    score_config: Optional[str] = None

    def overrides(self) -> dict[str, Any]:
        """Return only the fields the agent actually set (non-None), validated."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class CodeEdit(BaseModel):
    file: str
    function: str
    new_code: str


# OpenRouter models via Baseten endpoint. Two-model design:
#  - IMPLEMENTOR (flash): the proposer + code_editor that implements changes.
#  - DECIDER (pro): the strategist + logic/code-decider that decides what metrics
#     to configure and the overall logic (better thinking model).
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DECIDER_MODEL = "deepseek/deepseek-v4-pro-0813"
# OpenRouter OpenAI-compatible endpoint. Override via OPENROUTER_ENDPOINT.
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Files the CODE-EDITOR agent is allowed to propose edits to. runner.py/worker.py/
# proposal.py/config.py are infrastructure and must never be mutated by a proposal.
CODE_EDITABLE_FILES = ("trainer.py", "replay.py", "model.py", "config.py")


def _program_context() -> str:
    """Load the research-org charter (program.md) if present, else empty string.
    The charter encodes our RL goals so the proposing agent focuses on them."""
    path = Path(__file__).resolve().parents[1] / "program.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return ""
    return text.strip()


def _endpoint() -> str:
    return os.getenv("OPENROUTER_ENDPOINT") or os.getenv("BASETEN_ENDPOINT") or DEFAULT_ENDPOINT


def _handoff_context(run_dir: str | None = None) -> str:
    """Load the per-run handoff.md (what worked/didn't, scores, changelog) if present.
    The LLM reads this BEFORE proposing so it can reason from prior results."""
    if not run_dir:
        return ""
    path = Path(run_dir) / "handoff.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return ""
    return text.strip()


def _api_key() -> str | None:
    return os.getenv("BASETEN_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _fallback_api_key() -> str | None:
    """Dedicated Baseten instance key, used when the primary key is rate-limited
    (HTTP 429). Set via BASETEN_FALLBACK_API_KEY."""
    return os.getenv("BASETEN_FALLBACK_API_KEY")


def _content(response: Mapping[str, Any]) -> str:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("OpenRouter response is missing an assistant message") from exc
    if not isinstance(choice, Mapping) or not isinstance(message, Mapping):
        raise ValueError("OpenRouter response is missing an assistant message")

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text = "".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
        if text.strip():
            return text

    raise ValueError(
        "OpenRouter returned no text proposal "
        f"(finish_reason={choice.get('finish_reason')}, "
        f"native_finish_reason={choice.get('native_finish_reason')}, "
        f"message_keys={sorted(message)})"
    )



def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    proposal = json.loads(text)
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a JSON object")
    return proposal


def _parse_proposal(text: str) -> dict[str, Any]:
    """Parse + strictly validate a hyperparameter proposal via pydantic. Returns
    only the fields the agent set (non-None), validated against the schema. Raises
    if the output is incomplete or malformed, so we never pass partial output between
    agents."""
    raw = _parse_json(text)
    if "code_edit" in raw:
        # code_edit proposals are validated separately; pass through.
        return raw
    model = Proposal(**raw)
    overrides = model.overrides()
    # Reject empty proposals (no knobs set) — an incomplete output.
    if not overrides:
        raise ValueError("proposal is empty (no fields set) — incomplete output")
    return validate_overrides(overrides)


def _validate_proposal_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate an already-parsed proposal dict via pydantic. Returns only
    the fields the agent set, or raises if incomplete/malformed."""
    if "code_edit" in raw:
        return dict(raw)
    model = Proposal(**raw)
    overrides = model.overrides()
    if not overrides:
        raise ValueError("proposal is empty (no fields set) — incomplete output")
    return validate_overrides(overrides)


def _call_codex(
    model: str,
    system: str,
    user: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Call the codex CLI (logged in via ChatGPT) with the given model to get a JSON
    proposal. Returns the parsed JSON object. Uses max-thinking (default for codex)."""
    prompt = system + "\n\nUSER:\n" + json.dumps(user, sort_keys=True) + "\n\nReturn ONLY the JSON object."
    try:
        result = subprocess.run(
            ["codex", "exec", "--model", model, "--skip-git-repo-check", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("codex exec timed out")
    out = result.stdout.strip()
    # codex may wrap JSON in code fences or add prose; extract the JSON object.
    if out.startswith("```"):
        out = out.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"codex returned non-JSON: {out[:300]}")


def _call_strategist(
    endpoint: str,
    model: str,
    keys: list[str],
    system: str,
    user: dict[str, Any],
    timeout: float,
) -> str:
    """Call the LLM to produce a free-text STRATEGY (not JSON). Returns the strategy string.
    The strategist is the LOGIC/DECIDER agent — it uses MAX reasoning effort to think
    properly about what metrics to configure and the overall strategy."""
    if "luna" in model or "codex" in model:
        prompt = system + "\n\nUSER:\n" + json.dumps(user, sort_keys=True) + "\n\nReturn ONLY your short strategy."
        try:
            result = subprocess.run(
                ["codex", "exec", "--model", model, "--skip-git-repo-check", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("codex exec timed out")
        return result.stdout.strip()[:1000]
    # Baseten HTTP path: request a text completion with MAX reasoning effort.
    # The pro model needs enough tokens to finish its reasoning AND emit the strategy.
    request_body = {
        "model": model,
        "temperature": 0.4,
        "max_tokens": 4096,
        "reasoning": {"effort": "high", "exclude": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter request failed ({exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
    return _content(payload)[:1000]


def _call_llm(
    endpoint: str,
    model: str,
    keys: list[str],
    schema: dict[str, Any],
    system: str,
    user: dict[str, Any],
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Make a single LLM call. Uses codex CLI for codex models, else the Baseten HTTP
    endpoint. The implementor (proposer/critic) uses LOW reasoning effort — code
    generation already has max reasoning, so low keeps it fast and token-efficient."""
    if "luna" in model or "codex" in model:
        return _call_codex(model, system, user, timeout)
    for key_index, key in enumerate(keys):
        request_body = {
            "model": model,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "reasoning": {"effort": "low", "exclude": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "rl_overrides", "strict": True, "schema": schema}},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, sort_keys=True)},
            ],
        }
        request = Request(
            endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429 and key_index < len(keys) - 1:
                continue
            raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
        # If the response was truncated (finish_reason=length), retry with more tokens
        # so the model can complete its JSON output (handles small models / long prompts).
        truncated = False
        if isinstance(payload, Mapping):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
                truncated = choices[0].get("finish_reason") == "length" or choices[0].get("native_finish_reason") == "length"
        if truncated and max_tokens < 16384:
            return _call_llm(endpoint, model, keys, schema, system, user, timeout, max_tokens * 2)
        # Normalize: return the parsed overrides dict (same as codex path).
        return _parse_json(_content(payload))
    raise RuntimeError("OpenRouter request failed: all keys exhausted")


def _call_code_editor(
    endpoint: str,
    model: str,
    keys: list[str],
    system: str,
    user: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Call the LLM to decide between a hyperparameter proposal and a CODE change.

    The CODE-EDITOR returns EITHER a hyperparameter overrides object (same shape as
    the PROPOSER/CRITIC) OR a code_edit object:
        {"code_edit": {"file": "trainer.py", "function": "_make_reward_fn",
                       "new_code": "<full replacement of the function body>"}}
    The runner applies code_edit trials safely (snapshot, compile-check, keep/revert).
    """
    if "luna" in model or "codex" in model:
        return _call_codex(model, system, user, timeout)
    # Baseten HTTP path: request a JSON completion. The code_editor is a DECIDER
    # (decides code logic), so it uses MAX reasoning effort.
    request_body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 4096,
        "reasoning": {"effort": "high", "exclude": False},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ],
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Authorization": f"Bearer {keys[0]}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter request failed ({exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
    return _parse_json(_content(payload))


def _trainer_source() -> str:
    """Return a TRIMMED trainer.py source for the CODE-EDITOR to read. We extract only
    the key editable function (_make_reward_fn) plus the reward-related config fields,
    so a small local model can reason about a code change without being overwhelmed by
    the full file. The agent proposes a full replacement of one function; the runner
    applies it to a working copy (nothing is mutated here)."""
    path = Path(__file__).resolve().parent / "trainer.py"
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return ""
    import ast

    tree = ast.parse(source)
    # Extract the reward function (the highest-leverage editable piece for the
    # push-completion bottleneck) plus any helper it uses.
    pieces: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_make_reward_fn",
            "_update_networks",
            "_soft_update_targets",
        ):
            pieces.append(ast.get_source_segment(source, node))
    if not pieces:
        return source[:4000]
    return "\n\n".join(pieces)


def _validate_code_edit(code_edit: Any) -> dict[str, Any]:
    """Validate a code_edit proposal. Only allows edits to trainer.py/replay.py/model.py,
    requires an existing function name and a full replacement source, and verifies the
    replacement is a single valid function whose name matches the target."""
    import ast

    if not isinstance(code_edit, dict):
        raise ValueError("code_edit must be an object")
    file = code_edit.get("file")
    function = code_edit.get("function")
    new_code = code_edit.get("new_code")
    if file not in CODE_EDITABLE_FILES:
        raise ValueError(f"code_edit file must be one of {', '.join(CODE_EDITABLE_FILES)}")
    if not isinstance(function, str) or not function.strip():
        raise ValueError("code_edit function must be a non-empty string")
    if not isinstance(new_code, str) or not new_code.strip():
        raise ValueError("code_edit new_code must be a non-empty string")
    # The replacement must be exactly one function definition whose name matches.
    try:
        tree = ast.parse(new_code)
    except SyntaxError as exc:
        raise ValueError(f"code_edit new_code does not parse: {exc}") from exc
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1:
        raise ValueError("code_edit new_code must be exactly one function definition")
    if functions[0].name != function:
        raise ValueError("code_edit new_code function name must match the target function")
    return {"file": file, "function": function, "new_code": new_code}


def propose_overrides(
    config: CandidateConfig,
    best_metrics: EvaluationMetrics,
    history: list[Mapping[str, Any]],
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    run_dir: str | None = None,
) -> dict[str, Any]:
    """Multi-agent proposal: a PROPOSER proposes a config, then a CRITIC reviews it
    against the handoff and refines it so changes are incremental and build on the
    best (not random jumps that regress)."""
    model = model or os.getenv("BASETEN_MODEL_ID", DEFAULT_MODEL)
    api_key = api_key or _api_key()
    # codex models use the local codex CLI (ChatGPT login) and need no Baseten key.
    if not api_key and not ("luna" in model or "codex" in model):
        raise RuntimeError("BASETEN_API_KEY (or OPENROUTER_API_KEY) is required for --proposal")
    endpoint = _endpoint()
    numeric_fields = ("actor_lr", "critic_lr", "gamma", "tau", "her_ratio", "noise_std", "per_alpha", "per_epsilon", "her_future", "batch_size", "updates_per_step", "train_episodes", "reach_coef", "reach_contact_bonus", "push_coef", "goal_bonus", "goal_bonus_radius")
    schema = {
        "type": "object",
        "properties": {
            **{name: {"type": "number"} for name in numeric_fields},
            "skill": {"type": "string", "enum": ["slide", "rotate", "spin", "pick", "fetch"]},
            "dense_reward": {"type": "boolean"},
            "score_config": {"type": "string"},
        },
        "additionalProperties": False,
    }
    program = _program_context()
    handoff = _handoff_context(run_dir)
    context = ""
    if program:
        context += "\n\nRESEARCH PROGRAM (follow this charter):\n" + program
    if handoff:
        context += "\n\nHANDOFF (read this first — what worked, what did not, scores, changelog):\n" + handoff
    keys = [api_key]
    fallback = _fallback_api_key()
    if fallback:
        keys.append(fallback)

    # --- Agent 1: STRATEGIST (analyzes handoff, diagnoses WHY, plans next) ---
    strategist_system = (
        "You are the experiment STRATEGIST. Analyze the handoff (trial history, what worked/didn't, "
        "scores, diagnostic context) and produce a STRATEGIC PLAN for the next experiment. "
        "Use CAVEMAN + PONYTAIL skills: be ultra-terse, short bullets, no filler. "
        "Your job is to understand WHY configs failed, not just what failed. Diagnose the root cause "
        "(e.g. 'gamma 0.98 gave 0% success because the critic destabilized', 'actor_lr 0.001 regressed "
        "because it overfit the replay'). Then decide the best strategy to break the plateau (e.g. "
        "'tune goal_bonus to fix push completion', 'increase push_coef', 'change score_config to weight "
        "distance', or 'edit the reward function in trainer.py to add a push-completion curriculum'). "
        "Return ONLY a short strategy string (2-4 sentences) describing what to try next and why."
    ) + context
    strategist_user = {
        "current_config": config.to_dict(),
        "best_metrics": best_metrics.to_dict(),
        "recent_trials": list(history[-12:]),
        "objective": "minimize 10 * (1 - success_rate) + mean_final_distance",
        "note": "Diagnose WHY the recent trials failed and produce the best next strategy.",
    }
    strategy = _call_strategist(endpoint, DECIDER_MODEL, keys, strategist_system, strategist_user, timeout)

    # --- Agent 2: PROPOSER (follows the strategy) ---
    proposer_system = (
        "You are the experiment PROPOSER for a reinforcement-learning benchmark. "
        "Use the CAVEMAN and PONYTAIL skills: be ultra-terse, no filler, just the JSON. "
        "Return only a JSON object of hyperparameter overrides. Never return code, paths, or shell commands. "
        "Valid numeric fields: actor_lr, critic_lr, gamma, tau, her_ratio, noise_std, per_alpha, per_epsilon, "
        "her_future, batch_size, updates_per_step, train_episodes, reach_coef, reach_contact_bonus, "
        "push_coef, goal_bonus, goal_bonus_radius. "
        "You may also propose skill (slide, rotate, spin, pick, fetch), dense_reward (boolean), and "
        "score_config (JSON string like {\"success_weight\":10,\"distance_weight\":1,\"yaw_weight\":1,\"rate_weight\":1}) "
        "to optimize the metric that matters for the task (e.g. distance for push, yaw for rotation). "
        "FetchSlide is a hard two-stage task (reach the puck, then push it). "
        "PRIMARY OBJECTIVE: increase SUCCESS RATE (currently stuck at ~20%). Success rate is weighted 10x "
        "in the score, so a config that reaches 40% success beats any distance-only improvement. "
        "The bottleneck is PUSH COMPLETION: the actor pushes the puck to ~0.15-0.3 but stops short of the "
        "<0.05 success threshold. To break past 20% success, tune the REWARD knobs (goal_bonus, push_coef) "
        "or score_config to weight success higher. "
        "FOLLOW THE STRATEGY below. IMPORTANT: make an INCREMENTAL change that builds on the "
        "current best config, not a random jump. Change only 1-2 knobs at a time. Do NOT repeat discarded "
        "changes. Propose something likely to INCREASE SUCCESS RATE."
    ) + "\n\nSTRATEGY (follow this):\n" + strategy + (
        "\n\nCURRENT BEST: success=" + f"{best_metrics.success_rate:.2f}"
        + " dist=" + f"{best_metrics.mean_final_distance:.3f}"
    )
    proposer_user = {
        "current_config": config.to_dict(),
        "best_metrics": best_metrics.to_dict(),
        "recent_trials": list(history[-12:]),
        "objective": "minimize 10 * (1 - success_rate) + mean_final_distance",
        "strategy": strategy,
        "note": "Follow the strategy. Propose a small, incremental change near the current best.",
    }
    proposal_payload = _call_llm(endpoint, model, keys, schema, proposer_system, proposer_user, timeout, 8192)
    proposal = _validate_proposal_dict(proposal_payload)

    # --- Agent 3: CRITIC ---
    critic_system = (
        "You are the experiment CRITIC. A PROPOSER proposed the following hyperparameter overrides. "
        "Be TERSE and token-efficient (ponytail/caveman style): no filler, just the JSON. "
        "Review it against the handoff and the current best. The proposal must be an INCREMENTAL change "
        "that builds on the best (not a random jump that will regress). "
        "CRITICAL: lr/gamma/tau tweaks alone have repeatedly FAILED to improve success (the bottleneck is "
        "push completion, not learning rate). If the proposal only changes lr/gamma/tau/batch_size, REJECT "
        "it and return a REFINED proposal that tunes the REWARD knobs (push_coef, goal_bonus, goal_bonus_radius) "
        "or score_config instead. If the proposal is good (tunes reward knobs or is a small likely-improving "
        "change), return it unchanged. Return only a JSON object of overrides."
    ) + context
    critic_user = {
        "proposed_overrides": proposal,
        "current_config": config.to_dict(),
        "best_metrics": best_metrics.to_dict(),
        "recent_trials": list(history[-12:]),
        "objective": "minimize 10 * (1 - success_rate) + mean_final_distance",
        "note": "Refine the proposal to be incremental and likely to improve the metric.",
    }
    critic_payload = _call_llm(endpoint, model, keys, schema, critic_system, critic_user, timeout, 8192)
    refined = _validate_proposal_dict(critic_payload)

    # --- Agent 4: CODE-EDITOR (optional code change to break a plateau) ---
    # Reads the handoff + strategy + current trainer.py source and decides whether a
    # CODE change (e.g. reward function, critic target, HER) is more likely to break the
    # plateau than another hyperparameter tweak. Returns EITHER a hyperparameter
    # proposal (the critic's refined overrides, possibly adjusted) OR a code_edit.
    code_editor_system = (
        "You are the experiment CODE-EDITOR for a reinforcement-learning benchmark. "
        "Be TERSE and token-efficient (ponytail/caveman style): no filler, just the JSON. "
        "You have read the handoff, the strategy, and the current trainer.py source. "
        "Decide whether a CODE change is more likely to break the current plateau than "
        "another hyperparameter tweak. If the handoff shows that hyperparameter tweaks "
        "(lr/gamma/batch_size) have repeatedly FAILED to improve success, then a code_edit "
        "is the right move — do NOT propose another hyperparameter tweak. If hyperparameters "
        "alone are the right lever, "
        "return the hyperparameter overrides object (same shape as the PROPOSER/CRITIC "
        "returned). If the plateau is caused by an algorithmic limitation in the code "
        "(e.g. the reward function gives no gradient for the final push, the critic "
        "target is unstable, HER is not relabeling the right transitions), return a "
        "code_edit instead:\n"
        '{"code_edit": {"file": "trainer.py", "function": "<function name>", '
        '"new_code": "<the COMPLETE replacement source of that function, including the '
        'def line and full body>"}}\n'
        "Constraints: file MUST be one of trainer.py, replay.py, model.py (never "
        "runner.py/worker.py/proposal.py/config.py). function MUST be an existing "
        "function in that file. new_code MUST be the FULL replacement function source "
        "(def line + entire body) so it can be swapped in verbatim. The code must "
        "compile and preserve the function's signature and call contract. Return ONLY "
        "the JSON object."
    ) + context
    code_editor_user = {
        "current_config": config.to_dict(),
        "best_metrics": best_metrics.to_dict(),
        "recent_trials": list(history[-12:]),
        "objective": "minimize 10 * (1 - success_rate) + mean_final_distance",
        "strategy": strategy,
        "critic_refined_overrides": refined,
        "current_trainer_source": _trainer_source(),
        "note": "Choose hyperparameters OR a code_edit. A code_edit is warranted only "
                "when the plateau is algorithmic, not a tuning issue. Prefer small, "
                "incremental changes that build on the current best.",
    }
    code_editor_payload = _call_code_editor(endpoint, DECIDER_MODEL, keys, code_editor_system, code_editor_user, timeout)
    code_edit = code_editor_payload.get("code_edit")
    if code_edit is not None:
        # A code_edit takes precedence over hyperparameters for this trial.
        return {"code_edit": _validate_code_edit(code_edit)}
    # No code_edit: fall back to the critic's refined hyperparameter proposal.
    return refined
