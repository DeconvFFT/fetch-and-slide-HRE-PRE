from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any, Mapping

from .skills import VALID_SKILLS


@dataclass(frozen=True)
class CandidateConfig:
    """Bounded, serializable knobs an experiment is allowed to change."""

    env_id: str = "FetchSlide-v4"
    seed: int = 7
    actor_lr: float = 1e-3
    actor_l2: float = 0.1
    critic_lr: float = 1e-3
    gamma: float = 0.98
    tau: float = 0.05
    batch_size: int = 64
    hidden_dim: int = 256
    her_ratio: float = 0.8
    her_future: int = 8
    per: bool = True
    hper: bool = True
    per_alpha: float = 0.5
    per_beta: float = 0.4
    per_beta_final: float = 1.0
    per_epsilon: float = 0.1
    noise_std: float = 0.20
    random_prob: float = 0.30
    # TD3 target policy smoothing: noise added to target actions to reduce Q
    # overestimation, and delayed actor updates (every N critic steps).
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    actor_delay: int = 2
    algorithm: str = "td3"
    rehearse_critic: bool = True
    dense_reward: bool = True
    reach_coef: float = 2.0
    reach_contact_bonus: float = 0.5
    push_coef: float = 5.0
    goal_bonus: float = 4.0
    goal_bonus_radius: float = 0.4
    # Large one-time bonus on achieving the goal (puck within distance_threshold).
    # The reach/push shaping rewards contact and partial progress, so the actor
    # settles for "get near the puck, push a little" with no gradient to finish the
    # final push. A sparse success bonus teaches the critic that completing the goal
    # is worth far more than contact, driving push completion.
    success_bonus: float = 20.0
    # Number of scripted reach-then-push rollouts to seed the replay buffer with
    # before training. 0 disables the curriculum init.
    scripted_rollouts: int = 0
    # Interleave a scripted reach-then-push rollout every N training episodes so the
    # replay continuously gets contact-push examples (the actor's own exploration
    # rarely contacts the puck). 0 disables interleaving.
    scripted_every: int = 0
    # Reference cadence: collect `rollouts_per_cycle` rollouts, then do `optimsteps`
    # batched gradient updates, then soft-update targets once per cycle. This matches
    # the reference HER+DDPG recipe (200 epochs x 50 cycles x 2 rollouts x 40 optimsteps).
    # When 0, uses the legacy per-step cadence (updates_per_step updates per step).
    rollouts_per_cycle: int = 0
    optimsteps: int = 40
    warmup_steps: int = 50
    train_episodes: int = 100
    horizon: int = 50
    updates_per_step: int = 1
    log_every: int = 50
    eval_every: int = 200
    replay_capacity: int = 100_000
    eval_episodes: int = 5
    eval_seed_offset: int = 1_000
    device: str = "auto"
    skill: str = "slide"
    # Skill-aware scoring weights, JSON-serialized. Agents may configure these to
    # optimize the metric that matters for the task (e.g. success vs distance for
    # push, yaw error for rotation, rate error for spin). Format:
    # {"success_weight": 10, "distance_weight": 1, "yaw_weight": 1, "rate_weight": 1}
    score_config: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEARCHABLE_FIELDS = (
    "actor_lr",
    "actor_l2",
    "critic_lr",
    "gamma",
    "tau",
    "batch_size",
    "hidden_dim",
    "her_ratio",
    "her_future",
    "noise_std",
    "random_prob",
    "policy_noise",
    "noise_clip",
    "actor_delay",
    "algorithm",
    "warmup_steps",
    "train_episodes",
    "horizon",
    "updates_per_step",
    "log_every",
    "eval_every",
    "replay_capacity",
    "eval_episodes",
    "eval_seed_offset",
    "seed",
    "per",
    "hper",
    "rehearse_critic",
    "dense_reward",
    "reach_coef",
    "reach_contact_bonus",
    "push_coef",
    "goal_bonus",
    "goal_bonus_radius",
    "success_bonus",
    "scripted_rollouts",
    "scripted_every",
    "rollouts_per_cycle",
    "optimsteps",
    "per_alpha",
    "per_beta",
    "per_beta_final",
    "per_epsilon",
    "skill",
    "score_config",
)

_FIELD_NAMES = {field.name for field in fields(CandidateConfig)}
_INT_FIELDS = {
    "seed",
    "batch_size",
    "hidden_dim",
    "warmup_steps",
    "train_episodes",
    "horizon",
    "updates_per_step",
    "actor_delay",
    "log_every",
    "eval_every",
    "her_future",
    "replay_capacity",
    "eval_episodes",
    "eval_seed_offset",
    "scripted_rollouts",
    "scripted_every",
    "rollouts_per_cycle",
    "optimsteps",
}
_FLOAT_FIELDS = {"actor_lr", "actor_l2", "critic_lr", "gamma", "tau", "her_ratio", "noise_std", "random_prob", "policy_noise", "noise_clip", "per_alpha", "per_beta", "per_beta_final", "per_epsilon", "reach_coef", "reach_contact_bonus", "push_coef", "goal_bonus", "goal_bonus_radius", "success_bonus"}
_BOOL_FIELDS = {"per", "hper", "rehearse_critic", "dense_reward"}


def _validate_value(name: str, value: Any) -> Any:
    if name not in _FIELD_NAMES:
        raise ValueError(f"unknown parameter: {name}")
    if name in _BOOL_FIELDS:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value
    if name in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        bounds = {
            "seed": (0, 1_000_000_000),
            "batch_size": (4, 1_024),
            "hidden_dim": (32, 1_024),
            "warmup_steps": (0, 1_000_000),
            "train_episodes": (1, 100_000),
            "horizon": (1, 50),
            "updates_per_step": (1, 32),
            "actor_delay": (1, 32),
            "log_every": (1, 100_000),
            "eval_every": (1, 100_000),
            "her_future": (1, 16),
            "replay_capacity": (128, 1_000_000),
            "eval_episodes": (1, 1_000),
            "eval_seed_offset": (0, 1_000_000_000),
            "scripted_rollouts": (0, 100_000),
            "scripted_every": (0, 100_000),
            "rollouts_per_cycle": (0, 100),
            "optimsteps": (1, 1000),
        }
        low, high = bounds[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value
    if name in _FLOAT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name in {"actor_lr", "critic_lr"} and not 0.0 < value:
            raise ValueError(f"{name} must be greater than zero")
        if name == "actor_l2" and value < 0.0:
            raise ValueError("actor_l2 must be non-negative")
        if name == "noise_std" and value < 0.0:
            raise ValueError("noise_std must be non-negative")
        if name == "policy_noise" and not 0.0 <= value <= 1.0:
            raise ValueError("policy_noise must be in [0, 1]")
        if name == "noise_clip" and not 0.0 <= value <= 1.0:
            raise ValueError("noise_clip must be in [0, 1]")
        if name == "reach_coef" and value < 0.0:
            raise ValueError("reach_coef must be non-negative")
        if name == "reach_contact_bonus" and value < 0.0:
            raise ValueError("reach_contact_bonus must be non-negative")
        if name == "push_coef" and value < 0.0:
            raise ValueError("push_coef must be non-negative")
        if name == "goal_bonus" and value < 0.0:
            raise ValueError("goal_bonus must be non-negative")
        if name == "success_bonus" and value < 0.0:
            raise ValueError("success_bonus must be non-negative")
        if name == "goal_bonus_radius" and value <= 0.0:
            raise ValueError("goal_bonus_radius must be positive")
        if name == "per_alpha" and not 0.0 <= value <= 1.0:
            raise ValueError("per_alpha must be in [0, 1]")
        if name == "per_beta" and not 0.0 <= value <= 1.0:
            raise ValueError("per_beta must be in [0, 1]")
        if name == "per_beta_final" and not 0.0 <= value <= 1.0:
            raise ValueError("per_beta_final must be in [0, 1]")
        if name == "per_epsilon" and value < 0.0:
            raise ValueError("per_epsilon must be non-negative")
        if name in {"gamma", "tau"} and not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        if name == "her_ratio" and not 0.0 <= value <= 1.0:
            raise ValueError("her_ratio must be in [0, 1]")
        if name == "random_prob" and not 0.0 <= value <= 1.0:
            raise ValueError("random_prob must be in [0, 1]")
        return value
    if name == "env_id":
        if value != "FetchSlide-v4":
            raise ValueError("env_id is fixed to FetchSlide-v4")
        return value
    if name == "device":
        if value not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps, or auto")
        return value
    if name == "algorithm":
        if value not in {"td3", "ddpg"}:
            raise ValueError("algorithm must be td3 or ddpg")
        return value
    if name == "skill":
        if value not in VALID_SKILLS:
            raise ValueError(f"skill must be one of {', '.join(VALID_SKILLS)}")
        return value
    if name == "score_config":
        if not isinstance(value, str):
            raise ValueError("score_config must be a JSON string")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"score_config must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("score_config must be a JSON object")
        allowed = {"success_weight", "distance_weight", "yaw_weight", "rate_weight"}
        unknown = set(parsed) - allowed
        if unknown:
            raise ValueError(f"score_config unknown keys: {sorted(unknown)}")
        for key, val in parsed.items():
            if not isinstance(val, (float, int)) or val < 0.0:
                raise ValueError(f"score_config.{key} must be a non-negative number")
        return value
    raise ValueError(f"unsupported parameter: {name}")


def validate_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the fields an agent may propose; reject commands and paths."""
    unknown = set(overrides) - set(SEARCHABLE_FIELDS)
    if unknown:
        name = sorted(unknown)[0]
        raise ValueError(f"unknown parameter: {name}")
    return {name: _validate_value(name, value) for name, value in overrides.items()}


def config_from_mapping(values: Mapping[str, Any] | None = None) -> CandidateConfig:
    """Build a config after validating every supplied field."""
    values = {} if values is None else dict(values)
    unknown = set(values) - _FIELD_NAMES
    if unknown:
        name = sorted(unknown)[0]
        raise ValueError(f"unknown parameter: {name}")
    validated = {name: _validate_value(name, value) for name, value in values.items()}
    defaults = CandidateConfig().to_dict()
    defaults.update(validated)
    return CandidateConfig(**defaults)
