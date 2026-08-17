from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from torch import nn

from .config import CandidateConfig
from .metrics import EvaluationMetrics, score_metrics
from .model import Actor, Critic
from .replay import EpisodeReplay
from .skills import skill_dims


gym.register_envs(gymnasium_robotics)
MIN_STD = 1e-2


class RunningNormalizer:
    def __init__(self, size: int) -> None:
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self.count = 1e-4

    @classmethod
    def from_arrays(cls, mean: Any, std: Any) -> "RunningNormalizer":
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        normalizer = cls(mean.size)
        normalizer.mean = mean.copy()
        normalizer.var = np.square(np.maximum(std, MIN_STD))
        normalizer.count = 1.0
        return normalizer

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1, self.mean.size)
        if not len(values):
            return
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)
        batch_count = float(len(values))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = np.maximum((m_a + m_b + np.square(delta) * self.count * batch_count / total) / total, MIN_STD**2)
        self.count = total

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=np.float32) - self.mean) / np.sqrt(self.var), -5.0, 5.0)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": np.sqrt(self.var).tolist()}


def _device(config: CandidateConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    requested = torch.device(config.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return requested


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _make_env(config: CandidateConfig):
    from .skills import SKILL_ENV, SkillWrapper

    env_id = SKILL_ENV.get(config.skill, config.env_id)
    env = gym.make(env_id)
    if config.skill in ("rotate", "spin"):
        env = SkillWrapper(env, skill=config.skill)
    return env


def _load_actor_checkpoint(path: Path, hidden_dim: int, device: torch.device) -> tuple[Actor, RunningNormalizer, RunningNormalizer, Critic | None, Actor | None, Critic | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, (list, tuple)) and len(payload) == 5:
        obs_mean, obs_std, goal_mean, goal_std, state_dict = payload
        critic_state = None
    elif isinstance(payload, dict) and {"actor_state", "obs_mean", "goal_mean"} <= set(payload):
        obs_mean = payload["obs_mean"]
        obs_std = payload["obs_std"]
        goal_mean = payload["goal_mean"]
        goal_std = payload["goal_std"]
        state_dict = payload["actor_state"]
        critic_state = payload.get("critic_state")
    else:
        raise ValueError(f"Unsupported actor checkpoint format: {path}")
    obs_dim = int(np.asarray(obs_mean).size)
    goal_dim = int(np.asarray(goal_mean).size)
    actor = Actor(hidden_dim=hidden_dim, input_dim=obs_dim + goal_dim).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    critic = None
    target_actor = None
    target_critic = None
    if critic_state is not None:
        critic = Critic(hidden_dim=hidden_dim, input_dim=obs_dim + goal_dim + 4).to(device)
        critic.load_state_dict(critic_state)
        critic.eval()
        target_actor = Actor(hidden_dim=hidden_dim, input_dim=obs_dim + goal_dim).to(device)
        target_actor.load_state_dict(payload.get("target_actor_state", state_dict))
        target_actor.eval()
        target_critic = Critic(hidden_dim=hidden_dim, input_dim=obs_dim + goal_dim + 4).to(device)
        target_critic.load_state_dict(payload.get("target_critic_state", critic_state))
        target_critic.eval()
    return actor, RunningNormalizer.from_arrays(obs_mean, obs_std), RunningNormalizer.from_arrays(goal_mean, goal_std), critic, target_actor, target_critic


def evaluate_actor(
    config: CandidateConfig,
    actor: Actor,
    obs_normalizer: RunningNormalizer,
    goal_normalizer: RunningNormalizer,
) -> EvaluationMetrics:
    # Evaluate on CPU: deterministic and identical to evaluate_checkpoint,
    # so the warm-start incumbent matches the runner baseline (MPS numerics shift results).
    obs_dim, goal_dim = skill_dims(config.skill if hasattr(config, "skill") else "slide")
    eval_actor = Actor(hidden_dim=config.hidden_dim, input_dim=obs_dim + goal_dim)
    eval_actor.load_state_dict({name: value.detach().cpu() for name, value in actor.state_dict().items()})
    eval_actor.eval()
    device = torch.device("cpu")
    env = _make_env(config)
    successes: list[float] = []
    distances: list[float] = []
    returns: list[float] = []
    try:
        for episode in range(config.eval_episodes):
            observation, _ = env.reset(seed=config.seed + config.eval_seed_offset + episode)
            total_return = 0.0
            last_success = False
            final_distance = float("inf")
            for _ in range(config.horizon):
                state = np.concatenate(
                    [
                        obs_normalizer.normalize(observation["observation"]),
                        goal_normalizer.normalize(observation["desired_goal"]),
                    ]
                )
                with torch.inference_mode():
                    action = eval_actor(torch.from_numpy(state).to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                # Skill-aware final distance: slide -> position distance; rotate ->
                # position distance + yaw error; spin -> yaw-rate error. This keeps
                # the reported metric meaningful instead of mixing meters/radians.
                ag = np.asarray(observation["achieved_goal"], dtype=np.float32)
                dg = np.asarray(observation["desired_goal"], dtype=np.float32)
                if config.skill == "rotate":
                    final_distance = float(np.linalg.norm(ag[0:3] - dg[0:3]) + abs(float(ag[3]) - float(dg[3])))
                elif config.skill == "spin":
                    final_distance = float(abs(float(ag[0]) - float(dg[0])))
                else:
                    final_distance = float(np.linalg.norm(ag - dg))
                last_success = bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            successes.append(float(last_success))
            distances.append(final_distance)
            returns.append(total_return)
    finally:
        env.close()
    return EvaluationMetrics(
        success_rate=float(np.mean(successes)),
        mean_final_distance=float(np.mean(distances)),
        mean_return=float(np.mean(returns)),
        episodes=len(successes),
    )


def _make_reward_fn(config: CandidateConfig, env, distance_threshold: float = 0.05) -> Any:
    """Sparse (default) or dense progress reward.

    Dense reward = progress toward the goal (positive when moving toward it).
    For slide: d_now - d_next where d = ||puck - goal|| (position only).
    For rotate: position progress + yaw progress, so the agent is rewarded for
      both sliding the puck toward the goal AND rotating it to the target yaw.
    For spin: progress toward the target yaw-rate.
    The dense signal is directional (unlike an absolute-distance reward, which is
    ~0 for non-contact transitions and gives no directional gradient).

    For slide, the reach target is moved BEHIND the puck (opposite the goal).
    Approaching the puck directly from the goal side lets the gripper slide past
    without pushing; approaching the behind-puck point is the only geometry that
    produces a goal-directed push.
    """
    sparse = env.compute_reward if config.skill in ('rotate', 'spin') else env.unwrapped.compute_reward
    skill = getattr(config, 'skill', 'slide')

    def reward_fn(achieved_goal, goal, info, gripper_now=None, gripper_next=None, puck_now=None, puck_next=None):
        # Reach shaping is ALWAYS active (in both dense and sparse mode): it is the
        # mechanism that teaches the actor to contact the puck. Without it the actor
        # never contacts the puck (contact rate collapses to ~0) and the puck never
        # moves. The critic is stabilized against the positive Q this introduces by
        # the Huber loss + lower-bound-only Q clamp (see _update_networks).
        contact_dist = 0.06
        reach = 0.0
        if gripper_now is not None and gripper_next is not None and puck_now is not None:
            puck_pos = np.asarray(puck_now[0:3], dtype=np.float32)
            gnow = np.asarray(gripper_now, dtype=np.float32)
            gnext = np.asarray(gripper_next, dtype=np.float32)
            d_grip_now = np.linalg.norm(gnow - puck_pos)
            d_grip_next = np.linalg.norm(gnext - puck_pos)
            reach = (d_grip_now - d_grip_next) * config.reach_coef
            # Continuous contact bonus: grows smoothly as the gripper approaches the
            # puck (instead of a hard step at contact_dist, which gives no gradient).
            # This gives the actor a strong, differentiable signal to close the last
            # few cm to the puck.
            if d_grip_next < contact_dist:
                reach += config.reach_contact_bonus * (contact_dist - d_grip_next) / contact_dist
        if not config.dense_reward:
            return float(sparse(achieved_goal, goal, info)) + reach
        g = np.asarray(goal, dtype=np.float32)
        if puck_now is None or puck_next is None:
            # fallback (no delta available): negative absolute distance
            puck = np.asarray(achieved_goal, dtype=np.float32)
            return -float(np.linalg.norm(puck - g)) + reach
        now = np.asarray(puck_now, dtype=np.float32)
        nxt = np.asarray(puck_next, dtype=np.float32)
        if skill == "rotate":
            # Position progress (first 3 dims) + yaw progress (last dim, angular).
            pos_prog = float(np.linalg.norm(now[0:3] - g[0:3]) - np.linalg.norm(nxt[0:3] - g[0:3]))
            yaw_now = float(np.linalg.norm(now[3] - g[3]))
            yaw_nxt = float(np.linalg.norm(nxt[3] - g[3]))
            return pos_prog + (yaw_now - yaw_nxt) + reach
        if skill == "spin":
            # Progress toward target yaw-rate (single scalar).
            return float(np.linalg.norm(now[0] - g[0]) - np.linalg.norm(nxt[0] - g[0])) + reach
        # slide: position progress toward goal. The push reward is GATED on contact:
        # it only rewards goal-directed puck movement when the gripper is actually
        # touching the puck. This decouples the two phases — reach (get to the puck,
        # taught by `reach`) then push (direct it to the goal, taught by `progress`).
        # Without the gate, a high reach_coef dominates and the actor is rewarded for
        # approaching the puck regardless of where it pushes it (dist worsens).
        progress = 0.0
        if gripper_next is not None and puck_next is not None:
            gnext = np.asarray(gripper_next, dtype=np.float32)
            puck_nxt = np.asarray(puck_next[0:3], dtype=np.float32)
            if np.linalg.norm(gnext - puck_nxt) < contact_dist:
                progress = config.push_coef * float(np.linalg.norm(now - g) - np.linalg.norm(nxt - g))
        d_next = float(np.linalg.norm(nxt - g))
        goal_bonus = 0.0
        if d_next < config.goal_bonus_radius:
            # Quadratic goal bonus: grows steeply as the puck approaches the goal so
            # the FINAL push (e.g. 0.075 -> 0.05) is strongly rewarded. A linear bonus
            # gives the same per-meter gradient everywhere, so the actor stalls just
            # above the success threshold with no extra incentive to finish.
            frac = (config.goal_bonus_radius - d_next) / config.goal_bonus_radius
            goal_bonus = config.goal_bonus * frac * frac
        # Sparse success bonus: a large one-time reward when the puck reaches the goal
        # threshold. This is the key lever for push completion — the reach/push shaping
        # rewards contact and partial progress, so without a terminal success bonus the
        # actor settles for "get near the puck, push a little" and never finishes the
        # final push to <0.05.
        success = 0.0
        if d_next < distance_threshold:
            success = config.success_bonus
        return progress + reach + goal_bonus + success

    return reward_fn


def _update_networks(
    config: CandidateConfig,
    actor: Actor,
    critic: Critic,
    target_actor: Actor,
    target_critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    obs_normalizer: RunningNormalizer,
    goal_normalizer: RunningNormalizer,
    replay: EpisodeReplay,
    rng: np.random.Generator,
    reward_fn: Any,
    device: torch.device,
    distance_threshold: float = 0.05,
    actor_step: int = 0,
) -> tuple[float, float]:
    batch = replay.sample(config.batch_size, config.her_ratio, rng, reward_fn, config.her_future, distance_threshold, getattr(config, "skill", "slide"))
    state = np.concatenate([obs_normalizer.normalize(batch["state"]), goal_normalizer.normalize(batch["goal"])], axis=1)
    next_state = np.concatenate(
        [obs_normalizer.normalize(batch["next_state"]), goal_normalizer.normalize(batch["goal"])],
        axis=1,
    )
    state_t = torch.from_numpy(state).to(device)
    next_state_t = torch.from_numpy(next_state).to(device)
    action_t = torch.from_numpy(batch["action"]).to(device)
    reward_t = torch.from_numpy(batch["reward"]).to(device).unsqueeze(1)
    done_t = torch.from_numpy(batch["done"]).to(device).unsqueeze(1)

    with torch.no_grad():
        # TD3 target policy smoothing: add clipped noise to the target action so the
        # critic doesn't overestimate on sharp Q peaks, then take the MIN of the two
        # target critics (clipped double Q-learning) to curb overestimation.
        next_action = target_actor(next_state_t)
        noise = torch.randn_like(next_action) * config.policy_noise
        noise = torch.clamp(noise, -config.noise_clip, config.noise_clip)
        next_action = torch.clamp(next_action + noise, -1.0, 1.0)
        target_q1, target_q2 = target_critic(next_state_t, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_t + config.gamma * (1.0 - done_t) * target_q
        # Reach shaping is always active, so Q can be positive (reach/contact rewards).
        # Clamping max=0 would flatten the actor gradient for reach/contact. Keep only
        # the lower divergence guard ([-1/(1-gamma), inf)) to prevent critic blow-up;
        # the Huber loss bounds outlier TD errors from relabeled transitions.
        target_limit = 1.0 / max(1e-6, 1.0 - config.gamma)
        target_q = torch.clamp(target_q, min=-target_limit)
    current_q1, current_q2 = critic(state_t, action_t)
    # Huber (smooth L1) critic loss: MSE squares large TD errors, making the critic
    # hypersensitive to sparse-reward relabeling spikes (outlier TD errors from
    # table-edge collisions dominate the gradient and blow up critic weights).
    # smooth_l1 bounds outlier gradients and stabilizes the critic.
    if 'weight' in batch:
        # Importance-sampling weight corrects the priority bias (Schaul 2015).
        weight_t = torch.from_numpy(batch['weight']).to(device).unsqueeze(1)
        critic_loss = (weight_t * nn.functional.smooth_l1_loss(current_q1, target_q, reduction='none')).mean() + (weight_t * nn.functional.smooth_l1_loss(current_q2, target_q, reduction='none')).mean()
    else:
        critic_loss = nn.functional.smooth_l1_loss(current_q1, target_q) + nn.functional.smooth_l1_loss(current_q2, target_q)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_optimizer.step()
    if config.per:
        td_error = (target_q - current_q1).detach().abs().squeeze(1).cpu().numpy()
        # H-PER: relabeled rows carry ~zero TD error (hindsight goals are near-achieved),
        # so PER starves them. Prioritize by achieved-goal progress toward the hindsight
        # goal instead; original-goal rows keep TD priority. Use the explicit relabeled
        # flag (not reward==0, which is wrong under dense reward).
        goals = torch.from_numpy(batch["goal"])
        dist_now = torch.linalg.norm(torch.from_numpy(batch["achieved_goal"]) - goals, dim=1).numpy()
        dist_next = torch.linalg.norm(torch.from_numpy(batch["next_achieved_goal"]) - goals, dim=1).numpy()
        relabeled = np.asarray(batch["relabeled"]) > 0.0
        priority = np.abs(td_error)
        if config.hper:
            priority[relabeled] = np.maximum(priority[relabeled], dist_now[relabeled] - dist_next[relabeled])
        replay.update_priorities(batch["episode_index"], batch["transition_index"], priority)

    # Delayed actor update (TD3): update the actor every `actor_delay` critic steps so
    # the critic is accurate before the policy is pushed along its gradient.
    actor_loss = 0.0
    if actor_step % config.actor_delay == 0:
        actor_output = actor(state_t)
        actor_loss = -critic.q1_only(state_t, actor_output).mean() + config.actor_l2 * actor_output.square().mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_optimizer.step()
    if isinstance(actor_loss, torch.Tensor):
        actor_loss = float(actor_loss.detach().cpu().item())
    return float(actor_loss), float(critic_loss.item())


def _soft_update_targets(
    config: CandidateConfig,
    actor: Actor,
    critic: Critic,
    target_actor: Actor,
    target_critic: Critic,
) -> None:
    """Polyak-averaged target update. Applied ONCE per episode (after all gradient
    steps), matching the reference's once-per-cycle cadence. Applying tau=0.05 on
    EVERY gradient step would move the target by ~1-(0.95)^40≈0.87 over 40 steps
    instead of the reference's 0.05, letting the target chase the online network and
    defeating DDPG's slow-target stabilization."""
    with torch.no_grad():
        for target, source in zip(target_actor.parameters(), actor.parameters()):
            target.mul_(1.0 - config.tau).add_(source, alpha=config.tau)
        for target, source in zip(target_critic.parameters(), critic.parameters()):
            target.mul_(1.0 - config.tau).add_(source, alpha=config.tau)


def _scripted_rollout(
    config: CandidateConfig,
    env,
    rng: np.random.Generator,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one scripted reach-then-push rollout and return its trajectory.

    Two phases:
      1. Approach: move the gripper to a point BEHIND the puck (opposite the goal)
         using proportional control, closing the gripper to prepare for push.
      2. Push: move the gripper in the direction from the puck to the goal,
         maintaining contact and driving the puck toward the goal.

    The gripper must approach a point BEHIND the puck (opposite the goal) so that
    pushing toward the goal drives the gripper THROUGH the puck (verified: this
    moves the puck 0.377 -> 0.078). Approaching the puck itself lets the gripper
    slide past without pushing.
    """
    contact_dist = 0.06
    horizon = config.horizon
    offset_range = (0.02, 0.06)
    push_speed_range = (0.5, 1.0)
    obs, _ = env.reset(seed=seed)
    trajectory: list[dict[str, Any]] = []
    contacted = False
    for step in range(horizon):
        gripper_pos = obs["observation"][0:3].astype(np.float32)
        puck_pos = obs["achieved_goal"][0:3].astype(np.float32)
        goal = obs["desired_goal"][0:3].astype(np.float32)

        # Determine phase: approach until the gripper is BEHIND the puck
        # (opposite the goal) and close to it, then push.
        dist_grip_puck = np.linalg.norm(gripper_pos - puck_pos)
        dir_to_goal = goal - puck_pos
        norm = np.linalg.norm(dir_to_goal)
        if norm < 1e-6:
            behind = puck_pos
        else:
            behind = puck_pos - (dir_to_goal / norm) * 0.04
        behind_dist = np.linalg.norm(gripper_pos - behind)
        if not contacted and behind_dist < 0.08:
            contacted = True

        if not contacted:
            # Approach: move to a point behind the puck relative to the goal.
            if norm < 1e-6:
                target = puck_pos
            else:
                offset = rng.uniform(*offset_range)
                target = puck_pos - (dir_to_goal / norm) * offset
            delta = target - gripper_pos
            gain = 3.0
            action = np.zeros(4, dtype=np.float32)
            action[0:3] = np.clip(delta * gain, -1.0, 1.0)
            action[3] = 1.0  # close gripper to prepare for push
        else:
            # Push: move in the direction from puck to goal (THROUGH the puck).
            if norm < 1e-6:
                action = np.zeros(4, dtype=np.float32)
            else:
                # If the gripper lost contact (overshot past the puck), re-approach
                # the behind-point before pushing again; otherwise push through.
                if dist_grip_puck > 0.10:
                    target = behind
                    delta = target - gripper_pos
                    action = np.zeros(4, dtype=np.float32)
                    action[0:3] = np.clip(delta * 3.0, -1.0, 1.0)
                    action[3] = 1.0
                else:
                    push_speed = rng.uniform(*push_speed_range)
                    action = np.zeros(4, dtype=np.float32)
                    action[0:3] = np.clip((dir_to_goal / norm) * push_speed, -1.0, 1.0)
                    action[3] = 1.0  # keep gripper closed

        next_obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated or step + 1 >= horizon)
        trajectory.append(
            {
                "state": obs["observation"].copy(),
                "achieved_goal": obs["achieved_goal"].copy(),
                "goal": obs["desired_goal"].copy(),
                "action": action.copy(),
                "next_state": next_obs["observation"].copy(),
                "next_achieved_goal": next_obs["achieved_goal"].copy(),
                "gripper": gripper_pos.copy(),
                "next_gripper": next_obs["observation"][0:3].copy(),
                "done": done,
            }
        )
        obs = next_obs
        if done:
            break
    return trajectory


def _seed_scripted_rollouts(
    config: CandidateConfig,
    env,
    replay: EpisodeReplay,
    obs_normalizer: RunningNormalizer,
    goal_normalizer: RunningNormalizer,
    actor: Actor,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    """Seed the replay buffer with scripted reach-then-push rollouts.

    The scripted policy has two phases:
      1. Approach: move the gripper to a point behind the puck (relative to the
         goal) using proportional control, closing the gripper to prepare for push.
      2. Push: move the gripper in the direction from the puck to the goal,
         maintaining contact and driving the puck toward the goal.

    This provides the critic with early examples of contact and goal-directed
    pushing, which is critical for the sparse-reward slide task where random
    exploration rarely contacts the puck. The rollouts are stored in the replay
    buffer before training begins, and the normalizers are updated with the
    scripted data to give better initial statistics.
    """
    num_rollouts = getattr(config, "scripted_rollouts", 100)
    for rollout_idx in range(num_rollouts):
        # Use a distinct seed range to avoid overlapping with training seeds.
        seed = config.seed + 10_000 + rollout_idx
        trajectory = _scripted_rollout(config, env, rng, seed)
        replay.add(trajectory)
        # Update normalizers with the scripted data to improve initial statistics.
        obs_normalizer.update(np.asarray([row["state"] for row in trajectory]))
        goal_normalizer.update(np.asarray([row["goal"] for row in trajectory]))
    print(f"Seeded replay with {num_rollouts} scripted rollouts ({len(replay)} transitions)", flush=True)


def train_and_evaluate(
    config: CandidateConfig,
    output_dir: Path,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = _seed_everything(config.seed)
    device = _device(config)
    update_normalizers = True
    incumbent_metrics: EvaluationMetrics | None = None
    incumbent_actor_state: dict[str, torch.Tensor] | None = None
    incumbent_critic_state: dict[str, torch.Tensor] | None = None
    incumbent_target_actor_state: dict[str, torch.Tensor] | None = None
    incumbent_target_critic_state: dict[str, torch.Tensor] | None = None
    incumbent_obs_stats: tuple[np.ndarray, np.ndarray] | None = None
    incumbent_goal_stats: tuple[np.ndarray, np.ndarray] | None = None
    if init_checkpoint is not None:
        actor, obs_normalizer, goal_normalizer, loaded_critic, loaded_target_actor, loaded_target_critic = _load_actor_checkpoint(init_checkpoint, config.hidden_dim, device)
        incumbent_metrics = evaluate_actor(config, actor, obs_normalizer, goal_normalizer)
        incumbent_actor_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        if loaded_critic is not None:
            incumbent_critic_state = {name: value.detach().cpu().clone() for name, value in loaded_critic.state_dict().items()}
            incumbent_target_actor_state = {name: value.detach().cpu().clone() for name, value in loaded_target_actor.state_dict().items()}
            incumbent_target_critic_state = {name: value.detach().cpu().clone() for name, value in loaded_target_critic.state_dict().items()}
        incumbent_obs_stats = (obs_normalizer.mean.copy(), np.sqrt(obs_normalizer.var).copy())
        incumbent_goal_stats = (goal_normalizer.mean.copy(), np.sqrt(goal_normalizer.var).copy())
    else:
        obs_dim, goal_dim = skill_dims(config.skill)
        actor = Actor(hidden_dim=config.hidden_dim, input_dim=obs_dim + goal_dim).to(device)
        obs_normalizer = RunningNormalizer(obs_dim)
        goal_normalizer = RunningNormalizer(goal_dim)
        loaded_critic = None
        loaded_target_actor = None
        loaded_target_critic = None
    critic = loaded_critic if loaded_critic is not None else Critic(hidden_dim=config.hidden_dim, input_dim=obs_normalizer.mean.size + goal_normalizer.mean.size + 4).to(device)
    target_actor = loaded_target_actor if loaded_target_actor is not None else Actor(hidden_dim=config.hidden_dim, input_dim=obs_normalizer.mean.size + goal_normalizer.mean.size).to(device)
    target_critic = loaded_target_critic if loaded_target_critic is not None else Critic(hidden_dim=config.hidden_dim, input_dim=obs_normalizer.mean.size + goal_normalizer.mean.size + 4).to(device)
    if loaded_target_actor is None:
        target_actor.load_state_dict(actor.state_dict())
    if loaded_target_critic is None:
        target_critic.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    replay = EpisodeReplay(config.replay_capacity, prioritized=config.per, alpha=config.per_alpha, epsilon=config.per_epsilon, beta=config.per_beta)
    env = _make_env(config)
    distance_threshold = float(getattr(env.unwrapped, "distance_threshold", 0.05))
    reward_fn = _make_reward_fn(config, env, distance_threshold)
    # Seed the replay buffer with scripted reach-then-push rollouts so the actor
    # starts with contact-push examples (random exploration rarely contacts the puck).
    if getattr(config, "scripted_rollouts", 0) > 0:
        _seed_scripted_rollouts(config, env, replay, obs_normalizer, goal_normalizer, actor, device, rng)
    global_step = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    contact_rates: list[float] = []
    # Immediate startup banner: the first progress line only appears at log_every
    # episodes, so without this the run looks silent for the first ~2 minutes.
    print(
    f"train_start episodes={config.train_episodes} horizon={config.horizon} "
    f"device={device} replay={len(replay)} warm_start={init_checkpoint is not None}",
    flush=True,
    )
    # Track the best (lowest score) live-eval state seen during training. The final
    # eval is noisy and can regress past an earlier better state; keeping the best
    # observed state preserves mid-training progress instead of discarding it.
    # The FULL network state (actor + critic + targets) is captured so a restore
    # keeps the actor/critic pair consistent (no mismatched checkpoint).
    best_live_score: float | None = None
    best_live_actor_state: dict[str, torch.Tensor] | None = None
    best_live_critic_state: dict[str, torch.Tensor] | None = None
    best_live_target_actor_state: dict[str, torch.Tensor] | None = None
    best_live_target_critic_state: dict[str, torch.Tensor] | None = None
    best_live_obs_stats: tuple[np.ndarray, np.ndarray] | None = None
    best_live_goal_stats: tuple[np.ndarray, np.ndarray] | None = None
    try:
        if init_checkpoint is not None and loaded_critic is None and config.rehearse_critic:
            # CRI (critic rehearsal initialization): warm-start checkpoints carry only the actor,
            # so the critic starts random and ~flat; joint training then destroys the policy
            # (actor-only fine-tune collapses to the random-action floor). Rehearse the critic on
            # hindsight-relabeled rollouts of the frozen actor so Q is shaped before actor updates.
            actor.eval()
            for rehearsal_episode in range(20):
                observation, _ = env.reset(seed=config.seed + 10_000 + rehearsal_episode)
                trajectory: list[dict[str, Any]] = []
                for step in range(config.horizon):
                    state = np.concatenate(
                        [
                            obs_normalizer.normalize(observation["observation"]),
                            goal_normalizer.normalize(observation["desired_goal"]),
                        ]
                    )
                    with torch.inference_mode():
                        action = actor(torch.from_numpy(state).to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    next_observation, _, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated or step + 1 >= config.horizon)
                    trajectory.append(
                        {
                            "state": observation["observation"].copy(),
                            "achieved_goal": observation["achieved_goal"].copy(),
                            "goal": observation["desired_goal"].copy(),
                            "action": action.copy(),
                            "next_state": next_observation["observation"].copy(),
                            "next_achieved_goal": next_observation["achieved_goal"].copy(),
                            "done": done,
                        }
                    )
                    observation = next_observation
                    if done:
                        break
                replay.add(trajectory)
            for _ in range(300):
                batch = replay.sample(config.batch_size, 1.0, rng, reward_fn, config.her_future, distance_threshold, getattr(config, "skill", "slide"))
                state = np.concatenate([obs_normalizer.normalize(batch["state"]), goal_normalizer.normalize(batch["goal"])], axis=1)
                next_state = np.concatenate(
                    [obs_normalizer.normalize(batch["next_state"]), goal_normalizer.normalize(batch["goal"])],
                    axis=1,
                )
                state_t = torch.from_numpy(state).to(device)
                next_state_t = torch.from_numpy(next_state).to(device)
                action_t = torch.from_numpy(batch["action"]).to(device)
                reward_t = torch.from_numpy(batch["reward"]).to(device).unsqueeze(1)
                done_t = torch.from_numpy(batch["done"]).to(device).unsqueeze(1)
                with torch.no_grad():
                    next_action = target_actor(next_state_t)
                    noise = torch.randn_like(next_action) * config.policy_noise
                    noise = torch.clamp(noise, -config.noise_clip, config.noise_clip)
                    next_action = torch.clamp(next_action + noise, -1.0, 1.0)
                    tq1, tq2 = target_critic(next_state_t, next_action)
                    target_q = torch.min(tq1, tq2)
                    target_q = reward_t + config.gamma * (1.0 - done_t) * target_q
                    target_limit = 1.0 / max(1e-6, 1.0 - config.gamma)
                    target_q = torch.clamp(target_q, min=-target_limit)
                cq1, cq2 = critic(state_t, action_t)
                critic_loss = nn.functional.smooth_l1_loss(cq1, target_q) + nn.functional.smooth_l1_loss(cq2, target_q)
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                critic_optimizer.step()
            _soft_update_targets(config, actor, critic, target_actor, target_critic)
        for episode in range(config.train_episodes):
            # Interleave a scripted reach-then-push rollout every `scripted_every`
            # episodes so the replay continuously gets contact-push examples (the
            # actor's own exploration rarely contacts the puck).
            if config.scripted_every > 0 and episode % config.scripted_every == 0:
                scripted_traj = _scripted_rollout(config, env, rng, config.seed + 20_000 + episode)
                replay.add(scripted_traj)
                if update_normalizers:
                    obs_normalizer.update(np.asarray([row["state"] for row in scripted_traj]))
                    goal_normalizer.update(np.asarray([row["goal"] for row in scripted_traj]))
            observation, _ = env.reset(seed=config.seed + episode)
            trajectory: list[dict[str, Any]] = []
            # Contact telemetry: did the gripper get within contact distance of the
            # puck at any point this episode? (Contact rate is the key diagnostic for
            # whether the reach phase is working before the push phase.)
            contact_dist = 0.06
            episode_contact = False
            # Anneal random exploration to ~0 over training so the actor's own policy
            # dominates the collected data (DDPG needs to evaluate/improve its own
            # policy, not a permanently 30%-random one).
            anneal_progress = min(1.0, episode / max(1, config.train_episodes))
            random_prob = config.random_prob * (1.0 - 0.9 * anneal_progress)
            for step in range(config.horizon):
                if global_step < config.warmup_steps or rng.random() < random_prob:
                    action = env.action_space.sample().astype(np.float32)
                else:
                    state = np.concatenate(
                        [
                            obs_normalizer.normalize(observation["observation"]),
                            goal_normalizer.normalize(observation["desired_goal"]),
                        ]
                    )
                    with torch.inference_mode():
                        action = actor(torch.from_numpy(state).to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                    action = np.clip(action + rng.normal(0.0, config.noise_std, size=4), -1.0, 1.0).astype(np.float32)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated or step + 1 >= config.horizon)
                # Contact check: gripper within contact_dist of the puck's position.
                grip = np.asarray(observation["observation"][0:3], dtype=np.float32)
                puck = np.asarray(observation["achieved_goal"][0:3], dtype=np.float32)
                if np.linalg.norm(grip - puck) < contact_dist:
                    episode_contact = True
                trajectory.append(
                    {
                        "state": observation["observation"].copy(),
                        "achieved_goal": observation["achieved_goal"].copy(),
                        "goal": observation["desired_goal"].copy(),
                        "action": action.copy(),
                        "next_state": next_observation["observation"].copy(),
                        "next_achieved_goal": next_observation["achieved_goal"].copy(),
                        "gripper": observation["observation"][0:3].copy(),
                        "next_gripper": next_observation["observation"][0:3].copy(),
                        "done": done,
                    }
                )
                observation = next_observation
                global_step += 1
                if done:
                    break
            replay.add(trajectory)
            contact_rates.append(float(episode_contact))
            # Anneal importance-sampling exponent beta from per_beta to per_bet…
            # over training (Schaul 2015). beta=0 disables bias correction entirely.
            if config.per and config.per_beta_final != config.per_beta:
                total_steps = max(1, config.train_episodes * config.horizon)
                progress = min(1.0, global_step / total_steps)
                replay.set_beta(config.per_beta + (config.per_beta_final - config.per_beta) * progress)
            if update_normalizers:
                obs_normalizer.update(np.asarray([row["state"] for row in trajectory]))
                goal_normalizer.update(np.asarray([row["goal"] for row in trajectory]))
            if len(replay) >= config.batch_size:
                for _ in range(len(trajectory) * config.updates_per_step):
                    actor_loss, critic_loss = _update_networks(
                        config,
                        actor,
                        critic,
                        target_actor,
                        target_critic,
                        actor_optimizer,
                        critic_optimizer,
                        obs_normalizer,
                        goal_normalizer,
                        replay,
                        rng,
                        reward_fn,
                        device,
                        distance_threshold,
                        actor_step=global_step,
                    )
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                # Target networks soft-updated once per episode (after all gradient
                # steps), matching the reference's once-per-cycle cadence.
                _soft_update_targets(config, actor, critic, target_actor, target_critic)
            # Live progress: emit a bounded line every log_every episodes so the
            # dashboard can show in-flight work, not just finished trials.
            if (episode + 1) % config.log_every == 0:
                avg_actor = float(np.mean(actor_losses[-config.log_every * config.horizon:])) if actor_losses else 0.0
                avg_critic = float(np.mean(critic_losses[-config.log_every * config.horizon:])) if critic_losses else 0.0
                contact_rate = float(np.mean(contact_rates[-config.log_every:])) if contact_rates else 0.0
                print(
                    f"progress episode={episode + 1}/{config.train_episodes} steps={global_step} "
                    f"actor_loss={avg_actor:.4f} critic_loss={avg_critic:.4f} contact_rate={contact_rate:.3f}",
                    flush=True,
                )
            if (episode + 1) % config.eval_every == 0:
                live = evaluate_actor(config, actor, obs_normalizer, goal_normalizer)
                print(
                    f"live_eval episode={episode + 1} success={live.success_rate:.3f} "
                    f"dist={live.mean_final_distance:.4f} score={score_metrics(live, config.skill, config.score_config):.4f}",
                    flush=True,
                )
                live_score = score_metrics(live, config.skill, config.score_config)
                if best_live_score is None or live_score < best_live_score:
                    best_live_score = live_score
                    best_live_actor_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
                    best_live_critic_state = {name: value.detach().cpu().clone() for name, value in critic.state_dict().items()}
                    best_live_target_actor_state = {name: value.detach().cpu().clone() for name, value in target_actor.state_dict().items()}
                    best_live_target_critic_state = {name: value.detach().cpu().clone() for name, value in target_critic.state_dict().items()}
                    best_live_obs_stats = (obs_normalizer.mean.copy(), np.sqrt(obs_normalizer.var).copy())
                    best_live_goal_stats = (goal_normalizer.mean.copy(), np.sqrt(goal_normalizer.var).copy())
    finally:
        env.close()

    candidate_metrics = evaluate_actor(config, actor, obs_normalizer, goal_normalizer)
    # If a better state was observed mid-training (live eval), restore it: the final
    # eval is noisy and the incumbent gate would otherwise compare against a regressed
    # final state, discarding real progress.
    if best_live_score is not None and best_live_score < score_metrics(candidate_metrics, config.skill, config.score_config):
        assert best_live_actor_state is not None
        assert best_live_critic_state is not None
        assert best_live_target_actor_state is not None
        assert best_live_target_critic_state is not None
        assert best_live_obs_stats is not None
        assert best_live_goal_stats is not None
        actor.load_state_dict(best_live_actor_state)
        critic.load_state_dict(best_live_critic_state)
        target_actor.load_state_dict(best_live_target_actor_state)
        target_critic.load_state_dict(best_live_target_critic_state)
        obs_normalizer = RunningNormalizer.from_arrays(*best_live_obs_stats)
        goal_normalizer = RunningNormalizer.from_arrays(*best_live_goal_stats)
        candidate_metrics = evaluate_actor(config, actor, obs_normalizer, goal_normalizer)
    metrics = candidate_metrics
    warm_start_fallback = False
    if incumbent_metrics is not None and score_metrics(candidate_metrics, config.skill, config.score_config) > score_metrics(incumbent_metrics, config.skill, config.score_config):
        assert incumbent_actor_state is not None
        assert incumbent_obs_stats is not None
        assert incumbent_goal_stats is not None
        actor.load_state_dict(incumbent_actor_state)
        obs_normalizer = RunningNormalizer.from_arrays(*incumbent_obs_stats)
        goal_normalizer = RunningNormalizer.from_arrays(*incumbent_goal_stats)
        # Restore the full network pair when the incumbent checkpoint carried a critic;
        # actor-only checkpoints (no critic) keep the trained critic (matching prior behavior).
        if incumbent_critic_state is not None:
            assert incumbent_target_actor_state is not None
            assert incumbent_target_critic_state is not None
            critic.load_state_dict(incumbent_critic_state)
            target_actor.load_state_dict(incumbent_target_actor_state)
            target_critic.load_state_dict(incumbent_target_critic_state)
        metrics = incumbent_metrics
        warm_start_fallback = True
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "format_version": 2,
            "config": config.to_dict(),
            "actor_state": actor.cpu().state_dict(),
            "critic_state": critic.cpu().state_dict(),
            "target_actor_state": target_actor.cpu().state_dict(),
            "target_critic_state": target_critic.cpu().state_dict(),
            "obs_mean": obs_normalizer.mean,
            "obs_std": np.sqrt(obs_normalizer.var),
            "goal_mean": goal_normalizer.mean,
            "goal_std": np.sqrt(goal_normalizer.var),
        },
        checkpoint_path,
    )
    result = {
        "config": config.to_dict(),
        "metrics": metrics.to_dict(),
        "candidate_metrics": candidate_metrics.to_dict(),
        "warm_start_fallback": warm_start_fallback,
        "score": score_metrics(metrics, config.skill, config.score_config),
        "steps": global_step,
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else None,
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else None,
        "contact_rate": float(np.mean(contact_rates)) if contact_rates else None,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_checkpoint(config: CandidateConfig, checkpoint: Path) -> EvaluationMetrics:
    device = torch.device("cpu")
    actor, obs_normalizer, goal_normalizer, _, _, _ = _load_actor_checkpoint(checkpoint, config.hidden_dim, device)
    return evaluate_actor(config, actor, obs_normalizer, goal_normalizer)
