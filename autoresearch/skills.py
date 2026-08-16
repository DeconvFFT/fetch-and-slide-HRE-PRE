from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

VALID_SKILLS = ("slide", "rotate", "spin", "pick", "fetch")

# Skill -> underlying gymnasium-robotics env id.
#   slide/rotate/spin all run on FetchSlide (rotate/spin re-frame it via
#   SkillWrapper; slide is a byte-identical passthrough).
#   pick  -> FetchPickAndPlace-v4 (native obs/goal/reward, no wrapper)
#   fetch -> FetchReach-v4        (native obs/goal/reward, no wrapper)
SKILL_ENV = {
    "slide": "FetchSlide-v4",
    "rotate": "FetchSlide-v4",
    "spin": "FetchSlide-v4",
    "pick": "FetchPickAndPlace-v4",
    "fetch": "FetchReach-v4",
}

# FetchSlide object free joint layout (from the MuJoCo model):
#   qpos[15:22] = [x, y, z, qw, qx, qy, qz]
#   qvel[15:21] = [vx, vy, vz, wx, wy, wz]
_OBJECT_QPOS_ADR = 15
_OBJECT_QUAT_SLICE = slice(_OBJECT_QPOS_ADR + 3, _OBJECT_QPOS_ADR + 7)
_OBJECT_WZ_INDEX = 20  # qvel[20] = angular velocity about the world z axis (yaw rate)

# Skill-specific thresholds.
ROTATE_YAW_THRESHOLD = 0.2  # radians (~11 deg) of yaw error tolerated for success
SPIN_WZ_THRESHOLD = 0.5  # rad/s of angular-velocity error tolerated for success


def skill_dims(skill: str) -> tuple[int, int]:
    """Return the (obs_dim, goal_dim) the trainer must use for a skill.

    slide  -> (25, 3)  unchanged base observation / 3-dim puck goal
    rotate -> (26, 4)  base obs + object yaw / goal (x, y, z, target_yaw)
    spin   -> (26, 1)  base obs + yaw-rate / goal (target_wz)
    """
    if skill == "slide":
        return 25, 3
    if skill == "rotate":
        return 26, 4
    if skill == "spin":
        return 26, 1
    if skill == "pick":
        return 25, 3
    if skill == "fetch":
        return 10, 3
    raise ValueError(f"unknown skill: {skill!r} (valid: {', '.join(VALID_SKILLS)})")


def _object_yaw(env) -> float:
    """Current yaw (radians, about world z) of the object, from its quaternion."""
    qw, qx, qy, qz = env.unwrapped.data.qpos[_OBJECT_QUAT_SLICE]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def _object_wz(env) -> float:
    """Current yaw-rate (rad/s) of the object."""
    return float(env.unwrapped.data.qvel[_OBJECT_WZ_INDEX])


def _ang_diff(a: float, b: float) -> float:
    """Smallest signed absolute angular difference in [-pi, pi]."""
    return float(abs((a - b + np.pi) % (2.0 * np.pi) - np.pi))


class SkillWrapper:
    """Gym.Env-like adapter that re-frames FetchSlide for a skill.

    For skill="slide" the trainer does NOT wrap (byte-identical passthrough), so
    this class only ever runs rotate/spin. It augments the base observation with
    the object's yaw (rotate) or yaw-rate (spin) and redefines the goal so the
    agent can learn to orient or spin the object.
    """

    def __init__(
        self,
        env,
        skill: str = "slide",
        distance_threshold: float = 0.05,
        yaw_threshold: float = ROTATE_YAW_THRESHOLD,
        wz_threshold: float = SPIN_WZ_THRESHOLD,
    ) -> None:
        if skill not in VALID_SKILLS:
            raise ValueError(f"unknown skill: {skill!r}")
        self.env = env
        self.skill = skill
        self.distance_threshold = distance_threshold
        self.yaw_threshold = yaw_threshold
        self.wz_threshold = wz_threshold
        self.obs_dim, self.goal_dim = skill_dims(skill)
        self.action_space = env.action_space
        self.unwrapped = env.unwrapped
        self.metadata = env.metadata
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(self.goal_dim,), dtype=np.float32),
            }
        )
        self._target_yaw: float = 0.0
        self._target_wz: float = 0.0

    # -- goal sampling -----------------------------------------------------
    def _sample_target(self) -> None:
        if self.skill == "rotate":
            self._target_yaw = float(self.env.unwrapped.np_random.uniform(-np.pi, np.pi))
        elif self.skill == "spin":
            # Sustained rotation: ask for a moderate yaw-rate the gripper can drive.
            self._target_wz = float(self.env.unwrapped.np_random.uniform(2.0, 6.0))

    def _current_goal(self) -> np.ndarray:
        base_goal = self.env.unwrapped.goal
        if self.skill == "rotate":
            return np.concatenate([np.asarray(base_goal, dtype=np.float32), [self._target_yaw]])
        if self.skill == "spin":
            return np.asarray([self._target_wz], dtype=np.float32)
        return np.asarray(base_goal, dtype=np.float32)

    # -- observation adaptation --------------------------------------------
    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        base_obs = np.asarray(obs["observation"], dtype=np.float32)
        if self.skill == "rotate":
            yaw = _object_yaw(self.env)
            observation = np.concatenate([base_obs, [yaw]])
            achieved_goal = np.concatenate([np.asarray(obs["achieved_goal"], dtype=np.float32), [yaw]])
            desired_goal = self._current_goal()
        elif self.skill == "spin":
            wz = _object_wz(self.env)
            observation = np.concatenate([base_obs, [wz]])
            achieved_goal = np.asarray([wz], dtype=np.float32)
            desired_goal = self._current_goal()
        else:  # slide (unused in practice, kept for completeness)
            observation = base_obs
            achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)
            desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
        return {
            "observation": observation.astype(np.float32),
            "achieved_goal": achieved_goal.astype(np.float32),
            "desired_goal": desired_goal.astype(np.float32),
        }

    # -- reward ------------------------------------------------------------
    def compute_reward(self, achieved_goal: Any, goal: Any, info: Any) -> np.ndarray:
        """Sparse skill reward: -1 unless the skill-specific goal is reached."""
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        g = np.asarray(goal, dtype=np.float32)
        if self.skill == "rotate":
            pos_dist = float(np.linalg.norm(achieved[0:3] - g[0:3]))
            yaw_err = _ang_diff(float(achieved[3]), float(g[3]))
            reached = pos_dist < self.distance_threshold and yaw_err < self.yaw_threshold
        elif self.skill == "spin":
            wz_err = float(abs(float(achieved[0]) - float(g[0])))
            reached = wz_err < self.wz_threshold
        else:
            reached = float(np.linalg.norm(achieved - g)) < self.distance_threshold
        return -np.asarray(not reached, dtype=np.float32)

    def _is_success(self, achieved_goal: Any, goal: Any) -> float:
        return float(self.compute_reward(achieved_goal, goal, None) == 0.0)

    # -- gym.Env interface -------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._sample_target()
        return self._adapt_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        adapted = self._adapt_obs(obs)
        goal = self._current_goal()
        reward = float(self.compute_reward(adapted["achieved_goal"], goal, info))
        info["is_success"] = self._is_success(adapted["achieved_goal"], goal)
        return adapted, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
