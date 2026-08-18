from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np


class EpisodeReplay:
    """Bounded trajectory replay with future-goal relabeling and PER."""

    def __init__(self, max_transitions: int, prioritized: bool = True, alpha: float = 0.5, epsilon: float = 0.1, beta: float = 0.4) -> None:
        self.max_transitions = max_transitions
        self.prioritized = prioritized
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.episodes: deque[list[dict[str, Any]]] = deque()
        self.transition_count = 0

    def set_beta(self, beta: float) -> None:
        """Annealed importance-sampling exponent; beta=0 disables bias correction."""
        self.beta = float(beta)

    def add(self, episode: list[dict[str, Any]]) -> None:
        if not episode:
            return
        stored_episode = []
        for transition in episode:
            row = dict(transition)
            row["_priority"] = 1.0
            stored_episode.append(row)
        self.episodes.append(stored_episode)
        self.transition_count += len(stored_episode)
        while self.transition_count > self.max_transitions and self.episodes:
            self.transition_count -= len(self.episodes.popleft())

    def __len__(self) -> int:
        return self.transition_count

    def sample(
        self,
        batch_size: int,
        her_ratio: float,
        rng: np.random.Generator,
        reward_fn: Callable[[np.ndarray, np.ndarray, Any], Any],
        her_future: int = 4,
        distance_threshold: float = 0.05,
        skill: str = "slide",
        reference_her: bool = False,
    ) -> dict[str, np.ndarray]:
        if not self.episodes:
            raise RuntimeError("cannot sample an empty replay buffer")
        candidates = [(episode_index, transition_index) for episode_index, episode in enumerate(self.episodes) for transition_index in range(len(episode))]
        if self.prioritized:
            priorities = np.asarray([self.episodes[episode_index][transition_index]["_priority"] for episode_index, transition_index in candidates], dtype=np.float64)
            weights = np.power(np.maximum(priorities, self.epsilon), self.alpha)
            probabilities = weights / weights.sum()
            selected = rng.choice(len(candidates), size=batch_size, replace=True, p=probabilities)
        else:
            selected = rng.integers(0, len(candidates), size=batch_size)
        if reference_her:
            # Original HER implementation samples exactly batch_size transitions
            # and relabels each selected row at most once with one future goal.
            # It does not expand a row into 1+k transitions.
            out_rows = []
            if self.prioritized and self.beta > 0.0:
                max_weight = float(np.max(weights))
                n = float(len(candidates))
                candidate_weights = np.power(n * probabilities[selected], -self.beta) / max_weight
            row_weights = []
            for row_index, selected_index in enumerate(selected):
                episode_index, index = candidates[int(selected_index)]
                episode = self.episodes[episode_index]
                transition = episode[index]
                if rng.random() < her_ratio and index + 1 < len(episode):
                    future_index = int(rng.integers(index + 1, len(episode)))
                    goal = np.asarray(episode[future_index]["next_achieved_goal"], dtype=np.float32).copy()
                else:
                    goal = np.asarray(transition["goal"], dtype=np.float32).copy()
                out_rows.append(self._row(episode_index, index, transition, goal, reward_fn, distance_threshold, skill))
                if self.prioritized and self.beta > 0.0:
                    row_weights.append(float(candidate_weights[row_index]))
            result = {key: np.asarray([row[key] for row in out_rows], dtype=np.float32) for key in out_rows[0]}
            if self.prioritized and self.beta > 0.0:
                result["weight"] = np.asarray(row_weights, dtype=np.float32)
            return result
        rows: list[dict[str, Any]] = []
        for selected_index in selected:
            episode_index, index = candidates[int(selected_index)]
            episode = self.episodes[episode_index]
            transition = episode[index]
            future_goals = []
            if rng.random() < her_ratio and len(episode) > index + 1:
                # Sample future-goal indices WITHOUT replacement to avoid duplicate relabeled rows.
                n_future = len(episode) - (index + 1)
                k = min(her_future, n_future)
                future_goals = [int(i) for i in rng.choice(np.arange(index + 1, len(episode)), size=k, replace=False)]
            rows.append((episode_index, index, transition, future_goals))
        out_rows = []
        row_weights: list[float] = []
        if self.prioritized and self.beta > 0.0:
            # Importance-sampling weights correct the priority bias (Schaul 2015).
            # w_i = (N * P(i))^-beta, normalized by max weight. beta=0 disables (legacy).
            max_weight = float(np.max(weights))
            n = float(len(candidates))
            candidate_weights = np.power(n * probabilities[selected], -self.beta) / max_weight
        for row_index, (episode_index, index, transition, future_goals) in enumerate(rows):
            multiplicity = 1 + len(future_goals)
            if future_goals:
                for future_index in future_goals:
                    goal = np.asarray(episode[future_index]["next_achieved_goal"], dtype=np.float32).copy()
                    out_rows.append(self._row(episode_index, index, transition, goal, reward_fn, distance_threshold, skill))
                    if self.prioritized and self.beta > 0.0:
                        # Normalize weight by row multiplicity so relabeled expansion
                        # doesn't inflate the effective per-row weight.
                        row_weights.append(float(candidate_weights[row_index]) / multiplicity)
            else:
                goal = np.asarray(transition["goal"], dtype=np.float32).copy()
                out_rows.append(self._row(episode_index, index, transition, goal, reward_fn, distance_threshold, skill))
                if self.prioritized and self.beta > 0.0:
                    row_weights.append(float(candidate_weights[row_index]) / multiplicity)
        result = {key: np.asarray([row[key] for row in out_rows], dtype=np.float32) for key in out_rows[0]}
        if self.prioritized and self.beta > 0.0:
            result["weight"] = np.asarray(row_weights, dtype=np.float32)
        return result

    def _row(
        self,
        episode_index: int,
        index: int,
        transition: dict[str, Any],
        goal: np.ndarray,
        reward_fn: Callable[[np.ndarray, np.ndarray, Any], Any],
        distance_threshold: float,
        skill: str = "slide",
    ) -> dict[str, Any]:
        kwargs = {}
        if "next_gripper" in transition:
            kwargs = {
                "gripper_now": transition["gripper"],
                "gripper_next": transition["next_gripper"],
                "puck_now": transition["achieved_goal"],
                "puck_next": transition["next_achieved_goal"],
            }
        reward = reward_fn(transition["next_achieved_goal"], goal, None, **kwargs)
        # Explicit relabeling flag: goal differs from the stored original goal.
        relabeled = not np.allclose(goal, np.asarray(transition["goal"], dtype=np.float32))
        # Recompute done ONLY for the original goal: a transition is terminal if the
        # episode ended OR the puck reached the original goal. For relabeled rows the
        # goal is a future achieved goal, so "reaching" it is not a real terminal event
        # (under random actions the puck barely moves, so a relabeled future goal is
        # ~equal to next_achieved_goal and would wrongly mark ~100% of rows done,
        # collapsing target_q to 0 and flattening the critic/actor gradient).
        done = float(transition["done"])
        if not relabeled:
            # Skill-aware reached check: avoid mixing position (m) and yaw (rad) in a
            # single Euclidean norm. slide -> position distance; rotate -> position
            # distance AND yaw error; spin -> yaw-rate error.
            ag = np.asarray(transition["next_achieved_goal"], dtype=np.float32)
            g = np.asarray(goal, dtype=np.float32)
            if skill == "rotate":
                reached = float(np.linalg.norm(ag[0:3] - g[0:3]) < distance_threshold and abs(float(ag[3]) - float(g[3])) < distance_threshold)
            elif skill == "spin":
                reached = float(abs(float(ag[0]) - float(g[0])) < distance_threshold)
            else:
                reached = float(np.linalg.norm(ag - g) < distance_threshold)
            done = float(transition["done"] or bool(reached))
        return {
            "state": transition["state"],
            "achieved_goal": transition["achieved_goal"],
            "action": transition["action"],
            "next_state": transition["next_state"],
            "next_achieved_goal": transition["next_achieved_goal"],
            "goal": goal,
            "reward": float(np.asarray(reward).reshape(-1)[0]),
            "done": done,
            "relabeled": float(relabeled),
            "episode_index": episode_index,
            "transition_index": index,
        }

    def update_priorities(
        self,
        episode_indices: np.ndarray,
        transition_indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        # Cap priorities to avoid PER-HER conflict: relabeled hindsight transitions
        # produce extreme TD-error spikes on impact frames; uncapped, PER oversamples
        # those and starves the critic of regular sliding/settling trajectories.
        # Clamp to [eps, median*cap] so no single transition dominates sampling.
        prio = np.abs(np.asarray(priorities, dtype=np.float64)) + self.epsilon
        if prio.size:
            cap = float(np.median(prio)) * 10.0 + self.epsilon
            prio = np.clip(prio, self.epsilon, max(cap, self.epsilon))
        for episode_index, transition_index, priority in zip(episode_indices, transition_indices, prio):
            episode_index = int(episode_index)
            transition_index = int(transition_index)
            if episode_index >= len(self.episodes):
                continue
            episode = self.episodes[episode_index]
            if transition_index >= len(episode):
                continue
            episode[transition_index]["_priority"] = float(priority)
