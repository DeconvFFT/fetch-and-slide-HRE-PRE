from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Colab has no desktop display. EGL is the correct headless MuJoCo backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
import torch

from .config import config_from_mapping
from .trainer import _load_actor_checkpoint
from .skills import skill_dims


gym.register_envs(gymnasium_robotics)


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
    except Exception:
        # GIF is a useful fallback when ffmpeg is unavailable in a runtime.
        gif_path = path.with_suffix(".gif")
        imageio.mimsave(gif_path, [np.asarray(frame) for frame in frames], duration=1.0 / fps)


def render_checkpoint(
    config_path: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    episodes: int = 2,
    fps: int = 20,
    step: int = 0,
    tensorboard_dir: Path | None = None,
) -> dict[str, Any]:
    config = config_from_mapping(json.loads(config_path.read_text()))
    output_dir.mkdir(parents=True, exist_ok=True)
    if tensorboard_dir is None:
        tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, obs_normalizer, goal_normalizer, _, _, _ = _load_actor_checkpoint(
        checkpoint, config.hidden_dim, device, config.algorithm
    )
    actor.eval()

    # Render mode is only needed for this diagnostic process; training remains
    # headless and does not pay the rendering cost.
    env = gym.make(config.env_id, render_mode="rgb_array")
    successes: list[float] = []
    final_distances: list[float] = []
    min_goal_distances: list[float] = []
    contact_flags: list[float] = []
    min_grip_distances: list[float] = []
    puck_displacements: list[float] = []
    videos: list[tuple[str, np.ndarray]] = []

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=config.seed + config.eval_seed_offset + episode)
            initial_puck = np.asarray(observation["achieved_goal"][0:3], dtype=np.float32).copy()
            frames: list[np.ndarray] = []
            contacted = False
            min_goal = float("inf")
            min_grip = float("inf")
            final_distance = float("inf")
            success = False
            total_steps = 0

            for _ in range(config.horizon):
                grip = np.asarray(observation["observation"][0:3], dtype=np.float32)
                puck = np.asarray(observation["achieved_goal"][0:3], dtype=np.float32)
                goal = np.asarray(observation["desired_goal"][0:3], dtype=np.float32)
                min_grip = min(min_grip, float(np.linalg.norm(grip - puck)))
                contacted = contacted or min_grip < 0.06
                min_goal = min(min_goal, float(np.linalg.norm(puck - goal)))

                state = np.concatenate(
                    [
                        obs_normalizer.normalize(observation["observation"]),
                        goal_normalizer.normalize(observation["desired_goal"]),
                    ]
                )
                with torch.inference_mode():
                    action = actor(torch.from_numpy(state).to(device).unsqueeze(0)).squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                observation, _, terminated, truncated, info = env.step(action)
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
                final_puck = np.asarray(observation["achieved_goal"][0:3], dtype=np.float32)
                final_goal = np.asarray(observation["desired_goal"][0:3], dtype=np.float32)
                final_distance = float(np.linalg.norm(final_puck - final_goal))
                success = success or bool(info.get("is_success", False))
                total_steps += 1
                if terminated or truncated:
                    break

            final_puck = np.asarray(observation["achieved_goal"][0:3], dtype=np.float32)
            successes.append(float(success))
            final_distances.append(final_distance)
            min_goal_distances.append(min_goal)
            contact_flags.append(float(contacted))
            min_grip_distances.append(min_grip)
            puck_displacements.append(float(np.linalg.norm(final_puck - initial_puck)))
            video_path = output_dir / f"episode-{episode:03d}.mp4"
            _write_video(frames, video_path, fps)
            if frames:
                videos.append((f"eval/episode_{episode}", np.asarray(frames)))

    finally:
        env.close()

    telemetry = {
        "step": int(step),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_final_distance": float(np.mean(final_distances)) if final_distances else 0.0,
        "contact_rate": float(np.mean(contact_flags)) if contact_flags else 0.0,
        "mean_min_goal_distance": float(np.mean(min_goal_distances)) if min_goal_distances else 0.0,
        "mean_min_gripper_puck_distance": float(np.mean(min_grip_distances)) if min_grip_distances else 0.0,
        "mean_puck_displacement": float(np.mean(puck_displacements)) if puck_displacements else 0.0,
        "episodes": len(successes),
        "checkpoint": str(checkpoint),
        "videos": [str(output_dir / f"episode-{i:03d}.mp4") for i in range(len(videos))],
    }
    (output_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(tensorboard_dir))
        for name, value in telemetry.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"physics/{name}", value, step)
        for tag, frames in videos:
            video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
            writer.add_video(tag, video, global_step=step, fps=fps)
        writer.flush()
        writer.close()
    except Exception as exc:
        telemetry["tensorboard_error"] = str(exc)
        (output_dir / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    print(json.dumps(telemetry), flush=True)
    return telemetry


def main() -> int:
    parser = argparse.ArgumentParser(description="Render checkpoint physics diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard", type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    render_checkpoint(
        args.config,
        args.checkpoint,
        args.output,
        episodes=args.episodes,
        fps=args.fps,
        step=args.step,
        tensorboard_dir=args.tensorboard,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
