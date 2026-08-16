from __future__ import annotations

import json
import pytest
import numpy as np
from pathlib import Path

from autoresearch.config import CandidateConfig, config_from_mapping
from autoresearch.metrics import EvaluationMetrics, score_metrics


def test_candidate_config_rejects_unknown_or_unsafe_parameters() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        config_from_mapping({"optimizer_command": "rm -rf /"})

    with pytest.raises(ValueError, match="actor_lr"):
        config_from_mapping({"actor_lr": -0.1})


def test_candidate_config_round_trips_only_allowlisted_values() -> None:
    config = config_from_mapping({"actor_lr": 0.0005, "batch_size": 32})

    assert isinstance(config, CandidateConfig)
    assert config.actor_lr == 0.0005
    assert config.batch_size == 32
    assert config.critic_lr == CandidateConfig().critic_lr
    assert set(config.to_dict()) == set(CandidateConfig().to_dict())

def test_candidate_config_exposes_prioritized_her_replay_knobs() -> None:
    from autoresearch.config import CandidateConfig, config_from_mapping

    config = config_from_mapping({"per": False, "per_alpha": 0.7, "per_epsilon": 0.2})

    assert CandidateConfig().per is True
    assert config.per is False
    assert config.per_alpha == 0.7
    assert config.per_epsilon == 0.2

def test_episode_replay_returns_indices_and_updates_priority() -> None:
    from autoresearch.replay import EpisodeReplay

    episode = [
        {
            "state": np.zeros(25, dtype=np.float32),
            "achieved_goal": np.zeros(3, dtype=np.float32),
            "action": np.zeros(4, dtype=np.float32),
            "next_state": np.zeros(25, dtype=np.float32),
            "next_achieved_goal": np.zeros(3, dtype=np.float32),
            "goal": np.ones(3, dtype=np.float32),
            "done": True,
        }
    ]
    replay = EpisodeReplay(4, prioritized=True)
    replay.add(episode)
    batch = replay.sample(2, 1.0, np.random.default_rng(7), lambda *_: 0.0)

    assert "episode_index" in batch
    assert "transition_index" in batch
    replay.update_priorities(batch["episode_index"], batch["transition_index"], np.full(2, 3.0))
    assert episode[0].get("_priority") is None
    assert replay.episodes[0][0]["_priority"] == pytest.approx(3.1)


def test_score_prefers_success_before_distance() -> None:
    successful = EvaluationMetrics(
        success_rate=1.0,
        mean_final_distance=0.30,
        mean_return=-1.0,
        episodes=4,
    )

    failed = EvaluationMetrics(
        success_rate=0.75,
        mean_final_distance=0.01,
        mean_return=-1.0,
        episodes=4,
    )

    assert score_metrics(successful) < score_metrics(failed)


def test_score_is_skill_aware_and_configurable() -> None:
    from autoresearch.metrics import score_metrics

    m = EvaluationMetrics(success_rate=0.2, mean_final_distance=0.36, mean_return=-48.0, episodes=5)
    # slide default: success dominates (10*(1-0.2) + 0.36 = 8.36)
    assert score_metrics(m, "slide") == pytest.approx(8.36)
    # push: prioritize distance: 10*(1-0.2) + 5*0.36 = 9.8
    assert score_metrics(m, "slide", '{"distance_weight":5}') == pytest.approx(9.8)
    # rotate uses yaw error weighting
    m2 = EvaluationMetrics(success_rate=0.1, mean_final_distance=0.5, mean_return=-49.0, episodes=5)
    assert score_metrics(m2, "rotate") == pytest.approx(9.5)
    assert score_metrics(m2, "rotate", '{"yaw_weight":3}') == pytest.approx(10.5)
def test_agent_proposals_cannot_request_commands_or_paths() -> None:
    from autoresearch.config import validate_overrides

    assert validate_overrides({"actor_lr": 0.0005}) == {"actor_lr": 0.0005}
    with pytest.raises(ValueError, match="unknown parameter"):
        validate_overrides({"command": "echo unsafe"})


def test_proposal_requests_low_reasoning_budget(monkeypatch) -> None:
    from autoresearch import proposal

    captured: list[dict[str, object]] = []
    responses = iter(
        [
            b'{"choices":[{"message":{"content":"tune goal_bonus"}}]}',  # strategist (text)
            b'{"choices":[{"message":{"content":"{\\"actor_lr\\":0.0005}"}}]}',  # proposer
            b'{"choices":[{"message":{"content":"{\\"actor_lr\\":0.0005}"}}]}',  # critic
            b'{"choices":[{"message":{"content":"{\\"actor_lr\\":0.0005}"}}]}',  # code editor (hyperparams, no code_edit)
        ]
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return next(responses)

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(proposal, "urlopen", fake_urlopen)
    result = proposal.propose_overrides(
        CandidateConfig(),
        EvaluationMetrics(0.4, 0.064, -43.0, 5),
        [],
        api_key="test-key",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

    assert result == {"actor_lr": 0.0005}
    # Four LLM calls: strategist + proposer + critic + code editor.
    # Reasoning traces are excluded so the token budget goes to the JSON decision
    # (a long reasoning trace otherwise truncates the content).
    assert len(captured) == 4
    assert captured[0]["reasoning"] == {"effort": "high", "exclude": True}


def test_proposal_reports_empty_completion_context() -> None:
    from autoresearch.proposal import _content

    response = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "choices": [
            {
                "finish_reason": "length",
                "native_finish_reason": "length",
                "message": {"role": "assistant", "content": None, "reasoning": "hidden"},
            }
        ],
    }

    with pytest.raises(ValueError, match="no text proposal.*finish_reason=length"):
        _content(response)


def test_proposal_multi_agent_proposer_then_critic(monkeypatch) -> None:
    from autoresearch import proposal

    bodies: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode()

    # Strategist (text), proposer returns push_coef 8.0, critic refines to 6.0,
    # code editor keeps the critic's hyperparameters (no code_edit).
    responses = iter(
        [
            {"choices": [{"message": {"content": "tune push_coef"}}]},
            {"choices": [{"message": {"content": json.dumps({"push_coef": 8.0})}}]},
            {"choices": [{"message": {"content": json.dumps({"push_coef": 6.0})}}]},
            {"choices": [{"message": {"content": json.dumps({"push_coef": 6.0})}}]},
        ]
    )

    def fake_urlopen(request, timeout):
        bodies.append(json.loads(request.data))
        return FakeResponse(next(responses))

    monkeypatch.setattr(proposal, "urlopen", fake_urlopen)
    result = proposal.propose_overrides(
        CandidateConfig(),
        EvaluationMetrics(0.4, 0.064, -43.0, 5),
        [],
        api_key="test-key",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

    # The critic's refined proposal wins.
    assert result == {"push_coef": 6.0}
    # Four LLM calls: strategist then proposer then critic then code editor.
    assert len(bodies) == 4

def test_trace_writer_appends_events_and_updates_status(tmp_path) -> None:
    from autoresearch.runner import TraceWriter

    trace = TraceWriter(tmp_path)
    trace.record("trial_started", trial=3, phase="training")

    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    status = json.loads((tmp_path / "status.json").read_text())
    assert events[0]["event"] == "trial_started"
    assert events[0]["trial"] == 3
    assert status["phase"] == "training"
    assert status["trial"] == 3


def test_local_search_returns_bounded_numeric_override() -> None:
    from autoresearch.runner import local_candidate_overrides

    override = local_candidate_overrides(4, CandidateConfig())
    assert len(override) == 1
    assert set(override) <= {"actor_lr", "critic_lr", "gamma", "tau", "her_ratio", "noise_std", "batch_size", "updates_per_step"}


def test_research_readers_use_bounded_tails(tmp_path, monkeypatch) -> None:
    from demo.server import _read_results, _read_trace

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps({"event": "tick", "trial": index}) + "\n" for index in range(240)))
    results_path = tmp_path / "results.tsv"
    rows = ["trial\tscore\tstatus"] + [f"{index}\t{index / 10:.1f}\tdiscard" for index in range(240)]
    results_path.write_text("\n".join(rows) + "\n")

    def reject_full_text_read(_path: Path, *args, **kwargs):
        raise AssertionError("dashboard reader used full-file read_text")

    monkeypatch.setattr(Path, "read_text", reject_full_text_read)
    events = _read_trace(trace_path, limit=7)
    results = _read_results(results_path, limit=7)

    assert [event["trial"] for event in events] == list(range(233, 240))
    assert [row["trial"] for row in results] == [str(index) for index in range(233, 240)]

def test_running_normalizer_keeps_minimum_standard_deviation() -> None:
    from autoresearch.trainer import RunningNormalizer

    normalizer = RunningNormalizer(2)
    normalizer.update(np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    assert min(normalizer.to_dict()["std"]) == pytest.approx(0.01, abs=1e-6)


def test_warm_started_training_writes_valid_checkpoint(tmp_path) -> None:
    import torch

    from autoresearch.config import config_from_mapping
    from autoresearch.model import Actor
    from autoresearch.trainer import train_and_evaluate

    init_checkpoint = tmp_path / "init.pt"
    torch.save(
        {
            "format_version": 1,
            "actor_state": Actor(hidden_dim=32).state_dict(),
            "obs_mean": np.zeros(25, dtype=np.float32),
            "obs_std": np.full(25, 0.5, dtype=np.float32),
            "goal_mean": np.zeros(3, dtype=np.float32),
            "goal_std": np.full(3, 0.5, dtype=np.float32),
        },
        init_checkpoint,
    )
    config = config_from_mapping(
        {
            "hidden_dim": 32,
            "train_episodes": 1,
            "horizon": 1,
            "eval_episodes": 1,
            "batch_size": 64,
            "warmup_steps": 1,
        }
    )

    result = train_and_evaluate(config, tmp_path / "trial", init_checkpoint)
    saved = torch.load(tmp_path / "trial" / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert result["checkpoint"] is not None
    assert "actor_state" in saved
    assert "critic_state" in saved
    assert "format_version" in saved

def test_warm_start_gate_restores_incumbent_on_regression(tmp_path, monkeypatch) -> None:
    import torch

    from autoresearch import trainer as trainer_module
    from autoresearch.config import config_from_mapping
    from autoresearch.metrics import EvaluationMetrics
    from autoresearch.model import Actor

    init_checkpoint = tmp_path / "init.pt"
    actor = Actor(hidden_dim=32)
    torch.save(
        {
            "format_version": 1,
            "actor_state": actor.state_dict(),
            "obs_mean": np.zeros(25, dtype=np.float32),
            "obs_std": np.ones(25, dtype=np.float32),
            "goal_mean": np.zeros(3, dtype=np.float32),
            "goal_std": np.ones(3, dtype=np.float32),
        },
        init_checkpoint,
    )
    config = config_from_mapping(
        {
            "hidden_dim": 32,
            "train_episodes": 1,
            "horizon": 1,
            "eval_episodes": 1,
            "batch_size": 64,
            "warmup_steps": 1,
        }
    )
    incumbent = EvaluationMetrics(0.4, 0.06, -43.0, 5)
    regressed = EvaluationMetrics(0.0, 0.60, -50.0, 5)
    responses = iter((incumbent, regressed))
    monkeypatch.setattr(trainer_module, "evaluate_actor", lambda *args: next(responses))

    result = trainer_module.train_and_evaluate(config, tmp_path / "trial", init_checkpoint)

    assert result["warm_start_fallback"] is True
    assert result["candidate_metrics"] == regressed.to_dict()
    assert result["metrics"] == incumbent.to_dict()

def test_per_importance_sampling_weights_are_bounded_and_normalized() -> None:
    from autoresearch.replay import EpisodeReplay

    rng = np.random.default_rng(7)
    episode = [
        {
            "state": np.zeros(25, dtype=np.float32),
            "achieved_goal": np.zeros(3, dtype=np.float32),
            "action": np.zeros(4, dtype=np.float32),
            "next_state": np.zeros(25, dtype=np.float32),
            "next_achieved_goal": np.zeros(3, dtype=np.float32),
            "goal": np.ones(3, dtype=np.float32),
            "done": True,
        }
        for _ in range(10)
    ]
    replay = EpisodeReplay(100, prioritized=True, alpha=0.5, epsilon=0.1, beta=0.4)
    replay.add(episode)
    # give one transition high priority so weights are non-uniform
    replay.episodes[0][0]["_priority"] = 10.0
    batch = replay.sample(20, 0.0, rng, lambda *_: 0.0)

    assert "weight" in batch
    assert batch["weight"].shape[0] == 20
    # weights are normalized to max 1.0
    assert batch["weight"].max() <= 1.0
    assert batch["weight"].min() > 0.0


def test_per_importance_sampling_disabled_when_beta_zero() -> None:
    from autoresearch.replay import EpisodeReplay

    rng = np.random.default_rng(7)
    episode = [
        {
            "state": np.zeros(25, dtype=np.float32),
            "achieved_goal": np.zeros(3, dtype=np.float32),
            "action": np.zeros(4, dtype=np.float32),
            "next_state": np.zeros(25, dtype=np.float32),
            "next_achieved_goal": np.zeros(3, dtype=np.float32),
            "goal": np.ones(3, dtype=np.float32),
            "done": True,
        }
    ]
    replay = EpisodeReplay(4, prioritized=True, alpha=0.5, epsilon=0.1, beta=0.0)
    replay.add(episode)
    batch = replay.sample(2, 0.0, rng, lambda *_: 0.0)

    assert "weight" not in batch


def test_per_beta_annealing_ramps_to_final() -> None:
    from autoresearch.config import config_from_mapping

    cfg = config_from_mapping({"per_beta": 0.4, "per_beta_final": 1.0})
    assert cfg.per_beta == 0.4
    assert cfg.per_beta_final == 1.0

    with pytest.raises(ValueError, match="per_beta_final"):
        config_from_mapping({"per_beta_final": 1.5})


def test_replay_set_beta_updates_exponent() -> None:
    from autoresearch.replay import EpisodeReplay

    replay = EpisodeReplay(4, prioritized=True, alpha=0.5, epsilon=0.1, beta=0.4)
    replay.set_beta(1.0)
    assert replay.beta == 1.0
    replay.set_beta(0.0)
    assert replay.beta == 0.0


def test_validate_code_edit_rejects_unsafe_files_and_bad_replacement() -> None:
    from autoresearch.proposal import _validate_code_edit

    # Only trainer.py/replay.py/model.py are editable.
    with pytest.raises(ValueError, match="file must be one of"):
        _validate_code_edit({"file": "runner.py", "function": "run", "new_code": "def run(): pass"})
    # The replacement must be exactly one function whose name matches the target.
    with pytest.raises(ValueError, match="exactly one function"):
        _validate_code_edit({"file": "trainer.py", "function": "_make_reward_fn", "new_code": "x = 1"})
    with pytest.raises(ValueError, match="function name must match"):
        _validate_code_edit({"file": "trainer.py", "function": "_make_reward_fn", "new_code": "def other(): pass"})
    with pytest.raises(ValueError, match="does not parse"):
        _validate_code_edit({"file": "trainer.py", "function": "_make_reward_fn", "new_code": "def _make_reward_fn(:"})
    # A valid single-function replacement passes.
    result = _validate_code_edit(
        {"file": "trainer.py", "function": "_make_reward_fn", "new_code": "def _make_reward_fn(config, env):\n    return None"}
    )
    assert result["function"] == "_make_reward_fn"


def test_replace_function_source_swaps_one_function_verbatim() -> None:
    from autoresearch.runner import _replace_function_source

    source = (
        "def foo():\n"
        "    return 1\n\n"
        "def bar():\n"
        "    return 2\n"
    )
    new_code = "def foo():\n    return 99\n"
    result = _replace_function_source(source, "foo", new_code)
    assert "return 99" in result
    assert "return 1" not in result
    # bar is untouched
    assert "def bar():\n    return 2" in result

    with pytest.raises(ValueError, match="not found"):
        _replace_function_source(source, "nope", new_code)


def test_apply_and_restore_code_edit_snapshots_and_reverts(tmp_path, monkeypatch) -> None:
    import ast

    from autoresearch import runner as runner_module
    from autoresearch.proposal import _validate_code_edit

    # Point _code_file_path at a temp file so we don't touch the real trainer.py.
    target = tmp_path / "trainer.py"
    target.write_text("def _make_reward_fn(config, env):\n    return 0\n", encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "_code_file_path",
        lambda file: target if file == "trainer.py" else tmp_path / file,
    )
    code_edit = _validate_code_edit(
        {
            "file": "trainer.py",
            "function": "_make_reward_fn",
            "new_code": "def _make_reward_fn(config, env):\n    return 42\n",
        }
    )
    trial_dir = tmp_path / "trial"
    runner_module._apply_code_edit(code_edit, trial_dir)
    # The edit is applied in place and the original is snapshotted.
    assert "return 42" in target.read_text()
    assert (trial_dir / "trainer.py.orig").is_file()
    ast.parse(target.read_text())  # still compiles

    # Revert restores the original verbatim.
    runner_module._restore_code_edit(code_edit, trial_dir)
    assert "return 0" in target.read_text()
    assert "return 42" not in target.read_text()


def test_proposal_can_return_code_edit(monkeypatch) -> None:
    from autoresearch import proposal
    from autoresearch.config import CandidateConfig
    from autoresearch.metrics import EvaluationMetrics

    bodies: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode()

    # Strategist (text), proposer, critic, then code editor returns a code_edit.
    code_edit = {
        "file": "trainer.py",
        "function": "_make_reward_fn",
        "new_code": "def _make_reward_fn(config, env):\n    return None\n",
    }
    responses = iter(
        [
            {"choices": [{"message": {"content": "tune push_coef"}}]},
            {"choices": [{"message": {"content": json.dumps({"push_coef": 6.0})}}]},
            {"choices": [{"message": {"content": json.dumps({"push_coef": 6.0})}}]},
            {"choices": [{"message": {"content": json.dumps({"code_edit": code_edit})}}]},
        ]
    )

    def fake_urlopen(request, timeout):
        bodies.append(json.loads(request.data))
        return FakeResponse(next(responses))

    monkeypatch.setattr(proposal, "urlopen", fake_urlopen)
    result = proposal.propose_overrides(
        CandidateConfig(),
        EvaluationMetrics(0.4, 0.064, -43.0, 5),
        [],
        api_key="test-key",
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )

    assert "code_edit" in result
    assert result["code_edit"]["file"] == "trainer.py"
    assert result["code_edit"]["function"] == "_make_reward_fn"
    assert len(bodies) == 4
