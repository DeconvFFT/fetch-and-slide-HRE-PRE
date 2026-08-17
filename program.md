# autoresearch

This is an experiment to have the LLM do its own research. The LLM is the
autonomous loop: it directly edits the training file, runs it, reads the
results, and keeps or discards each change via git. There is no external
orchestrator deciding what to try — the LLM decides, edits, runs, and judges.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `aug16`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current
   `main`.
3. **Read the in-scope files**: The repo is small. Read these files for full
   context:
   - `program.md` — this charter (what you may/may not do, the metric, the loop).
   - `autoresearch/trainer.py` — the file you modify. Model architecture,
     reward function, training loop. This is the analog of karpathy's `train.py`.
   - `autoresearch/config.py` — bounded, validated training knobs (may be edited).
   - `autoresearch/replay.py`, `autoresearch/model.py` — replay buffer and
     network definitions (may be edited).
   - `autoresearch/metrics.py` — the ground-truth scoring function.
   Do NOT read or modify `runner.py`, `worker.py`, or `proposal.py` — they are
   infrastructure.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a bounded training of the FetchSlide-v4 DDPG+HER agent.
The metric is the **score** (lower is better):

```
score = 10 * (1 - success_rate) + mean_final_distance
```

Success rate is weighted 10x — getting any success is worth far more than
shaving distance. The current best is ~0% success / ~0.56 distance. The goal is
to break past that toward the reference's 80%.

**What you CAN do:**
- Modify `autoresearch/trainer.py` — the only file you edit for the algorithm.
  Everything is fair game: model architecture, reward function, training loop,
  hyperparameters, critic/actor updates, HER logic.
- Modify `autoresearch/config.py`, `autoresearch/replay.py`,
  `autoresearch/model.py` — supporting training files.

**What you CANNOT do:**
- Modify `autoresearch/runner.py`, `autoresearch/worker.py`,
  `autoresearch/proposal.py`, `autoresearch/agent_loop.py` — infrastructure.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `score_metrics` function in
  `autoresearch/metrics.py` is the ground truth metric.

**The goal is simple: get the lowest score.** Everything is fair game: change
the reward function, the architecture, the hyperparameters, the training loop.
The only constraint is that the code runs without crashing and finishes within
the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Conversely, removing
something and getting equal or better results is a great outcome — that's a
simplification win.

**The first run**: Your very first run should always be to establish the
baseline, so you will run the training script as is.

## Physics ground truth (VERIFIED — read before proposing)

These are empirically verified facts about the FetchSlide-v4 environment. They
are the ground truth you must reason from. Do NOT guess about the physics —
use these facts.

**Environment geometry:**
- Observation is 25-dim: gripper position = `obs[0:3]`, puck position =
  `obs[3:6]`, gripper velocity, puck velocity, etc.
- `achieved_goal[0:3]` = puck position. `desired_goal[0:3]` = goal position.
- Action is 4-dim: `action[0:3]` = gripper position delta (scaled to [-1,1]),
  `action[3]` = gripper open/close (1.0 = closed).
- Success threshold: puck within 0.05m of goal (`distance_threshold=0.05`).

**Critical physics facts (verified by direct env diagnostics):**
1. **Random actions NEVER move the puck.** Over 100 random steps, the puck moved
   only 7.6e-7 (essentially zero). The gripper starts ~0.1m from the puck but
   random actions never bring it into contact. So pure random exploration gives
   the agent ZERO examples of "pushing the puck".
2. **The gripper starts on the GOAL-SIDE of the puck in ~8/10 episodes.** The
   puck is 0.68m from the goal; the gripper is 0.1m from the puck but usually on
   the wrong side. To push the puck toward the goal, the gripper must get BEHIND
   the puck (opposite the goal).
3. **Pushing THROUGH the puck works.** When the gripper is positioned behind the
   puck and pushes toward the goal, the puck moves 0.377 → 0.078 (verified).
   Approaching the puck itself (not behind it) lets the gripper slide past
   without pushing.
4. **The puck is 0.68m from the goal.** The actor must push it in the right
   direction repeatedly — a hard credit-assignment problem.

**Telemetry you MUST monitor (in the progress lines and metrics.json):**
- `contact_rate`: fraction of episodes where the gripper got within 0.06m of the
  puck. This is the KEY diagnostic for whether the reach phase works.
  - contact_rate < 0.3 → the actor isn't reaching the puck. Fix: raise
    `reach_coef` (10.0 was proven to reach ~90% contact).
  - contact_rate high but success 0% → the actor contacts but doesn't push
    toward the goal. Fix: the push reward must be GATED on contact and the
    actor must push THROUGH from behind the puck.
- `mean_final_distance` (dist): if stuck at ~0.56-0.68, the puck isn't moving
  toward the goal at all. If it's higher (~0.7-0.9), the puck is being pushed
  around but not toward the goal.
- `actor_loss` / `critic_loss`: critic_loss climbing past ~1.0 = critic
  divergence (fix: Huber/smooth_l1 loss, or lower reward magnitudes).

**Proven fixes (from prior runs — do not re-discover these):**
- **Critic divergence** (critic_loss climbing): use Huber (smooth_l1) loss, not
  MSE. MSE squares large TD errors from relabeled transitions and blows up the
  critic.
- **No contact** (contact_rate < 0.3): raise `reach_coef` to ~10.0. Reach
  shaping rewards the gripper approaching the puck.
- **Contact but no goal-directed push** (dist flat): gate the push reward on
  gripper-puck contact, and ensure the actor learns to push THROUGH from behind
  the puck.
- **The reference hit 80% success with pure sparse reward + 1M steps of
  exploration.** Short runs (100-200k steps) cannot reproduce this without
  shaping or a curriculum. The scripted reach-then-push curriculum
  (`_seed_scripted_rollouts`, `scripted_rollouts`, `scripted_every`) seeds the
  replay with contact-push examples.

**Diagnostics you SHOULD run** (before proposing a reward/architecture change,
write a small script that resets the env and checks the physics):
- Sample random actions and verify whether the puck moves.
- Check which side of the puck the gripper starts on.
- Test a scripted push and measure how far the puck moves.
- Measure the actual puck-goal distance for the eval seeds.

## Output format

Once the script finishes it writes a `metrics.json` with the result. Extract
the key metrics:

```
score, success_rate, mean_final_distance
```

If the metrics are missing, the run crashed (see below).

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	score	status	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. 8.123456) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	score	status	description
a1b2c3d	8.347314	keep	baseline
b2c3d4e	6.289718	keep	increase goal_bonus to 4.0
c3d4e5f	10.563817	discard	switch reward to absolute distance
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/aug16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Read the latest telemetry (contact_rate, dist, actor/critic loss) from the
   last run's metrics.json and progress lines. Diagnose WHY the current best
   stalls using the Physics ground truth section above.
3. If you are unsure about the physics (e.g. does the puck move under this
   action?), write a small diagnostic script that resets the env and measures it
   directly. Do NOT guess about the physics.
4. Tune `autoresearch/trainer.py` with an experimental idea by directly hacking
   the code.
5. git commit
6. Run the experiment
7. Read out the results (score from metrics.json) AND the telemetry trends
   (contact_rate, dist, losses).
8. If the metrics are missing, the run crashed. Read the stack trace and attempt
   a fix. If you can't get things to work after more than a few attempts, give up.
9. Record the results in the tsv (NOTE: do not commit the results.tsv file,
   leave it untracked by git)
10. If score improved (lower), you "advance" the branch, keeping the git commit
11. If score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep. If they don't, discard. And you're advancing the branch so
that you can iterate. Judge each experiment on the TELEMETRY TRENDS, not just
the raw success number — a run that raises contact_rate or lowers dist is
making progress even if success is still 0%.

**Timeout**: A full training run takes ~15-60 minutes (2000 episodes × 50 steps
on MPS). Use `eval_every` to get live success/distance telemetry during the run
and `log_every` for contact_rate. Do NOT kill a run early unless it has clearly
diverged (critic_loss > 1.0 and climbing). The reference needed ~1M steps to
hit 80% success, so a run under ~200k steps that shows 0% success is expected —
judge it on the TELEMETRY TRENDS (contact_rate climbing, dist decreasing), not
the raw success number.

**Crashes**: If a run crashes, use your judgment: If it's something dumb and
easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea
itself is fundamentally broken, just skip it, log "crash" as the status in the
tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup),
do NOT pause to ask the human if you should continue. Do NOT ask "should I keep
going?" or "is this a good stopping point?". The human might be asleep, or gone
from a computer and expects you to continue working *indefinitely* until you
are manually stopped. You are autonomous. If you run out of ideas, think harder
— re-read the in-scope files for new angles, try combining previous near-misses,
try more radical changes to the reward function or architecture. The loop runs
until the human interrupts you, period.
