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
shaving distance. The current best is ~20% success / ~0.3 distance. The goal is
to break past that.

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
2. Tune `autoresearch/trainer.py` with an experimental idea by directly hacking
   the code.
3. git commit
4. Run the experiment
5. Read out the results (score from metrics.json)
6. If the metrics are missing, the run crashed. Read the stack trace and attempt
   a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file,
   leave it untracked by git)
8. If score improved (lower), you "advance" the branch, keeping the git commit
9. If score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep. If they don't, discard. And you're advancing the branch so
that you can iterate.

**Timeout**: Each experiment should take ~30 seconds. If a run exceeds a few
minutes, kill it and treat it as a failure (discard and revert).

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
