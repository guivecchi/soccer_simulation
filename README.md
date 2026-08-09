# Soccer Simulation

A 2D continuous-physics soccer pitch simulation, built as a hobby project to practice ML/neural-network concepts — scripted rules, classical supervised tactics models, imitation learning, single-agent RL, and multi-agent RL/self-play — by building and comparing them against each other.

## Status

Stage 2 (visualization & replay) complete — a `pygame` renderer draws live matches or recorded replays, and a JSON-Lines replay recorder/loader persists `(state, action, event)` sequences to disk. Stage 1's deterministic, headless `step()` kernel (ball/player movement, kicking, dribbling/trapping/bouncing off players, goal restarts, goal/out-of-bounds detection) has since picked up a round of pre-Stage-3 physics/rules refinements — see the addenda in [docs/stages/stage1.md](docs/stages/stage1.md) — and still has no rendering dependency. No agents yet. See [ROADMAP.md](ROADMAP.md) for the full staged plan and [docs/stages/](docs/stages/) for per-stage notes and design decisions.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync         # create the .venv and install dependencies from uv.lock
uv run pytest   # run the test suite
```

## Running a match

```
uv run python scripts/run_match.py                        # live window
uv run python scripts/run_match.py --record replays/demo.jsonl   # also record to disk
uv run python scripts/watch_replay.py replays/demo.jsonl   # play back a recorded replay
```

Controls (live window): arrow keys move; hold Space to charge a kick and release to fire it in whichever direction you're currently facing (a bar shows the charge level); Tab switches which player you're controlling; Esc or closing the window quits.

## Stack

- Python, managed with `uv`
- Gymnasium-style env API, PettingZoo for multi-agent RL
- PyTorch, stable-baselines3, scikit-learn/xgboost
- pygame/matplotlib for visualization
- Ruff + Black (pre-commit) for lint/format, pytest for tests
