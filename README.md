# Soccer Simulation

A 2D continuous-physics soccer pitch simulation, built as a hobby project to practice ML/neural-network concepts — scripted rules, classical supervised tactics models, imitation learning, single-agent RL, and multi-agent RL/self-play — by building and comparing them against each other.

## Status

Stage 2 (visualization & replay) complete — a `pygame` renderer draws live matches or recorded replays, and a JSON-Lines replay recorder/loader persists `(state, action, event)` sequences to disk. Stage 1's deterministic, headless `step()` kernel (ball/player movement, kicking, possession, goal/out-of-bounds detection) is unchanged and still has no rendering dependency. No agents yet. See [ROADMAP.md](ROADMAP.md) for the full staged plan and [docs/stages/](docs/stages/) for per-stage notes and design decisions.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync         # create the .venv and install dependencies from uv.lock
uv run pytest   # run the test suite
```

## Running a match

```
uv run python scripts/run_match.py                        # live window; arrow keys + space control player 0
uv run python scripts/run_match.py --record replays/demo.jsonl   # also record to disk
uv run python scripts/watch_replay.py replays/demo.jsonl   # play back a recorded replay
```

## Stack

- Python, managed with `uv`
- Gymnasium-style env API, PettingZoo for multi-agent RL
- PyTorch, stable-baselines3, scikit-learn/xgboost
- pygame/matplotlib for visualization
- Ruff + Black (pre-commit) for lint/format, pytest for tests
