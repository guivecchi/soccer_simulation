# Soccer Simulation

A 2D continuous-physics soccer pitch simulation, built as a hobby project to practice ML/neural-network concepts — scripted rules, classical supervised tactics models, imitation learning, single-agent RL, and multi-agent RL/self-play — by building and comparing them against each other.

## Status

Stage 1 (core physics/rules engine) complete — a deterministic, headless `step()` kernel simulates ball/player movement, kicking, possession, and goal/out-of-bounds detection. No rendering or agents yet. See [ROADMAP.md](ROADMAP.md) for the full staged plan and [docs/stages/](docs/stages/) for per-stage notes and design decisions.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync         # create the .venv and install dependencies from uv.lock
uv run pytest   # run the test suite
```

## Stack

- Python, managed with `uv`
- Gymnasium-style env API, PettingZoo for multi-agent RL
- PyTorch, stable-baselines3, scikit-learn/xgboost
- pygame/matplotlib for visualization
- Ruff + Black (pre-commit) for lint/format, pytest for tests
