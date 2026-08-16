# Soccer Simulation

A 2D continuous-physics soccer pitch simulation, built as a hobby project to practice ML/neural-network concepts — scripted rules, classical supervised tactics models, imitation learning, single-agent RL, and multi-agent RL/self-play — by building and comparing them against each other.

## Status

Stage 3 (baseline scripted AI) complete — two full scripted teams now play each other end-to-end: a per-player `Agent` interface, a per-team role-assignment layer (keeper/chaser/marker/support), and a `ScriptedAgent` implementing chase-ball, mark-nearest, pass-to-open-teammate, and basic keeper behaviors. The physics kernel picked up throw-in/corner/goal-kick restarts and `Ball.last_touch_team` tracking to support them. See [docs/stages/stage3.md](docs/stages/stage3.md) for scope/decisions (including a known, deliberately deferred gap: no tackle/dispossession mechanic yet). Stage 2's `pygame` renderer/JSONL replay recorder and Stage 1's deterministic, headless `step()` kernel are unchanged in spirit. See [ROADMAP.md](ROADMAP.md) for the full staged plan and [docs/stages/](docs/stages/) for per-stage notes and design decisions.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync         # create the .venv and install dependencies from uv.lock
uv run pytest   # run the test suite
```

## Running a match

```
uv run python scripts/run_match.py                        # live window, two scripted teams + one human-controlled player
uv run python scripts/run_match.py --record replays/demo.jsonl   # also record to disk
uv run python scripts/run_match.py --headless --record replays/demo.jsonl   # no window/keyboard, fully scripted
uv run python scripts/watch_replay.py replays/demo.jsonl   # play back a recorded replay
```

Every player except one is driven by `ScriptedAgent` (chase-ball, mark-nearest, pass-to-open-teammate, basic keeper — see [docs/stages/stage3.md](docs/stages/stage3.md)). Controls for the remaining human-controlled player (live window only): arrow keys move; hold Space to charge a kick and release to fire it in whichever direction you're currently facing (a bar shows the charge level); Tab switches which of team 0's players you're controlling (highlighted with a yellow halo). Every player shows a small white nose pointing in their facing direction. Esc or closing the window quits.

## Stack

- Python, managed with `uv`
- Gymnasium-style env API, PettingZoo for multi-agent RL
- PyTorch, stable-baselines3, scikit-learn/xgboost
- pygame/matplotlib for visualization
- Ruff + Black (pre-commit) for lint/format, pytest for tests
