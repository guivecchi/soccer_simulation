# Soccer Simulation — Learning-Oriented Side Project Roadmap

## Context

This project combines an interest in soccer with deliberate practice of ML/neural-network concepts. The idea: build a 2D physics-based soccer pitch simulation, then use it as a testbed to build and compare progressively more sophisticated approaches — scripted rules, classical supervised tactics models, imitation learning, single-agent RL, and multi-agent RL/self-play.

Confirmed decisions:
- **ML focus:** both tactical decision-making *and* player-level control, introduced in stages (tactics/scripted first, learned player behavior later).
- **Simulation fidelity:** 2D continuous physics (continuous x/y coordinates, velocity/friction, acceleration/speed limits) — not a discrete grid, not a full physics engine like Box2D.
- **Stack:** Python, Gymnasium-style env API, PyTorch, PettingZoo when multi-agent RL arrives, pygame/matplotlib for visualization.

The intent behind the ordering below: get something *visible and deterministic* early (so simulation bugs are a 5-second visual check, not a debugging session), establish a stable `Agent` interface early so every later ML approach can be swapped in without touching the simulation core, and end with a "league" harness that fairly compares every approach against each other — which is itself the payoff of the whole project.

## Staged Roadmap

**Stage 0 — Scaffolding.** Repo skeleton (`physics/`, `env/`, `agents/`, `tactics/`, `training/`, `eval/`, `viz/`), config system (Hydra/YAML or dataclasses), pytest + pre-commit wired up. Keep it minimal — expand structure as real needs appear.

**Stage 1 — Core physics/rules engine (headless).** A pure `step(state, actions) -> state, events` kernel: ball with velocity/friction, players with acceleration/speed limits, possession radius, pitch boundaries, goal/out-of-bounds detection. No rendering, no ML. Get determinism right now (seeded RNG, no wall-clock/iteration-order dependence) — retrofitting it later is painful. Decide now whether state will be numpy-array-shaped for future vectorization (matters a lot once RL needs high throughput in Stage 7+).
- *Test with:* analytical checks (stopping distance under friction, time-to-max-speed) + property-based tests (`hypothesis`) for invariants like "speed never exceeds max" and "no NaNs/tunneling at high dt."

**Stage 2 — Visualization & replay.** `pygame` renderer (pitch/ball/players/HUD) plus a replay recorder that dumps `(state, action, event)` sequences to disk — this schema becomes your training-data format later. Keep rendering strictly optional/pluggable so headless training isn't coupled to pygame.

**Stage 3 — Baseline scripted AI.** Deterministic heuristics (chase-ball, mark-nearest, pass-to-open-teammate, basic keeper). Define the `Agent` interface (`act(observation) -> action`) here — this is the abstraction every later ML approach plugs into, so get it general enough now.

**Stage 4 — Gymnasium env wrapper.** Wrap the physics core as a proper `gymnasium.Env` (observation/action spaces, reward hook even if unused yet). Decide observation design carefully (egocentric vs. global, normalization, fixed vs. variable roster) — validate with `gymnasium.utils.env_checker.check_env`. Stand up lightweight experiment tracking (W&B or MLflow) now, before there's even a model to track.

**Stage 5 — Tactics layer (classical/supervised ML).** A higher-level module choosing formation/set-pieces/substitutions, which feeds target positions to Stage 3's scripted execution. Generate a labeled dataset from scripted self-play, train with `scikit-learn`/`xgboost`. Evaluate by actually running many matches against a *fixed* baseline opponent and comparing win rates with confidence intervals — not a single anecdotal match. This is where classic DS skills (feature engineering, avoiding leakage, proper train/test splits by match not by row) map directly.

**Stage 6 — Imitation learning (behavior cloning).** Log dense (observation, action) pairs from scripted agents at scale, train a PyTorch MLP to clone that behavior, swap it in as a player via the Stage 3 `Agent` interface. Watch for **distributional shift/compounding error** — the network drifts into states the teacher never visited once it's driving the sim itself. This failure mode is the natural motivator for Stage 7.

**Stage 7 — Single-agent RL.** Freeze all but one player as scripted; train that player with PPO (`stable-baselines3`) against the fixed baseline. Start with sparse reward (goal/no-goal), iterate toward shaping deliberately, and always sanity-check learned behavior *visually* (via Stage 2's renderer) against reward curves — shaped rewards are easy to hack (e.g., hovering near the ball without advancing play). This is usually where simulation throughput starts to matter — profile `step()` speed here.

**Stage 8 — Multi-agent RL / self-play.** Convert the env to `pettingzoo.ParallelEnv` + `SuperSuit`; start with one learning team (shared policy across teammates) against a fixed opponent, before full simultaneous self-play. Use an opponent pool/checkpoint history to avoid self-play instability, and always keep a frozen external baseline (Stage 3/7) in the evaluation loop so progress isn't measured against a co-drifting target. Expect this to be the largest, most time-consuming stage — timebox it and scale the pitch/roster down if training stalls.

**Stage 9 — Evaluation "league."** A tournament runner that takes any agent conforming to the Stage 3 interface — scripted, tactics-layer, cloned, single-agent RL, self-play — and round-robins them with enough repeats per pairing to be statistically meaningful. Use `openskill`/`trueskill` for a unified rating across heterogeneous agent types, and produce a leaderboard + rating-trajectory plots + notable replay highlights. This stage is the payoff: a fair, quantified answer to "which approach actually worked best, and where."

**Stage 10 — Stretch goals (optional).** Hierarchical control (feed tactics output into the multi-agent policy as context), opponent modeling, curriculum learning for RL sample efficiency, league-style training against a diverse checkpoint pool, richer rules (offside, fatigue).

## Repository structure

```
soccersim/
  physics/    # pure state/step kernel — no ML, no rendering (Stage 1)
  viz/        # pygame renderer + replay I/O (Stage 2)
  agents/     # Agent interface + scripted/cloned/RL implementations (Stage 3,6,7,8)
  tactics/    # formation/set-piece decision models (Stage 5)
  env/        # gymnasium.Env / pettingzoo.ParallelEnv wrappers (Stage 4,8)
  training/   # BC/RL/MARL training scripts, reward configs
  eval/       # league runner, rating system, leaderboard generation (Stage 9)
configs/      # per-experiment YAML/Hydra configs
tests/        # physics invariants, env-contract tests, agent smoke tests
scripts/      # CLI entry points: run_match.py, train_bc.py, train_rl.py, run_league.py
```
Dependency direction is one-way: `physics` never imports from `agents`/`training`/`eval` — this is what keeps every later ML stage swappable without touching the simulation.

## First files to create (in order)

- `soccersim/physics/state.py` — `State`/`Action` data structures used by everything downstream
- `soccersim/physics/step.py` — the deterministic `step(state, actions) -> state, events` kernel
- `soccersim/agents/base.py` — the `Agent` interface all scripted/cloned/RL agents implement
- `soccersim/env/soccer_env.py` — the Gymnasium wrapper around the physics kernel
- `tests/test_physics_invariants.py` — analytical + property-based physics correctness tests

## Verification approach

- Each stage has its own "done" bar (see above) — mostly: unit/property tests pass, a match runs without NaNs/instability, and (from Stage 3 onward) two agents can play a full match end-to-end.
- From Stage 5 onward, "better" is only claimed based on many repeated simulated matches against a fixed opponent with confidence intervals — not single-match anecdotes.
- Stage 9's league is the end-to-end verification of the whole project: every agent built in Stages 3–8 should be runnable through the same league harness and produce a comparable ranking.
