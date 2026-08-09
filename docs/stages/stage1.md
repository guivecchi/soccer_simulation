# Stage 1 — Core physics/rules engine (headless)

## Scope

A pure, deterministic `step(state, actions, config) -> (state, events)` kernel: ball movement under friction, player movement under acceleration/speed limits, kicking (a player can impart velocity to the ball while in possession range), pitch-boundary and goal detection. No rendering, no ML, no game-restart logic (throw-ins/corners/kickoffs after a goal) yet — those are gameplay/behavior concerns for later stages.

## Decisions

- **State representation:** plain Python dataclasses (`Ball`, `Player`, `PlayerAction`, `MatchState`), not numpy-array-of-everything. Chosen for readability while the physics rules are still being designed and debugged; positions/velocities within each dataclass are small numpy vectors (`Vec2`, shape `(2,)`) purely for convenient vector math. If Stage 7 (single-agent RL) needs higher throughput, the state can be converted to a fully array-based batched representation behind the same `step()` call signature — this is a pure function of `(state, actions, config)`, so nothing outside `physics/` should need to change.
- **Roster size:** configurable via `SimConfig.players_per_team`, defaulting to 5 (5v5) rather than 11v11, to keep early debugging/visualization simpler. Scaling up later is a one-line config change.
- **Coordinate system:** origin at the pitch center. `+x` runs from team 0's goal toward team 1's goal; `+y` runs across the pitch. Team 0 defends the goal at `x = -pitch_length/2` and attacks `+x`; team 1 is the mirror image. The goal mouth spans `[-goal_width/2, +goal_width/2]` in `y` at each goal line. This keeps the pitch symmetric around `(0, 0)`, which simplifies mirroring team 1's logic off team 0's later.
- **Friction model:** ball velocity decays as `velocity *= ball_friction ** dt` each step, where `ball_friction` is the fraction of speed retained *per second*. This is an exponential-decay approximation of rolling friction (real kinetic friction is closer to a constant deceleration, not exponential) — chosen because it composes *exactly* under repeated stepping regardless of `dt` (see `physics/step.py` module docstring), which makes it easy to write exact analytical tests and keeps behavior consistent if `dt` ever changes.
- **Kick model:** a kick instantaneously *replaces* the ball's velocity (clipped to `max_kick_speed`), rather than adding an impulse on top of the ball's current velocity. This ignores ball mass / impact-angle physics, but is simple, intuitive, and enough to let a scripted agent (Stage 3) pass and shoot.
- **Possession:** the *possessor* is the nearest player within `possession_radius` of the ball (ties broken by lowest `player_id`, for determinism). Only the possessor's kick action (if any) affects the ball each step; any other player's kick action is ignored that step.
- **Boundaries:** both players and the ball are clamped to stay within the pitch rectangle. A ball crossing a goal line inside the goal mouth is a `GOAL` event (score updates immediately); crossing anywhere else out of bounds is an `OUT_OF_BOUNDS` event. In both cases the ball is stopped at the boundary (position clamped, the outward velocity component zeroed) since there's no restart/set-piece logic yet to relocate it.
- **No player-player collision physics.** Players can currently overlap; this is a known simplification, revisit only if it becomes a visible problem (candidate for Stage 10 "richer rules").

## Done criteria

- `step()` is a pure function: same `(state, actions, config)` in, same `(state, events)` out, no hidden randomness or global state.
- Analytical tests pass: ball friction decay matches the exact closed-form exponential solution; a player under constant max acceleration reaches exactly `max_speed` (then stays clamped there).
- Property-based tests (via `hypothesis`) pass: player speed never exceeds `max_speed`; no NaN/Inf values appear in ball or player state across a wide range of random actions and `dt` values, including unusually large `dt`.
- Goal / out-of-bounds / possession-change events fire correctly in constructed scenarios.
- `uv run pytest`, `uv run ruff check .`, and `uv run black --check .` all pass.
