"""The deterministic physics kernel: `step(state, actions, config) -> (state, events)`.

`step()` is a pure function — same inputs always produce the same outputs, with
no hidden randomness, wall-clock reads, or reliance on dict/set iteration
order. That determinism is deliberate and worth protecting as this project
grows: it's what lets a "replay" be exactly reproducible, and it's what makes
regression tests ("this exact scenario used to produce this exact result")
possible at all. Every helper below is written to preserve it — e.g.
`_find_possessor` breaks distance ties by `player_id`, never by whatever
order players happen to be in.

## Integration scheme: semi-implicit ("symplectic") Euler

For player motion, each step:
  1. clip the requested acceleration to `player_max_accel`,
  2. update velocity using that acceleration,
  3. clip the *updated* velocity to `player_max_speed`,
  4. update position using the *updated* velocity (not the old one).

Using the just-updated velocity to move the position (step 4) — rather than
the velocity from the start of the step — is what makes this "semi-implicit"
Euler instead of plain ("explicit") Euler. It's a one-line difference that
matters: explicit Euler tends to add energy and drift outward on curved
motion, while semi-implicit Euler is much more stable for exactly this kind
of "accelerate, then move" simulation. This is the same integrator commonly
used in game physics engines for that reason.

## Friction model: why `ball_friction ** dt` and not `ball_friction` directly

We want "the ball loses friction at some fixed rate" to behave the same
whether the simulation runs at `dt = 1/60` or `dt = 1/30` — i.e. friction
should be a property of *time*, not of the step size. A quantity that decays
by a constant *fraction* per second follows `v(t) = v0 * r**t` for some rate
`r` (`config.ball_friction`). Because `(r**dt)**n == r**(dt*n)`, multiplying
velocity by `r**dt` every step for `n` steps is *exactly* equivalent to
multiplying by `r**t` once, where `t = n * dt` — not an approximation, just
an algebraic identity. That gives us a friction model that's dt-invariant and
has an exact closed form, which is exactly what `tests/test_physics_invariants.py`
checks it against.

Position integration (`position += velocity * dt`), by contrast, *is* only an
approximation of the true integral of a time-varying velocity — it converges
to the exact answer as `dt -> 0`, but isn't exact at finite `dt`. That
distinction (which parts of a simulation are exact vs. approximate at a given
step size) is a recurring numerical-methods concept worth noticing here.
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.vector import Vec2, clip_magnitude, magnitude, vec2


def step(
    state: MatchState,
    actions: dict[int, PlayerAction],
    config: SimConfig,
) -> tuple[MatchState, list[Event]]:
    """Advance the match by one `config.dt` and report what happened.

    `actions` maps `player_id -> PlayerAction`; a player with no entry is
    treated as issuing no acceleration and no kick this step.
    """
    events: list[Event] = []
    dt = config.dt

    # Possession is checked against the *start-of-step* ball/player positions,
    # so a kick this step is judged by "were you actually close enough a
    # moment ago" rather than by where everyone ends up after moving.
    possessor_id = _find_possessor(state.ball, state.players, config.possession_radius)

    ball_velocity = state.ball.velocity
    if possessor_id is not None:
        possessor_action = actions.get(possessor_id)
        if possessor_action is not None and possessor_action.kick is not None:
            # A kick *replaces* the ball's velocity (clipped to max_kick_speed)
            # rather than adding an impulse to it — see the "Kick model"
            # decision in docs/stages/stage1.md for why.
            ball_velocity = clip_magnitude(possessor_action.kick, config.max_kick_speed)

    friction_factor = config.ball_friction**dt
    ball_velocity = ball_velocity * friction_factor
    ball_position = state.ball.position + ball_velocity * dt

    ball_position, ball_velocity, ball_event = _resolve_ball_bounds(
        ball_position, ball_velocity, config
    )
    if ball_event is not None:
        events.append(ball_event)

    new_score = state.score
    if ball_event is not None and ball_event.type is EventType.GOAL:
        scoring_team = ball_event.data["team"]
        new_score = (
            state.score[0] + (scoring_team == 0),
            state.score[1] + (scoring_team == 1),
        )

    new_ball = Ball(position=ball_position, velocity=ball_velocity)

    new_players = [
        _step_player(player, actions.get(player.player_id), config) for player in state.players
    ]

    new_possessor_id = _find_possessor(new_ball, new_players, config.possession_radius)
    if new_possessor_id != possessor_id:
        events.append(Event(EventType.POSSESSION_CHANGE, {"player_id": new_possessor_id}))

    new_state = MatchState(
        time=state.time + dt,
        ball=new_ball,
        players=new_players,
        score=new_score,
    )
    return new_state, events


def _step_player(player: Player, action: PlayerAction | None, config: SimConfig) -> Player:
    """Semi-implicit Euler update for one player — see module docstring."""
    requested_accel = action.move if action is not None else vec2(0.0, 0.0)
    accel = clip_magnitude(requested_accel, config.player_max_accel)

    velocity = clip_magnitude(player.velocity + accel * config.dt, config.player_max_speed)
    position = player.position + velocity * config.dt
    position = _clamp_to_pitch(position, config)

    return Player(player.player_id, player.team, position, velocity)


def _find_possessor(ball: Ball, players: list[Player], possession_radius: float) -> int | None:
    """The nearest player within `possession_radius` of the ball, or None.

    Ties (equal distance) are broken by lowest `player_id` — an arbitrary but
    *fixed* rule, chosen only so the result never depends on dict/list
    ordering. That's what keeps `step()` deterministic.
    """
    candidates = [
        (magnitude(player.position - ball.position), player.player_id)
        for player in players
        if magnitude(player.position - ball.position) <= possession_radius
    ]
    if not candidates:
        return None
    _, nearest_id = min(candidates, key=lambda pair: (pair[0], pair[1]))
    return nearest_id


def _clamp_to_pitch(position: Vec2, config: SimConfig) -> Vec2:
    """Keep a player's position inside the pitch rectangle.

    Stage 1 doesn't model player-player collisions or goal-area-specific
    rules (e.g. keeping outfield players out of the goal area) — this is
    just "don't let anyone wander off the edge of the world."
    """
    half_length = config.pitch_length / 2
    half_width = config.pitch_width / 2
    x = min(max(position[0], -half_length), half_length)
    y = min(max(position[1], -half_width), half_width)
    return vec2(x, y)


def _resolve_ball_bounds(
    position: Vec2, velocity: Vec2, config: SimConfig
) -> tuple[Vec2, Vec2, Event | None]:
    """Detect goals/out-of-bounds and stop the ball at the boundary it crossed.

    There's no restart logic yet (throw-ins, corners, kickoffs after a goal),
    so "stop the ball at the boundary" is a placeholder until a later stage
    adds one. Both axes are always clamped into the pitch rectangle — even
    when only one axis triggers the reported event — because a single large
    step (e.g. a big `dt`) can push the ball out on *both* x and y at once
    (a corner of the pitch); leaving one axis unclamped would let the ball
    escape the rectangle it's supposed to be confined to. When both axes are
    out simultaneously, x (the goal line) takes priority for *which event* is
    reported — a known, intentionally simple tie-break rather than a
    physically exact corner case.
    """
    half_length = config.pitch_length / 2
    half_width = config.pitch_width / 2
    goal_half_width = config.goal_width / 2
    x, y = position
    vx, vy = velocity

    x_out = x > half_length or x < -half_length
    y_out = y > half_width or y < -half_width

    clamped_position = vec2(
        min(max(x, -half_length), half_length),
        min(max(y, -half_width), half_width),
    )
    # Zero the velocity component on whichever axis actually got clamped —
    # the ball shouldn't keep "pushing" against a boundary it's stopped at.
    stopped_velocity = vec2(0.0 if x_out else vx, 0.0 if y_out else vy)

    if x_out:
        if abs(y) <= goal_half_width:
            scoring_team = 0 if x > half_length else 1
            return clamped_position, stopped_velocity, Event(EventType.GOAL, {"team": scoring_team})
        return (
            clamped_position,
            stopped_velocity,
            Event(EventType.OUT_OF_BOUNDS, {"side": "goal_line"}),
        )

    if y_out:
        return (
            clamped_position,
            stopped_velocity,
            Event(EventType.OUT_OF_BOUNDS, {"side": "touchline"}),
        )

    return position, velocity, None
