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

## Ball carrying: possession vs. carrying, and trap-vs-bounce contact

Two related but distinct ideas share the word "possession" in everyday
football talk, and it's worth keeping them separate here:

- **`possessor_id`** (`_find_possessor`): purely geometric, recomputed fresh
  every step from *start-of-step* positions — "who's the nearest player
  within `possession_radius` right now." This is what decides whose `kick`
  action (if any) gets to affect the ball this step; it always exists,
  independent of whether the ball is actually under control.
- **`carrier_id`** (`state.ball.carrier_id`): genuine persistent state — "who
  is currently *dribbling* the ball." Unlike `possessor_id`, this can't be
  recomputed from a single instant; whether the ball is attached depends on
  what happened the moment contact was made, in a *previous* step.

When a free ball (no current carrier) comes within `possession_radius` of its
nearest player, `_is_receiving` decides which of two things happens:

- **Trap**: the player's requested movement is roughly aligned with the
  ball's incoming velocity (they're moving *with* it, cushioning its
  momentum — like taking a pass by stepping back with it) — the ball
  attaches (`carrier_id` is set) and, from then on, its velocity is eased
  toward the carrier's own velocity each step (`_dribble_velocity`), so it
  rides along with them until they kick it away.
- **Bounce**: no such alignment (the player is standing in the way, not
  moving to receive) — the ball deflects off them like hitting a fixed,
  immovable obstacle (`_bounce_velocity`): the velocity component *along*
  the contact normal is reversed and damped by `bounce_restitution`, while
  the tangential component passes through unchanged, so an off-center graze
  deflects sideways rather than bouncing straight back.

A kick always takes priority over both: any player identified as
`possessor_id` who issues a `kick` this step overrides carrying/bouncing
entirely (one-touch shots and passes don't require having "trapped" the
ball first), and releases the ball from any existing carrier.
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.vector import Vec2, clip_magnitude, dot, magnitude, normalize, vec2


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

    # Possession/contact is *decided* against the start-of-step ball/player
    # positions, so a kick (or a new trap/bounce) this step is judged by
    # "were you actually close enough a moment ago" rather than by where
    # everyone ends up after moving.
    possessor_id = _find_possessor(state.ball, state.players, config.possession_radius)
    possessor = _find_player(state.players, possessor_id)
    possessor_action = actions.get(possessor_id) if possessor_id is not None else None
    carrier_id = state.ball.carrier_id

    # Players move independently of the ball, so we can advance them first
    # and use each one's *just-updated* velocity as the dribble target below
    # — the ball then rides along with where its carrier actually ends up
    # this step, instead of trailing one step behind a carrier who's still
    # accelerating.
    new_players = [
        _step_player(player, actions.get(player.player_id), config) for player in state.players
    ]

    if possessor is not None and possessor_action is not None and possessor_action.kick is not None:
        # A kick *replaces* the ball's velocity (clipped to max_kick_speed)
        # rather than adding an impulse to it — see the "Kick model" decision
        # in docs/stages/stage1.md for why. It always wins over carrying or
        # bouncing: a one-touch shot doesn't require having trapped the ball
        # first, and it releases the ball from whoever was carrying it.
        ball_velocity = clip_magnitude(possessor_action.kick, config.max_kick_speed)
        carrier_id = None
    elif carrier_id is not None:
        # Already being dribbled — keep riding along with the carrier. See
        # the module docstring's "Ball carrying" section.
        new_carrier = _find_player(new_players, carrier_id)
        ball_velocity = _dribble_velocity(state.ball.velocity, new_carrier.velocity, config)
    elif possessor is not None and _is_approaching(
        state.ball.velocity, state.ball.position, possessor.position
    ):
        # A free ball is on a collision course with its nearest player:
        # decide whether that's a deliberate trap or just a bounce off their
        # body. A ball merely passing within possession_radius *without*
        # closing on them (e.g. it's already bounced away, or was never
        # headed their way) isn't a contact to resolve at all — it just
        # coasts, same as the `else` branch below.
        if _is_receiving(state.ball.velocity, possessor_action, config.receive_alignment_threshold):
            carrier_id = possessor_id
            new_carrier = _find_player(new_players, possessor_id)
            ball_velocity = _dribble_velocity(state.ball.velocity, new_carrier.velocity, config)
            events.append(Event(EventType.BALL_TRAPPED, {"player_id": possessor_id}))
        else:
            ball_velocity = _bounce_velocity(
                state.ball.velocity,
                state.ball.position,
                possessor.position,
                config.bounce_restitution,
            )
            events.append(Event(EventType.BALL_BOUNCED, {"player_id": possessor_id}))
    else:
        ball_velocity = state.ball.velocity

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

    new_ball = Ball(position=ball_position, velocity=ball_velocity, carrier_id=carrier_id)

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
    """Semi-implicit Euler update for one player — see module docstring.

    Acceleration is capped at one of two limits depending on whether the
    requested direction opposes the player's current velocity:
    `player_brake_accel` (higher) when braking/reversing, `player_max_accel`
    (lower) otherwise — including accelerating from a standstill. This
    mirrors real movement (and most sports games' "game feel"): digging in
    to stop or cut back is quicker than building up speed from rest.
    """
    requested_accel = action.move if action is not None else vec2(0.0, 0.0)

    is_braking = magnitude(player.velocity) > 0.0 and dot(requested_accel, player.velocity) < 0.0
    accel_limit = config.player_brake_accel if is_braking else config.player_max_accel
    accel = clip_magnitude(requested_accel, accel_limit)

    velocity = clip_magnitude(player.velocity + accel * config.dt, config.player_max_speed)
    position = player.position + velocity * config.dt
    position = _clamp_to_pitch(position, config)

    # Facing tracks the last *nonzero* requested direction, so a player who
    # stops moving keeps facing the way they were last heading rather than
    # snapping to some default — see the field's docstring in state.py.
    facing = normalize(requested_accel) if magnitude(requested_accel) > 0.0 else player.facing

    return Player(player.player_id, player.team, position, velocity, facing)


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


def _find_player(players: list[Player], player_id: int | None) -> Player | None:
    """Look up a player by id, or None if `player_id` is None (a convenience
    so callers don't need a separate branch for "no such player")."""
    if player_id is None:
        return None
    return next((p for p in players if p.player_id == player_id), None)


def _is_approaching(ball_velocity: Vec2, ball_position: Vec2, player_position: Vec2) -> bool:
    """Is the ball actually on a collision course with this player?

    Without this check, a ball that's already bounced away (or was simply
    passing by without ever heading toward them) would keep re-triggering a
    trap/bounce decision on every step it happens to stay within
    `possession_radius`, instead of just coasting past. A ball with
    (near-)zero speed counts as "approaching" unconditionally — there's no
    trajectory to check, and a dead ball should still be collectible.
    """
    if magnitude(ball_velocity) < 1e-6:
        return True
    return dot(ball_velocity, player_position - ball_position) > 0.0


def _is_receiving(ball_velocity: Vec2, action: PlayerAction | None, threshold: float) -> bool:
    """Decide whether contact with the ball is a deliberate trap rather than a bounce.

    A player whose requested movement points roughly the *same* way the ball
    is already traveling is "giving" with it — cushioning its momentum, like
    stepping back while receiving a pass — rather than presenting a rigid
    obstacle it just cannons off. We compare that requested direction to the
    ball's incoming velocity direction via cosine similarity (their dot
    product, since both are unit vectors): 1.0 is dead-on the same direction,
    0.0 is perpendicular, negative is moving to meet it head-on.

    A ball that's already nearly stopped (e.g. sitting at kickoff) has no
    real momentum to cushion, so it's always treated as receivable — anyone
    nearby can just walk up and gather it, no particular movement required.
    """
    if magnitude(ball_velocity) < 1e-6:
        return True
    if action is None or magnitude(action.move) < 1e-9:
        return False
    return dot(normalize(action.move), normalize(ball_velocity)) >= threshold


def _dribble_velocity(ball_velocity: Vec2, carrier_velocity: Vec2, config: SimConfig) -> Vec2:
    """Ease a carried ball's velocity toward its carrier's velocity.

    Modeled as a capped acceleration toward the target (the same
    accelerate-then-clip pattern as `_step_player`) rather than an instant
    snap: if the carrier suddenly changes direction, the ball takes a brief
    moment to "catch up" instead of teleporting to match. `dribble_accel` is
    deliberately much larger than a player's own `player_max_accel`, so in
    practice this catch-up is fast enough to look like the ball is glued to
    their feet, while staying a continuous function of state rather than a
    discontinuous jump.
    """
    velocity_error = carrier_velocity - ball_velocity
    accel = clip_magnitude(velocity_error / config.dt, config.dribble_accel)
    return ball_velocity + accel * config.dt


def _bounce_velocity(
    ball_velocity: Vec2, ball_position: Vec2, player_position: Vec2, restitution: float
) -> Vec2:
    """Reflect the ball off a player's body, treated as a fixed, immovable obstacle.

    This is the standard reflection-with-restitution formula for a collision
    against a surface of (effectively) infinite mass: only the velocity
    component *along the contact normal* (the line from the player to the
    ball) is reversed and damped by `restitution`; the tangential component
    (along the "surface") passes through unchanged. That's what makes an
    off-center graze deflect sideways instead of bouncing straight back —
    the same logic as a ball glancing off a wall at an angle.
    """
    normal = normalize(ball_position - player_position)
    if magnitude(normal) == 0.0:
        # Degenerate contact (ball and player at the same point) — no
        # well-defined normal to reflect off, so just damp the velocity.
        return ball_velocity * restitution

    normal_speed = dot(ball_velocity, normal)
    if normal_speed >= 0.0:
        # Already moving away from the player along the normal — nothing to
        # resolve (e.g. this player isn't the one the ball is approaching).
        return ball_velocity

    return ball_velocity - (1.0 + restitution) * normal_speed * normal


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
