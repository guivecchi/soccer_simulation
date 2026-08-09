"""Correctness tests for the Stage 1 physics kernel.

Two complementary styles, per docs/stages/stage1.md:

- Analytical tests: derive the exact expected answer by hand from the physics
  model itself and check `step()` against it in a specific scenario.
- Property-based tests (via `hypothesis`): rather than one hand-picked
  scenario, assert something that must hold for *any* input in a range (e.g.
  "speed never exceeds max_speed"), and let hypothesis search for a
  counterexample across many randomly generated inputs/edge cases.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from soccersim.config import SimConfig
from soccersim.physics.events import EventType
from soccersim.physics.reset import restart_after_goal
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import magnitude, vec2

# --- Analytical tests -------------------------------------------------------


def test_ball_friction_matches_exact_exponential_decay():
    """velocity(T) should equal v0 * friction**T exactly (up to float error),
    regardless of how many steps of size dt we split T into.

    This is the "friction composes exactly under repeated stepping" property
    explained in physics/step.py's module docstring: it's not a numerical
    approximation, so we can assert near-exact equality rather than a loose
    tolerance.
    """
    config = SimConfig(dt=1.0 / 60.0, ball_friction=0.98, players_per_team=0)
    v0 = 10.0
    state = MatchState(
        time=0.0,
        ball=Ball(position=vec2(0.0, 0.0), velocity=vec2(v0, 0.0)),
        players=[],
        score=(0, 0),
    )

    total_time = 2.0  # seconds
    n_steps = round(total_time / config.dt)
    for _ in range(n_steps):
        state, _ = step(state, {}, config)

    expected_speed = v0 * config.ball_friction**total_time
    assert math.isclose(magnitude(state.ball.velocity), expected_speed, rel_tol=1e-9)


def test_ball_stopping_distance_matches_closed_form_integral():
    """Distance traveled should match the closed-form integral of v0 * r**t.

    Unlike velocity decay, position integration is only approximate at finite
    dt (see module docstring on step.py) so this uses a small dt and a
    generous-but-meaningful tolerance rather than exact equality.
    """
    config = SimConfig(dt=1.0 / 240.0, ball_friction=0.9, players_per_team=0)
    v0 = 10.0
    state = MatchState(
        time=0.0,
        ball=Ball(position=vec2(0.0, 0.0), velocity=vec2(v0, 0.0)),
        players=[],
        score=(0, 0),
    )

    total_time = 3.0
    n_steps = round(total_time / config.dt)
    for _ in range(n_steps):
        state, _ = step(state, {}, config)

    r = config.ball_friction
    expected_distance = v0 * (1 - r**total_time) / (-math.log(r))
    assert math.isclose(state.ball.position[0], expected_distance, rel_tol=1e-3)


def test_braking_uses_the_higher_brake_accel_cap_not_max_accel():
    """Decelerating/reversing should be capped at `player_brake_accel`
    (higher), while accelerating from a standstill uses `player_max_accel`
    (lower) — even for the identical, oversized requested acceleration.
    """
    config = SimConfig(
        dt=1.0 / 60.0, player_max_accel=6.0, player_brake_accel=14.0, players_per_team=0
    )
    huge_request = {0: PlayerAction(move=vec2(-100.0, 0.0))}
    ball = Ball(vec2(50.0, 50.0), vec2(0.0, 0.0))

    moving_player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(5.0, 0.0))
    moving_state = MatchState(time=0.0, ball=ball, players=[moving_player], score=(0, 0))
    braked_state, _ = step(moving_state, huge_request, config)

    resting_player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    resting_state = MatchState(time=0.0, ball=ball, players=[resting_player], score=(0, 0))
    accelerated_state, _ = step(resting_state, huge_request, config)

    expected_braked_vx = 5.0 - config.player_brake_accel * config.dt
    expected_accelerated_vx = 0.0 - config.player_max_accel * config.dt
    assert math.isclose(braked_state.players[0].velocity[0], expected_braked_vx, rel_tol=1e-9)
    assert math.isclose(
        accelerated_state.players[0].velocity[0], expected_accelerated_vx, rel_tol=1e-9
    )


def test_player_facing_tracks_movement_input_and_persists_when_idle():
    """`facing` should follow the last *nonzero* requested direction, and
    stay put (not reset or drift) once the player stops pushing any key.
    """
    config = SimConfig(dt=1.0 / 60.0, players_per_team=0)
    ball = Ball(vec2(50.0, 50.0), vec2(0.0, 0.0))
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    state, _ = step(state, {0: PlayerAction(move=vec2(0.0, 5.0))}, config)
    assert list(state.players[0].facing) == [0.0, 1.0]

    state, _ = step(state, {0: PlayerAction(move=vec2(0.0, 0.0))}, config)
    assert list(state.players[0].facing) == [0.0, 1.0]


def test_player_reaches_and_stays_clamped_at_max_speed():
    """Under constant full-throttle acceleration, speed should hit exactly
    max_speed (via clamping) and stay there — not overshoot, not approach it
    asymptotically.
    """
    config = SimConfig(
        dt=1.0 / 60.0, player_max_accel=6.0, player_max_speed=8.0, players_per_team=0
    )
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(
        time=0.0, ball=Ball(vec2(50.0, 50.0), vec2(0.0, 0.0)), players=[player], score=(0, 0)
    )

    full_throttle = {0: PlayerAction(move=vec2(config.player_max_accel * 10, 0.0))}

    n_steps = math.ceil(config.player_max_speed / config.player_max_accel / config.dt) + 5
    for _ in range(n_steps):
        state, _ = step(state, full_throttle, config)

    assert math.isclose(magnitude(state.players[0].velocity), config.player_max_speed, rel_tol=1e-9)


# --- Event tests -------------------------------------------------------------


def test_ball_crossing_goal_mouth_scores_and_updates_score():
    config = SimConfig(
        dt=1.0,
        pitch_length=100.0,
        pitch_width=60.0,
        goal_width=10.0,
        ball_friction=1.0,
        players_per_team=0,
    )
    half_length = config.pitch_length / 2
    ball = Ball(position=vec2(half_length - 1.0, 0.0), velocity=vec2(5.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[], score=(0, 0))

    state, events = step(state, {}, config)

    goal_events = [e for e in events if e.type is EventType.GOAL]
    assert len(goal_events) == 1
    assert goal_events[0].data["team"] == 0
    assert state.score == (1, 0)
    # Ball should be stopped at the boundary, not left drifting past it.
    assert state.ball.velocity[0] == 0.0


def test_restart_after_goal_composes_with_step_to_relocate_the_ball():
    """The intended usage pattern (see physics/reset.py): callers that want a
    goal to actually restart play call `restart_after_goal()` themselves,
    right after `step()` reports a GOAL — `step()` alone still just stops
    the ball at the line, as asserted by the test above.
    """
    config = SimConfig(
        dt=1.0,
        pitch_length=100.0,
        pitch_width=60.0,
        goal_width=10.0,
        ball_friction=1.0,
        players_per_team=0,
    )
    half_length = config.pitch_length / 2
    ball = Ball(position=vec2(half_length - 1.0, 0.0), velocity=vec2(5.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[], score=(0, 0))

    state, events = step(state, {}, config)
    assert [e.type for e in events] == [EventType.GOAL]

    state = restart_after_goal(state, config)

    assert list(state.ball.position) == [0.0, 0.0]
    assert state.score == (1, 0)  # the goal itself is still on the scoreboard


def test_ball_crossing_touchline_is_out_of_bounds_not_a_goal():
    config = SimConfig(
        dt=1.0,
        pitch_length=100.0,
        pitch_width=60.0,
        goal_width=10.0,
        ball_friction=1.0,
        players_per_team=0,
    )
    half_width = config.pitch_width / 2
    ball = Ball(position=vec2(0.0, half_width - 1.0), velocity=vec2(0.0, 5.0))
    state = MatchState(time=0.0, ball=ball, players=[], score=(0, 0))

    state, events = step(state, {}, config)

    assert [e.type for e in events] == [EventType.OUT_OF_BOUNDS]
    assert state.score == (0, 0)
    assert state.ball.velocity[1] == 0.0


def test_possession_change_event_fires_when_nearest_player_changes():
    """Possession is judged against the *start* of the step (see step.py's
    docstring on `possessor_id`), so to observe a change we need the ball to
    move from "out of anyone's range" to "in range" *during* one step call —
    not just start pre-possessed, which wouldn't be a change at all.
    """
    config = SimConfig(dt=1.0, possession_radius=2.0, ball_friction=1.0, players_per_team=0)
    stationary_player = Player(
        player_id=0, team=0, position=vec2(5.0, 0.0), velocity=vec2(0.0, 0.0)
    )
    # Ball starts 5 units away (out of the radius-2 possession range) and is
    # moving straight at the player fast enough to land exactly on them
    # after this one step (dt=1, friction=1 => still-unit velocity retained).
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(5.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[stationary_player], score=(0, 0))

    _, events = step(state, {}, config)

    possession_events = [e for e in events if e.type is EventType.POSSESSION_CHANGE]
    assert len(possession_events) == 1
    assert possession_events[0].data["player_id"] == 0


def test_kick_from_non_possessor_is_ignored():
    config = SimConfig(
        dt=1.0, possession_radius=1.0, ball_friction=1.0, max_kick_speed=50.0, players_per_team=0
    )
    close_player = Player(player_id=0, team=0, position=vec2(0.5, 0.0), velocity=vec2(0.0, 0.0))
    far_player = Player(player_id=1, team=1, position=vec2(20.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[close_player, far_player], score=(0, 0))

    # far_player is not the possessor (outside possession_radius) — their
    # kick action must have no effect on the ball.
    actions = {1: PlayerAction(move=vec2(0.0, 0.0), kick=vec2(30.0, 0.0))}
    new_state, _ = step(state, actions, config)

    assert magnitude(new_state.ball.velocity) == 0.0


# --- Property-based tests ----------------------------------------------------


finite_floats = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


@given(
    accel_x=finite_floats,
    accel_y=finite_floats,
    dt=st.floats(min_value=1e-4, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_player_speed_never_exceeds_max_speed(accel_x, accel_y, dt):
    """No matter how large the requested acceleration or dt, speed stays clamped.

    `dt` up to 2.0 stands in for "unrealistically large timestep" — the kind
    of input a naive integrator could blow up on — per the roadmap's "no
    NaNs/tunneling at high dt" invariant.
    """
    config = SimConfig(dt=dt, player_max_accel=6.0, player_max_speed=8.0, players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(
        time=0.0, ball=Ball(vec2(100.0, 100.0), vec2(0.0, 0.0)), players=[player], score=(0, 0)
    )

    actions = {0: PlayerAction(move=vec2(accel_x, accel_y))}
    new_state, _ = step(state, actions, config)

    assert magnitude(new_state.players[0].velocity) <= config.player_max_speed + 1e-9
    assert np.all(np.isfinite(new_state.players[0].position))
    assert np.all(np.isfinite(new_state.players[0].velocity))


@given(
    ball_vx=finite_floats,
    ball_vy=finite_floats,
    dt=st.floats(min_value=1e-4, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_ball_state_stays_finite_and_in_bounds_at_high_dt(ball_vx, ball_vy, dt):
    """No NaN/Inf ball state, and the ball never ends up outside the pitch
    rectangle it was just clamped to — even for a single, possibly very large,
    step.
    """
    config = SimConfig(dt=dt, players_per_team=0)
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(ball_vx, ball_vy))
    state = MatchState(time=0.0, ball=ball, players=[], score=(0, 0))

    new_state, _ = step(state, {}, config)

    assert np.all(np.isfinite(new_state.ball.position))
    assert np.all(np.isfinite(new_state.ball.velocity))
    half_length = config.pitch_length / 2
    half_width = config.pitch_width / 2
    assert -half_length - 1e-9 <= new_state.ball.position[0] <= half_length + 1e-9
    assert -half_width - 1e-9 <= new_state.ball.position[1] <= half_width + 1e-9
