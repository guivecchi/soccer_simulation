"""Tests for out-of-bounds restart placement (throw-ins, corners, goal-kicks).

See the "Throw-in / corner / goal-kick placement" section of
physics/reset.py's module docstring, and docs/stages/stage3.md's decisions,
for why attribution depends on `Ball.last_touch_team`.
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.reset import (
    CORNER_INSET_M,
    GOAL_KICK_DEPTH_M,
    THROW_IN_INSET_M,
    restart_after_out_of_bounds,
)
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import vec2


def _state_with_ball(config: SimConfig, ball: Ball) -> MatchState:
    players = [Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))]
    return MatchState(time=10.0, ball=ball, players=players, score=(1, 0))


def test_touchline_restart_places_ball_inset_from_the_exit_point_with_no_carrier_or_touch():
    """Inset (not placed exactly on the line) — see the module docstring's
    note on why: a ball sitting right on the boundary can be dragged back
    across it by the very next dribble step, causing an infinite dead-ball
    loop at the sideline (caught by running a full scripted match)."""
    config = SimConfig()
    half_width = config.pitch_width / 2
    ball = Ball(position=vec2(12.0, half_width), velocity=vec2(0.0, 0.0), last_touch_team=0)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "touchline"})

    restarted = restart_after_out_of_bounds(state, event, config)

    assert list(restarted.ball.position) == [12.0, half_width - THROW_IN_INSET_M]
    assert list(restarted.ball.velocity) == [0.0, 0.0]
    assert restarted.ball.carrier_id is None
    assert restarted.ball.last_touch_team is None
    # Only the ball resets — players, time, and score carry over unchanged.
    assert restarted.time == state.time
    assert restarted.score == state.score
    assert restarted.players is state.players


def test_touchline_restart_insets_toward_the_pitch_on_the_negative_side_too():
    config = SimConfig()
    half_width = config.pitch_width / 2
    ball = Ball(position=vec2(-8.0, -half_width), velocity=vec2(0.0, 0.0), last_touch_team=1)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "touchline"})

    restarted = restart_after_out_of_bounds(state, event, config)

    assert list(restarted.ball.position) == [-8.0, -half_width + THROW_IN_INSET_M]


def test_goal_line_exit_touched_last_by_attacking_team_is_a_goal_kick():
    """Team 0 attacks the +x line; if team 0 touched it last on the way out,
    that's an overhit shot/pass — team 1 (the defence) gets a goal-kick.
    """
    config = SimConfig()
    half_length = config.pitch_length / 2
    ball = Ball(position=vec2(half_length, 3.0), velocity=vec2(0.0, 0.0), last_touch_team=0)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "goal_line"})

    restarted = restart_after_out_of_bounds(state, event, config)

    expected_x = half_length - GOAL_KICK_DEPTH_M
    assert restarted.ball.position[0] == expected_x
    assert restarted.ball.position[1] == 0.0


def test_goal_line_exit_touched_last_by_defending_team_is_a_corner():
    """If team 1 (defending the +x line) touched it last — a clearance or
    deflection — team 0 (the attack) gets a corner instead."""
    config = SimConfig()
    half_length = config.pitch_length / 2
    half_width = config.pitch_width / 2
    ball = Ball(position=vec2(half_length, 5.0), velocity=vec2(0.0, 0.0), last_touch_team=1)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "goal_line"})

    restarted = restart_after_out_of_bounds(state, event, config)

    assert restarted.ball.position[0] == half_length - CORNER_INSET_M
    assert restarted.ball.position[1] == half_width - CORNER_INSET_M  # exit y was positive


def test_goal_line_exit_with_no_recorded_touch_defaults_to_goal_kick():
    config = SimConfig()
    half_length = config.pitch_length / 2
    ball = Ball(position=vec2(-half_length, -2.0), velocity=vec2(0.0, 0.0), last_touch_team=None)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "goal_line"})

    restarted = restart_after_out_of_bounds(state, event, config)

    expected_x = -half_length + GOAL_KICK_DEPTH_M
    assert restarted.ball.position[0] == expected_x
    assert restarted.ball.position[1] == 0.0
