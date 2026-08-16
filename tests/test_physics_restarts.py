"""Tests for out-of-bounds restart placement (throw-ins, corners, goal-kicks)
and the restart-ownership entitlement that keeps the wrong team from
retaking a restart they just conceded.

See the "Throw-in / corner / goal-kick placement" section of
physics/reset.py's module docstring, and docs/stages/stage3.md's decisions,
for why attribution depends on `Ball.last_touch_team`, and
`Ball.restart_owner_team`'s docstring in physics/state.py for how the
entitlement itself is enforced (via `physics/step.py::find_possessor`).
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
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.step import find_possessor, step
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
    # Team 0 kicked it out, so team 1 -- the other side -- gets the throw-in.
    assert restarted.ball.restart_owner_team == 1
    # Only the ball resets — players, time, and score carry over unchanged.
    assert restarted.time == state.time
    assert restarted.score == state.score
    assert restarted.players is state.players


def test_touchline_restart_with_no_recorded_touch_leaves_the_restart_unowned():
    config = SimConfig()
    half_width = config.pitch_width / 2
    ball = Ball(position=vec2(12.0, half_width), velocity=vec2(0.0, 0.0), last_touch_team=None)
    state = _state_with_ball(config, ball)
    event = Event(EventType.OUT_OF_BOUNDS, {"side": "touchline"})

    restarted = restart_after_out_of_bounds(state, event, config)

    assert restarted.ball.restart_owner_team is None


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
    assert restarted.ball.restart_owner_team == 1  # the defending team takes the goal-kick


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
    assert restarted.ball.restart_owner_team == 0  # the attacking team takes the corner


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


# --- Restart entitlement enforcement (Ball.restart_owner_team) --------------
#
# Regression tests for a real bug: `restart_after_out_of_bounds` correctly
# decided which team *should* take a restart, but nothing stopped the other
# team — including the very team that just kicked the ball out — from
# walking up and touching it first. Reported directly as "the last team to
# touch the ball is able to kick it," confirmed by tracing real match
# replays where the offending team regained the ball on the very next step.


def test_find_possessor_ignores_the_non_owning_team_even_when_closer():
    config = SimConfig(possession_radius=2.0)
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0), restart_owner_team=1)
    # Team 0's player is right on top of the ball; team 1's is farther away
    # but is the one actually entitled to it.
    team_0_player = Player(player_id=0, team=0, position=vec2(0.1, 0.0), velocity=vec2(0.0, 0.0))
    team_1_player = Player(player_id=1, team=1, position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    possessor_id = find_possessor(ball, [team_0_player, team_1_player], config.possession_radius)

    assert possessor_id == 1


def test_find_possessor_returns_none_if_no_entitled_player_is_close_enough():
    config = SimConfig(possession_radius=2.0)
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0), restart_owner_team=1)
    team_0_player = Player(player_id=0, team=0, position=vec2(0.1, 0.0), velocity=vec2(0.0, 0.0))

    possessor_id = find_possessor(ball, [team_0_player], config.possession_radius)

    assert possessor_id is None


def test_non_owning_team_cannot_trap_or_kick_a_restarted_ball_even_standing_on_it():
    """End-to-end through `step()`: a team-0 player standing right on a ball
    that's awaiting team 1's throw-in shouldn't be able to trap it (even
    though a stationary ball is normally always receivable) or have a kick
    action register at all.
    """
    config = SimConfig(dt=1.0, possession_radius=2.0, players_per_team=0)
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0), restart_owner_team=1)
    team_0_player = Player(player_id=0, team=0, position=vec2(0.1, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[team_0_player], score=(0, 0))

    actions = {0: PlayerAction(move=vec2(0.0, 0.0), kick=vec2(10.0, 0.0))}
    new_state, events = step(state, actions, config)

    assert new_state.ball.carrier_id is None
    assert list(new_state.ball.velocity) == [0.0, 0.0]  # the kick had no effect
    assert new_state.ball.restart_owner_team == 1  # still locked -- no legitimate touch happened
    assert events == []


def test_owning_team_touching_the_ball_clears_the_restart_lock():
    config = SimConfig(dt=1.0, possession_radius=2.0, players_per_team=0, ball_friction=1.0)
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0), restart_owner_team=1)
    team_1_player = Player(player_id=1, team=1, position=vec2(0.1, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[team_1_player], score=(0, 0))

    actions = {1: PlayerAction(move=vec2(0.0, 0.0), kick=vec2(10.0, 0.0))}
    new_state, _ = step(state, actions, config)

    assert new_state.ball.restart_owner_team is None
    assert list(new_state.ball.velocity) == [10.0, 0.0]
