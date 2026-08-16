"""Tests for `ScriptedAgent`'s per-role decision-making (agents/scripted.py).

Each test calls `ScriptedAgent.act()` directly with a hand-built `Role`,
rather than going through `assign_roles` — role assignment has its own tests
(test_agents_roles.py); these exercise what a player *does* once it already
knows its job.
"""

from __future__ import annotations

import math

from soccersim.agents.base import Role, RoleKind
from soccersim.agents.scripted import (
    KEEPER_LINE_DEPTH_M,
    MARKER_GOAL_SIDE_OFFSET_M,
    SHOOT_RANGE_M,
    SUPPORT_ATTACK_THIRD_MARGIN_M,
    ScriptedAgent,
)
from soccersim.config import SimConfig
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import vec2

AGENT = ScriptedAgent()


def _state(ball: Ball, players: list[Player]) -> MatchState:
    return MatchState(time=0.0, ball=ball, players=players, score=(0, 0))


def test_keeper_tracks_the_balls_y_within_the_goal_mouth_when_ball_is_far():
    config = SimConfig()
    half_length = config.pitch_length / 2
    keeper = Player(
        player_id=0,
        team=0,
        position=vec2(-half_length + KEEPER_LINE_DEPTH_M, 0.0),
        velocity=vec2(0.0, 0.0),
    )
    ball = Ball(position=vec2(0.0, 2.0), velocity=vec2(0.0, 0.0))  # far from goal, not "danger"
    state = _state(ball, [keeper])

    action = AGENT.act(0, state, Role(RoleKind.KEEPER), config)

    assert action.kick is None
    assert action.move[1] > 0.0  # tracks the ball upward in y
    assert math.isclose(action.move[0], 0.0, abs_tol=1e-9)  # already at target x


def test_keeper_advances_off_the_line_when_the_ball_is_close_to_goal():
    config = SimConfig()
    half_length = config.pitch_length / 2
    keeper = Player(
        player_id=0,
        team=0,
        position=vec2(-half_length + KEEPER_LINE_DEPTH_M, 0.0),
        velocity=vec2(0.0, 0.0),
    )
    ball = Ball(
        position=vec2(-half_length + 3.0, 0.0), velocity=vec2(0.0, 0.0)
    )  # well within danger range
    state = _state(ball, [keeper])

    action = AGENT.act(0, state, Role(RoleKind.KEEPER), config)

    assert action.move[0] > 0.0  # advances inward (+x, toward the ball) off the line


def test_keeper_clears_a_ball_that_has_rolled_to_them():
    config = SimConfig()
    half_length = config.pitch_length / 2
    keeper_position = vec2(-half_length + KEEPER_LINE_DEPTH_M, 0.0)
    keeper = Player(player_id=0, team=0, position=keeper_position, velocity=vec2(0.0, 0.0))
    ball = Ball(position=keeper_position, velocity=vec2(0.0, 0.0), carrier_id=0)
    state = _state(ball, [keeper])

    action = AGENT.act(0, state, Role(RoleKind.KEEPER), config)

    assert action.kick is not None
    assert action.kick[0] > 0.0  # clears upfield, toward center (+x from this corner of the goal)


def test_chaser_without_the_ball_moves_toward_it():
    config = SimConfig()
    player = Player(player_id=1, team=0, position=vec2(-10.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))  # loose
    state = _state(ball, [player])

    action = AGENT.act(1, state, Role(RoleKind.CHASER), config)

    assert action.kick is None
    assert action.move[0] > 0.0


def test_carrier_dribbles_forward_with_no_shot_or_pass_available():
    config = SimConfig()
    carrier = Player(player_id=1, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=carrier.position, velocity=vec2(0.0, 0.0), carrier_id=1)
    state = _state(ball, [carrier])  # no teammates, no opponents, far from goal

    action = AGENT.act(1, state, Role(RoleKind.CHASER), config)

    assert action.kick is None
    assert action.move[0] > 0.0  # dribbles toward the attacking goal (+x for team 0)
    assert math.isclose(action.move[1], 0.0, abs_tol=1e-9)


def test_carrier_shoots_when_in_range_with_a_clear_lane_aimed_away_from_the_keeper():
    config = SimConfig(players_per_team=2)
    half_length = config.pitch_length / 2
    carrier = Player(
        player_id=1, team=0, position=vec2(half_length - 10.0, 0.0), velocity=vec2(0.0, 0.0)
    )
    # Opponent keeper (id = (1 - team) * players_per_team = 2) standing off to
    # one side of goal, not directly in the shot's path.
    opponent_keeper = Player(
        player_id=2, team=1, position=vec2(half_length, 2.5), velocity=vec2(0.0, 0.0)
    )
    ball = Ball(position=carrier.position, velocity=vec2(0.0, 0.0), carrier_id=1)
    state = _state(ball, [carrier, opponent_keeper])

    action = AGENT.act(1, state, Role(RoleKind.CHASER), config)

    assert list(action.move) == [0.0, 0.0]
    assert action.kick is not None
    assert action.kick[0] > 0.0  # aimed toward the goal line
    assert action.kick[1] < 0.0  # away from the keeper, who's on the +y side


def test_carrier_passes_to_the_teammate_with_the_clearest_lane_over_dribbling():
    config = SimConfig()
    carrier = Player(player_id=1, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    teammate = Player(player_id=2, team=0, position=vec2(5.0, 10.0), velocity=vec2(0.0, 0.0))
    # Sitting directly on the straight-ahead dribble path, but well clear of
    # the passing lane out to the teammate.
    blocker = Player(player_id=3, team=1, position=vec2(15.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=carrier.position, velocity=vec2(0.0, 0.0), carrier_id=1)
    state = _state(ball, [carrier, teammate, blocker])  # far from goal -> shooting isn't considered

    action = AGENT.act(1, state, Role(RoleKind.CHASER), config)

    assert list(action.move) == [0.0, 0.0]
    assert action.kick is not None
    assert action.kick[0] > 0.0
    assert action.kick[1] > 0.0  # passed toward the open teammate, not straight ahead


def test_marker_holds_a_goal_side_offset_from_their_mark():
    config = SimConfig()
    marker = Player(player_id=10, team=0, position=vec2(-5.0, 5.0), velocity=vec2(0.0, 0.0))
    mark = Player(player_id=20, team=1, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = _state(ball, [marker, mark])

    action = AGENT.act(10, state, Role(RoleKind.MARKER, mark_target_id=20), config)

    # Target is `MARKER_GOAL_SIDE_OFFSET_M` from the mark, toward team 0's
    # own goal (-x) — i.e. at (-MARKER_GOAL_SIDE_OFFSET_M, 0), which is up
    # and to the right of the marker's own (-5, 5) position.
    assert MARKER_GOAL_SIDE_OFFSET_M < 5.0  # sanity: target really is toward +x from the marker
    assert action.kick is None
    assert action.move[0] > 0.0
    assert action.move[1] < 0.0


def test_support_player_advances_into_the_attacking_third_holding_their_width():
    config = SimConfig()
    half_length = config.pitch_length / 2
    player = Player(player_id=1, team=0, position=vec2(0.0, 7.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = _state(ball, [player])

    action = AGENT.act(1, state, Role(RoleKind.SUPPORT), config)

    target_x = half_length - SUPPORT_ATTACK_THIRD_MARGIN_M
    assert target_x > player.position[0]  # sanity: the target really is ahead of them
    assert action.kick is None
    assert action.move[0] > 0.0
    assert math.isclose(action.move[1], 0.0, abs_tol=1e-9)  # holds their y (width)


def test_shoot_range_constant_is_reasonable_relative_to_the_pitch():
    """Not a behavior test — just guards against SHOOT_RANGE_M silently
    drifting to something larger than the pitch itself."""
    config = SimConfig()
    assert 0.0 < SHOOT_RANGE_M < config.pitch_length / 2
