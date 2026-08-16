"""Tests for per-team role assignment (see agents/roles.py and
docs/stages/stage3-concepts.md's "Assignment as a matching problem").
"""

from __future__ import annotations

from soccersim.agents.base import RoleKind
from soccersim.agents.roles import assign_roles
from soccersim.config import SimConfig
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import vec2


def _player(player_id: int, team: int, x: float, y: float) -> Player:
    return Player(player_id=player_id, team=team, position=vec2(x, y), velocity=vec2(0.0, 0.0))


def test_keeper_is_always_the_fixed_roster_index_regardless_of_position():
    """Keeper is a fixed designation (roster index 0 per team), not computed
    from position — even a keeper standing at the center circle stays KEEPER.
    """
    config = SimConfig(players_per_team=2)
    players = [
        _player(0, team=0, x=0.0, y=0.0),  # team 0 keeper, standing at center circle
        _player(1, team=0, x=-40.0, y=0.0),
        _player(2, team=1, x=40.0, y=0.0),  # team 1 keeper
        _player(3, team=1, x=1.0, y=0.0),
    ]
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[0].kind is RoleKind.KEEPER
    assert roles[2].kind is RoleKind.KEEPER


def test_loose_ball_far_from_everyone_gives_each_team_one_chaser_and_rest_support():
    config = SimConfig(players_per_team=3)
    players = [
        _player(0, team=0, x=-50.0, y=0.0),  # keeper
        _player(1, team=0, x=-5.0, y=0.0),  # nearest team-0 outfield player to the ball
        _player(2, team=0, x=-20.0, y=10.0),
        _player(3, team=1, x=50.0, y=0.0),  # keeper
        _player(4, team=1, x=8.0, y=0.0),  # nearest team-1 outfield player to the ball
        _player(5, team=1, x=25.0, y=-10.0),
    ]
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))  # loose, nobody has it
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[1].kind is RoleKind.CHASER
    assert roles[4].kind is RoleKind.CHASER
    # Nobody has the ball yet, so there's nothing to mark — the rest hold
    # supporting positions rather than shadowing an opponent.
    assert roles[2].kind is RoleKind.SUPPORT
    assert roles[5].kind is RoleKind.SUPPORT


def test_ball_carrier_is_always_the_chaser_even_if_not_nearest():
    """A player already carrying the ball is the chaser regardless of
    distance — role assignment shouldn't hand CHASER to some other teammate
    just because they happen to be a hair closer to the ball's raw position.
    """
    config = SimConfig(players_per_team=2)
    players = [
        _player(0, team=0, x=-50.0, y=0.0),
        _player(1, team=0, x=-5.0, y=5.0),  # carrying the ball, but not the closest position-wise
        _player(2, team=1, x=50.0, y=0.0),
        _player(3, team=1, x=-4.9, y=5.0),  # slightly closer to raw ball position than player 1
    ]
    ball = Ball(position=vec2(-4.95, 5.0), velocity=vec2(0.0, 0.0), carrier_id=1)
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[1].kind is RoleKind.CHASER


def test_defending_team_markers_get_one_to_one_assignment_with_no_duplicates():
    """Two defenders, three attackers: greedy nearest-neighbor matching should
    assign each defender to a *different* attacker, never double-covering one
    attacker while another goes unmarked when a defender is available.
    """
    config = SimConfig(players_per_team=4)
    players = [
        _player(0, team=0, x=-50.0, y=0.0),  # team 0 keeper
        _player(1, team=0, x=-1.0, y=0.0),  # carrying the ball -> chaser
        _player(2, team=0, x=-3.0, y=4.0),
        _player(3, team=0, x=-10.0, y=-10.0),
        _player(4, team=1, x=50.0, y=0.0),  # team 1 keeper
        _player(5, team=1, x=6.0, y=0.0),  # nearest to ball -> team 1's chaser
        _player(6, team=1, x=8.0, y=3.0),  # marker, closest to player 1
        _player(7, team=1, x=20.0, y=20.0),  # marker, closest to player 2
    ]
    ball = Ball(position=vec2(-1.0, 0.0), velocity=vec2(0.0, 0.0), carrier_id=1)
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[5].kind is RoleKind.CHASER
    assert roles[6].kind is RoleKind.MARKER
    assert roles[7].kind is RoleKind.MARKER
    assert roles[6].mark_target_id == 1
    assert roles[7].mark_target_id == 2
    assert roles[6].mark_target_id != roles[7].mark_target_id


def test_attacking_teams_non_chasers_support_rather_than_mark():
    """The team *with* the ball has nobody to mark — its non-chaser outfield
    players hold supporting positions instead.
    """
    config = SimConfig(players_per_team=3)
    players = [
        _player(0, team=0, x=-50.0, y=0.0),
        _player(1, team=0, x=-1.0, y=0.0),  # carrying the ball -> chaser
        _player(2, team=0, x=-20.0, y=15.0),
        _player(3, team=1, x=50.0, y=0.0),
        _player(4, team=1, x=6.0, y=0.0),
        _player(5, team=1, x=15.0, y=-15.0),
    ]
    ball = Ball(position=vec2(-1.0, 0.0), velocity=vec2(0.0, 0.0), carrier_id=1)
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[2].kind is RoleKind.SUPPORT


def test_team_locked_out_of_a_restart_gets_no_chaser_at_all():
    """A throw-in/corner/goal-kick entitles one team to first touch (see
    Ball.restart_owner_team) — the other team shouldn't even have a player
    sent to chase the dead ball, however close one of them happens to be
    standing to it.
    """
    config = SimConfig(players_per_team=2)
    players = [
        _player(0, team=0, x=-50.0, y=0.0),
        _player(1, team=0, x=0.1, y=0.0),  # right next to the restart spot
        _player(2, team=1, x=50.0, y=0.0),
        _player(3, team=1, x=10.0, y=0.0),  # entitled team, still far away
    ]
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0), restart_owner_team=1)
    state = MatchState(time=0.0, ball=ball, players=players, score=(0, 0))

    roles = assign_roles(state, config)

    assert roles[1].kind is not RoleKind.CHASER  # locked-out team: nobody chases
    assert roles[3].kind is RoleKind.CHASER  # entitled team: plays normally
