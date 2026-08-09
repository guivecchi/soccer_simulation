"""Tests for ball-player contact: dribbling (carrying), trapping, and bouncing.

See the "Ball carrying" section of physics/step.py's module docstring for the
concepts these exercise: `possessor_id` (geometric, recomputed every step) vs
`carrier_id` (persistent — whether the ball is actually attached to someone).
"""

from __future__ import annotations

import math

from soccersim.config import SimConfig
from soccersim.physics.events import EventType
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import vec2


def _isclose_vec(v, expected, tol=1e-9):
    return all(math.isclose(a, b, abs_tol=tol) for a, b in zip(v, expected))


def test_moving_to_receive_traps_the_ball():
    """A player whose movement is aligned with the ball's incoming velocity
    (retreating *with* it) should trap it: the ball attaches and its
    velocity is pulled to match the player's, rather than bouncing off.
    """
    config = SimConfig(
        dt=1.0,
        possession_radius=2.0,
        ball_friction=1.0,
        dribble_accel=40.0,
        receive_alignment_threshold=0.3,
        players_per_team=0,
    )
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(-2.0, 0.0))
    ball = Ball(position=vec2(1.0, 0.0), velocity=vec2(-3.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    # Requested movement points the same way the ball is already traveling.
    actions = {0: PlayerAction(move=vec2(-2.0, 0.0))}
    new_state, events = step(state, actions, config)

    assert new_state.ball.carrier_id == 0
    # The ball's dribble target is the carrier's *just-updated* velocity for
    # this step (see step.py) — here the player kept accelerating in the
    # same direction, from -2.0 to -4.0, so that's what the ball matches too.
    assert _isclose_vec(new_state.ball.velocity, [-4.0, 0.0])
    assert EventType.BALL_TRAPPED in [e.type for e in events]
    trapped_event = next(e for e in events if e.type is EventType.BALL_TRAPPED)
    assert trapped_event.data["player_id"] == 0


def test_standing_still_bounces_the_ball_off_instead_of_trapping():
    """A player who doesn't move to meet the ball just presents a body it
    deflects off, losing speed but not being brought under control.
    """
    config = SimConfig(
        dt=1.0,
        possession_radius=2.0,
        ball_friction=1.0,
        bounce_restitution=0.4,
        players_per_team=0,
    )
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(1.0, 0.0), velocity=vec2(-3.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    new_state, events = step(state, {}, config)

    assert new_state.ball.carrier_id is None
    # Head-on hit: bounces straight back, at restitution * incoming speed.
    assert _isclose_vec(new_state.ball.velocity, [1.2, 0.0])
    assert EventType.BALL_BOUNCED in [e.type for e in events]
    bounced_event = next(e for e in events if e.type is EventType.BALL_BOUNCED)
    assert bounced_event.data["player_id"] == 0


def test_stationary_ball_is_always_receivable_regardless_of_movement():
    """A ball with no real momentum (e.g. sitting dead) has nothing to
    cushion, so any nearby player can gather it without a special motion.
    """
    config = SimConfig(dt=1.0, possession_radius=2.0, ball_friction=1.0, players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    new_state, events = step(state, {}, config)

    assert new_state.ball.carrier_id == 0
    assert [e.type for e in events] == [EventType.BALL_TRAPPED]


def test_carried_ball_keeps_tracking_the_carrier_across_multiple_steps():
    """Once attached, the ball should keep riding along with its carrier —
    not just on the step it was trapped, and without needing to stay within
    `possession_radius` to "re-earn" the attachment each step.
    """
    config = SimConfig(dt=1.0 / 60.0, dribble_accel=40.0, players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.3, 0.0), velocity=vec2(0.0, 0.0), carrier_id=0)
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    actions = {0: PlayerAction(move=vec2(config.player_max_accel, 0.0))}
    for _ in range(30):
        state, _ = step(state, actions, config)

    assert state.ball.carrier_id == 0
    assert math.isclose(state.ball.velocity[0], state.players[0].velocity[0], rel_tol=1e-2)


def test_ball_moving_away_from_a_nearby_player_does_not_bounce_again():
    """A ball already receding from a player (e.g. right after a bounce)
    shouldn't keep re-triggering contact just for staying within
    possession_radius for a few more steps — it should coast normally.
    """
    config = SimConfig(dt=1.0, possession_radius=2.0, ball_friction=1.0, players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))
    ball = Ball(position=vec2(0.5, 0.0), velocity=vec2(5.0, 0.0))
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    new_state, events = step(state, {}, config)

    assert new_state.ball.carrier_id is None
    assert _isclose_vec(new_state.ball.velocity, [5.0, 0.0])
    assert EventType.BALL_BOUNCED not in [e.type for e in events]
    assert EventType.BALL_TRAPPED not in [e.type for e in events]


def test_kicking_releases_the_ball_from_its_carrier():
    """A kick always overrides carrying — one-touch passes/shots shouldn't
    require the ball to be "let go" first.
    """
    config = SimConfig(dt=1.0, ball_friction=1.0, max_kick_speed=15.0, players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(1.0, 0.0))
    ball = Ball(position=vec2(0.2, 0.0), velocity=vec2(1.0, 0.0), carrier_id=0)
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))

    actions = {0: PlayerAction(move=vec2(0.0, 0.0), kick=vec2(0.0, 10.0))}
    new_state, _ = step(state, actions, config)

    assert new_state.ball.carrier_id is None
    assert _isclose_vec(new_state.ball.velocity, [0.0, 10.0])
