"""Round-trip correctness for the JSONL replay format (see viz/replay.py).

The core property under test: writing a sequence of (state, action, event)
triples to disk and reading them back must reproduce the exact same data —
this format is meant to become a training-data source later, so silent
precision loss or field drops would be a real (if quiet) bug.
"""

from __future__ import annotations

import numpy as np

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.reset import build_kickoff_state
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import vec2
from soccersim.viz.replay import ReplayWriter, load_replay


def test_replay_round_trip_preserves_config_state_actions_and_events(tmp_path):
    config = SimConfig(players_per_team=2, dt=1.0 / 30.0)
    state = build_kickoff_state(config)
    replay_path = tmp_path / "demo.jsonl"

    recorded_actions = []
    recorded_states = [state]
    recorded_events = [[]]

    with ReplayWriter(replay_path, config) as writer:
        writer.write_step(state, {}, [])
        for _ in range(3):
            actions = {0: PlayerAction(move=vec2(1.0, -2.0), kick=vec2(5.0, 0.0))}
            state, events = step(state, actions, config)
            writer.write_step(state, actions, events)
            recorded_actions.append(actions)
            recorded_states.append(state)
            recorded_events.append(events)

    replay = load_replay(replay_path)

    assert replay.config == config
    assert len(replay.steps) == len(recorded_states)

    for i, recorded_state in enumerate(recorded_states):
        loaded_state = replay.steps[i].state
        assert loaded_state.time == recorded_state.time
        assert loaded_state.score == recorded_state.score
        np.testing.assert_array_equal(loaded_state.ball.position, recorded_state.ball.position)
        np.testing.assert_array_equal(loaded_state.ball.velocity, recorded_state.ball.velocity)
        for loaded_player, recorded_player in zip(loaded_state.players, recorded_state.players):
            assert loaded_player.player_id == recorded_player.player_id
            assert loaded_player.team == recorded_player.team
            np.testing.assert_array_equal(loaded_player.position, recorded_player.position)
            np.testing.assert_array_equal(loaded_player.velocity, recorded_player.velocity)
            np.testing.assert_array_equal(loaded_player.facing, recorded_player.facing)
        assert loaded_state.ball.carrier_id == recorded_state.ball.carrier_id

    # Step 0 has no actions/events; steps 1..3 carry the kick action we sent.
    assert replay.steps[0].actions == {}
    assert replay.steps[0].events == []
    for i, actions in enumerate(recorded_actions, start=1):
        loaded_action = replay.steps[i].actions[0]
        np.testing.assert_array_equal(loaded_action.move, actions[0].move)
        np.testing.assert_array_equal(loaded_action.kick, actions[0].kick)


def test_replay_round_trip_preserves_a_non_none_carrier_id(tmp_path):
    """The kickoff-based round-trip test above never has a carried ball (the
    scripted actions always kick, which overrides carrying — see
    physics/step.py). Exercise the `carrier_id` field directly so a replay
    of a real dribble doesn't silently lose who's carrying the ball.
    """
    config = SimConfig(players_per_team=0)
    player = Player(player_id=0, team=0, position=vec2(0.0, 0.0), velocity=vec2(1.0, 0.0))
    ball = Ball(position=vec2(0.2, 0.0), velocity=vec2(1.0, 0.0), carrier_id=0)
    state = MatchState(time=0.0, ball=ball, players=[player], score=(0, 0))
    replay_path = tmp_path / "carried.jsonl"

    with ReplayWriter(replay_path, config) as writer:
        writer.write_step(state, {}, [])

    replay = load_replay(replay_path)

    assert replay.steps[0].state.ball.carrier_id == 0


def test_replay_round_trip_preserves_none_kick_and_events(tmp_path):
    config = SimConfig(players_per_team=0)
    state = build_kickoff_state(config)
    replay_path = tmp_path / "no_kick.jsonl"

    action = PlayerAction(move=vec2(0.0, 0.0), kick=None)
    event = Event(EventType.GOAL, {"team": 1})

    with ReplayWriter(replay_path, config) as writer:
        writer.write_step(state, {0: action}, [event])

    replay = load_replay(replay_path)

    loaded_action = replay.steps[0].actions[0]
    assert loaded_action.kick is None
    np.testing.assert_array_equal(loaded_action.move, vec2(0.0, 0.0))

    loaded_event = replay.steps[0].events[0]
    assert loaded_event.type is EventType.GOAL
    assert loaded_event.data == {"team": 1}
