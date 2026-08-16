"""End-to-end smoke test: two full scripted teams playing each other.

Cheap but important — this is Stage 3's actual done criterion ("two full
scripted teams can play a complete match end-to-end... without crashing,
going out of sync, or producing NaNs"), exercised the same way
`scripts/run_match.py --headless` runs it, just without pygame.
"""

from __future__ import annotations

import math

from soccersim.agents.roles import assign_roles
from soccersim.agents.scripted import ScriptedAgent
from soccersim.config import SimConfig
from soccersim.physics.events import EventType
from soccersim.physics.reset import (
    build_kickoff_state,
    restart_after_goal,
    restart_after_out_of_bounds,
)
from soccersim.physics.step import step


def _is_finite_vec(v) -> bool:
    return all(math.isfinite(component) for component in v)


def test_a_full_scripted_match_runs_many_steps_without_nans_or_crashing():
    config = SimConfig(players_per_team=4)
    state = build_kickoff_state(config)
    agent = ScriptedAgent()

    for _ in range(600):  # 10 simulated seconds at the default dt
        roles = assign_roles(state, config)
        actions = {
            player.player_id: agent.act(player.player_id, state, roles[player.player_id], config)
            for player in state.players
        }
        state, events = step(state, actions, config)

        for event in events:
            if event.type is EventType.GOAL:
                state = restart_after_goal(state, config)
            elif event.type is EventType.OUT_OF_BOUNDS:
                state = restart_after_out_of_bounds(state, event, config)

        assert _is_finite_vec(state.ball.position)
        assert _is_finite_vec(state.ball.velocity)
        for player in state.players:
            assert _is_finite_vec(player.position)
            assert _is_finite_vec(player.velocity)
            # Pitch bounds are enforced by the kernel itself (_clamp_to_pitch)
            # — a scripted target outside the pitch shouldn't be able to push
            # a player past it.
            assert -config.pitch_length / 2 <= player.position[0] <= config.pitch_length / 2
            assert -config.pitch_width / 2 <= player.position[1] <= config.pitch_width / 2

    assert state.time > 0.0
