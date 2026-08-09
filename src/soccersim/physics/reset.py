"""Builds the initial `MatchState` for kickoff.

Deliberately fixed/symmetric rather than randomized: a reproducible starting
position means two runs with the same action sequence produce byte-identical
matches, which is what makes replay files and regression tests meaningful.
`config.seed` isn't used here yet — nothing in Stage 1 is stochastic — but is
threaded through `SimConfig` already for later stages that will need it
(e.g. randomized training scenarios).
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import vec2


def build_kickoff_state(config: SimConfig) -> MatchState:
    """A symmetric starting formation: ball at center, teams mirrored across x=0."""
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))

    players: list[Player] = []
    players.extend(_line_up(team=0, config=config))
    players.extend(_line_up(team=1, config=config))

    return MatchState(time=0.0, ball=ball, players=players, score=(0, 0))


def _line_up(team: int, config: SimConfig) -> list[Player]:
    """Place one team's players in a single vertical line facing the ball.

    Just enough structure to have a non-degenerate starting state (players
    aren't stacked on top of each other) — actual formations are a Stage 5
    (tactics layer) concern, not a physics-engine one.
    """
    n = config.players_per_team
    spacing = config.pitch_width / (n + 1)
    # Team 0 lines up on the -x half, facing +x (toward team 1's goal);
    # team 1 is the mirror image on the +x half.
    x = -config.pitch_length / 4 if team == 0 else config.pitch_length / 4

    players = []
    for i in range(n):
        y = -config.pitch_width / 2 + spacing * (i + 1)
        player_id = team * n + i
        players.append(
            Player(player_id=player_id, team=team, position=vec2(x, y), velocity=vec2(0.0, 0.0))
        )
    return players
