"""Builds `MatchState`s for kickoff and for restarting play after a goal.

Deliberately fixed/symmetric rather than randomized: a reproducible starting
position means two runs with the same action sequence produce byte-identical
matches, which is what makes replay files and regression tests meaningful.
`config.seed` isn't used here yet — nothing in Stage 1 is stochastic — but is
threaded through `SimConfig` already for later stages that will need it
(e.g. randomized training scenarios).

## Why `restart_after_goal` isn't inside `step()`

`step()` (see `physics/step.py`) deliberately only *reports* a `GOAL` event
when the ball crosses a goal line inside the goal mouth — it stops the ball
at the boundary and leaves it there, rather than also resetting positions
itself. Two reasons:

1. **Keeps `step()`'s existing tests meaningful.** Tests like
   `test_ball_crossing_goal_mouth_scores_and_updates_score` check the exact
   state produced by *one* `step()` call at the moment a goal is scored
   (ball stopped at the line, velocity zeroed). If `step()` silently
   teleported the ball back to center inside the same call, that assertion
   would no longer be checking what it says it's checking.
2. **Restart is a game-flow decision, not physics.** "What happens next
   after a goal" is a rule about *match structure*, not about how the ball
   moves — closer in kind to Stage 3's future kickoff/throw-in/corner logic
   than to friction or collision. Keeping it as a separate, explicitly
   *composed* step (`step()` then, if a `GOAL` fired, `restart_after_goal()`)
   means any caller can choose whether it wants that behavior, instead of it
   being baked silently into the kernel everyone calls.

Throw-ins and corners (restarting after an `OUT_OF_BOUNDS` event) are a
known, still-deferred gap — same category of fix, just not this one.
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import vec2


def build_kickoff_state(config: SimConfig) -> MatchState:
    """The very first state of a match: score 0-0, clock at zero."""
    ball, players = _kickoff_positions(config)
    return MatchState(time=0.0, ball=ball, players=players, score=(0, 0))


def restart_after_goal(state: MatchState, config: SimConfig) -> MatchState:
    """Reset ball and players to kickoff positions after a goal.

    Unlike `build_kickoff_state`, this *preserves* `time` and `score` from
    the state it's given — a restart continues the same match, it doesn't
    start a new one. Callers are expected to call this only in response to
    a `GOAL` event from `step()`'s returned event list (see module
    docstring for why this lives outside `step()` itself).
    """
    ball, players = _kickoff_positions(config)
    return MatchState(time=state.time, ball=ball, players=players, score=state.score)


def _kickoff_positions(config: SimConfig) -> tuple[Ball, list[Player]]:
    """The shared symmetric formation used by both a fresh kickoff and a restart."""
    ball = Ball(position=vec2(0.0, 0.0), velocity=vec2(0.0, 0.0))

    players: list[Player] = []
    players.extend(_line_up(team=0, config=config))
    players.extend(_line_up(team=1, config=config))

    return ball, players


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
