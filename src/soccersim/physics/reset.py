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

## Throw-in / corner / goal-kick placement (`restart_after_out_of_bounds`)

Added in Stage 3, alongside the rest of `restart_after_goal`'s pattern:
`step()` only reports *that* the ball left the pitch and *which* team
touched it last (`Event.data["side"]`, `state.ball.last_touch_team`); this
module decides where play resumes *and* who's entitled to take it
(`Ball.restart_owner_team`, enforced by `physics/step.py::find_possessor` —
see that field's docstring in state.py). An earlier version of this
function placed the ball correctly but didn't attribute an owning team for
throw-ins at all, and didn't stop the *other* team from just walking up and
retaking a restart even when it did know the right team (corners/
goal-kicks) — reported directly as "the last team to touch the ball is able
to kick it," confirmed by tracing real match replays where the team that
kicked it out of bounds regained it on the very next step.

`GOAL_KICK_DEPTH_M`/`CORNER_INSET_M` are simplified stand-ins for a proper
six-yard-box / corner-arc geometry, which Stage 1 never modeled (see
`physics/step.py`'s `_resolve_ball_bounds` — there's no goal-area concept in
the kernel at all). A fixed depth/inset from the goal line is enough to place
the ball somewhere sensible without inventing pitch-marking data the rest of
the sim doesn't otherwise track.
"""

from __future__ import annotations

from soccersim.config import SimConfig
from soccersim.physics.events import Event
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.vector import Vec2, vec2

# How far in front of the goal line a goal-kick is placed, and how far inset
# from the exact corner point a corner kick is placed — see the module
# docstring's note on why these are fixed constants rather than modeled
# pitch-marking geometry.
GOAL_KICK_DEPTH_M = 5.0
CORNER_INSET_M = 1.0
THROW_IN_INSET_M = 1.0


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


def restart_after_out_of_bounds(state: MatchState, event: Event, config: SimConfig) -> MatchState:
    """Place the ball for a throw-in, goal-kick, or corner after an `OUT_OF_BOUNDS` event.

    Unlike `restart_after_goal`, only the ball moves — a throw-in/corner/
    goal-kick doesn't reset every player back to kickoff formation, just
    where the ball is placed. Callers are expected to call this only in
    response to an `OUT_OF_BOUNDS` event from `step()`'s returned event list,
    using the exact same `state` that event was produced from (this function
    reads `state.ball.position` — already clamped to the boundary by
    `step()`'s `_resolve_ball_bounds` — and `state.ball.last_touch_team` to
    decide where and for which team).
    """
    side = event.data["side"]
    x, y = state.ball.position
    last_touch_team = state.ball.last_touch_team

    if side == "touchline":
        # Inset from the exact touchline, not placed right on it: a ball
        # sitting precisely on the boundary can be dragged back across it by
        # a player's very next dribble step (their trapping/carrying motion
        # doesn't know or care that it's standing on a line), immediately
        # re-triggering another OUT_OF_BOUNDS and producing an infinite
        # dead-ball loop at the sideline. A small inward nudge gives a
        # carrier room to actually move before the boundary is a concern
        # again.
        half_width = config.pitch_width / 2
        inset_y = half_width - THROW_IN_INSET_M if y >= 0.0 else -half_width + THROW_IN_INSET_M
        restart_position = vec2(x, inset_y)
        # Real football: whichever team *didn't* touch it last gets the
        # throw-in. `last_touch_team is None` (shouldn't happen once the
        # ball's left kickoff, but has no well-defined answer if it somehow
        # does) leaves the restart unowned rather than guessing.
        owner_team = None if last_touch_team is None else 1 - last_touch_team
    else:
        restart_position, owner_team = _goal_line_restart_position(x, y, last_touch_team, config)

    ball = Ball(position=restart_position, velocity=vec2(0.0, 0.0), restart_owner_team=owner_team)
    return MatchState(time=state.time, ball=ball, players=state.players, score=state.score)


def _goal_line_restart_position(
    exit_x: float, exit_y: float, last_touch_team: int | None, config: SimConfig
) -> tuple[Vec2, int | None]:
    """Where the ball is placed after going out over a goal line (not a goal),
    and which team is entitled to take it.

    Team 0 attacks the `+x` goal line, team 1 attacks the `-x` one (see the
    coordinate convention in `physics/state.py`), so whichever line the ball
    crossed tells us which team was defending it and which was attacking.
    Real football: if the *attacking* team touched it last (an overhit shot
    or pass), it's a goal-kick for the defence; if the *defending* team
    touched it last (a clearance or deflection), it's a corner for the
    attack. `last_touch_team is None` (no recorded touch — shouldn't really
    happen once the ball has left kickoff, but has no well-defined "who
    touched it" answer if it somehow does) defaults to the more conservative
    goal-kick outcome rather than gifting a corner from an untracked touch.
    """
    defending_team = 1 if exit_x > 0 else 0
    attacking_team = 1 - defending_team
    goal_line_x = config.pitch_length / 2 if defending_team == 1 else -config.pitch_length / 2
    # Direction from the goal line back toward the center circle — both a
    # goal-kick and a corner are placed some distance *inward* along this
    # direction from the exact line/corner point.
    inward_sign = -1.0 if defending_team == 1 else 1.0

    is_goal_kick = last_touch_team is None or last_touch_team == attacking_team
    if is_goal_kick:
        position = vec2(goal_line_x + inward_sign * GOAL_KICK_DEPTH_M, 0.0)
        return position, defending_team

    corner_y_sign = 1.0 if exit_y >= 0.0 else -1.0
    position = vec2(
        goal_line_x + inward_sign * CORNER_INSET_M,
        corner_y_sign * (config.pitch_width / 2 - CORNER_INSET_M),
    )
    return position, attacking_team


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

    # Face each team toward the goal they attack — see the coordinate
    # convention in physics/state.py.
    facing = vec2(1.0, 0.0) if team == 0 else vec2(-1.0, 0.0)

    players = []
    for i in range(n):
        y = -config.pitch_width / 2 + spacing * (i + 1)
        player_id = team * n + i
        players.append(
            Player(
                player_id=player_id,
                team=team,
                position=vec2(x, y),
                velocity=vec2(0.0, 0.0),
                facing=facing,
            )
        )
    return players
