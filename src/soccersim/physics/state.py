"""Data structures describing "the world at one instant" for the physics kernel.

Coordinate convention (this matters for every module downstream — viz,
agents, tactics — so it's defined once, here):

- The origin ``(0, 0)`` is the *center* of the pitch.
- ``+x`` points from team 0's goal toward team 1's goal. Team 0 defends the
  goal line at ``x = -pitch_length / 2`` and attacks toward ``+x``; team 1 is
  the mirror image, defending ``x = +pitch_length / 2``.
- ``+y`` runs across the pitch (from one touchline to the other). Each goal
  mouth is centered at ``y = 0``, spanning ``[-goal_width / 2, +goal_width / 2]``.

Keeping the pitch symmetric around the origin means team 1's geometry is
always "team 0's geometry, mirrored in x" — useful later when writing scripted
agents or observations that should work the same way for either team.

State is modeled as a tree of small, readable dataclasses rather than one big
numpy array. See docs/stages/stage1.md for why (short version: readability
while the rules are still being designed; can change later without touching
the public `step()` API if RL throughput ever demands it).
"""

from __future__ import annotations

import dataclasses

from soccersim.physics.vector import Vec2, vec2


@dataclasses.dataclass
class Ball:
    position: Vec2
    velocity: Vec2
    # The player currently dribbling the ball (see physics/step.py's "Ball
    # carrying" section), or None if it's a free ball. Unlike `possessor_id`
    # (recomputed fresh every step from positions alone — see `find_possessor`),
    # this is genuine persistent state: whether the ball is *attached* to a
    # player depends on what happened on contact in a *previous* step, not
    # just on current distance.
    carrier_id: int | None = None
    # Which team's player last made contact with the ball — a kick, a trap,
    # or a bounce off a player's body (a deflection still counts as "that
    # team touched it last", even though nobody controlled it). Like
    # `carrier_id`, this is genuine history that can't be recomputed from a
    # single frozen state; it's what `reset.py::restart_after_out_of_bounds`
    # uses to decide throw-in/corner/goal-kick attribution. `None` means the
    # ball hasn't been touched since the last kickoff/restart.
    last_touch_team: int | None = None


@dataclasses.dataclass
class Player:
    player_id: int
    team: int  # 0 or 1 — see module docstring for what each team defends/attacks
    position: Vec2
    velocity: Vec2
    # Unit vector for "which way this player is oriented" — used for kicks
    # aimed by facing rather than by an explicit target (see
    # `scripts/run_match.py`'s charge-and-release kick control) and available
    # to future scripted/RL agents for the same purpose. Tracks the last
    # *nonzero* requested movement direction (see `_step_player` in step.py);
    # a player who stops moving keeps facing the way they were last heading,
    # rather than snapping back to some default.
    facing: Vec2 = dataclasses.field(default_factory=lambda: vec2(1.0, 0.0))


@dataclasses.dataclass
class PlayerAction:
    """What one player is trying to do this step.

    `move` is a desired acceleration (direction + magnitude); `step()` clips
    it to `config.player_max_accel` before applying it, so callers don't need
    to worry about exceeding the limit themselves.

    `kick` is a desired *ball velocity* (not a force) — see the "Kick model"
    decision in docs/stages/stage1.md. It only has any effect if this player
    is the ball's current possessor (nearest player within
    `config.possession_radius`); otherwise it's silently ignored for this step.
    """

    move: Vec2
    kick: Vec2 | None = None


@dataclasses.dataclass
class MatchState:
    time: float
    ball: Ball
    players: list[Player]
    score: tuple[int, int]  # (team 0 goals, team 1 goals)
