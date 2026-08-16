"""Per-team role assignment: `assign_roles(state, config) -> dict[player_id, Role]`.

Runs once per team, twice per `step()` call — each team's roles are computed
independently from raw positions, never looking at the other team's
assignment, which keeps the result symmetric and side-agnostic. This is
deliberately **not** the Stage 5 tactics layer: it's a fixed, rule-based
assignment (nearest-player/greedy-matching), not a learned or configurable
formation system. It exists purely to give Stage 3's individually-simple
`Agent`s coherent team shape — see docs/stages/stage3-concepts.md Part 1 for
why this needs to be a shared coordination step rather than something each
player's `Agent` infers independently.

Like `physics/step.py::step`, `assign_roles` is a pure function of
`(state, config)` — same inputs always produce the same role assignment, with
ties broken by `player_id` throughout, so replays and tests stay
reproducible.
"""

from __future__ import annotations

from soccersim.agents.base import Role, RoleKind
from soccersim.config import SimConfig
from soccersim.physics.state import MatchState, Player
from soccersim.physics.step import find_possessor
from soccersim.physics.vector import magnitude


def assign_roles(state: MatchState, config: SimConfig) -> dict[int, Role]:
    """Compute every player's role for the current step, one team at a time."""
    roles: dict[int, Role] = {}
    roles.update(_assign_team_roles(team=0, state=state, config=config))
    roles.update(_assign_team_roles(team=1, state=state, config=config))
    return roles


def _assign_team_roles(team: int, state: MatchState, config: SimConfig) -> dict[int, Role]:
    n = config.players_per_team
    keeper_id = team * n
    outfield_ids = sorted(
        p.player_id for p in state.players if p.team == team and p.player_id != keeper_id
    )

    roles: dict[int, Role] = {keeper_id: Role(RoleKind.KEEPER)}

    # A throw-in/corner/goal-kick entitles the *other* team to first touch
    # (see Ball.restart_owner_team's docstring) — this team gets no CHASER
    # at all while that holds, so nobody's even sent toward the ball. The
    # entitled team plays entirely normally; `find_possessor` (physics/
    # step.py) is what makes the entitlement actually stick even if this
    # team's players are already standing right next to the restart spot.
    is_locked_out_of_restart = (
        state.ball.restart_owner_team is not None and state.ball.restart_owner_team != team
    )
    if is_locked_out_of_restart:
        chaser_id = None
    else:
        ball_handler_id = _team_ball_handler(team, state, config)
        chaser_id = (
            ball_handler_id
            if ball_handler_id in outfield_ids
            else _nearest_to_ball(outfield_ids, state)
        )
        roles[chaser_id] = Role(RoleKind.CHASER)

    remaining = [pid for pid in outfield_ids if pid != chaser_id]
    opponent_team = 1 - team
    if _team_ball_handler(opponent_team, state, config) is not None and remaining:
        opponent_keeper_id = opponent_team * n
        opponent_outfield_ids = [
            p.player_id
            for p in state.players
            if p.team == opponent_team and p.player_id != opponent_keeper_id
        ]
        marks = _assign_markers(remaining, opponent_outfield_ids, state.players)
        for pid in remaining:
            roles[pid] = Role(RoleKind.MARKER, mark_target_id=marks.get(pid))
    else:
        for pid in remaining:
            roles[pid] = Role(RoleKind.SUPPORT)

    return roles


def _team_ball_handler(team: int, state: MatchState, config: SimConfig) -> int | None:
    """The player_id of `team`'s player currently carrying/possessing the ball, or None.

    "Possessing" falls back to `find_possessor` (the same possession-radius
    lookup `step()` uses to decide whose kick affects the ball) when nobody
    is currently carrying it — a player about to receive or shoot still
    counts as "their team has the ball" for role-assignment purposes, even
    a step before `carrier_id` would reflect it.
    """
    by_id = {p.player_id: p for p in state.players}

    carrier_id = state.ball.carrier_id
    if carrier_id is not None:
        return carrier_id if by_id[carrier_id].team == team else None

    possessor_id = find_possessor(state.ball, state.players, config.possession_radius)
    if possessor_id is not None:
        return possessor_id if by_id[possessor_id].team == team else None

    return None


def _nearest_to_ball(player_ids: list[int], state: MatchState) -> int:
    """The player (from `player_ids`) closest to the ball, ties broken by `player_id`."""
    by_id = {p.player_id: p for p in state.players}
    return min(
        player_ids,
        key=lambda pid: (magnitude(by_id[pid].position - state.ball.position), pid),
    )


def _assign_markers(
    defender_ids: list[int], opponent_ids: list[int], players: list[Player]
) -> dict[int, int]:
    """Greedy nearest-neighbor one-to-one matching of defenders to opponents.

    Repeatedly pairs off the single closest remaining (defender, opponent)
    pair and removes both from further consideration, until either side runs
    out. This doesn't guarantee the globally-optimal total-distance matching
    (the Hungarian algorithm would), but it does guarantee the property that
    actually matters for team shape: no two defenders end up assigned to the
    same opponent while a third goes completely unassigned. See
    docs/stages/stage3-concepts.md's "Assignment as a matching problem".

    If the rosters are uneven (more defenders than opponents, or vice
    versa), leftover players on the larger side simply have no entry in the
    returned mapping — callers treat a missing key as "no specific mark".
    """
    positions = {p.player_id: p.position for p in players}
    remaining_defenders = list(defender_ids)
    remaining_opponents = list(opponent_ids)
    assignment: dict[int, int] = {}

    while remaining_defenders and remaining_opponents:
        best_defender, best_opponent, best_key = None, None, None
        for defender_id in remaining_defenders:
            for opponent_id in remaining_opponents:
                distance = magnitude(positions[defender_id] - positions[opponent_id])
                key = (distance, defender_id, opponent_id)
                if best_key is None or key < best_key:
                    best_key = key
                    best_defender, best_opponent = defender_id, opponent_id

        assignment[best_defender] = best_opponent
        remaining_defenders.remove(best_defender)
        remaining_opponents.remove(best_opponent)

    return assignment
