"""The `Agent` interface: the seam every scripted/cloned/RL player plugs into.

`Agent` is deliberately **per-player**, not per-team: `act()` is asked "what
should *this one* player do right now", never "what should the whole team
do". Every later stage that produces a policy — Stage 6's behavior cloning,
Stage 7's single-agent RL, Stage 8's multi-agent RL — trains exactly that
shape of decision (one player's policy), so keeping the interface
single-player now means a learned `Agent` can later replace a scripted one
for a single roster slot without anything else (the match loop, the
renderer, the replay format) changing at all. See docs/stages/stage3-concepts.md
Part 1 for the full reasoning, including why team-level coordination lives
outside this interface (in `agents/roles.py`) rather than inside it.

`Agent` is a `typing.Protocol` rather than an ABC: any object with a matching
`act()` method satisfies it structurally, with no forced inheritance — a
thin wrapper around a trained PyTorch policy is exactly as much an `Agent` as
`agents/scripted.py::ScriptedAgent` is, as long as its `act()` signature
matches.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol

from soccersim.config import SimConfig
from soccersim.physics.state import MatchState, PlayerAction


class RoleKind(enum.Enum):
    """What job a player has been assigned for the current step.

    Computed once per team per step by `agents/roles.py::assign_roles` — not
    decided by the player's own `Agent`. See stage3-concepts.md's "Assignment
    as a matching problem" for why role assignment is a separate, shared
    computation rather than something each `Agent` infers independently.
    """

    KEEPER = "keeper"  # fixed designation, anchors near their own goal
    CHASER = "chaser"  # pursuing or currently carrying the ball
    MARKER = "marker"  # shadowing a specific opponent (see `mark_target_id`)
    SUPPORT = "support"  # own team has the ball elsewhere; find open space


@dataclasses.dataclass(frozen=True)
class Role:
    """One player's assignment for the current step, from `assign_roles`."""

    kind: RoleKind
    # Which opponent this player is shadowing — only meaningful when
    # `kind is RoleKind.MARKER`; `None` otherwise.
    mark_target_id: int | None = None


class Agent(Protocol):
    """Something that can decide one player's action for one step.

    `state` is the raw `MatchState` — not a bespoke observation wrapper (see
    stage3-concepts.md for why: formal, normalized observation design is a
    Stage 4 concern, and scripted heuristics have no need for it). `role` is
    this player's assignment from the current step's `assign_roles()` call.
    """

    def act(
        self, player_id: int, state: MatchState, role: Role, config: SimConfig
    ) -> PlayerAction: ...
