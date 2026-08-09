"""Events emitted by `step()` to report things that happened during a step.

`step()` returns `(new_state, events)` rather than mutating a log internally.
State transitions stay a pure function of `(state, actions, config)`, and
anything that wants to react to what happened (a renderer, a match-restart
rule, a training script logging goals) reads it from the returned event list
instead of having to diff two `MatchState`s itself.
"""

from __future__ import annotations

import dataclasses
import enum


class EventType(enum.Enum):
    GOAL = "goal"
    OUT_OF_BOUNDS = "out_of_bounds"
    POSSESSION_CHANGE = "possession_change"
    BALL_TRAPPED = "ball_trapped"  # a free ball was brought under control (see step.py)
    BALL_BOUNCED = "ball_bounced"  # a free ball deflected off a player without being controlled


@dataclasses.dataclass
class Event:
    type: EventType
    data: dict
