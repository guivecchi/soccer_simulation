"""Replay recording/loading: dumps `(state, action, event)` sequences to disk.

Per ROADMAP.md's Stage 2 scope, this file format *is* the training-data
schema future stages (behavior cloning, RL logging) will read — worth being
deliberate about now rather than treating it as a throwaway debug dump.

## Format: JSON Lines (one JSON object per line)

Chosen over pickle or a fixed-schema numpy array (the other options
considered) because:

- It's human-readable and diffable — you can `head` a replay file and see
  what happened, which matters a lot while the simulation itself is still
  being debugged.
- It's resilient to the dataclasses in `physics/state.py` changing shape
  over time — unlike pickle, which serializes by class identity and breaks
  the moment a dataclass's fields change.
- Line-oriented means a replay can be streamed (read one line at a time)
  rather than requiring the whole file to be valid before any of it is
  usable — useful for very long matches later, even though `load_replay()`
  below just reads everything into memory for now (simplicity first; revisit
  if replay files get large enough for that to matter).

The trade-off is verbosity (JSON text is bigger than packed binary) and
parse speed — acceptable at this project's scale (short, low-roster matches),
worth reconsidering only if Stage 6+ needs to stream *many* large replays
into a training pipeline.

## Record shape

The first line is always a `"meta"` record carrying the full `SimConfig` the
match was run with (so a replay is self-describing: `dt`, pitch dimensions,
etc. don't need to be guessed or passed separately when loading it back for
playback). Every following line is a `"step"` record: the resulting
`MatchState` after one `step()` call, the `actions` dict that produced it,
and the `events` that fired — i.e. exactly the `(state, action, event)`
triple the roadmap asks for. The very first step record (before any `step()`
call) carries the initial state with empty actions/events, so a replay always
starts from a complete, renderable frame.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.state import Ball, MatchState, Player, PlayerAction
from soccersim.physics.vector import vec2


@dataclasses.dataclass
class ReplayStep:
    state: MatchState
    actions: dict[int, PlayerAction]
    events: list[Event]


@dataclasses.dataclass
class Replay:
    config: SimConfig
    steps: list[ReplayStep]


class ReplayWriter:
    """Append-only JSONL writer. Use as a context manager or call `close()`."""

    def __init__(self, path: str | Path, config: SimConfig):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately not a `with` block: the handle must stay open across
        # every `write_step()` call for the writer's lifetime, closed only
        # by `close()`/`__exit__` — that's the reason this class exists.
        self._file = path.open("w", encoding="utf-8")
        self._write({"kind": "meta", "config": dataclasses.asdict(config)})

    def write_step(
        self,
        state: MatchState,
        actions: dict[int, PlayerAction],
        events: list[Event],
    ) -> None:
        self._write(
            {
                "kind": "step",
                "time": state.time,
                "score": list(state.score),
                "ball": _ball_to_jsonable(state.ball),
                "players": [_player_to_jsonable(p) for p in state.players],
                "actions": {
                    str(player_id): _action_to_jsonable(action)
                    for player_id, action in actions.items()
                },
                "events": [_event_to_jsonable(e) for e in events],
            }
        )

    def close(self) -> None:
        self._file.close()

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")

    def __enter__(self) -> ReplayWriter:  # noqa: PYI034 (no `Self` type on py310)
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def load_replay(path: str | Path) -> Replay:
    """Read an entire replay file into memory (see module docstring on why)."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    meta, *step_records = records
    config = SimConfig(**meta["config"])
    steps = [_step_from_record(r) for r in step_records]
    return Replay(config=config, steps=steps)


# --- state/action/event <-> JSON-safe dict conversions -----------------------
#
# numpy arrays (`Vec2`) aren't JSON-serializable, so every conversion here
# goes through plain lists. Kept separate from `physics/` so the physics
# kernel itself never needs to know this format exists.


def _ball_to_jsonable(ball: Ball) -> dict:
    return {
        "position": ball.position.tolist(),
        "velocity": ball.velocity.tolist(),
        "carrier_id": ball.carrier_id,
        "last_touch_team": ball.last_touch_team,
    }


def _ball_from_record(record: dict) -> Ball:
    return Ball(
        position=vec2(*record["position"]),
        velocity=vec2(*record["velocity"]),
        # `.get` (not `["carrier_id"]`/`["last_touch_team"]`): replay files
        # written before these fields existed don't have them, and "no
        # carrier" / "no recorded touch" is the correct reading for them
        # anyway.
        carrier_id=record.get("carrier_id"),
        last_touch_team=record.get("last_touch_team"),
    )


def _player_to_jsonable(player: Player) -> dict:
    return {
        "player_id": player.player_id,
        "team": player.team,
        "position": player.position.tolist(),
        "velocity": player.velocity.tolist(),
        "facing": player.facing.tolist(),
    }


def _player_from_record(record: dict) -> Player:
    facing = vec2(*record["facing"]) if "facing" in record else vec2(1.0, 0.0)
    return Player(
        player_id=record["player_id"],
        team=record["team"],
        position=vec2(*record["position"]),
        velocity=vec2(*record["velocity"]),
        facing=facing,
    )


def _action_to_jsonable(action: PlayerAction) -> dict:
    return {
        "move": action.move.tolist(),
        "kick": action.kick.tolist() if action.kick is not None else None,
    }


def _action_from_record(record: dict) -> PlayerAction:
    kick = vec2(*record["kick"]) if record["kick"] is not None else None
    return PlayerAction(move=vec2(*record["move"]), kick=kick)


def _event_to_jsonable(event: Event) -> dict:
    return {"type": event.type.value, "data": event.data}


def _event_from_record(record: dict) -> Event:
    return Event(type=EventType(record["type"]), data=record["data"])


def _step_from_record(record: dict) -> ReplayStep:
    state = MatchState(
        time=record["time"],
        ball=_ball_from_record(record["ball"]),
        players=[_player_from_record(p) for p in record["players"]],
        score=tuple(record["score"]),
    )
    actions = {int(player_id): _action_from_record(a) for player_id, a in record["actions"].items()}
    events = [_event_from_record(e) for e in record["events"]]
    return ReplayStep(state=state, actions=actions, events=events)
