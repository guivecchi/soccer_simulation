from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class SimConfig:
    # Pitch is centered at the origin: pitch_length runs along x (goal-to-goal,
    # the *long* side of a real pitch), pitch_width runs along y (touchline-to-
    # touchline, the *short* side). See physics/state.py for the full coordinate
    # convention.
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    goal_width: float = 7.32  # standard FIFA goal width, centered at y=0

    dt: float = 1.0 / 60.0

    # Fraction of speed the ball retains per *second* (not per step) — see
    # physics/step.py for why the model is expressed this way.
    ball_friction: float = 0.98

    player_max_speed: float = 8.0
    player_max_accel: float = 6.0
    # Acceleration cap used instead of player_max_accel when the requested
    # acceleration opposes the player's current velocity (braking / reversing
    # direction). Real players (and most sports games) can dig in and stop or
    # cut back much faster than they can build up speed from a standstill —
    # see `_step_player` in physics/step.py.
    player_brake_accel: float = 14.0
    possession_radius: float = 1.0
    max_kick_speed: float = 20.0

    # How fast a carried ball's velocity is pulled to match its carrier's —
    # deliberately much higher than player_max_accel so the ball "keeps up"
    # almost immediately rather than visibly lagging. See physics/step.py's
    # "Ball carrying" section.
    dribble_accel: float = 40.0
    # Fraction of the incoming normal-direction speed a ball keeps after
    # bouncing off a player it wasn't received by (1.0 = perfectly elastic,
    # 0.0 = fully absorbed). A body isn't a bouncy wall, so this is low.
    bounce_restitution: float = 0.4
    # Minimum cosine similarity between a player's movement input and the
    # ball's incoming velocity for contact to count as a deliberate "receive"
    # (trap) rather than an uncontrolled bounce. See `_is_receiving`.
    receive_alignment_threshold: float = 0.3

    players_per_team: int = 5

    seed: int = 0


def load_config(path: str | Path | None = None) -> SimConfig:
    """Build a SimConfig from defaults, optionally overridden by a YAML file."""
    if path is None:
        return SimConfig()

    overrides = yaml.safe_load(Path(path).read_text()) or {}
    known_fields = {f.name for f in dataclasses.fields(SimConfig)}
    unknown = set(overrides) - known_fields
    if unknown:
        raise ValueError(f"Unknown config field(s): {sorted(unknown)}")

    return SimConfig(**overrides)
