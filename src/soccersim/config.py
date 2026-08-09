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
    possession_radius: float = 1.0
    max_kick_speed: float = 20.0

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
