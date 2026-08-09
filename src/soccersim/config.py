from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class SimConfig:
    pitch_width: float = 105.0
    pitch_height: float = 68.0
    dt: float = 1.0 / 60.0
    ball_friction: float = 0.98
    player_max_speed: float = 8.0
    player_max_accel: float = 6.0
    possession_radius: float = 1.0
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
