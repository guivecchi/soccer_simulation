"""Small 2D vector helpers shared across the physics kernel.

We represent a 2D vector as a plain numpy array of shape ``(2,)`` rather than
introducing a custom ``Vector2`` class. This keeps the state dataclasses in
``state.py`` simple while still getting fast, readable vector arithmetic
(``+``, ``-``, ``*`` scalar, ``np.linalg.norm``) for free.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec2 = npt.NDArray[np.float64]


def vec2(x: float, y: float) -> Vec2:
    """Build a Vec2 from two scalars."""
    return np.array([x, y], dtype=np.float64)


def magnitude(v: Vec2) -> float:
    """Euclidean length of a vector."""
    return float(np.linalg.norm(v))


def clip_magnitude(v: Vec2, max_magnitude: float) -> Vec2:
    """Scale `v` down to `max_magnitude` if it exceeds it, otherwise return it unchanged.

    This is how we enforce both "max acceleration" and "max speed": rather than
    capping each axis independently (which would let a diagonal vector exceed
    the limit by a factor of sqrt(2)), we cap the vector's overall length so a
    player's top speed is the same in every direction.
    """
    mag = magnitude(v)
    if mag <= max_magnitude or mag == 0.0:
        return np.array(v, dtype=np.float64)
    return v * (max_magnitude / mag)
