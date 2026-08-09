"""Smoke tests for the pygame renderer (see viz/render.py).

These draw onto a plain off-screen `pygame.Surface` rather than opening a
window — `pygame.draw`/`pygame.font` work fine without a display, which is
what makes the renderer testable in CI/headless environments. We're not
asserting on exact pixel output (too brittle for what it'd verify); the bar
here is "drawing a real match state doesn't crash and actually paints
something," which is enough to catch the common failure modes (bad
world_to_screen math, drawing off-surface, unhandled team id, etc.).
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from soccersim.config import SimConfig
from soccersim.physics.events import Event, EventType
from soccersim.physics.reset import build_kickoff_state
from soccersim.viz.render import draw_match_state, window_size, world_to_screen


def test_window_size_scales_with_pitch_dimensions():
    small = SimConfig(pitch_length=50.0, pitch_width=30.0)
    large = SimConfig(pitch_length=105.0, pitch_width=68.0)
    assert window_size(large)[0] > window_size(small)[0]
    assert window_size(large)[1] > window_size(small)[1]


def test_world_to_screen_maps_pitch_center_and_corner_consistently():
    config = SimConfig()
    center_px = world_to_screen((0.0, 0.0), config)
    top_left_corner_px = world_to_screen((-config.pitch_length / 2, config.pitch_width / 2), config)
    # The center must be strictly inside the corner on both axes, and moving
    # in +y (world "up") must move *up* the screen (smaller pixel y) — this
    # is the axis-flip the module docstring calls out.
    assert top_left_corner_px[0] < center_px[0]
    assert top_left_corner_px[1] < center_px[1]


def test_draw_match_state_runs_headlessly_and_paints_pitch_and_ball():
    pygame.init()
    config = SimConfig(players_per_team=3)
    state = build_kickoff_state(config)
    surface = pygame.Surface(window_size(config))

    draw_match_state(surface, state, config)

    # A freshly created Surface is all-black; a successful draw must have
    # painted at least the pitch-green background somewhere.
    from soccersim.viz.render import PITCH_COLOR

    colors_present = {
        surface.get_at((x, y))[:3]
        for x in range(0, surface.get_width(), 7)
        for y in range(0, surface.get_height(), 7)
    }
    assert PITCH_COLOR in colors_present
    pygame.quit()


def test_draw_match_state_handles_goal_event_state_without_crashing():
    pygame.init()
    config = SimConfig(players_per_team=0)
    state = build_kickoff_state(config)
    surface = pygame.Surface(window_size(config))

    # Not exercising real event data through the renderer (it only reads
    # `state`), just confirming a post-goal score is drawn without error.
    _ = Event(EventType.GOAL, {"team": 0})
    state.score = (1, 0)
    draw_match_state(surface, state, config)
    pygame.quit()
