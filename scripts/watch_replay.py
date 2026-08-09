"""Play back a replay file recorded by `run_match.py` in a pygame window.

Uses the exact same `draw_match_state()` as the live viewer (see
`viz/render.py`'s module docstring on why the renderer is a pure function
shared by both) — this script just feeds it pre-recorded frames instead of
frames from a running `step()` loop.

Usage:
    uv run python scripts/watch_replay.py replays/demo.jsonl
    uv run python scripts/watch_replay.py replays/demo.jsonl --speed 2.0
"""

from __future__ import annotations

import argparse

import pygame

from soccersim.viz.render import draw_match_state, window_size
from soccersim.viz.replay import load_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_path", type=str)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed multiplier (1.0 = real-time)"
    )
    args = parser.parse_args()

    replay = load_replay(args.replay_path)
    config = replay.config

    pygame.init()
    screen = pygame.display.set_mode(window_size(config))
    pygame.display.set_caption(f"Soccer Simulation — replay: {args.replay_path}")
    clock = pygame.time.Clock()

    frames_per_second = (1.0 / config.dt) * args.speed

    running = True
    frame_index = 0
    while running and frame_index < len(replay.steps):
        for pygame_event in pygame.event.get():
            is_quit = pygame_event.type == pygame.QUIT
            is_escape = pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_ESCAPE
            if is_quit or is_escape:
                running = False
        if not running:
            break

        draw_match_state(screen, replay.steps[frame_index].state, config)
        pygame.display.flip()
        clock.tick(frames_per_second)
        frame_index += 1

    pygame.quit()


if __name__ == "__main__":
    main()
