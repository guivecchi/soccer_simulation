"""Run a live match in a pygame window; optionally record it to a replay file.

There's no `Agent` interface yet (that's Stage 3) and no scripted opponents,
so this script drives player 0 directly from the keyboard — purely a debug
tool for eyeballing the physics kernel, not a preview of how matches will
eventually be played. Every other player just sits still under friction-only
physics: the point is to *see* the kernel from Stage 1 behave correctly
(acceleration, speed clamping, friction, kicking, bounds/goals), which is a
five-second visual check instead of reading numbers in a debugger.

Controls:
    Arrow keys   accelerate player 0
    Space        kick the ball toward the opponent's goal (only has an
                 effect if player 0 currently has possession)
    Esc / close  quit

Usage:
    uv run python scripts/run_match.py
    uv run python scripts/run_match.py --record replays/demo.jsonl
"""

from __future__ import annotations

import argparse

import pygame

from soccersim.config import load_config
from soccersim.physics.reset import build_kickoff_state
from soccersim.physics.state import PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import vec2
from soccersim.viz.render import draw_match_state, window_size
from soccersim.viz.replay import ReplayWriter

CONTROLLED_PLAYER_ID = 0


def _read_keyboard_action(config, ball_position, controlled_player) -> PlayerAction:
    keys = pygame.key.get_pressed()
    ax = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * config.player_max_accel
    ay = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * config.player_max_accel
    move = vec2(ax, ay)

    kick = None
    if keys[pygame.K_SPACE]:
        # Aim at the opponent's goal line, at the ball's own y — a simple,
        # deterministic "shoot" rather than anything resembling real aiming.
        target_x = (
            config.pitch_length / 2 if controlled_player.team == 0 else -config.pitch_length / 2
        )
        aim = vec2(target_x, ball_position[1]) - ball_position
        norm = float((aim[0] ** 2 + aim[1] ** 2) ** 0.5)
        kick = (aim / norm) * config.max_kick_speed if norm > 0 else vec2(0.0, 0.0)

    return PlayerAction(move=move, kick=kick)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=str, default=None, help="path to write a replay to")
    parser.add_argument("--config", type=str, default=None, help="optional YAML config override")
    args = parser.parse_args()

    config = load_config(args.config)
    state = build_kickoff_state(config)

    pygame.init()
    screen = pygame.display.set_mode(window_size(config))
    pygame.display.set_caption("Soccer Simulation — live match")
    clock = pygame.time.Clock()

    writer = ReplayWriter(args.record, config) if args.record else None
    if writer is not None:
        writer.write_step(state, {}, [])

    running = True
    while running:
        for pygame_event in pygame.event.get():
            is_quit = pygame_event.type == pygame.QUIT
            is_escape = pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_ESCAPE
            if is_quit or is_escape:
                running = False

        controlled_player = next(p for p in state.players if p.player_id == CONTROLLED_PLAYER_ID)
        actions = {
            CONTROLLED_PLAYER_ID: _read_keyboard_action(
                config, state.ball.position, controlled_player
            )
        }

        state, events = step(state, actions, config)
        if writer is not None:
            writer.write_step(state, actions, events)

        draw_match_state(screen, state, config)
        pygame.display.flip()
        clock.tick(round(1.0 / config.dt))

    if writer is not None:
        writer.close()
    pygame.quit()


if __name__ == "__main__":
    main()
