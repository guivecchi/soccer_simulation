"""Run a live match in a pygame window; optionally record it to a replay file.

There's no `Agent` interface yet (that's Stage 3) and no scripted opponents,
so this script drives one player directly from the keyboard — purely a debug
tool for eyeballing the physics kernel, not a preview of how matches will
eventually be played. Every other player just sits still under friction-only
physics: the point is to *see* the kernel behave correctly (acceleration,
speed clamping, friction, kicking, dribbling, bounds/goals), which is a
five-second visual check instead of reading numbers in a debugger.

Controls:
    Arrow keys   accelerate the controlled player (braking/reversing is
                 quicker than accelerating from rest — see player_brake_accel
                 in config.py)
    Space        hold to charge a kick, release to fire it in whichever
                 direction the player is currently facing, at a strength
                 proportional to how long Space was held (only has an effect
                 if the controlled player currently has possession)
    Tab          switch which of team 0's players is controlled
    Esc / close  quit

Usage:
    uv run python scripts/run_match.py
    uv run python scripts/run_match.py --record replays/demo.jsonl
"""

from __future__ import annotations

import argparse

import pygame

from soccersim.config import load_config
from soccersim.physics.events import EventType
from soccersim.physics.reset import build_kickoff_state, restart_after_goal
from soccersim.physics.state import PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import vec2
from soccersim.viz.render import draw_match_state, window_size
from soccersim.viz.replay import ReplayWriter

# Kick charging lives entirely in this keyboard-demo script, not the physics
# kernel: `PlayerAction.kick` is still just "one desired ball velocity for
# this step" (see physics/state.py) — charging is an *input* concern layered
# on top by whatever produces actions, human or (later) scripted.
KICK_CHARGE_SECONDS = 1.0  # time holding Space to reach full max_kick_speed

CHARGE_BAR_SIZE_PX = (120, 10)
CHARGE_BAR_MARGIN_PX = 10
CHARGE_BAR_BG_COLOR = (40, 40, 40)
CHARGE_BAR_FILL_COLOR = (230, 200, 60)


class _KickCharger:
    """Tracks a hold-to-charge, release-to-fire kick across frames."""

    def __init__(self) -> None:
        self.charge = 0.0  # 0..1, exposed for the HUD bar

    def update(self, space_held: bool, dt: float) -> None:
        if space_held:
            self.charge = min(1.0, self.charge + dt / KICK_CHARGE_SECONDS)

    def release(self) -> float:
        """Consume and reset the charge, returning the power fraction (0..1) to kick with."""
        charge, self.charge = self.charge, 0.0
        return charge


def _read_keyboard_action(config, controlled_player, kick_power: float | None) -> PlayerAction:
    keys = pygame.key.get_pressed()
    ax = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * config.player_max_accel
    ay = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * config.player_max_accel
    move = vec2(ax, ay)

    kick = None
    if kick_power is not None:
        # Direction comes from the player's facing (their last nonzero
        # movement direction — see Player.facing in state.py), not from
        # wherever the ball happens to be, so a kick can be a deliberate pass
        # in any direction rather than always aimed at the opponent's goal.
        kick = controlled_player.facing * (kick_power * config.max_kick_speed)

    return PlayerAction(move=move, kick=kick)


def _draw_kick_charge_bar(screen: pygame.Surface, charge: float) -> None:
    width, height = CHARGE_BAR_SIZE_PX
    x = CHARGE_BAR_MARGIN_PX
    y = screen.get_height() - height - CHARGE_BAR_MARGIN_PX
    pygame.draw.rect(screen, CHARGE_BAR_BG_COLOR, (x, y, width, height))
    pygame.draw.rect(screen, CHARGE_BAR_FILL_COLOR, (x, y, int(width * charge), height))


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

    team_0_ids = sorted(p.player_id for p in state.players if p.team == 0)
    controlled_index = 0
    kick_charger = _KickCharger()

    running = True
    while running:
        kick_power = None
        for pygame_event in pygame.event.get():
            is_quit = pygame_event.type == pygame.QUIT
            is_escape = pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_ESCAPE
            if is_quit or is_escape:
                running = False

            is_switch = pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_TAB
            if is_switch and team_0_ids:
                controlled_index = (controlled_index + 1) % len(team_0_ids)

            is_kick_release = (
                pygame_event.type == pygame.KEYUP and pygame_event.key == pygame.K_SPACE
            )
            if is_kick_release:
                kick_power = kick_charger.release()

        controlled_player_id = team_0_ids[controlled_index]
        controlled_player = next(p for p in state.players if p.player_id == controlled_player_id)

        keys = pygame.key.get_pressed()
        kick_charger.update(keys[pygame.K_SPACE], config.dt)

        actions = {
            controlled_player_id: _read_keyboard_action(config, controlled_player, kick_power)
        }

        state, events = step(state, actions, config)
        if any(event.type is EventType.GOAL for event in events):
            state = restart_after_goal(state, config)
        if writer is not None:
            writer.write_step(state, actions, events)

        draw_match_state(screen, state, config)
        _draw_kick_charge_bar(screen, kick_charger.charge)
        pygame.display.flip()
        clock.tick(round(1.0 / config.dt))

    if writer is not None:
        writer.close()
    pygame.quit()


if __name__ == "__main__":
    main()
