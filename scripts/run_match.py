"""Run a live match in a pygame window; optionally record it to a replay file.

As of Stage 3, every player except one keyboard-controlled debug player is
driven by `agents.scripted.ScriptedAgent` — two full scripted teams actually
play each other, with the human slot free to jump in (or, via `--headless`,
absent entirely) to eyeball the physics kernel or the scripted decision-making
in real time. Role assignment (`agents.roles.assign_roles`) runs once per
step, before any agent acts, so every scripted player's decision is
consistent with the rest of its team that same step.

Controls (ignored entirely with `--headless`):
    Arrow keys   accelerate the controlled player (braking/reversing is
                 quicker than accelerating from rest — see player_brake_accel
                 in config.py); release all of them and the player actively
                 brakes to a stop rather than coasting (see
                 `_read_keyboard_action`) — players have no passive friction
                 in the physics kernel itself (only the ball does), so
                 without this the controlled player would just glide
                 forever at whatever speed they had
    Space        hold to charge a kick, release to fire it in whichever
                 direction the player is currently facing, at a strength
                 proportional to how long Space was held (only has an effect
                 if the controlled player currently has possession)
    Tab          switch which of team 0's players is controlled
    Esc / close  quit

Usage:
    uv run python scripts/run_match.py
    uv run python scripts/run_match.py --record replays/demo.jsonl
    uv run python scripts/run_match.py --headless --record replays/demo.jsonl
"""

from __future__ import annotations

import argparse

import pygame

from soccersim.agents.roles import assign_roles
from soccersim.agents.scripted import ScriptedAgent
from soccersim.config import load_config
from soccersim.physics.events import EventType
from soccersim.physics.reset import (
    build_kickoff_state,
    restart_after_goal,
    restart_after_out_of_bounds,
)
from soccersim.physics.state import PlayerAction
from soccersim.physics.step import step
from soccersim.physics.vector import magnitude, normalize, vec2
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


# Below this speed, cancel the controlled player's residual velocity
# outright instead of braking toward zero. This can't be an arbitrary small
# constant: braking always removes `player_brake_accel * dt` of speed in one
# step (see `_read_keyboard_action`), so a threshold smaller than that would
# let braking overshoot straight past zero and into the opposite direction
# every single frame once the residual speed drops below it — a tight,
# persistent jitter right at zero, not a clean stop (this was caught by
# actually simulating idle deceleration, not by inspection). A function
# rather than a module constant since it depends on `config`.
def _stop_deadzone_mps(config) -> float:
    return config.player_brake_accel * config.dt


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

    if magnitude(move) < 1e-9:
        # No directional key held. `step()`'s braking only applies when the
        # requested acceleration actively *opposes* the current velocity
        # (see physics/step.py's `is_braking`) — a zero request doesn't
        # qualify, and players have no passive friction of their own (only
        # the ball does), so without this the player would just coast
        # forever at whatever speed they had. Actively brake to a stop
        # instead, at the same player_brake_accel the kernel already applies
        # whenever you steer against your own motion.
        speed = magnitude(controlled_player.velocity)
        if speed < _stop_deadzone_mps(config):
            move = -controlled_player.velocity / config.dt  # cancel outright this step
        else:
            move = -normalize(controlled_player.velocity) * config.player_brake_accel

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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run with no window/keyboard control — every player is scripted",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many steps (default: unlimited with a window, 3600 [~60s] headless)",
    )
    args = parser.parse_args()
    max_steps = args.max_steps if args.max_steps is not None else (3600 if args.headless else None)

    config = load_config(args.config)
    state = build_kickoff_state(config)
    agent = ScriptedAgent()

    writer = ReplayWriter(args.record, config) if args.record else None
    if writer is not None:
        writer.write_step(state, {}, [])

    screen = None
    clock = None
    controlled_player_id = None
    controlled_index = 0
    kick_charger = _KickCharger()
    team_0_ids = sorted(p.player_id for p in state.players if p.team == 0)

    if not args.headless:
        pygame.init()
        screen = pygame.display.set_mode(window_size(config))
        pygame.display.set_caption("Soccer Simulation — live match")
        clock = pygame.time.Clock()
        controlled_player_id = team_0_ids[controlled_index]

    running = True
    step_count = 0
    while running:
        kick_power = None
        if not args.headless:
            for pygame_event in pygame.event.get():
                is_quit = pygame_event.type == pygame.QUIT
                is_escape = (
                    pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_ESCAPE
                )
                if is_quit or is_escape:
                    running = False

                is_switch = pygame_event.type == pygame.KEYDOWN and pygame_event.key == pygame.K_TAB
                if is_switch and team_0_ids:
                    controlled_index = (controlled_index + 1) % len(team_0_ids)
                    controlled_player_id = team_0_ids[controlled_index]

                is_kick_release = (
                    pygame_event.type == pygame.KEYUP and pygame_event.key == pygame.K_SPACE
                )
                if is_kick_release:
                    kick_power = kick_charger.release()

            keys = pygame.key.get_pressed()
            kick_charger.update(keys[pygame.K_SPACE], config.dt)

        # Role assignment runs once per step, before any agent acts, so every
        # scripted player's decision this step is consistent with the rest
        # of its team — see agents/roles.py.
        roles = assign_roles(state, config)
        actions = {
            player.player_id: agent.act(player.player_id, state, roles[player.player_id], config)
            for player in state.players
        }
        if controlled_player_id is not None:
            controlled_player = next(
                p for p in state.players if p.player_id == controlled_player_id
            )
            actions[controlled_player_id] = _read_keyboard_action(
                config, controlled_player, kick_power
            )

        state, events = step(state, actions, config)
        for event in events:
            if event.type is EventType.GOAL:
                state = restart_after_goal(state, config)
            elif event.type is EventType.OUT_OF_BOUNDS:
                state = restart_after_out_of_bounds(state, event, config)
        if writer is not None:
            writer.write_step(state, actions, events)

        if not args.headless:
            draw_match_state(screen, state, config, controlled_player_id=controlled_player_id)
            _draw_kick_charge_bar(screen, kick_charger.charge)
            pygame.display.flip()
            clock.tick(round(1.0 / config.dt))

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            running = False

    if writer is not None:
        writer.close()
    if not args.headless:
        pygame.quit()


if __name__ == "__main__":
    main()
