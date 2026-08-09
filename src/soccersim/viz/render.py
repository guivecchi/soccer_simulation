"""Pure `MatchState -> pixels` renderer, built with pygame.

Kept as a pure function of `(surface, state, config)` — no simulation logic
lives here, and nothing in `physics/` imports this module (see ROADMAP.md's
"dependency direction" rule: `physics` never imports from `viz`). That's what
lets the *same* drawing code serve two different callers: a live running sim
(`scripts/run_match.py`) and a replay loaded from disk
(`scripts/watch_replay.py`), and lets headless training (Stage 4+) skip
importing pygame entirely, since only `viz/` and the scripts that use it ever
touch this module.

## Coordinate transform: world meters -> screen pixels

Physics state lives in the "world" coordinate system described in
`physics/state.py`: origin at the pitch center, `+x` toward team 1's goal,
`+y` across the pitch, units in meters. Screen/pixel coordinates have their
origin at the top-left corner with `+y` pointing *down* — the opposite
vertical convention from a normal math coordinate system (a raster display is
drawn top row first, so "down the screen" is the natural "increasing"
direction). `world_to_screen()` is the one place that does both the axis flip
and the meters -> pixels scale, so every drawing helper below only ever deals
in already-flipped screen coordinates and never touches world units directly.
"""

from __future__ import annotations

import pygame

from soccersim.config import SimConfig
from soccersim.physics.state import MatchState

# Visual constants below are cosmetic choices, not physics — free to retune.
PIXELS_PER_METER = 9
MARGIN_PX = 30
HUD_HEIGHT_PX = 36
BALL_RADIUS_PX = 6
PLAYER_RADIUS_PX = 10
GOAL_DEPTH_M = 2.0  # how far the drawn goal box protrudes past the goal line

# How far the facing "nose" tip sits beyond the player's own radius, and how
# wide its base is — small enough not to read as a second player, big enough
# to see which way someone's oriented at a glance.
FACING_INDICATOR_LENGTH_PX = 7
FACING_INDICATOR_WIDTH_PX = 5

# How much bigger than the player the highlight halo is drawn (see
# _draw_players): the halo is a solid disc drawn *underneath* the player, so
# it only needs to be wide enough to peek out from behind them.
CONTROLLED_HIGHLIGHT_MARGIN_PX = 5

TEAM_COLORS = {0: (214, 69, 65), 1: (65, 105, 225)}
PITCH_COLOR = (33, 122, 62)
LINE_COLOR = (230, 230, 230)
BALL_COLOR = (245, 235, 90)
HUD_BG_COLOR = (18, 18, 18)
HUD_TEXT_COLOR = (235, 235, 235)
FACING_INDICATOR_COLOR = (255, 255, 255)
CONTROLLED_HIGHLIGHT_COLOR = (255, 215, 0)


def window_size(config: SimConfig) -> tuple[int, int]:
    """Pixel size of the window needed to fit the pitch, margins, and HUD."""
    width = round(config.pitch_length * PIXELS_PER_METER) + 2 * MARGIN_PX
    height = round(config.pitch_width * PIXELS_PER_METER) + 2 * MARGIN_PX + HUD_HEIGHT_PX
    return width, height


def world_to_screen(position, config: SimConfig) -> tuple[int, int]:
    """Map a world position (meters, origin at pitch center) to screen pixels."""
    x, y = position
    px = MARGIN_PX + (x + config.pitch_length / 2) * PIXELS_PER_METER
    py = HUD_HEIGHT_PX + MARGIN_PX + (config.pitch_width / 2 - y) * PIXELS_PER_METER
    return round(px), round(py)


def draw_match_state(
    surface: pygame.Surface,
    state: MatchState,
    config: SimConfig,
    controlled_player_id: int | None = None,
) -> None:
    """Draw one full frame: pitch markings, goals, players, ball, HUD.

    Pure with respect to `state`/`config` — the only side effect is drawing
    onto `surface`. Safe to call against a plain off-screen `pygame.Surface`
    (no display needed), which is what makes it testable headlessly.

    `controlled_player_id` is display-only overlay information, not part of
    `MatchState` — it's who a human is currently driving from the keyboard in
    `scripts/run_match.py`, which has no meaning during replay playback
    (`scripts/watch_replay.py` just leaves it as the default `None`).
    """
    surface.fill(PITCH_COLOR)
    _draw_pitch_markings(surface, config)
    _draw_goals(surface, config)
    _draw_players(surface, state, config, controlled_player_id)

    ball_center = world_to_screen(state.ball.position, config)
    pygame.draw.circle(surface, BALL_COLOR, ball_center, BALL_RADIUS_PX)

    _draw_hud(surface, state)


def _draw_players(
    surface: pygame.Surface,
    state: MatchState,
    config: SimConfig,
    controlled_player_id: int | None,
) -> None:
    for player in state.players:
        center = world_to_screen(player.position, config)

        if player.player_id == controlled_player_id:
            # A solid halo drawn *underneath* the smaller player circle —
            # simpler than an outline ring, and it can't visually clash with
            # the facing indicator drawn on top afterward.
            pygame.draw.circle(
                surface,
                CONTROLLED_HIGHLIGHT_COLOR,
                center,
                PLAYER_RADIUS_PX + CONTROLLED_HIGHLIGHT_MARGIN_PX,
            )

        pygame.draw.circle(surface, TEAM_COLORS[player.team], center, PLAYER_RADIUS_PX)
        pygame.draw.polygon(
            surface, FACING_INDICATOR_COLOR, _facing_indicator_points(center, player.facing)
        )


def _facing_indicator_points(center: tuple[int, int], facing) -> list[tuple[float, float]]:
    """Triangle "nose" pointing in `facing`'s direction, from the rim outward.

    `facing` is a world-space unit vector (`+y` = "up" the pitch); screen
    space has `+y` pointing *down*, so its y component is negated here — the
    same axis flip `world_to_screen()` applies to positions, just applied to
    a direction instead. The x component is unaffected by the flip.
    """
    dx, dy = float(facing[0]), -float(facing[1])
    # Perpendicular to (dx, dy), for the two base corners of the triangle.
    px, py = -dy, dx

    tip_x = center[0] + dx * (PLAYER_RADIUS_PX + FACING_INDICATOR_LENGTH_PX)
    tip_y = center[1] + dy * (PLAYER_RADIUS_PX + FACING_INDICATOR_LENGTH_PX)
    base_x = center[0] + dx * (PLAYER_RADIUS_PX - 2)
    base_y = center[1] + dy * (PLAYER_RADIUS_PX - 2)
    half_width = FACING_INDICATOR_WIDTH_PX / 2

    return [
        (tip_x, tip_y),
        (base_x + px * half_width, base_y + py * half_width),
        (base_x - px * half_width, base_y - py * half_width),
    ]


def _draw_pitch_markings(surface: pygame.Surface, config: SimConfig) -> None:
    half_length = config.pitch_length / 2
    half_width = config.pitch_width / 2

    top_left = world_to_screen((-half_length, half_width), config)
    bottom_right = world_to_screen((half_length, -half_width), config)
    pitch_rect = pygame.Rect(
        top_left, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    )
    pygame.draw.rect(surface, LINE_COLOR, pitch_rect, width=2)

    halfway_top = world_to_screen((0.0, half_width), config)
    halfway_bottom = world_to_screen((0.0, -half_width), config)
    pygame.draw.line(surface, LINE_COLOR, halfway_top, halfway_bottom, width=2)

    center = world_to_screen((0.0, 0.0), config)
    center_circle_radius_px = round(9.15 * PIXELS_PER_METER)  # FIFA center-circle radius
    pygame.draw.circle(surface, LINE_COLOR, center, center_circle_radius_px, width=2)


def _draw_goals(surface: pygame.Surface, config: SimConfig) -> None:
    half_length = config.pitch_length / 2
    goal_half_width = config.goal_width / 2

    for goal_line_x, outward_x in ((-half_length, -1.0), (half_length, 1.0)):
        near_top = world_to_screen((goal_line_x, goal_half_width), config)
        far_bottom = world_to_screen(
            (goal_line_x + outward_x * GOAL_DEPTH_M, -goal_half_width), config
        )
        goal_rect = pygame.Rect(
            min(near_top[0], far_bottom[0]),
            near_top[1],
            abs(far_bottom[0] - near_top[0]),
            far_bottom[1] - near_top[1],
        )
        pygame.draw.rect(surface, LINE_COLOR, goal_rect, width=2)


def _draw_hud(surface: pygame.Surface, state: MatchState) -> None:
    hud_rect = pygame.Rect(0, 0, surface.get_width(), HUD_HEIGHT_PX)
    pygame.draw.rect(surface, HUD_BG_COLOR, hud_rect)

    # Not cached across calls: a `pygame.font.Font` is tied to the font
    # subsystem's current init state, and this module has no hook into when
    # a caller re-runs `pygame.init()`/`pygame.quit()` (e.g. between tests) —
    # holding onto a stale Font past a `quit()` raises "Text has zero width"
    # rather than a clear error, so it's simpler and more robust to just
    # build a fresh one per frame. HUD text is a handful of characters, so
    # the cost is negligible next to the per-frame drawing above.
    font = pygame.font.SysFont("consolas", HUD_HEIGHT_PX - 12)
    text = f"{state.score[0]:>2} - {state.score[1]:<2}    t = {state.time:6.2f}s"
    text_surface = font.render(text, True, HUD_TEXT_COLOR)
    surface.blit(text_surface, (MARGIN_PX, (HUD_HEIGHT_PX - text_surface.get_height()) // 2))
