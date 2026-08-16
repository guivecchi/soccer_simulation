"""`ScriptedAgent`: deterministic heuristics for chase-ball, mark-nearest,
pass-to-open-teammate, and basic keeper play.

One stateless class, dispatching purely on `Role.kind` (see `agents/base.py`)
— there's no per-player memory carried between steps; every decision is
recomputed fresh from the current `MatchState`, the same way `step()` itself
has no hidden state. This means a single `ScriptedAgent` instance can (and,
in `scripts/run_match.py`, does) drive every scripted player on the pitch.

The steering primitive throughout is "seek a target point with kinematic
arrival braking" (`_move_toward`): full-speed pursuit while far away, then a
smooth slowdown timed to arrive at (rather than blast through) the target.
An earlier version of this file always requested full `player_max_accel`
straight at the target with no slowdown at all — every role's target
settled into a slow, decaying back-and-forth oscillation around the point it
was aiming for (overshoot, hard-brake, overshoot the other way, repeat) as
soon as a player got close, since there was nothing to stop them blasting
straight through it every time. See `_move_toward`'s docstring for the fix.
"""

from __future__ import annotations

import math

from soccersim.agents.base import Role, RoleKind
from soccersim.config import SimConfig
from soccersim.physics.state import MatchState, Player, PlayerAction
from soccersim.physics.vector import Vec2, dot, magnitude, normalize, vec2

# --- Tunable heuristic constants ---------------------------------------------
#
# These aren't physics (nothing in `config.SimConfig` needs them; the kernel
# doesn't know "shooting range" is a concept) — they're parameters of the
# *scripted decision-making* in this file, kept here rather than in
# `SimConfig` so they can be tuned freely without touching the physics
# kernel's public config surface.

SHOOT_RANGE_M = 22.0  # distance to goal center within which the chaser considers shooting
SHOOT_MIN_CLEARANCE_M = 2.0  # min lane clearance (see _lane_clearance) required to actually shoot
GOAL_AIM_INSET_FRACTION = 0.8  # aim within the goal mouth, not right at the post
DRIBBLE_LOOKAHEAD_M = 15.0  # how far "ahead" the dribble baseline lane is measured
PASS_KICK_SPEED_FRACTION = 0.6  # fraction of max_kick_speed used for passes (vs. full for shots)

KEEPER_LINE_DEPTH_M = 2.0  # how far off their own goal line the keeper stands by default
KEEPER_DANGER_RANGE_M = 12.0  # ball-to-goal-line distance within which the keeper advances
KEEPER_DANGER_Y_MARGIN_M = (
    3.0  # extra width beyond the goal mouth that still counts as "in front of goal"
)
KEEPER_MAX_ADVANCE_M = 5.0  # furthest the keeper will come off their line
KEEPER_CLEARANCE_SPEED_FRACTION = (
    0.7  # fraction of max_kick_speed used when the keeper clears the ball
)

MARKER_GOAL_SIDE_OFFSET_M = 2.0  # how far goal-side of their mark a marker positions themselves
SUPPORT_ATTACK_THIRD_MARGIN_M = (
    15.0  # how far infield of the attacking goal line support players hold
)

# Not a gameplay constant like the ones above — a numerical dead-zone for
# `_move_toward`'s arrival behavior. Right at the target, "the direction to
# steer in" is barely defined (dividing by a near-zero distance), which left
# uncorrected produces a tiny persistent velocity chatter rather than
# settling to a true stop. Below this distance, stop outright instead of
# computing a noisy unit direction toward a point this close.
ARRIVAL_DEADZONE_M = 0.05


class ScriptedAgent:
    """A rule-based `Agent` (see `agents/base.py`) usable for any player/role."""

    def act(self, player_id: int, state: MatchState, role: Role, config: SimConfig) -> PlayerAction:
        player = _find_player(state.players, player_id)

        if role.kind is RoleKind.KEEPER:
            return self._act_keeper(player, state, config)
        if role.kind is RoleKind.CHASER:
            return self._act_chaser(player, state, config)
        if role.kind is RoleKind.MARKER:
            return self._act_marker(player, role, state, config)
        return self._act_support(player, state, config)

    def _act_keeper(self, player: Player, state: MatchState, config: SimConfig) -> PlayerAction:
        half_length = config.pitch_length / 2
        goal_half_width = config.goal_width / 2
        own_line_x = -half_length if player.team == 0 else half_length
        inward = 1.0 if player.team == 0 else -1.0  # own goal line -> field center

        ball_x, ball_y = state.ball.position
        distance_to_line = abs(ball_x - own_line_x)
        base_x = own_line_x + inward * KEEPER_LINE_DEPTH_M

        is_danger = (
            distance_to_line < KEEPER_DANGER_RANGE_M
            and abs(ball_y) < goal_half_width + KEEPER_DANGER_Y_MARGIN_M
        )
        if is_danger:
            advance = KEEPER_MAX_ADVANCE_M * (1.0 - distance_to_line / KEEPER_DANGER_RANGE_M)
            target_x = base_x + inward * advance
        else:
            target_x = base_x

        target_y = min(max(ball_y, -goal_half_width), goal_half_width)
        move = _move_toward(player, vec2(target_x, target_y), config)

        kick = None
        if state.ball.carrier_id == player.player_id:
            # A ball that's rolled to the keeper gets cleared upfield, aimed
            # at the center of the pitch rather than dribbled — a keeper
            # holding onto the ball indefinitely isn't useful scripted play.
            clear_target = vec2(0.0, 0.0)
            kick = normalize(clear_target - player.position) * (
                config.max_kick_speed * KEEPER_CLEARANCE_SPEED_FRACTION
            )

        return PlayerAction(move=move, kick=kick)

    def _act_chaser(self, player: Player, state: MatchState, config: SimConfig) -> PlayerAction:
        if state.ball.carrier_id != player.player_id:
            move = _move_toward(player, state.ball.position, config)
            return PlayerAction(move=move, kick=None)
        return self._act_with_ball(player, state, config)

    def _act_with_ball(self, player: Player, state: MatchState, config: SimConfig) -> PlayerAction:
        half_length = config.pitch_length / 2
        goal_half_width = config.goal_width / 2
        n = config.players_per_team
        attack_dir = 1.0 if player.team == 0 else -1.0
        goal_x = half_length * attack_dir

        teammates = [
            p for p in state.players if p.team == player.team and p.player_id != player.player_id
        ]
        opponents = [p for p in state.players if p.team != player.team]

        goal_center = vec2(goal_x, 0.0)
        distance_to_goal = magnitude(goal_center - player.position)
        shot_clearance = _lane_clearance(player.position, goal_center, opponents)

        if distance_to_goal <= SHOOT_RANGE_M and shot_clearance >= SHOOT_MIN_CLEARANCE_M:
            opponent_keeper_id = (1 - player.team) * n
            opponent_keeper = _find_player(state.players, opponent_keeper_id)
            aim_point = _best_goal_aim_point(goal_x, goal_half_width, opponent_keeper.position[1])
            kick = normalize(aim_point - player.position) * config.max_kick_speed
            return PlayerAction(move=vec2(0.0, 0.0), kick=kick)

        forward_point = player.position + vec2(attack_dir * DRIBBLE_LOOKAHEAD_M, 0.0)
        dribble_clearance = _lane_clearance(player.position, forward_point, opponents)

        best_pass = _best_passing_lane(player.position, teammates, opponents)
        if best_pass is not None and best_pass[1] > dribble_clearance:
            teammate, _ = best_pass
            kick = normalize(teammate.position - player.position) * (
                config.max_kick_speed * PASS_KICK_SPEED_FRACTION
            )
            return PlayerAction(move=vec2(0.0, 0.0), kick=kick)

        move = _move_toward(player, forward_point, config)
        return PlayerAction(move=move, kick=None)

    def _act_marker(
        self, player: Player, role: Role, state: MatchState, config: SimConfig
    ) -> PlayerAction:
        half_length = config.pitch_length / 2
        own_goal_x = -half_length if player.team == 0 else half_length

        if role.mark_target_id is None:
            # Uneven rosters only (see `agents/roles.py::_assign_markers`):
            # no specific mark to shadow, so just hold a defensive position.
            target = vec2(own_goal_x * 0.5, player.position[1])
        else:
            mark = _find_player(state.players, role.mark_target_id)
            toward_own_goal = normalize(vec2(own_goal_x, 0.0) - mark.position)
            target = mark.position + toward_own_goal * MARKER_GOAL_SIDE_OFFSET_M

        move = _move_toward(player, target, config)
        return PlayerAction(move=move, kick=None)

    def _act_support(self, player: Player, state: MatchState, config: SimConfig) -> PlayerAction:
        half_length = config.pitch_length / 2
        attack_dir = 1.0 if player.team == 0 else -1.0
        target_x = attack_dir * (half_length - SUPPORT_ATTACK_THIRD_MARGIN_M)
        target = vec2(target_x, player.position[1])
        move = _move_toward(player, target, config)
        return PlayerAction(move=move, kick=None)


def _move_toward(player: Player, target: Vec2, config: SimConfig) -> Vec2:
    """Seek `target`, braking to arrive rather than blasting through it.

    The desired speed is capped by `sqrt(2 * player_brake_accel * distance)`
    — the standard kinematic relation for "how fast could I be going right
    now and still brake to a dead stop exactly `distance` away, using the
    braking rate the physics kernel already enforces (`player_brake_accel`,
    see `_step_player` in physics/step.py)." Far from the target this cap
    exceeds `player_max_speed` and has no effect at all (full-speed pursuit,
    same as a plain "accelerate toward the point"); only once the player is
    close enough that their current speed would overshoot does it start
    pulling the desired speed down, smoothly, all the way to zero at the
    target itself. That's what makes a player settle at a role's target
    point instead of the overshoot-brake-overshoot oscillation a constant
    full-acceleration pursuit produces once you're close enough to reach the
    target at max speed — see the module docstring.

    Returns a raw, unclipped acceleration (`(desired_velocity - velocity) /
    dt`); `physics/step.py::_step_player` still does the actual clipping to
    `player_max_accel`/`player_brake_accel` depending on whether this
    opposes the player's current velocity, exactly as it does for any other
    action.
    """
    direction = target - player.position
    distance = magnitude(direction)
    if distance < ARRIVAL_DEADZONE_M:
        return -player.velocity / config.dt

    max_approach_speed = math.sqrt(2 * config.player_brake_accel * distance)
    desired_speed = min(config.player_max_speed, max_approach_speed)
    desired_velocity = (direction / distance) * desired_speed

    return (desired_velocity - player.velocity) / config.dt


def _find_player(players: list[Player], player_id: int) -> Player:
    return next(p for p in players if p.player_id == player_id)


def _point_to_segment_distance(point: Vec2, seg_a: Vec2, seg_b: Vec2) -> float:
    """Shortest distance from `point` to the segment `seg_a -> seg_b`.

    Standard projection formula: project `point` onto the *line* through
    `seg_a`/`seg_b`, clamp the projection parameter to `[0, 1]` so it can't
    land beyond either endpoint, then measure the distance to that clamped
    point. See docs/stages/stage3-concepts.md's "Passing as a geometry
    problem" for why this (rather than a flat distance threshold) is what
    decides whether a passing lane is actually obstructed.
    """
    segment = seg_b - seg_a
    segment_length_sq = dot(segment, segment)
    if segment_length_sq < 1e-9:
        return magnitude(point - seg_a)

    t = dot(point - seg_a, segment) / segment_length_sq
    t = min(max(t, 0.0), 1.0)
    closest_point = seg_a + segment * t
    return magnitude(point - closest_point)


def _lane_clearance(start: Vec2, end: Vec2, opponents: list[Player]) -> float:
    """How much room a straight line from `start` to `end` has from `opponents`.

    The minimum, over every opponent, of their distance to the segment — the
    single closest opponent to the lane is what determines whether it's
    actually usable.
    """
    if not opponents:
        return float("inf")
    return min(_point_to_segment_distance(o.position, start, end) for o in opponents)


def _best_passing_lane(
    carrier_position: Vec2, teammates: list[Player], opponents: list[Player]
) -> tuple[Player, float] | None:
    """The teammate whose passing lane from `carrier_position` is least obstructed."""
    if not teammates:
        return None
    return max(
        ((mate, _lane_clearance(carrier_position, mate.position, opponents)) for mate in teammates),
        key=lambda pair: pair[1],
    )


def _best_goal_aim_point(goal_x: float, goal_half_width: float, keeper_y: float) -> Vec2:
    """Aim at whichever side of the goal mouth is farther from the keeper.

    Inset by `GOAL_AIM_INSET_FRACTION` so the aim point is comfortably inside
    the goal mouth rather than right on a post — the same "maximize
    clearance" idea as passing-lane selection, applied to the goal mouth
    instead of a teammate.
    """
    near_post_y = goal_half_width * GOAL_AIM_INSET_FRACTION
    far_post_y = -near_post_y
    aim_y = near_post_y if abs(near_post_y - keeper_y) > abs(far_post_y - keeper_y) else far_post_y
    return vec2(goal_x, aim_y)
