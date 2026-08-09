from soccersim.config import SimConfig
from soccersim.physics.reset import build_kickoff_state, restart_after_goal
from soccersim.physics.state import Ball, MatchState, Player
from soccersim.physics.step import step
from soccersim.physics.vector import vec2


def test_kickoff_state_is_well_formed():
    config = SimConfig(players_per_team=5)
    state = build_kickoff_state(config)

    assert len(state.players) == 2 * config.players_per_team
    assert state.score == (0, 0)
    assert list(state.ball.position) == [0.0, 0.0]

    team_0_ids = {p.player_id for p in state.players if p.team == 0}
    team_1_ids = {p.player_id for p in state.players if p.team == 1}
    assert len(team_0_ids) == config.players_per_team
    assert len(team_1_ids) == config.players_per_team
    assert team_0_ids.isdisjoint(team_1_ids)  # no duplicate player_ids across teams

    for player in state.players:
        assert -config.pitch_width / 2 <= player.position[1] <= config.pitch_width / 2


def test_kickoff_state_is_steppable():
    """Cheap end-to-end smoke test: a freshly built match should survive a
    few steps with no actions at all, with no crashes or NaNs.
    """
    config = SimConfig()
    state = build_kickoff_state(config)

    for _ in range(10):
        state, _ = step(state, {}, config)

    assert state.time > 0.0


def test_restart_after_goal_resets_ball_and_players_but_keeps_time_and_score():
    """A restart continues the same match — only positions/velocities reset."""
    config = SimConfig(players_per_team=3)
    kickoff = build_kickoff_state(config)

    # Simulate "mid-match, just after a goal": ball is dead in the goal, one
    # team is ahead, players have wandered from their kickoff formation.
    mid_match_state = MatchState(
        time=123.4,
        ball=Ball(position=vec2(config.pitch_length / 2, 0.0), velocity=vec2(0.0, 0.0)),
        players=[
            Player(p.player_id, p.team, position=vec2(1.0, 1.0), velocity=vec2(3.0, -3.0))
            for p in kickoff.players
        ],
        score=(2, 1),
    )

    restarted = restart_after_goal(mid_match_state, config)

    assert restarted.time == mid_match_state.time
    assert restarted.score == mid_match_state.score
    assert list(restarted.ball.position) == [0.0, 0.0]
    assert list(restarted.ball.velocity) == [0.0, 0.0]
    # Player positions/velocities go back to the same formation build_kickoff_state uses.
    for restarted_player, kickoff_player in zip(restarted.players, kickoff.players):
        assert list(restarted_player.position) == list(kickoff_player.position)
        assert list(restarted_player.velocity) == [0.0, 0.0]


def test_restart_after_goal_is_steppable():
    """A restarted match shouldn't produce a state step() then chokes on."""
    config = SimConfig(players_per_team=2)
    state = build_kickoff_state(config)
    state = restart_after_goal(state, config)

    state, _ = step(state, {}, config)

    assert state.time > 0.0
