from soccersim.config import SimConfig
from soccersim.physics.reset import build_kickoff_state
from soccersim.physics.step import step


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
