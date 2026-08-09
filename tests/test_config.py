from soccersim.config import SimConfig, load_config


def test_defaults():
    cfg = load_config()
    assert cfg == SimConfig()


def test_yaml_override(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("pitch_length: 50.0\nseed: 42\n")

    cfg = load_config(override_path)

    assert cfg.pitch_length == 50.0
    assert cfg.seed == 42
    assert cfg.pitch_width == SimConfig().pitch_width


def test_unknown_field_raises(tmp_path):
    override_path = tmp_path / "override.yaml"
    override_path.write_text("not_a_real_field: 1\n")

    try:
        load_config(override_path)
        assert False, "expected ValueError"
    except ValueError:
        pass
