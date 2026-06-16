import yaml
import pytest
from pathlib import Path

from src.config.config import ExperimentConfig, EncoderType, AggregationType

# Helper function that writes a YAML config file for certain given parameters
def write_config(tmp_path, **overrides):
    data_dir = tmp_path / "data"
    exp_dir = tmp_path / "experiments"

    data_dir.mkdir()
    exp_dir.mkdir()

    # Default values
    config = {
        "data_dir": str(data_dir),
        "exp_dir": str(exp_dir),
        "encoding_scheme": "canonical",
        "aggregation_1": "max",
        "aggregation_2": "sum",
        "derivation_threshold": 0.5,
        "clamping": 0.1,
        "use_dummy_constants": True,
        "non_negative_weights": False,
    }

    # Overwrite default values with any extra given arguments
    config.update(overrides)

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    return config_file


def test_valid_config_loads(tmp_path):
    config_file = write_config(tmp_path)

    cfg = ExperimentConfig(str(config_file))

    assert cfg.encoding_scheme == EncoderType.CANONICAL
    assert cfg.agg_function_1 == AggregationType.MAX
    assert cfg.agg_function_2 == AggregationType.SUM
    assert cfg.derivation_threshold == 0.5
    assert cfg.clamping == 0.1
    assert cfg.use_dummies is True
    assert cfg.non_negative_weights is False


def test_invalid_data_dir_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        data_dir=str(tmp_path / "missing_directory"),
    )

    with pytest.raises(ValueError, match="data path"):
        ExperimentConfig(str(config_file))


def test_invalid_encoder_type_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        encoding_scheme="invalid_encoder",
    )

    with pytest.raises(ValueError, match="encoder type not valid"):
        ExperimentConfig(str(config_file))


def test_invalid_aggregation_type_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        aggregation_1="average",
    )

    with pytest.raises(ValueError, match="aggregation function not valid"):
        ExperimentConfig(str(config_file))


def test_threshold_out_of_range_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        derivation_threshold=1.5,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        ExperimentConfig(str(config_file))


def test_threshold_not_float_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        derivation_threshold="not_a_number",
    )

    with pytest.raises(ValueError, match="threshold value must be a float"):
        ExperimentConfig(str(config_file))


def test_negative_clamping_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        clamping=-0.5,
    )

    with pytest.raises(ValueError, match="clamping value must be non-negative"):
        ExperimentConfig(str(config_file))