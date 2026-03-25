# config.py
import yaml
from enum import Enum
from pathlib import Path

class EncoderType(Enum):
    CANONICAL = "canonical"
    ICLR22 = "iclr22"

class ExperimentType(Enum):
    NODE_CLASSIFICATION = "node_classification"
    LINK_PREDICTION = "link_prediction"

class AggregationType(Enum):
    MAX_MAX = "max-max"
    MAX_SUM = "max-sum"
    SUM_MAX = "sum-max"
    SUM_SUM = "sum-sum"

class ExperimentConfig:

    # Folder with the specific dataset
    data_dir: Path
    # Folder with all the experiment configuration and results
    exp_dir: Path
    # Type of experiment (node classification or link prediction)
    experiment_type: ExperimentType
    # Encoding/decoding scheme (canonical or iclr22)
    encoding_scheme: EncoderType
    # Aggregation functions
    agg_function: AggregationType
    # Model threshold for derivation; must be between 0 and 1
    derivation_threshold: float
    # Use dummy nodes during training (this is a training optimisation that sometimes helps)
    use_dummies: bool
    # Clamp weights whose absolute value is smaller than this to 0.
    clamping: float
    # Use only non-negative weights in the model's matrices.
    non_negative_weights: bool

    # Which operations should be done in this experiment?
    train: bool
    valid: bool
    test: bool
    explain: bool
    minimal_rule: bool

    def __init__(self, config_path: str):

        with open(config_path) as f:
            data = yaml.safe_load(f)

        path = Path(data["paths"]["data_dir"])
        if not path.exists():
            raise ValueError(f"data folder does not exist: {data['paths']['data_dir']!r}")
        if not path.is_dir():
            raise ValueError(f"data path is not a folder: {data['paths']['data_dir']!r}")
        self.data_dir = path

        path = Path(data["paths"]["exp_dir"])
        if not path.exists():
            raise ValueError(f"experiment folder does not exist: {data['paths']['exp_dir']!r}")
        if not path.is_dir():
            raise ValueError(f"experiment path is not a folder: {data['paths']['exp_dir']!r}")
        self.exp_dir = path

        try:
            self.experiment_type = ExperimentType(data["experiment"]["type"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in ExperimentType)
            raise ValueError(f"experiment type not valid: please choose {valid}")

        try:
            self.encoding_scheme = EncoderType(data["experiment"]["encoding_scheme"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in EncoderType)
            raise ValueError(f"encoder type not valid: please choose {valid}")

        try:
            self.agg_function = AggregationType(data["experiment"]["aggregation"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in AggregationType)
            raise ValueError(f"aggregation function not valid: please choose {valid}")

        try:
            self.derivation_threshold = float(data["experiment"]["derivation_threshold"])
        except ValueError:
            raise ValueError(f"threshold value must be a float, got {data['experiment']['derivation_threshold']!r}")
        if not 0 <= self.derivation_threshold <= 1:
            raise ValueError(f"threshold value must be between 0 and 1, got {self.derivation_threshold!r}")

        try:
            self.clamping = float(data["experiment"]["clamping"])
        except ValueError:
            raise ValueError(f"clamping value must be a float, got {data['experiment']['clamping']!r}")
        if self.clamping < 0:
            raise ValueError(f"clamping value must be non-negative, got {self.clamping!r}")

        self.use_dummies = data["experiment"]["use_dummy_constants"]
        self.non_negative_weights = data["experiment"]["non_negative_weights"]
        self.train = data["run"]["train"]
        self.valid = data["run"]["valid"]
        self.test = data["run"]["test"]
        self.explain = data["run"]["explain"]
        self.minimal_rule = data["run"]["minimal_rule"]

