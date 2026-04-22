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
    MAX = "max"
    SUM = "sum"

class ExperimentConfig:

    # Folder with the specific dataset
    data_dir: Path
    # Folder where you store the experiment folders
    exp_dir: Path
    # Type of experiment (node classification or link prediction)
    experiment_type: ExperimentType
    # Encoding/decoding scheme (canonical or iclr22)
    encoding_scheme: EncoderType
    # Aggregation functions for layer 1
    agg_function_1: AggregationType
    # Aggregation functions for layer 2
    agg_function_2: AggregationType
    # Model threshold for derivation; must be between 0 and 1
    derivation_threshold: float
    # Use dummy nodes during training (this is a training optimisation that sometimes helps)
    use_dummies: bool
    # Clamp weights whose absolute value is smaller than this to 0.
    clamping: float
    # Use only non-negative weights in the model's matrices.
    non_negative_weights: bool

    def get_exp_dir(self) -> Path:
        return self.exp_dir

    def get_data_dir(self) -> Path:
        return self.data_dir

    def __init__(self, config_path: str):

        with open(config_path) as f:
            data = yaml.safe_load(f)

        data_path = Path(data["data_dir"])
        if not data_path.is_dir():
            raise ValueError(f"data path is not an existing folder: {data_path}")
        self.data_dir = data_path

        exp_path = Path(data["exp_dir"])
        if not exp_path.is_dir():
            raise ValueError(f"experiment path is not an existing folder: {exp_path}")
        self.exp_dir = exp_path

        try:
            self.experiment_type = ExperimentType(data["type"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in ExperimentType)
            raise ValueError(f"experiment type not valid: please choose {valid}")

        try:
            self.encoding_scheme = EncoderType(data["encoding_scheme"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in EncoderType)
            raise ValueError(f"encoder type not valid: please choose {valid}")

        try:
            self.agg_function_1 = AggregationType(data["aggregation_1"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in AggregationType)
            raise ValueError(f"aggregation function not valid: please choose {valid}")

        try:
            self.agg_function_2 = AggregationType(data["aggregation_2"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in AggregationType)
            raise ValueError(f"aggregation function not valid: please choose {valid}")

        try:
            self.derivation_threshold = float(data["derivation_threshold"])
        except ValueError:
            raise ValueError(f"threshold value must be a float, got {data['derivation_threshold']!r}")
        if not 0 <= self.derivation_threshold <= 1:
            raise ValueError(f"threshold value must be between 0 and 1, got {self.derivation_threshold!r}")

        try:
            self.clamping = float(data["clamping"])
        except ValueError:
            raise ValueError(f"clamping value must be a float, got {data['clamping']!r}")
        if self.clamping < 0:
            raise ValueError(f"clamping value must be non-negative, got {self.clamping!r}")

        self.use_dummies = data["use_dummy_constants"]
        self.non_negative_weights = data["non_negative_weights"]

