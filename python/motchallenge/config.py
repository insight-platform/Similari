from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf, MISSING


class TrackerType(Enum):
    """Supported tracker types."""

    OriginalSort = 0
    Sort = 1
    # VisualSort = 2


@dataclass
class Tracker:
    """Base tracker configuration (without parameters specification)."""

    type: TrackerType
    params: Dict[str, Any]


@dataclass
class OriginalSortParams:
    """Original Sort tracker parameters.
    https://github.com/abewley/sort
    """

    max_age: int = 1
    """Maximum number of frames to keep alive a track without associated detections."""

    min_hits: int = 3
    """Minimum number of associated detections before track is initialised."""

    iou_threshold: float = 0.3
    """Minimum IOU for match."""


class PositionalMetricType(Enum):
    """Positional metric type."""

    IoU = 0
    Maha = 1


@dataclass
class PositionalMetric:
    """Positional metric configuration."""

    type: PositionalMetricType
    threshold: float = 0.3


@dataclass
class SortParams:
    """Sort tracker parameters."""

    shards: int = 4
    """Amount of cpu threads to process the data, 
    keep 1 for up to 100 simultaneously tracked objects, 
    try it before setting high - higher numbers may lead to unexpected latencies.
    """

    bbox_history: int = 10
    """How many last bboxes are kept within stored track 
    (valuable for offline trackers), for online - keep 1    
    """

    max_idle_epochs: int = 10
    """How long track survives without being updated."""

    positional_metric: PositionalMetric = PositionalMetric(PositionalMetricType.IoU)
    """Setting the positional metric used by a tracker. 
    Two positional metrics are supported: IoU and Mahalanobis.
    """

    spatio_temporal_constraints: Optional[List[List]] = None
    """Defining the constraints for objects compared across different epochs.
    https://docs.rs/similari/latest/similari/trackers/spatio_temporal_constraints/struct.SpatioTemporalConstraints.html
    """

    use_confidence: bool = False
    """Whether to use bounding box confidences."""


@dataclass
class VisualSortParams:
    """Visual Sort tracker parameters.
    TODO
    """


@dataclass
class Evaluator:
    """Evaluator configuration."""

    num_cores: int = 1
    """Number of cores to use."""


@dataclass
class ConfigSchema:
    """Configuration schema."""

    name: str
    data_path: str
    output_path: str
    tracker: Tracker
    evaluator: Evaluator = Evaluator()


@dataclass
class Config(ConfigSchema):
    """Configuration object."""

    tracker: Union[OriginalSortParams, SortParams, VisualSortParams] = MISSING


class ConfigException(Exception):
    """Configuration exception."""


def load_config(config_file_path: str) -> Config:
    """Loads, pareses and validate specified configuration file."""
    config = OmegaConf.unsafe_merge(ConfigSchema, OmegaConf.load(config_file_path))

    tracker_params_schema = SortParams
    if config.tracker.type == TrackerType.OriginalSort:
        tracker_params_schema = OriginalSortParams
    # elif config.toml.tracker.type == TrackerType.VisualSort:
    #     tracker_params_schema = VisualSortParams
    tracker_params = OmegaConf.to_object(
        OmegaConf.unsafe_merge(
            tracker_params_schema, config.tracker.params
        )
    )

    print(f'Configuration:\n{OmegaConf.to_yaml(config)}')

    return Config(
        name=config.name,
        data_path=config.data_path,
        output_path=config.output_path,
        tracker=tracker_params,
        evaluator=config.evaluator
    )
