"""Unified tracker interface for supported trackers."""
from abc import abstractmethod
from dataclasses import asdict
from typing import Dict, List, Tuple, Union
import numpy as np

from similari import (
    Sort as SortImpl,
    VisualSort as VisualSortImpl,
    BoundingBox,
    SpatioTemporalConstraints,
    PositionalMetricType,
)
from .config import (
    OriginalSortParams,
    SortParams,
    VisualSortParams,
    PositionalMetricType as PositionalMetricConfigType,
)
from .original_sort import Sort as OriginalSortImpl


class Tracker:
    @abstractmethod
    def process_frame(
        self, frame_num: int, detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Tuple[int, float, float, float, float, float]]:
        """(left, top, width, height, confidence) =>
        (track_id, left, top, width, height, confidence)
        """
        pass


class OriginalSort(Tracker):
    def __init__(self, params: OriginalSortParams):
        self._tracker = OriginalSortImpl(**asdict(params))

    def process_frame(
        self, frame_num: int, detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Tuple[int, float, float, float, float, float]]:
        # tuple(top, left, width, height) to np.array([x1, y1, x2, y2])
        np_detections = np.array(detections)
        np_detections[:, 2:4] += np_detections[:, 0:2]
        tracks = self._tracker.update(np_detections)
        return [
            (
                int(track[4]),
                track[0],
                track[1],
                track[2] - track[0],
                track[3] - track[1],
                1.0,
            )
            for track in tracks
        ]


class SimilariTracker(Tracker):
    def __init__(self, params: Union[SortParams, VisualSortParams]):
        constraints = None
        if params.spatio_temporal_constraints:
            constraints = SpatioTemporalConstraints()
            constraints.add_constraints(
                list(map(tuple, params.spatio_temporal_constraints))
            )

        positional_metric = None
        if params.positional_metric:
            if params.positional_metric.type == PositionalMetricConfigType.IoU:
                positional_metric = PositionalMetricType.iou(
                    threshold=params.positional_metric.threshold
                )
            else:
                positional_metric = PositionalMetricType.maha()

        if isinstance(params, SortParams):
            self._tracker = SortImpl(
                shards=params.shards,
                bbox_history=params.bbox_history,
                max_idle_epochs=params.max_idle_epochs,
                method=positional_metric,
                spatio_temporal_constraints=constraints,
            )
        else:
            raise NotImplementedError

        self._use_confidence = params.use_confidence
        self._track_id_map: Dict[int, int] = {}  # to have 1-based track id

    def process_frame(
        self, frame_num: int, detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Tuple[int, float, float, float, float, float]]:
        if self._use_confidence:
            dets = [
                (BoundingBox.new_with_confidence(*detection).as_xyaah(), 0)
                for detection in detections
            ]
        else:
            dets = [
                (BoundingBox(*detection[:-1]).as_xyaah(), 0) for detection in detections
            ]

        tracks = self._tracker.predict(dets)

        rows = []
        for track in tracks:
            track_id = track.id
            _bbox = track.predicted_bbox.as_ltwh()
            if track_id not in self._track_id_map:
                self._track_id_map[track_id] = len(self._track_id_map) + 1
            rows.append(
                (
                    self._track_id_map[track_id],
                    _bbox.left,
                    _bbox.top,
                    _bbox.width,
                    _bbox.height,
                    1.0,
                )
            )

        # TODO
        # if frame_num % ...:
        #     self._tracker.wasted()

        return rows
