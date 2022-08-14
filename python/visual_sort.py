from similari import VisualSortOptions, VisualObservation, VisualObservationSet, VisualSort, BoundingBox, \
    VisualMetricType, PositionalMetricType, SpatioTemporalConstraints

import numpy as np

if __name__ == '__main__':
    opts = VisualSortOptions()
    opts.max_idle_epochs(3)
    opts.kept_history_length(10)
    opts.visual_metric(VisualMetricType.euclidean(1.0))
    opts.positional_metric(PositionalMetricType.maha())
    opts.visual_minimal_track_length(3)
    opts.visual_minimal_area(5.0)
    opts.visual_minimal_quality_use(0.45)
    opts.visual_minimal_quality_collect(0.5)
    opts.visual_max_observations(5)
    opts.visual_min_votes(2)
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    opts.spatio_temporal_constraints(constraints)
    print(opts)

    tracker = VisualSort(shards=4, opts=opts)
    observation_set = VisualObservationSet()
    observation_set.add(VisualObservation(feature=np.array([0.1, 0.1]),
                                          feature_quality=0.96,
                                          bounding_box=BoundingBox(0, 0, 5, 10).as_xyaah(),
                                          custom_object_id=10))
    tracks = tracker.predict(observation_set)
    print(tracks[0])
    tracker.skip_epochs(10)
    wasted = tracker.wasted()
    print(wasted[0])
