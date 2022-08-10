from similari import VisualSortOptions, VisualObservation, VisualObservationSet, VisualSort, BoundingBox, \
    VisualMetricType, PositionalMetricType

if __name__ == '__main__':
    opts = VisualSortOptions()
    opts.max_idle_epochs(3)
    opts.history_length(10)
    opts.visual_metric(VisualMetricType.euclidean())
    opts.positional_metric(PositionalMetricType.maha())
    opts.visual_minimal_track_length(3)
    opts.visual_minimal_area(5.0)
    opts.visual_minimal_quality_use(0.45)
    opts.visual_minimal_quality_collect(0.5)
    opts.visual_max_observations(25)
    opts.visual_max_distance(1.0)
    opts.visual_min_votes(5)
    print(opts)

    tracker = VisualSort(shards=4, opts=opts)
    observation_set = VisualObservationSet()
    observation_set.add(VisualObservation(feature=[0.1, 0.1],
                                          feature_quality=0.96,
                                          bounding_box=BoundingBox(0, 0, 5, 10).as_xyaah(),
                                          custom_object_id=10))
    tracks = tracker.predict(observation_set)
    print(tracks[0])
