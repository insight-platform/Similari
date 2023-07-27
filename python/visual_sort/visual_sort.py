from similari import (
    VisualSort,
    SpatioTemporalConstraints,
    PositionalMetricType,
    VisualSortOptions,
    VisualSortMetricType,
    BoundingBox, VisualSortObservation, VisualSortObservationSet
)

def get_opts():
    # init the parameters of the algorithm
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])

    # choose the values according to your case
    opts = VisualSortOptions()
    opts.spatio_temporal_constraints(constraints)
    opts.max_idle_epochs(15)
    opts.kept_history_length(25)
    opts.max_idle_epochs(15)
    opts.kept_history_length(25)

    # two alternative visual metrics are available
    opts.visual_metric(VisualSortMetricType.euclidean(0.7))
    # opts.visual_metric(VisualSortMetricType.cosine(0.2))

    # two alternative positional metrics are available to be used as a fallback
    # when the visual metric match fails
    opts.positional_metric(PositionalMetricType.maha())
    # opts.positional_metric(PositionalMetricType.iou(threshold=0.3))

    # options specific to the VisualSort algorithm
    # choose the values according to your case
    opts.visual_minimal_track_length(7)
    opts.visual_minimal_area(5.0)
    opts.visual_minimal_quality_use(0.45)
    opts.visual_minimal_quality_collect(0.5)
    opts.visual_max_observations(25)
    opts.visual_min_votes(5)
    return opts

tracker = VisualSort(shards=1, opts=get_opts())


assert False

# let's say frame_objs is a list of objs detected in a frame
for frame_objs in frames:
    # for each set of detected objects
    # a VisualSortObservationSet object needs to be initialized
    # and filled with VisualSortObservation structures
    observation_set = VisualSortObservationSet()
    for obj in frame_objs:
        # let's say each obj is a list where
        # obj[:4] are left, top, width, height of the bounding box
        # and obj[4:132] is a 128 length feature vector
        # then the mandatory values for a VisualSortObservation structure are
        bbox = BoundingBox(*obj[:4]).as_xyaah()
        feature = obj[4:132]

        # optionally, it's possible to set a feature_quality value
        # it can be the quality score from the reid model
        # or the bounding box confidence from the detector.
        # The algorithm will use feature quality values as weights
        # for potential matches based on visual metrics.
        # let's say obj[132] is the feature_quality value
        feature_quality = obj[132]

        # custom_object_id is optional
        # if set, it will allow establishing which input object was assigned to which output track
        observation = VisualSortObservation(
            feature=feature,
            feature_quality=feature_quality,
            bounding_box=bbox,
            custom_object_id=None,
        )
        observation_set.add(observation)

    # tracker is called for each frame
    # even if there were no objects to add to the VisualSortObservationSet
    tracks = tracker.predict(observation_set)

    # results are parsed like so
    results = []
    for track in tracks:
        bbox = track.predicted_bbox.as_ltwh()
        results.append(
            (
                track.id,
                bbox.left,
                bbox.top,
                bbox.width,
                bbox.height,
                bbox.confidence,
            )
        )