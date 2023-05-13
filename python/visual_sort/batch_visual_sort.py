import numpy as np
from similari import (
    BatchVisualSort,
    SpatioTemporalConstraints,
    PositionalMetricType,
    VisualSortOptions,
    VisualSortMetricType,
    BoundingBox, VisualSortObservation, VisualSortObservationSet,
    VisualSortPredictionBatchRequest
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

    # two alternative positional metrics are used as a fallback
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


def build_observation(obj):
    # let's say each obj is a list where
    # obj[:4] are left, top, width, height of the bounding box
    # and obj[4:132] is a 128 length feature vector
    # then the mandatory values for a VisualSortObservationSet structure are
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
    # if set, it will allow to establish which input object was assigned to which output track
    return VisualSortObservation(
        feature=feature,
        feature_quality=feature_quality,
        bounding_box=bbox,
        custom_object_id=None,
    )


def generate_objs(n_objs):
    # in this example
    # each object has 4 values for the bounding box
    # 128 values for the visual features
    # and 1 value for the feature_quality
    return np.random.rand(n_objs, 4 + 128 + 1)


def build_prediction_request(objs, n_batches):
    # for each set of detected objects
    # a VisualSortPredictionBatchRequest object needs to be initialized
    # and filled with VisualSortObservation structures
    batch_request = VisualSortPredictionBatchRequest()
    # split objs into n batches
    for batch_i, batch_objs in enumerate(np.split(objs, n_batches)):
        for obj in batch_objs:
            # to add a VisualSortObservation to a batch request
            # scene id is passed as a first argument
            # the algorithm will only match observations with tracks
            # with the same scene id
            batch_request.add(batch_i, build_observation(obj))
    return batch_request


def main(n_frames=10, n_objs=6, n_batches=2):
    assert n_objs % n_batches == 0, 'For simplicity, batches of equal size are expected.'

    tracker = BatchVisualSort(distance_shards=1, voting_shards=1, opts=get_opts())

    objs = generate_objs(n_objs)

    for frame_i in range(n_frames):
        print(f'======== {frame_i} ========')
        # same objects are reused for simplicity
        # in a real case the prediction request is built
        # from the objects detected on a frame
        batch_request = build_prediction_request(objs, n_batches)

        # tracker is called for each frame
        # even if there were no objects to add to the VisualSortPredictionBatchRequest
        result = tracker.predict(batch_request)

        # the result is parsed like so
        for _ in range(result.batch_size()):
            scene_id, tracks = result.get()
            print("Scene", scene_id)
            for track in tracks:
                print(track)

            # example conversion from Similari track
            # to a left, top, width, height bbox + track id
            track = tracks[0]
            bbox = track.predicted_bbox.as_ltwh()
            print((
                track.id,
                bbox.left,
                bbox.top,
                bbox.width,
                bbox.height,
                bbox.confidence,
            ))

    # you have to call wasted from time to time to purge wasted tracks
    # out of the waste bin. Without doing that the memory utilization will grow.
    print('++++ Wasted ++++')
    wasted = tracker.wasted()
    for w in wasted:
        print(w)

    # or just
    # tracker.clear_wasted()


if __name__ == '__main__':
    main()
