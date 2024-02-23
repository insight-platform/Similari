from similari import BatchSort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType, SortPredictionBatchRequest

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = BatchSort(distance_shards=4, voting_shards=4, bbox_history=10, max_idle_epochs=5,
                     method=PositionalMetricType.iou(threshold=0.3),
                     spatio_temporal_constraints=constraints,
                     kalman_position_weight=0.1,
                     kalman_velocity_weight=0.1)

    box1 = BoundingBox(10., 5., 7., 7.).as_xyaah()
    box2 = BoundingBox(5., 5., 3., 7.).as_xyaah()
    batch = SortPredictionBatchRequest()
    batch.add(0, box1, 11111)
    batch.add(1, box2, 22222)
    prediction_result = sort.predict(batch)
    for _ in range(prediction_result.batch_size()):
        scene_id, tracks = prediction_result.get()
        print("Scene", scene_id)
        print(tracks[0])

    sort.skip_epochs_for_scene(0, 10)
    sort.skip_epochs_for_scene(1, 10)

    # you have to call wasted from time to time to purge wasted tracks
    # out of the waste bin. Without doing that the memory utilization will grow.
    wasted = sort.wasted()
    for w in wasted:
        print(w)
