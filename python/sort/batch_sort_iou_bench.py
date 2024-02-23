import timeit

from similari import BatchSort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType, SortPredictionBatchRequest

if __name__ == '__main__':
    # all train

    def bench(n):
        dets = []
        for i in range(n):
            dets.append(BoundingBox(1000 * i, 1000 * i, 50, 60).as_xyaah())

        shards = 4
        if n <= 100:
            shards = 1
        elif n <= 200:
            shards = 2

        constraints = SpatioTemporalConstraints()
        constraints.add_constraints([(1, 1.0)])
        mot_tracker = BatchSort(distance_shards=shards,
                                voting_shards=shards,
                                bbox_history=10,
                                max_idle_epochs=5,
                                method=PositionalMetricType.iou(threshold=0.3),
                                spatio_temporal_constraints=constraints,
                                kalman_position_weight=0.1,
                                kalman_velocity_weight=0.1)

        def run_it(mot_tracker, detections):
            batch = SortPredictionBatchRequest()
            for d in detections:
                batch.add(0, d, 11111)
            prediction_result = mot_tracker.predict(batch)
            for _ in range(prediction_result.batch_size()):
                _scene_id, _tracks = prediction_result.get()

        count = 100
        duration = timeit.timeit(lambda: run_it(mot_tracker, dets), number=count)
        print("Run for {} took {} ms".format(n, duration / float(count) * 1000.0))


    bench(10)
    bench(100)
    bench(200)
    bench(300)
    bench(500)
    bench(1000)
