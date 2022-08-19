from similari import Sort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType
import timeit

if __name__ == '__main__':
    # all train

    def bench(n):
        dets = []
        for i in range(n):
            dets.append((BoundingBox(1000 * i, 1000 * i, 50, 60).as_xyaah(), None))

        shards = 4
        if n <= 100:
            shards = 1
        elif n <= 200:
            shards = 2


        constraints = SpatioTemporalConstraints()
        constraints.add_constraints([(1, 1.0)])
        mot_tracker = Sort(shards=shards, bbox_history=1, max_idle_epochs=10,
                    method=PositionalMetricType.iou(threshold=0.3),
                    spatio_temporal_constraints=constraints)

        count = 100
        duration = timeit.timeit(lambda: mot_tracker.predict(dets), number=count)
        print("Run for {} took {} ms".format(n, duration / float(count) * 1000.0))

    bench(10)
    bench(100)
    bench(200)
    bench(300)
    bench(500)
    bench(1000)