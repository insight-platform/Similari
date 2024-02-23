from similari import Sort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = Sort(shards=4, bbox_history=10, max_idle_epochs=5,
                method=PositionalMetricType.iou(threshold=0.3),
                spatio_temporal_constraints=constraints,
                kalman_position_weight=0.1,
                kalman_velocity_weight=0.1)

    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict([(box, 11111)])
    for t in tracks:
        print(t)
    sort.skip_epochs(10)

    # you have to call wasted from time to time to purge wasted tracks
    # out of the waste bin. Without doing that the memory utilization will grow.
    wasted = sort.wasted()
    print(wasted[0])

    # or just clear wasted
    sort.clear_wasted()
