from similari import Sort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = Sort(shards=4, bbox_history=10, max_idle_epochs=5,
                method=PositionalMetricType.iou(threshold=0.3),
                spatio_temporal_constraints=constraints)

    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict_with_scene(1, [(box, 11111)])
    for t in tracks:
        print(t)

    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict_with_scene(2, [(box, 22222)])
    for t in tracks:
        print(t)

    sort.skip_epochs_for_scene(1, 10)
    sort.skip_epochs_for_scene(2, 10)

    # you have to call wasted from time to time to purge wasted tracks
    # out of the waste bin. Without doing that the memory utilization will grow.
    wasted = sort.wasted()
    print(wasted[0])


    # or just clear wasted
    sort.clear_wasted()