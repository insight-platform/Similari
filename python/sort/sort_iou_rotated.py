from similari import Sort, Universal2DBox, SpatioTemporalConstraints, PositionalMetricType

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = Sort(shards=4, bbox_history=10, max_idle_epochs=5,
                method=PositionalMetricType.iou(threshold=0.3),
                spatio_temporal_constraints=constraints)

    box = Universal2DBox(10., 5., 0.32, 1.0, 7.)
    tracks = sort.predict([(box, 11111)])
    for t in tracks:
        print(t)

    box = Universal2DBox(11., 5.2, 0.35, 1.05, 7.02)
    tracks = sort.predict([(box, 11111)])
    for t in tracks:
        print(t)

    box = Universal2DBox(11.6, 5.26, 0.37, 1.06, 7.025)
    tracks = sort.predict([(box, 11111)])
    for t in tracks:
        print(t)

    # you have to call wasted from time to time to purge wasted tracks
    # out of the waste bin. Without doing that the memory utilization will grow.
    sort.skip_epochs(10)
    wasted = sort.wasted()
    print(wasted[0])


    # or just clear wasted
    sort.clear_wasted()