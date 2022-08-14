from similari import Sort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = Sort(shards = 4, bbox_history = 10, max_idle_epochs = 5, method=PositionalMetricType.maha())
    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict([(box, 1111)])
    for t in tracks:
        print(t)
    sort.skip_epochs(10)
    wasted = sort.wasted()
    print(wasted[0])
