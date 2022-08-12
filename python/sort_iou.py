from similari import IoUSort, BoundingBox, SpatioTemporalConstraints

if __name__ == '__main__':
    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])
    sort = IoUSort(shards=4, bbox_history=10, max_idle_epochs=5, threshold=0.3, spatio_temporal_constraints=constraints)
    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict([box])
    for t in tracks:
        print(t)
    sort.skip_epochs(10)
    wasted = sort.wasted()
    # print(wasted[0])
