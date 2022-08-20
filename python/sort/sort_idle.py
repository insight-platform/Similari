from similari import Sort, BoundingBox, SpatioTemporalConstraints, PositionalMetricType

if __name__ == '__main__':
    sort = Sort(shards=4, bbox_history=10, max_idle_epochs=5,
                method=PositionalMetricType.iou(threshold=0.3))

    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict([(box, 11111)])
    for t in tracks:
        print(t)

    tracks = sort.predict([])
    print("Tracks:",  tracks)

    idle_tracks = sort.idle_tracks()
    print("Idle Tracks:", idle_tracks)

    sort.skip_epochs(10)

    idle_tracks = sort.idle_tracks()
    print("Idle Tracks:", idle_tracks)

    # or just clear wasted
    sort.clear_wasted()