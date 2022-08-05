from similari import KalmanFilter, BoundingBox

if __name__ == '__main__':
    f = KalmanFilter()
    state = f.initiate(BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah())
    state = f.predict(state)
    box_xywh = state.bbox()
    print(box_xywh)
    # if work with oriented box
    # import Universal2DBox and use it
    #
    #box_xyaah = state.universal_bbox()
    #print(box_xyaah)

    state = f.update(state, BoundingBox(0.2, 0.2, 5.1, 9.9).as_xyaah())
    state = f.predict(state)
    box_xywh = state.bbox()
    print(box_xywh)

    for i in range(1, 21):
        state = f.predict(state)
        box_xywh = state.universal_bbox()
        print("Prediction:", box_xywh)

        obs = BoundingBox(0.2 + i * 0.2, 0.2 + i * 0.2, 5.0, 10.0).as_xyaah()
        state = f.update(state, obs)
        print("Observation:", obs)

