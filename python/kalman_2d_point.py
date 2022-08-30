from similari import Point2DKalmanFilter, Point2D

if __name__ == '__main__':
    f = Point2DKalmanFilter()
    state = f.initiate(Point2D(1.0, 2.0))

    for i in range(1, 21):
        state = f.predict(state)
        print("Predicted", state.x(), state.y())

        pt = Point2D(1.0 + i * 0.1, 2.0 + i * 0.1)
        print("Observation:", pt)
        state = f.update(state, pt)


