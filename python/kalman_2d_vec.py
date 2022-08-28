from similari import Vec2DKalmanFilter

if __name__ == '__main__':
    f = Vec2DKalmanFilter()
    state = f.initiate([(1.0, 2.0), (3.0, 4.0)])

    for i in range(1, 21):
        state = f.predict(state)
        print("Predicted [0]:", state[0].x(), state[0].y())
        print("Predicted [1]:", state[1].x(), state[1].y())

        pt1 = (1.0 + i * 0.1, 2.0 + i * 0.1)
        pt2 = (3.0 + i * 0.05, 4.0 + i * 0.05)
        print("Observation:",  [pt1, pt2])

        state = f.update(state, [pt1, pt2])


