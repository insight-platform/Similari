use crate::utils::kalman::kalman_2d_point::DIM_2D_POINT_X2;
use crate::utils::kalman::kalman_2d_point_vec::Vec2DKalmanFilter;
use crate::utils::kalman::KalmanState;
use crate::utils::point_2d::Point2D;

pub trait KeypointVectorKalmanPrediction {
    fn get_state(&self) -> Option<Vec<KalmanState<{ DIM_2D_POINT_X2 }>>>;
    fn set_state(&mut self, state: Vec<KalmanState<{ DIM_2D_POINT_X2 }>>);

    fn make_prediction(&mut self, points: &[Point2D]) -> Vec<Point2D> {
        let f = Vec2DKalmanFilter::default();
        let state = if let Some(state) = self.get_state() {
            f.update(&state, points)
        } else {
            f.initiate(points)
        };

        let prediction = f.predict(&state);
        self.set_state(prediction.clone());
        prediction
            .into_iter()
            .zip(points.iter())
            .map(|(s, p)| Point2D::from(s).confidence(p.confidence))
            .collect()
    }
}
