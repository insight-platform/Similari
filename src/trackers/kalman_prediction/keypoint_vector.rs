use crate::utils::kalman::kalman_2d_point::DIM_2D_POINT_X2;
use crate::utils::kalman::kalman_2d_point_vec::Vec2DKalmanFilter;
use crate::utils::kalman::KalmanState;
use nalgebra::Point2;

pub trait KeypointVectorKalmanPrediction {
    fn get_state(&self) -> Option<Vec<KalmanState<{ DIM_2D_POINT_X2 }>>>;
    fn set_state(&mut self, state: Vec<KalmanState<{ DIM_2D_POINT_X2 }>>);

    fn make_prediction(&mut self, points: &[Point2<f32>]) -> Vec<Point2<f32>> {
        let f = Vec2DKalmanFilter::default();
        let state = if let Some(state) = self.get_state() {
            f.update(&state, points)
        } else {
            f.initiate(points)
        };

        let prediction = f.predict(&state);
        self.set_state(prediction.clone());
        prediction.into_iter().map(Point2::from).collect()
    }
}
