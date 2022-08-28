use crate::utils::kalman::kalman_2d_point::{Point2DKalmanFilter, DIM_2D_POINT_X2};
use crate::utils::kalman::KalmanState;
use nalgebra::Point2;

#[derive(Debug)]
pub struct Vec2DKalmanFilter {
    f: Point2DKalmanFilter,
}

/// Default initializer
impl Default for Vec2DKalmanFilter {
    fn default() -> Self {
        Self {
            f: Point2DKalmanFilter::new(1.0 / 20.0, 1.0 / 160.0),
        }
    }
}

impl Vec2DKalmanFilter {
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        Self {
            f: Point2DKalmanFilter::new(position_weight, velocity_weight),
        }
    }

    pub fn initiate(&self, points: &[Point2<f32>]) -> Vec<KalmanState<DIM_2D_POINT_X2>> {
        points.iter().map(|p| self.f.initiate(p)).collect()
    }

    pub fn predict(
        &self,
        state: &[KalmanState<DIM_2D_POINT_X2>],
    ) -> Vec<KalmanState<DIM_2D_POINT_X2>> {
        state.iter().map(|s| self.f.predict(s)).collect()
    }

    pub fn update(
        &self,
        state: &[KalmanState<DIM_2D_POINT_X2>],
        points: &[Point2<f32>],
    ) -> Vec<KalmanState<DIM_2D_POINT_X2>> {
        assert_eq!(
            state.len(),
            points.len(),
            "Lengths of state and points must match"
        );
        state
            .iter()
            .zip(points.iter())
            .map(|(s, p)| self.f.update(s, p))
            .collect()
    }

    pub fn distance(
        &self,
        state: &[KalmanState<DIM_2D_POINT_X2>],
        points: &[Point2<f32>],
    ) -> Vec<f32> {
        assert_eq!(
            state.len(),
            points.len(),
            "Lengths of state and points must match"
        );
        state
            .iter()
            .zip(points.iter())
            .map(|(s, p)| self.f.distance(s, p))
            .collect()
    }

    pub fn calculate_cost(distances: &[f32], inverted: bool) -> Vec<f32> {
        distances
            .iter()
            .map(|d| Point2DKalmanFilter::calculate_cost(*d, inverted))
            .collect()
    }
}
