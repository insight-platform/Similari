use crate::utils::kalman::kalman_2d_point::{Point2DKalmanFilter, DIM_2D_POINT_X2};
use crate::utils::kalman::KalmanState;
use crate::utils::point_2d::Point2D;

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

    pub fn initiate(&self, points: &[Point2D]) -> Vec<KalmanState<DIM_2D_POINT_X2>> {
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
        points: &[Point2D],
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

    pub fn distance(&self, state: &[KalmanState<DIM_2D_POINT_X2>], points: &[Point2D]) -> Vec<f32> {
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

pub mod python {
    use crate::utils::kalman::kalman_2d_point::python::PyPoint2DKalmanFilterState;
    use crate::utils::kalman::kalman_2d_point_vec::Vec2DKalmanFilter;
    use crate::utils::point_2d::Point2D;
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "Vec2DKalmanFilter")]
    pub struct PyVec2DKalmanFilter {
        filter: Vec2DKalmanFilter,
    }

    #[pymethods]
    impl PyVec2DKalmanFilter {
        #[new]
        #[args(position_weight = "0.05", velocity_weight = "0.00625")]
        pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
            Self {
                filter: Vec2DKalmanFilter::new(position_weight, velocity_weight),
            }
        }

        #[pyo3(text_signature = "($self, points)")]
        pub fn initiate(&self, points: Vec<Point2D>) -> Vec<PyPoint2DKalmanFilterState> {
            self.filter
                .initiate(&points)
                .into_iter()
                .map(PyPoint2DKalmanFilterState::new)
                .collect()
        }

        #[pyo3(text_signature = "($self, state)")]
        pub fn predict(
            &self,
            state: Vec<PyPoint2DKalmanFilterState>,
        ) -> Vec<PyPoint2DKalmanFilterState> {
            let args = state.into_iter().map(|s| *s.inner()).collect::<Vec<_>>();
            self.filter
                .predict(&args)
                .into_iter()
                .map(PyPoint2DKalmanFilterState::new)
                .collect()
        }

        #[pyo3(text_signature = "($self, state, points)")]
        pub fn update(
            &self,
            state: Vec<PyPoint2DKalmanFilterState>,
            points: Vec<Point2D>,
        ) -> Vec<PyPoint2DKalmanFilterState> {
            let state_args = state.iter().map(|s| *s.inner()).collect::<Vec<_>>();
            self.filter
                .update(&state_args, &points)
                .into_iter()
                .map(PyPoint2DKalmanFilterState::new)
                .collect()
        }

        #[pyo3(text_signature = "($self, state, points)")]
        pub fn distance(
            &self,
            state: Vec<PyPoint2DKalmanFilterState>,
            points: Vec<Point2D>,
        ) -> Vec<f32> {
            let state_args = state.iter().map(|s| *s.inner()).collect::<Vec<_>>();
            self.filter.distance(&state_args, &points)
        }

        #[staticmethod]
        #[pyo3(text_signature = "(distances, inverted)")]
        pub fn calculate_cost(distances: Vec<f32>, inverted: bool) -> Vec<f32> {
            Vec2DKalmanFilter::calculate_cost(&distances, inverted)
        }
    }
}
