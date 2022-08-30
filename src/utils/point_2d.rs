use crate::utils::kalman::kalman_2d_point::DIM_2D_POINT_X2;
use crate::utils::kalman::KalmanState;
use nalgebra::Point2;
use pyo3::{pyclass, pymethods, Py, PyAny};

#[derive(Clone, Debug)]
#[pyclass]
pub struct Point2D {
    pub p: Point2<f32>,
    pub confidence: f32,
}

impl Point2D {
    pub fn confidence(mut self, confidence: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must be between 0.0 and 1.0"
        );
        self.confidence = confidence;
        self
    }
}

#[pymethods]
impl Point2D {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            p: Point2::from([x, y]),
            confidence: 1.0,
        }
    }

    #[staticmethod]
    pub fn with_confidence(x: f32, y: f32, confidence: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must be between 0.0 and 1.0"
        );
        Self {
            p: Point2::from([x, y]),
            confidence,
        }
    }
}

impl From<KalmanState<{ DIM_2D_POINT_X2 }>> for Point2D {
    fn from(s: KalmanState<{ DIM_2D_POINT_X2 }>) -> Self {
        Self::new(s.mean().x, s.mean().y)
    }
}
