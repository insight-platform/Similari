use crate::utils::bbox::{BoundingBox, Universal2DBox};
use crate::utils::kalman::{KalmanFilter, State};
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "KalmanFilter")]
pub struct PyKalmanFilter {
    filter: KalmanFilter,
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "KalmanFilterState")]
pub struct PyKalmanFilterState {
    state: State,
}

#[pymethods]
impl PyKalmanFilterState {
    #[pyo3(text_signature = "($self)")]
    pub fn universal_bbox(&self) -> Universal2DBox {
        self.state.universal_bbox()
    }

    #[pyo3(text_signature = "($self)")]
    pub fn bbox(&self) -> PyResult<BoundingBox> {
        self.universal_bbox().as_xywh_py()
    }
}

#[pymethods]
impl PyKalmanFilter {
    #[new]
    #[args(position_weight = "0.05", velocity_weight = "0.00625")]
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        Self {
            filter: KalmanFilter::new(position_weight, velocity_weight),
        }
    }

    #[pyo3(text_signature = "($self, bbox)")]
    pub fn initiate(&self, bbox: Universal2DBox) -> PyKalmanFilterState {
        PyKalmanFilterState {
            state: self.filter.initiate(bbox),
        }
    }

    #[pyo3(text_signature = "($self, state)")]
    pub fn predict(&self, state: PyKalmanFilterState) -> PyKalmanFilterState {
        PyKalmanFilterState {
            state: self.filter.predict(state.state),
        }
    }

    #[pyo3(text_signature = "($self, state, bbox)")]
    pub fn update(&self, state: PyKalmanFilterState, bbox: Universal2DBox) -> PyKalmanFilterState {
        PyKalmanFilterState {
            state: self.filter.update(state.state, bbox),
        }
    }

    #[pyo3(text_signature = "($self, state, bbox)")]
    pub fn distance(&self, state: PyKalmanFilterState, bbox: Universal2DBox) -> f32 {
        self.filter.distance(state.state, &bbox)
    }

    #[staticmethod]
    #[pyo3(text_signature = "(distance, inverted)")]
    pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
        KalmanFilter::calculate_cost(distance, inverted)
    }
}
