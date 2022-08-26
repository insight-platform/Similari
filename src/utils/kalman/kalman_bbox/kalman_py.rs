use crate::utils::bbox::{BoundingBox, Universal2DBox};
use crate::utils::kalman::kalman_bbox::{Universal2DBoxKalmanFilter, DIM_X2};
use crate::utils::kalman::KalmanState;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Universal2DBoxKalmanFilter")]
pub struct PyUniversal2DBoxKalmanFilter {
    filter: Universal2DBoxKalmanFilter,
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Universal2DBoxKalmanFilterState")]
pub struct PyUniversal2DBoxKalmanFilterState {
    state: KalmanState<{ DIM_X2 }>,
}

#[pymethods]
impl PyUniversal2DBoxKalmanFilterState {
    #[pyo3(text_signature = "($self)")]
    pub fn universal_bbox(&self) -> Universal2DBox {
        Universal2DBox::try_from(self.state).unwrap()
    }

    #[pyo3(text_signature = "($self)")]
    pub fn bbox(&self) -> PyResult<BoundingBox> {
        self.universal_bbox().as_ltwh_py()
    }
}

#[pymethods]
impl PyUniversal2DBoxKalmanFilter {
    #[new]
    #[args(position_weight = "0.05", velocity_weight = "0.00625")]
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        Self {
            filter: Universal2DBoxKalmanFilter::new(position_weight, velocity_weight),
        }
    }

    #[pyo3(text_signature = "($self, bbox)")]
    pub fn initiate(&self, bbox: Universal2DBox) -> PyUniversal2DBoxKalmanFilterState {
        PyUniversal2DBoxKalmanFilterState {
            state: self.filter.initiate(bbox),
        }
    }

    #[pyo3(text_signature = "($self, state)")]
    pub fn predict(
        &self,
        state: PyUniversal2DBoxKalmanFilterState,
    ) -> PyUniversal2DBoxKalmanFilterState {
        PyUniversal2DBoxKalmanFilterState {
            state: self.filter.predict(state.state),
        }
    }

    #[pyo3(text_signature = "($self, state, bbox)")]
    pub fn update(
        &self,
        state: PyUniversal2DBoxKalmanFilterState,
        bbox: Universal2DBox,
    ) -> PyUniversal2DBoxKalmanFilterState {
        PyUniversal2DBoxKalmanFilterState {
            state: self.filter.update(state.state, bbox),
        }
    }

    #[pyo3(text_signature = "($self, state, bbox)")]
    pub fn distance(&self, state: PyUniversal2DBoxKalmanFilterState, bbox: Universal2DBox) -> f32 {
        self.filter.distance(state.state, &bbox)
    }

    #[staticmethod]
    #[pyo3(text_signature = "(distance, inverted)")]
    pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
        Universal2DBoxKalmanFilter::calculate_cost(distance, inverted)
    }
}
