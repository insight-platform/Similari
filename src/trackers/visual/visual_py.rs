use crate::trackers::visual::metric::PositionalMetricType;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "PositionalMetricType")]
pub struct PyPositionalMetricType(PositionalMetricType);

#[pymethods]
impl PyPositionalMetricType {
    #[staticmethod]
    pub fn maha() -> Self {
        PyPositionalMetricType(PositionalMetricType::Mahalanobis)
    }

    #[staticmethod]
    pub fn iou(threshold: f32) -> Self {
        assert!(
            threshold > 0.0 && threshold < 1.0,
            "Threshold must lay between (0.0 and 1.0)"
        );
        PyPositionalMetricType(PositionalMetricType::IoU(threshold))
    }

    #[staticmethod]
    pub fn ignore() -> Self {
        PyPositionalMetricType(PositionalMetricType::Ignore)
    }
}
