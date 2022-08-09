use crate::prelude::{IoUSort, MahaSort, SortTrack};
use crate::trackers::visual::metric::VisualMetricType;
use crate::trackers::visual::simple_visual::options::VisualSortOptions;
use crate::trackers::visual::simple_visual::VisualSort;
use crate::trackers::visual::visual_py::PyPositionalMetricType;
use crate::trackers::visual::VisualObservation;
use crate::utils::bbox::{BoundingBox, Universal2DBox};
use crate::utils::clipping::clipping_py::{
    intersection_area_py, sutherland_hodgman_clip_py, PyPolygon,
};
use crate::utils::kalman::kalman_py::{PyKalmanFilter, PyKalmanFilterState};
use crate::utils::nms::nms_py::nms_py;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "similari")]
fn similari(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<Universal2DBox>()?;
    m.add_class::<SortTrack>()?;
    m.add_class::<IoUSort>()?;
    m.add_class::<MahaSort>()?;
    m.add_class::<PyPolygon>()?;
    m.add_class::<PyKalmanFilterState>()?;
    m.add_class::<PyKalmanFilter>()?;
    m.add_class::<PyPositionalMetricType>()?;
    m.add_class::<VisualMetricType>()?;
    m.add_class::<VisualSortOptions>()?;
    m.add_class::<VisualObservation>()?;
    m.add_class::<VisualSort>()?;
    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(sutherland_hodgman_clip_py, m)?)?;
    m.add_function(wrap_pyfunction!(intersection_area_py, m)?)?;
    Ok(())
}
