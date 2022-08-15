use crate::prelude::{Sort, SortTrack};
use crate::trackers::sort::sort_py::PySortPredictionBatchRequest;
use crate::trackers::sort::{PyPositionalMetricType, PyWastedSortTrack};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::trackers::visual_sort::metric::PyVisualSortMetricType;
use crate::trackers::visual_sort::simple_api::options::VisualSortOptions;
use crate::trackers::visual_sort::simple_api::simple_visual_py::{
    PyVisualSortObservation, PyVisualSortObservationSet,
};
use crate::trackers::visual_sort::simple_api::VisualSort;
use crate::trackers::visual_sort::visual_sort_py::PyVisualSortPredictionBatchRequest;
use crate::trackers::visual_sort::PyWastedVisualSortTrack;
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
    m.add_class::<PyPolygon>()?;
    m.add_class::<SortTrack>()?;
    m.add_class::<PyWastedSortTrack>()?;
    m.add_class::<PyKalmanFilterState>()?;
    m.add_class::<PyKalmanFilter>()?;

    m.add_class::<PySortPredictionBatchRequest>()?;
    m.add_class::<SpatioTemporalConstraints>()?;
    m.add_class::<Sort>()?;

    m.add_class::<PyPositionalMetricType>()?;
    m.add_class::<PyVisualSortMetricType>()?;
    m.add_class::<VisualSortOptions>()?;
    m.add_class::<PyVisualSortObservation>()?;
    m.add_class::<PyVisualSortObservationSet>()?;
    m.add_class::<PyVisualSortPredictionBatchRequest>()?;
    m.add_class::<PyWastedVisualSortTrack>()?;
    m.add_class::<VisualSort>()?;

    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(sutherland_hodgman_clip_py, m)?)?;
    m.add_function(wrap_pyfunction!(intersection_area_py, m)?)?;
    Ok(())
}
