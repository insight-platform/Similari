//!
//! # Similari
//!
//! The purpose of the crate is to provide tools to config.toml in-memory vector (feature) similarity engines.
//! Similarity calculation is an important resource demanding task broadly used in machine learning and AI systems.
//! Read more about it at GitHub [README](https://github.com/insight-platform/Similari/blob/main/README.md)
//!

/// Holds auxiliary functions that calculate distances between two features.
///
pub mod distance;

/// Various auxiliary testing and example components
///
pub mod examples;

/// Frequently used components
///
pub mod prelude;

/// Holds basic abstractions for tracking - [Track](track::Track), auxiliary structures, traits, and functions.
///
/// It defines the track's look and feel, provides `Track` structure that holds track attributes and features,
/// can accumulate track features and calculate feature distances between pair of tracks.
///
pub mod track;

/// Ready-to-use trackers - IoU SORT, Mahalanobis SORT, Hybrid Visual/Positional Sort
///
pub mod trackers;

/// Utility objects - bounding boxes, kalman_2d_box filter, polygon clipping, nms
///
pub mod utils;

pub use track::store;
pub use track::voting;

use thiserror::Error;

/// Package errors
///
#[derive(Error, Debug, Clone)]
pub enum Errors {
    /// Compared tracks have incompatible attributes, so cannot be used in calculations.
    #[error("Attributes are incompatible between tracks and cannot be used in calculations.")]
    IncompatibleAttributes,
    /// One of tracks doesn't have features for specified class
    ///
    #[error("Requested observations for class={2} are missing in track={0} or track={1} - distance cannot be calculated.")]
    ObservationForClassNotFound(u64, u64, u64),
    /// Requested track is not found in the store
    ///
    #[error("Missing track={0}.")]
    TrackNotFound(u64),

    #[error("Missing requested tracks.")]
    TracksNotFound,

    /// The distance is calculated against self. Ignore it.
    ///
    #[error("Calculation with self id={0} not permitted")]
    SameTrackCalculation(u64),

    /// Track ID is duplicate
    ///
    #[error("Duplicate track id={0}")]
    DuplicateTrackId(u64),

    /// Object cannot be converted
    ///
    #[error("Generic BBox cannot be converted to a requested type")]
    GenericBBoxConversionError,

    /// Index is out of range
    #[error("The index is out of range")]
    OutOfRange,
}

pub const EPS: f32 = 0.00001;

#[cfg(feature = "python")]
mod python {
    use crate::trackers::batch::python::PyPredictionBatchResult;
    use crate::trackers::sort::batch_api::python::{PyBatchSort, PySortPredictionBatchRequest};
    use crate::trackers::sort::python::{PyPositionalMetricType, PySortTrack, PyWastedSortTrack};
    use crate::trackers::sort::simple_api::python::PySort;
    use crate::trackers::spatio_temporal_constraints::python::PySpatioTemporalConstraints;
    use crate::trackers::visual_sort::batch_api::python::{
        PyBatchVisualSort, PyVisualSortPredictionBatchRequest,
    };
    use crate::trackers::visual_sort::metric::python::PyVisualSortMetricType;
    use crate::trackers::visual_sort::options::python::PyVisualSortOptions;
    use crate::trackers::visual_sort::python::{
        PyVisualSortObservation, PyVisualSortObservationSet, PyWastedVisualSortTrack,
    };
    use crate::trackers::visual_sort::simple_api::python::PyVisualSort;
    use crate::utils::bbox::python::{PyBoundingBox, PyUniversal2DBox};
    use crate::utils::clipping::clipping_py::{
        intersection_area_py, sutherland_hodgman_clip_py, PyPolygon,
    };
    use crate::utils::kalman::kalman_2d_box::python::{
        PyUniversal2DBoxKalmanFilter, PyUniversal2DBoxKalmanFilterState,
    };
    use crate::utils::kalman::kalman_2d_point::python::{
        PyPoint2DKalmanFilter, PyPoint2DKalmanFilterState,
    };
    use crate::utils::kalman::kalman_2d_point_vec::python::PyVec2DKalmanFilter;
    use crate::utils::nms::nms_py::nms_py;
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[pymodule]
    #[pyo3(name = "similari")]
    fn similari(_py: Python, m: &PyModule) -> PyResult<()> {
        pyo3_log::init();

        m.add_class::<PyBoundingBox>()?;
        m.add_class::<PyUniversal2DBox>()?;
        m.add_class::<PyPolygon>()?;
        m.add_class::<PySortTrack>()?;
        m.add_class::<PyWastedSortTrack>()?;

        m.add_class::<PyUniversal2DBoxKalmanFilterState>()?;
        m.add_class::<PyUniversal2DBoxKalmanFilter>()?;

        m.add_class::<PyPoint2DKalmanFilterState>()?;
        m.add_class::<PyPoint2DKalmanFilter>()?;

        m.add_class::<PyVec2DKalmanFilter>()?;

        m.add_class::<PySortPredictionBatchRequest>()?;
        m.add_class::<PySpatioTemporalConstraints>()?;
        m.add_class::<PySort>()?;

        m.add_class::<PyPositionalMetricType>()?;
        m.add_class::<PyVisualSortMetricType>()?;
        m.add_class::<PyVisualSortOptions>()?;
        m.add_class::<PyVisualSortObservation>()?;
        m.add_class::<PyVisualSortObservationSet>()?;
        m.add_class::<PyVisualSortPredictionBatchRequest>()?;
        m.add_class::<PyWastedVisualSortTrack>()?;
        m.add_class::<PyVisualSort>()?;

        m.add_class::<PyPredictionBatchResult>()?;

        m.add_class::<PySortPredictionBatchRequest>()?;
        m.add_class::<PyBatchSort>()?;

        m.add_class::<PyBatchVisualSort>()?;

        m.add_function(wrap_pyfunction!(version, m)?)?;
        m.add_function(wrap_pyfunction!(nms_py, m)?)?;
        m.add_function(wrap_pyfunction!(sutherland_hodgman_clip_py, m)?)?;
        m.add_function(wrap_pyfunction!(intersection_area_py, m)?)?;
        Ok(())
    }
}
