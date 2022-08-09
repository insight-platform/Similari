use crate::utils::bbox::Universal2DBox;
use pyo3::prelude::*;

/// Track metric implementation
pub mod metric;

/// Implementation of Python-only structs and their implementations
///
pub mod visual_py;

/// Cascade voting engine for visual tracker. Combines TopN voting first for features and
/// Hungarian voting for the rest of unmatched (objects, tracks)
pub mod voting;

/// Track attributes for visual tracker
pub mod track_attributes;

/// Observation attributes for visual tracker
pub mod observation_attributes;

/// Implementation of Visual tracker with simple API
pub mod simple_visual;

#[pyclass]
#[derive(Debug, Clone)]
pub struct VisualObservation {
    feature: Option<Vec<f32>>,
    feature_quality: Option<f32>,
    bounding_box: Universal2DBox,
    custom_object_id: Option<i64>,
}

#[pymethods]
impl VisualObservation {
    #[new]
    pub fn new(
        feature: Option<Vec<f32>>,
        feature_quality: Option<f32>,
        bounding_box: Universal2DBox,
        custom_object_id: Option<i64>,
    ) -> Self {
        Self {
            feature,
            feature_quality,
            bounding_box,
            custom_object_id,
        }
    }
}
