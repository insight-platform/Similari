use crate::utils::bbox::Universal2DBox;
use pyo3::prelude::*;

/// Track metric implementation
pub mod metric;

/// Cascade voting engine for visual_sort tracker. Combines TopN voting first for features and
/// Hungarian voting for the rest of unmatched (objects, tracks)
pub mod voting;

/// Track attributes for visual_sort tracker
pub mod track_attributes;

/// Observation attributes for visual_sort tracker
pub mod observation_attributes;

/// Implementation of Visual tracker with simple API
pub mod simple_api;

/// Python API implementation
pub mod visual_py;

#[derive(Debug, Clone)]
pub struct VisualObservation<'a> {
    feature: Option<&'a Vec<f32>>,
    feature_quality: Option<f32>,
    bounding_box: Universal2DBox,
    custom_object_id: Option<i64>,
}

impl<'a> VisualObservation<'a> {
    pub fn new(
        feature: Option<&'a Vec<f32>>,
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

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
#[pyclass]
#[pyo3(name = "WastedVisualSortTrack")]
pub struct PyWastedVisualSortTrack {
    /// id of the track
    ///
    #[pyo3(get)]
    pub id: u64,
    /// when the track was lastly updated
    ///
    #[pyo3(get)]
    pub epoch: usize,
    /// the bbox predicted by KF
    ///
    #[pyo3(get)]
    pub predicted_bbox: Universal2DBox,
    /// the bbox passed by detector
    ///
    #[pyo3(get)]
    pub observed_bbox: Universal2DBox,
    /// user-defined scene id that splits tracking space on isolated realms
    ///
    #[pyo3(get)]
    pub scene_id: u64,
    /// current track length
    ///
    #[pyo3(get)]
    pub length: usize,
    /// history of predicted boxes
    ///
    #[pyo3(get)]
    pub predicted_boxes: Vec<Universal2DBox>,
    /// history of observed boxes
    ///
    #[pyo3(get)]
    pub observed_boxes: Vec<Universal2DBox>,
    /// history of features
    ///
    #[pyo3(get)]
    pub observed_features: Vec<Option<Vec<f32>>>,
}

#[pymethods]
impl PyWastedVisualSortTrack {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}
