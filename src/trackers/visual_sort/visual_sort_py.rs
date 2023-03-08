use crate::prelude::Universal2DBox;
use crate::trackers::batch::{PredictionBatchRequest, PredictionBatchResult};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
#[pyo3(name = "VisualSortPredictionBatchRequest")]
pub(crate) struct PyVisualSortPredictionBatchRequest {
    pub(crate) batch: PredictionBatchRequest<PyVisualSortObservation>,
    result: Option<PredictionBatchResult>,
}

#[pymethods]
impl PyVisualSortPredictionBatchRequest {
    #[new]
    fn new() -> Self {
        let (batch, result) = PredictionBatchRequest::new();
        Self {
            batch,
            result: Some(result),
        }
    }

    fn prediction(&mut self) -> Option<PredictionBatchResult> {
        self.result.take()
    }

    fn add(&mut self, scene_id: u64, elt: PyVisualSortObservation) {
        self.batch.add(scene_id, elt);
    }
}

#[pyclass(
    text_signature = "(feature_opt, feature_quality_opt, bounding_box, custom_object_id_opt)"
)]
#[derive(Debug, Clone)]
#[pyo3(name = "VisualSortObservation")]
pub struct PyVisualSortObservation {
    pub feature: Option<Vec<f32>>,
    pub feature_quality: Option<f32>,
    pub bounding_box: Universal2DBox,
    pub custom_object_id: Option<i64>,
}

#[pymethods]
impl PyVisualSortObservation {
    #[new]
    #[pyo3(signature = (feature, feature_quality, bounding_box, custom_object_id))]
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

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:#?}")
    }
}

#[pyclass(
    text_signature = "(feature_opt, feature_quality_opt, bounding_box, custom_object_id_opt)"
)]
#[derive(Debug)]
#[pyo3(name = "VisualSortObservationSet")]
pub struct PyVisualSortObservationSet {
    pub inner: Vec<PyVisualSortObservation>,
}

#[pymethods]
impl PyVisualSortObservationSet {
    #[new]
    fn new() -> Self {
        Self {
            inner: Vec::default(),
        }
    }

    #[pyo3(text_signature = "($self, observation)")]
    fn add(&mut self, observation: PyVisualSortObservation) {
        self.inner.push(observation);
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        format!("{self:#?}")
    }
}
