use crate::trackers::batch::{PredictionBatchRequest, PredictionBatchResult};
use crate::trackers::visual_sort::simple_api::simple_visual_py::PyVisualSortObservation;
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "VisualSortPredictionBatchRequest")]
pub(crate) struct PyVisualSortPredictionBatchRequest {
    batch: PredictionBatchRequest<PyVisualSortObservation>,
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

    fn get_future_result(&mut self) -> Option<PredictionBatchResult> {
        self.result.take()
    }

    fn add(&mut self, scene_id: u64, elt: PyVisualSortObservation) {
        self.batch.add(scene_id, elt)
    }
}
