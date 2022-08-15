use crate::trackers::batch::PredictionBatchRequest;
use crate::trackers::visual_sort::simple_api::simple_visual_py::PyVisualSortObservation;
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "VisualSortPredictionBatchRequest")]
pub(crate) struct PyVisualSortPredictionBatchRequest {
    batch: PredictionBatchRequest<PyVisualSortObservation>,
}

#[pymethods]
impl PyVisualSortPredictionBatchRequest {
    #[new]
    fn new() -> Self {
        Self {
            batch: PredictionBatchRequest::new(),
        }
    }

    fn add(&mut self, scene_id: u64, elt: PyVisualSortObservation) {
        self.batch.add(scene_id, elt)
    }
}
