use crate::trackers::batch::PredictionBatchRequest;
use crate::trackers::visual::simple_api::simple_visual_py::PyVisualObservation;
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "SortPredictionBatchRequest")]
pub(crate) struct PySortPredictionBatchRequest {
    batch: PredictionBatchRequest<PyVisualObservation>,
}

#[pymethods]
impl PySortPredictionBatchRequest {
    #[new]
    fn new() -> Self {
        Self {
            batch: PredictionBatchRequest::new(),
        }
    }

    fn add(&mut self, scene_id: u64, elt: PyVisualObservation) {
        self.batch.add(scene_id, elt)
    }
}
