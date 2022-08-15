use crate::prelude::Universal2DBox;
use crate::trackers::batch::PredictionBatchRequest;
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "SortPredictionBatchRequest")]
pub(crate) struct PySortPredictionBatchRequest {
    batch: PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
}

#[pymethods]
impl PySortPredictionBatchRequest {
    #[new]
    fn new() -> Self {
        Self {
            batch: PredictionBatchRequest::new(),
        }
    }

    fn add(&mut self, scene_id: u64, elt: (Universal2DBox, Option<i64>)) {
        self.batch.add(scene_id, elt)
    }
}
