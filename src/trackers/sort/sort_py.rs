use crate::prelude::Universal2DBox;
use crate::trackers::batch::{PredictionBatchRequest, PredictionBatchResult};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
#[pyo3(name = "SortPredictionBatchRequest")]
pub(crate) struct PySortPredictionBatchRequest {
    pub(crate) batch: PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
    result: Option<PredictionBatchResult>,
}

#[pymethods]
impl PySortPredictionBatchRequest {
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

    fn add(&mut self, scene_id: u64, bbox: Universal2DBox, custom_object_id: Option<i64>) {
        self.batch.add(scene_id, (bbox, custom_object_id))
    }
}
