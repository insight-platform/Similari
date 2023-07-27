use crate::prelude::SortTrack;
use crossbeam::channel::{Receiver, Sender};
use log::debug;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type BatchRecords<T> = HashMap<u64, Vec<T>>;
pub type SceneTracks = (u64, Vec<SortTrack>);

#[derive(Debug, Clone)]
pub struct PredictionBatchRequest<T> {
    batch: BatchRecords<T>,
    sender: Sender<SceneTracks>,
    batch_size: Arc<Mutex<usize>>,
}

#[derive(Clone, Debug)]
pub struct PredictionBatchResult {
    receiver: Receiver<SceneTracks>,
    batch_size: Arc<Mutex<usize>>,
}

impl PredictionBatchResult {
    pub fn ready(&self) -> bool {
        !self.receiver.is_empty()
    }

    pub fn get(&self) -> SceneTracks {
        self.receiver
            .recv()
            .expect("Receiver must always receive batch computation result")
    }

    pub fn batch_size(&self) -> usize {
        *self.batch_size.lock().unwrap()
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::trackers::sort::python::PySortTrack;

    use super::PredictionBatchResult;
    use pyo3::prelude::*;

    pub type PySceneTracks = (u64, Vec<PySortTrack>);

    #[pyclass]
    #[derive(Clone, Debug)]
    #[pyo3(name = "PredictionBatchResult")]
    pub struct PyPredictionBatchResult(pub(crate) PredictionBatchResult);

    #[pymethods]
    impl PyPredictionBatchResult {
        pub fn ready(&self) -> bool {
            self.0.ready()
        }

        #[pyo3(signature = ())]
        fn get(&self) -> PySceneTracks {
            Python::with_gil(|py| py.allow_threads(|| unsafe { std::mem::transmute(self.0.get()) }))
        }

        pub fn batch_size(&self) -> usize {
            self.0.batch_size()
        }
    }
}

impl<T> PredictionBatchRequest<T> {
    pub fn get_sender(&self) -> Sender<SceneTracks> {
        self.sender.clone()
    }

    #[allow(dead_code)]
    pub(crate) fn send(&self, res: SceneTracks) -> bool {
        let res = self.sender.send(res);
        if let Err(e) = res {
            debug!(
                "Error occurred when sending results to the batch result object. Error is: {:?}",
                e
            );
            false
        } else {
            true
        }
    }

    pub fn batch_size(&self) -> usize {
        *self.batch_size.lock().unwrap()
    }

    pub fn add(&mut self, scene_id: u64, elt: T) {
        let vec = self.batch.get_mut(&scene_id);
        if let Some(vec) = vec {
            vec.push(elt);
        } else {
            self.batch.insert(scene_id, vec![elt]);
        }
        let mut batch_size = self.batch_size.lock().unwrap();
        *batch_size = self.batch.len();
    }

    pub fn new() -> (Self, PredictionBatchResult) {
        let (sender, receiver) = crossbeam::channel::bounded(1);
        let batch_size = Arc::new(Mutex::new(0));
        (
            Self {
                batch: BatchRecords::default(),
                sender,
                batch_size: batch_size.clone(),
            },
            PredictionBatchResult {
                receiver,
                batch_size,
            },
        )
    }

    pub fn get_batch(&self) -> &BatchRecords<T> {
        &self.batch
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::Universal2DBox;
    use crate::trackers::batch::PredictionBatchRequest;

    #[test]
    fn test() {
        let (mut request, result) = PredictionBatchRequest::<Universal2DBox>::new();
        request.add(0, Universal2DBox::new(0.0, 0.0, Some(0.5), 1.0, 5.0));
        request.add(0, Universal2DBox::new(5.0, 5.0, Some(0.0), 1.5, 10.0));
        request.add(1, Universal2DBox::new(0.0, 0.0, Some(1.0), 0.7, 5.1));
        let _batch = request.get_batch();
        assert_eq!(result.batch_size(), 2);

        assert!(request.send((0, vec![])));
        assert_eq!(result.ready(), true);
        let res = result.get();
        assert_eq!(res.0, 0);
        assert!(res.1.is_empty());
        drop(result);
        assert!(!request.send((0, vec![])));
    }
}
