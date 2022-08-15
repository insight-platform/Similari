use crate::prelude::SortTrack;
use crossbeam::channel::{Receiver, Sender};
use log::debug;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type BatchRecords<T> = HashMap<u64, Vec<T>>;
pub type SceneTracks = (u64, Vec<SortTrack>);

#[derive(Debug)]
pub struct PredictionBatchRequest<T> {
    batch: BatchRecords<T>,
    sender: Sender<SceneTracks>,
    batch_size: Arc<Mutex<usize>>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PredictionBatchResult {
    receiver: Receiver<SceneTracks>,
    batch_size: Arc<Mutex<usize>>,
}

#[pymethods]
impl PredictionBatchResult {
    pub fn ready(&self) -> bool {
        !self.receiver.is_empty()
    }

    pub fn batch_size(&self) -> usize {
        *self.batch_size.lock().unwrap()
    }

    pub fn get(&self) -> SceneTracks {
        let gil = Python::acquire_gil();
        let py = gil.python();
        py.allow_threads(|| {
            self.receiver
                .recv()
                .expect("Receiver must always receive batch computation result")
        })
    }
}

impl<T> PredictionBatchRequest<T> {
    pub fn send(&self, res: SceneTracks) {
        let res = self.sender.send(res);
        if let Err(e) = res {
            debug!(
                "Error occurred when sending results to the batch result object. Error is: {:?}",
                e
            );
        }
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
        let (request, result) = PredictionBatchRequest::<Universal2DBox>::new();
        drop(request);
        drop(result);
    }
}
