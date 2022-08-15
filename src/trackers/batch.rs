use std::collections::HashMap;

pub type BatchRecords<T> = HashMap<u64, T>;

#[derive(Debug)]
pub struct PredictionBatchRequest<T> {
    batch: BatchRecords<T>,
}

impl<T> PredictionBatchRequest<T> {
    pub fn new() -> Self {
        Self {
            batch: HashMap::default(),
        }
    }

    pub fn add(&mut self, scene_id: u64, elt: T) {
        self.batch.insert(scene_id, elt);
    }

    pub fn get(&self) -> &BatchRecords<T> {
        &self.batch
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::Universal2DBox;
    use crate::trackers::batch::PredictionBatchRequest;

    #[test]
    fn test() {
        let b = PredictionBatchRequest::<Universal2DBox>::new();
        drop(b);
    }
}
