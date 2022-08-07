use crate::track::TrackStatus;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;

pub trait EpochDb {
    fn epoch_db(&self) -> &Option<RwLock<HashMap<u64, usize>>>;
    fn max_idle_epochs(&self) -> usize;

    fn skip_epochs_for_scene(&self, scene_id: u64, n: usize) {
        if let Some(epoch_store) = self.epoch_db() {
            let mut epoch_store = epoch_store.write().unwrap();
            if let Some(epoch) = epoch_store.get_mut(&scene_id) {
                *epoch += n;
            } else {
                epoch_store.insert(scene_id, n);
            }
        }
    }

    fn current_epoch_with_scene(&self, scene_id: u64) -> Option<usize> {
        if let Some(epoch_store) = self.epoch_db() {
            let mut epoch_store = epoch_store.write().unwrap();
            let epoch = epoch_store.get_mut(&scene_id);
            if let Some(epoch) = epoch {
                Some(*epoch)
            } else {
                Some(0)
            }
        } else {
            None
        }
    }

    fn next_epoch(&self, scene_id: u64) -> Option<usize> {
        if let Some(epoch_store) = self.epoch_db() {
            let mut epoch_store = epoch_store.write().unwrap();
            let epoch = epoch_store.get_mut(&scene_id);
            if let Some(epoch) = epoch {
                *epoch += 1;
                Some(*epoch)
            } else {
                epoch_store.insert(scene_id, 1);
                Some(1)
            }
        } else {
            None
        }
    }

    fn baked(&self, scene_id: u64, last_updated: usize) -> Result<TrackStatus> {
        if let Some(current_epoch) = &self.epoch_db() {
            let current_epoch = current_epoch.read().unwrap();
            if last_updated + self.max_idle_epochs() < *current_epoch.get(&scene_id).unwrap_or(&0) {
                Ok(TrackStatus::Wasted)
            } else {
                Ok(TrackStatus::Pending)
            }
        } else {
            // If epoch expiration is not set the tracks are always ready.
            // If set, then only when certain amount of epochs pass they are Wasted.
            //
            Ok(TrackStatus::Ready)
        }
    }
}
