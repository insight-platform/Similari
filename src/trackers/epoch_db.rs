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

#[cfg(test)]
mod tests {
    use crate::track::TrackStatus;
    use crate::trackers::epoch_db::EpochDb;
    use std::collections::HashMap;
    use std::sync::RwLock;

    #[test]
    fn test_epoch_db() {
        #[derive(Debug, Default)]
        pub struct DbOptions {
            /// The map that stores current epochs for the scene_id
            epoch_db: Option<RwLock<HashMap<u64, usize>>>,
            /// The maximum number of epochs without update while the track is alive
            max_idle_epochs: usize,
        }

        impl EpochDb for DbOptions {
            fn epoch_db(&self) -> &Option<RwLock<HashMap<u64, usize>>> {
                &self.epoch_db
            }

            fn max_idle_epochs(&self) -> usize {
                self.max_idle_epochs
            }
        }

        let db = DbOptions {
            epoch_db: Some(RwLock::new(HashMap::default())),
            max_idle_epochs: 2,
        };

        assert_eq!(db.next_epoch(0), Some(1));
        assert_eq!(db.next_epoch(0), Some(2));

        assert_eq!(db.next_epoch(1), Some(1));
        assert_eq!(db.next_epoch(1), Some(2));

        assert_eq!(db.current_epoch_with_scene(0), Some(2));
        assert_eq!(db.current_epoch_with_scene(1), Some(2));
        assert_eq!(db.current_epoch_with_scene(2), Some(0));

        db.skip_epochs_for_scene(0, 10);
        db.skip_epochs_for_scene(1, 2);

        assert!(matches!(db.baked(0, 2), Ok(TrackStatus::Wasted)));
        assert!(matches!(db.baked(1, 2), Ok(TrackStatus::Pending)));

        db.skip_epochs_for_scene(1, 1);
        assert!(matches!(db.baked(1, 2), Ok(TrackStatus::Wasted)));
    }
}
