pub mod simple_iou_py;

use crate::prelude::{ObservationBuilder, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::{Track, TrackStatus};
use crate::trackers::sort::iou::IOUSortMetric;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{PyWastedSortTrack, SortAttributes, SortAttributesUpdate, SortTrack};
use crate::utils::bbox::Universal2DBox;
use crate::voting::Voting;
use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
/// Easy to use SORT tracker implementation
///

#[pyclass(text_signature = "(shards, bbox_history, max_idle_epochs, threshold)")]
pub struct IoUSort {
    store: TrackStore<SortAttributes, IOUSortMetric, Universal2DBox>,
    epoch: Arc<RwLock<HashMap<u64, usize>>>,
    threshold: f32,
}

impl IoUSort {
    /// Creates new tracker
    ///
    /// # Parameters
    /// * `shards` - amount of cpu threads to process the data, keep 1 for up to 100 simultaneously tracked objects, try it before setting high - higher numbers may lead to unexpected latencies.
    /// * `bbox_history` - how many last bboxes are kept within stored track (valuable for offline trackers), for online - keep 1
    /// * `max_idle_epochs` - how long track survives without being updated
    /// * `threshold` - how low IoU must be to establish a new track (default from the authors of SORT is 0.3)
    ///
    pub fn new(shards: usize, bbox_history: usize, max_idle_epochs: usize, threshold: f32) -> Self {
        assert!(bbox_history > 0);
        let epoch = Arc::new(RwLock::new(HashMap::default()));
        let store = TrackStoreBuilder::new(shards)
            .default_attributes(SortAttributes::new_with_epochs(
                bbox_history,
                max_idle_epochs,
                epoch.clone(),
            ))
            .metric(IOUSortMetric::new(threshold))
            .build();

        Self {
            epoch,
            store,
            threshold,
        }
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id` == 0
    ///
    pub fn skip_epochs(&mut self, n: usize) {
        self.skip_epochs_for_scene(0, n)
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id`
    /// * `scene_id` - scene to skip epochs
    ///
    pub fn skip_epochs_for_scene(&mut self, scene_id: u64, n: usize) {
        let mut epoch_store = self.epoch.write().unwrap();
        if let Some(epoch) = epoch_store.get_mut(&scene_id) {
            *epoch += n;
        } else {
            epoch_store.insert(scene_id, n);
        }
    }

    /// Get the amount of stored tracks per shard
    ///
    pub fn shard_stats(&self) -> Vec<usize> {
        self.store.shard_stats()
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    pub fn current_epoch(&self) -> usize {
        self.current_epoch_with_scene(0)
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    pub fn current_epoch_with_scene(&self, scene_id: u64) -> usize {
        let mut epoch_map = self.epoch.write().unwrap();
        let epoch = epoch_map.get_mut(&scene_id);
        if let Some(epoch) = epoch {
            *epoch
        } else {
            0
        }
    }

    /// Receive tracking information for observed bboxes of `scene_id` == 0
    ///
    /// # Parameters
    /// * `bboxes` - bounding boxes received from a detector
    ///
    pub fn predict(&mut self, bboxes: &[Universal2DBox]) -> Vec<SortTrack> {
        self.predict_with_scene(0, bboxes)
    }

    /// Receive tracking information for observed bboxes of `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id provided by a user (class, camera id, etc...)
    /// * `bboxes` - bounding boxes received from a detector
    ///
    pub fn predict_with_scene(
        &mut self,
        scene_id: u64,
        bboxes: &[Universal2DBox],
    ) -> Vec<SortTrack> {
        let mut rng = rand::thread_rng();
        let epoch = {
            let mut epoch_map = self.epoch.write().unwrap();
            let epoch = epoch_map.get_mut(&scene_id);
            if let Some(epoch) = epoch {
                *epoch += 1;
                *epoch
            } else {
                epoch_map.insert(scene_id, 1);
                1
            }
        };

        let tracks = bboxes
            .iter()
            .map(|bb| {
                self.store
                    .track_builder(rng.gen())
                    .observation(
                        ObservationBuilder::new(0)
                            .observation_attributes(bb.clone())
                            .track_attributes_update(SortAttributesUpdate::new_with_scene(
                                epoch, scene_id,
                            ))
                            .build(),
                    )
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let num_tracks = tracks.len();
        let (dists, errs) = self.store.foreign_track_distances(tracks.clone(), 0, false);
        assert!(errs.all().is_empty());
        let voting = SortVoting::new(
            self.threshold,
            num_tracks,
            self.store.shard_stats().iter().sum(),
        );
        let winners = voting.winners(dists);
        let mut res = Vec::default();
        for t in tracks {
            let source = t.get_track_id();
            let track_id: u64 = if let Some(dest) = winners.get(&source) {
                let dest = dest[0];
                if dest == source {
                    self.store.add_track(t).unwrap();
                    source
                } else {
                    self.store
                        .merge_external(dest, &t, Some(&[0]), false)
                        .unwrap();
                    dest
                }
            } else {
                self.store.add_track(t).unwrap();
                source
            };

            let store = self.store.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap().clone();

            res.push(track.into())
        }

        res
    }

    /// Receive all the tracks with expired life
    ///
    pub fn wasted(&mut self) -> Vec<Track<SortAttributes, IOUSortMetric, Universal2DBox>> {
        let res = self.store.find_usable();
        let wasted = res
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.store.fetch_tracks(&wasted)
    }
}

impl From<Track<SortAttributes, IOUSortMetric, Universal2DBox>> for SortTrack {
    fn from(track: Track<SortAttributes, IOUSortMetric, Universal2DBox>) -> Self {
        let attrs = track.get_attributes();
        SortTrack {
            id: track.get_track_id(),
            epoch: attrs.epoch,
            observed_bbox: attrs.last_observation.clone(),
            scene_id: attrs.scene_id,
            predicted_bbox: attrs.last_prediction.clone(),
            length: attrs.length,
        }
    }
}

impl From<Track<SortAttributes, IOUSortMetric, Universal2DBox>> for PyWastedSortTrack {
    fn from(track: Track<SortAttributes, IOUSortMetric, Universal2DBox>) -> Self {
        let attrs = track.get_attributes();
        PyWastedSortTrack {
            id: track.get_track_id(),
            epoch: attrs.epoch,
            observed_bbox: attrs.last_observation.clone(),
            scene_id: attrs.scene_id,
            predicted_bbox: attrs.last_prediction.clone(),
            length: attrs.length,
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::sort::simple_iou::IoUSort;
    use crate::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
    use crate::utils::bbox::BoundingBox;
    use crate::{EstimateClose, EPS};

    #[test]
    fn sort() {
        let mut t = IoUSort::new(1, 10, 2, DEFAULT_SORT_IOU_THRESHOLD);
        assert_eq!(t.current_epoch(), 0);
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        let v = t.predict(&vec![bb.into()]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        let track_id = v.id;
        assert_eq!(v.length, 1);
        assert!(v.observed_bbox.almost_same(&bb.into(), EPS));
        assert_eq!(v.epoch, 1);
        assert_eq!(t.current_epoch(), 1);

        let bb = BoundingBox::new(0.1, 0.1, 10.1, 20.0);
        let v = t.predict(&vec![bb.into()]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        assert_eq!(v.id, track_id);
        assert_eq!(v.length, 2);
        assert!(v.observed_bbox.almost_same(&bb.into(), EPS));
        assert_eq!(v.epoch, 2);
        assert_eq!(t.current_epoch(), 2);

        let bb = BoundingBox::new(10.1, 10.1, 10.1, 20.0);
        let v = t.predict(&[bb.into()]);
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        assert_ne!(v.id, track_id);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(t.current_epoch(), 3);

        let bb = t.predict(&[]);
        assert!(bb.is_empty());
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(t.current_epoch(), 4);
        assert_eq!(t.current_epoch(), 4);

        let bb = t.predict(&[]);
        assert!(bb.is_empty());
        let wasted = t.wasted();
        assert_eq!(wasted.len(), 1);
        assert_eq!(wasted[0].get_track_id(), track_id);
        assert_eq!(t.current_epoch(), 5);
    }

    #[test]
    fn sort_with_scenes() {
        let mut t = IoUSort::new(1, 10, 2, DEFAULT_SORT_IOU_THRESHOLD);
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        assert_eq!(t.current_epoch_with_scene(1), 0);
        assert_eq!(t.current_epoch_with_scene(2), 0);

        let _v = t.predict_with_scene(1, &vec![bb.into()]);
        let _v = t.predict_with_scene(1, &vec![bb.into()]);

        assert_eq!(t.current_epoch_with_scene(1), 2);
        assert_eq!(t.current_epoch_with_scene(2), 0);

        let _v = t.predict_with_scene(2, &vec![bb.into()]);

        assert_eq!(t.current_epoch_with_scene(1), 2);
        assert_eq!(t.current_epoch_with_scene(2), 1);
    }
}
