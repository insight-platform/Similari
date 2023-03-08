use crate::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::Track;
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::metric::SortMetric;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{
    AutoWaste, PositionalMetricType, PyPositionalMetricType, SortLookup,
    DEFAULT_AUTO_WASTE_PERIODICITY, MAHALANOBIS_NEW_TRACK_THRESHOLD,
};
use crate::trackers::sort::{
    PyWastedSortTrack, SortAttributes, SortAttributesOptions, SortAttributesUpdate, SortTrack,
    VotingType,
};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::trackers::tracker_api::TrackerAPI;
use crate::utils::bbox::Universal2DBox;
use crate::voting::Voting;
use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Easy to use SORT tracker implementation
///
#[pyclass(text_signature = "(shards, bbox_history, max_idle_epochs, threshold)")]
pub struct Sort {
    store: RwLock<TrackStore<SortAttributes, SortMetric, Universal2DBox>>,
    wasted_store: RwLock<TrackStore<SortAttributes, SortMetric, Universal2DBox>>,
    method: PositionalMetricType,
    opts: Arc<SortAttributesOptions>,
    auto_waste: AutoWaste,
    track_id: u64,
}

impl Sort {
    /// Creates new tracker
    ///
    /// # Parameters
    /// * `shards` - amount of cpu threads to process the data, keep 1 for up to 100 simultaneously tracked objects, try it before setting high - higher numbers may lead to unexpected latencies.
    /// * `bbox_history` - how many last bboxes are kept within stored track (valuable for offline trackers), for online - keep 1
    /// * `max_idle_epochs` - how long track survives without being updated
    /// * `threshold` - how low IoU must be to establish a new track (default from the authors of SORT is 0.3)
    ///
    pub fn new(
        shards: usize,
        bbox_history: usize,
        max_idle_epochs: usize,
        method: PositionalMetricType,
        min_confidence: f32,
        spatio_temporal_constraints: Option<SpatioTemporalConstraints>,
    ) -> Self {
        assert!(bbox_history > 0);
        let epoch_db = RwLock::new(HashMap::default());
        let opts = Arc::new(SortAttributesOptions::new(
            Some(epoch_db),
            max_idle_epochs,
            bbox_history,
            spatio_temporal_constraints.unwrap_or_default(),
        ));
        let store = RwLock::new(
            TrackStoreBuilder::new(shards)
                .default_attributes(SortAttributes::new(opts.clone()))
                .metric(SortMetric::new(method, min_confidence))
                .notifier(NoopNotifier)
                .build(),
        );

        let wasted_store = RwLock::new(
            TrackStoreBuilder::new(shards)
                .default_attributes(SortAttributes::new(opts.clone()))
                .metric(SortMetric::new(method, min_confidence))
                .notifier(NoopNotifier)
                .build(),
        );

        Self {
            store,
            track_id: 0,
            wasted_store,
            method,
            opts,
            auto_waste: AutoWaste {
                periodicity: DEFAULT_AUTO_WASTE_PERIODICITY,
                counter: DEFAULT_AUTO_WASTE_PERIODICITY,
            },
        }
    }

    /// Receive tracking information for observed bboxes of `scene_id` == 0
    ///
    /// # Parameters
    /// * `bboxes` - bounding boxes received from a detector
    ///
    pub fn predict(&mut self, bboxes: &[(Universal2DBox, Option<i64>)]) -> Vec<SortTrack> {
        self.predict_with_scene(0, bboxes)
    }

    fn gen_track_id(&mut self) -> u64 {
        self.track_id += 1;
        self.track_id
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
        bboxes: &[(Universal2DBox, Option<i64>)],
    ) -> Vec<SortTrack> {
        if self.auto_waste.counter == 0 {
            self.auto_waste();
            self.auto_waste.counter = self.auto_waste.periodicity;
        } else {
            self.auto_waste.counter -= 1;
        }

        let mut rng = rand::thread_rng();
        let epoch = self.opts.next_epoch(scene_id).unwrap();

        let tracks = bboxes
            .iter()
            .map(|(bb, custom_object_id)| {
                self.store
                    .read()
                    .unwrap()
                    .new_track(rng.gen())
                    .observation(
                        ObservationBuilder::new(0)
                            .observation_attributes(bb.clone())
                            .track_attributes_update(SortAttributesUpdate::new_with_scene(
                                epoch,
                                scene_id,
                                *custom_object_id,
                            ))
                            .build(),
                    )
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let num_candidates = tracks.len();
        let (dists, errs) =
            self.store
                .write()
                .unwrap()
                .foreign_track_distances(tracks.clone(), 0, false);
        assert!(errs.all().is_empty());
        let dists = dists.all();
        let voting = SortVoting::new(
            match self.method {
                PositionalMetricType::Mahalanobis => MAHALANOBIS_NEW_TRACK_THRESHOLD,
                PositionalMetricType::IoU(t) => t,
            },
            num_candidates,
            self.store.read().unwrap().shard_stats().iter().sum(),
        );
        let winners = voting.winners(dists);
        let mut res = Vec::default();

        for mut t in tracks {
            let source = t.get_track_id();
            let track_id: u64 = if let Some(dest) = winners.get(&source) {
                let dest = dest[0];
                if dest == source {
                    let track_id = self.gen_track_id();
                    t.set_track_id(track_id);
                    self.store.write().unwrap().add_track(t).unwrap();
                    track_id
                } else {
                    self.store
                        .write()
                        .unwrap()
                        .merge_external(dest, &t, Some(&[0]), false)
                        .unwrap();
                    dest
                }
            } else {
                let track_id = self.gen_track_id();
                t.set_track_id(track_id);
                self.store.write().unwrap().add_track(t).unwrap();
                track_id
            };

            let lock = self.store.read().unwrap();
            let store = lock.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap();
            res.push(SortTrack::from(track));
        }

        res
    }

    pub fn idle_tracks(&mut self) -> Vec<SortTrack> {
        self.idle_tracks_with_scene(0)
    }

    pub fn idle_tracks_with_scene(&mut self, scene_id: u64) -> Vec<SortTrack> {
        let store = self.store.read().unwrap();

        store
            .lookup(SortLookup::IdleLookup(scene_id))
            .iter()
            .map(|(track_id, _status)| {
                let shard = store.get_store(*track_id as usize);
                let track = shard.get(track_id).unwrap();
                SortTrack::from(track)
            })
            .collect()
    }
}

impl TrackerAPI<SortAttributes, SortMetric, Universal2DBox, SortAttributesOptions, NoopNotifier>
    for Sort
{
    fn get_auto_waste_obj_mut(&mut self) -> &mut AutoWaste {
        &mut self.auto_waste
    }

    fn get_opts(&self) -> &SortAttributesOptions {
        &self.opts
    }

    fn get_main_store_mut(
        &mut self,
    ) -> RwLockWriteGuard<TrackStore<SortAttributes, SortMetric, Universal2DBox, NoopNotifier>>
    {
        self.store.write().unwrap()
    }

    fn get_wasted_store_mut(
        &mut self,
    ) -> RwLockWriteGuard<TrackStore<SortAttributes, SortMetric, Universal2DBox, NoopNotifier>>
    {
        self.wasted_store.write().unwrap()
    }

    fn get_main_store(
        &self,
    ) -> RwLockReadGuard<TrackStore<SortAttributes, SortMetric, Universal2DBox, NoopNotifier>> {
        self.store.read().unwrap()
    }

    fn get_wasted_store(
        &self,
    ) -> RwLockReadGuard<TrackStore<SortAttributes, SortMetric, Universal2DBox, NoopNotifier>> {
        self.wasted_store.read().unwrap()
    }
}

impl From<&Track<SortAttributes, SortMetric, Universal2DBox>> for SortTrack {
    fn from(track: &Track<SortAttributes, SortMetric, Universal2DBox>) -> Self {
        let attrs = track.get_attributes();
        SortTrack {
            id: track.get_track_id(),
            custom_object_id: attrs.custom_object_id,
            voting_type: VotingType::Positional,
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            length: attrs.track_length,
        }
    }
}

impl From<Track<SortAttributes, SortMetric, Universal2DBox>> for PyWastedSortTrack {
    fn from(track: Track<SortAttributes, SortMetric, Universal2DBox>) -> Self {
        let attrs = track.get_attributes();
        PyWastedSortTrack {
            id: track.get_track_id(),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            length: attrs.track_length,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
    use crate::trackers::sort::simple_api::Sort;
    use crate::trackers::sort::PositionalMetricType::IoU;
    use crate::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
    use crate::trackers::tracker_api::TrackerAPI;
    use crate::utils::bbox::BoundingBox;

    #[test]
    fn sort() {
        let mut t = Sort::new(
            1,
            10,
            2,
            IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );
        assert_eq!(t.current_epoch(), 0);
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        let v = t.predict(&[(bb.into(), None)]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        let track_id = v.id;
        assert_eq!(v.custom_object_id, None);
        assert_eq!(v.length, 1);
        assert_eq!(v.observed_bbox, bb.into());
        assert_eq!(v.epoch, 1);
        assert_eq!(t.current_epoch(), 1);

        let bb = BoundingBox::new(0.1, 0.1, 10.1, 20.0);
        let v = t.predict(&[(bb.into(), Some(2))]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        assert_eq!(v.custom_object_id, Some(2));
        assert_eq!(v.id, track_id);
        assert_eq!(v.length, 2);
        assert_eq!(v.observed_bbox, bb.into());
        assert_eq!(v.epoch, 2);
        assert_eq!(t.current_epoch(), 2);

        let bb = BoundingBox::new(10.1, 10.1, 10.1, 20.0);
        let v = t.predict(&[(bb.into(), Some(3))]);
        assert_eq!(v.len(), 1);
        let v = v[0].clone();
        assert_eq!(v.custom_object_id, Some(3));
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
        let mut t = Sort::new(
            1,
            10,
            2,
            IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
        assert_eq!(t.current_epoch_with_scene(1), 0);
        assert_eq!(t.current_epoch_with_scene(2), 0);

        let _v = t.predict_with_scene(1, &[(bb.into(), Some(4))]);
        let _v = t.predict_with_scene(1, &[(bb.into(), Some(5))]);

        assert_eq!(t.current_epoch_with_scene(1), 2);
        assert_eq!(t.current_epoch_with_scene(2), 0);

        let _v = t.predict_with_scene(2, &[(bb.into(), Some(6))]);

        assert_eq!(t.current_epoch_with_scene(1), 2);
        assert_eq!(t.current_epoch_with_scene(2), 1);
    }

    #[test]
    fn idle_tracks() {
        let mut t = Sort::new(
            1,
            10,
            2,
            IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);

        let _v = t.predict_with_scene(1, &[(bb.into(), Some(4))]);
        let idle = t.idle_tracks_with_scene(1);
        assert!(idle.is_empty());

        let _v = t.predict_with_scene(1, &[]);

        let idle = t.idle_tracks_with_scene(1);
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0].id, 1);
    }

    #[test]
    fn clear_wasted_tracks() {
        let mut t = Sort::new(
            1,
            10,
            2,
            IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
        );
        let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);

        let _v = t.predict_with_scene(1, &[(bb.into(), Some(4))]);
        t.skip_epochs_for_scene(1, 3);
        assert_eq!(
            t.wasted_store
                .read()
                .unwrap()
                .shard_stats()
                .iter()
                .sum::<usize>(),
            1
        );
        t.clear_wasted();
        assert_eq!(
            t.wasted_store
                .read()
                .unwrap()
                .shard_stats()
                .iter()
                .sum::<usize>(),
            0
        );
    }
}

#[pymethods]
impl Sort {
    #[new]
    #[pyo3(signature = (shards=4, bbox_history=1, max_idle_epochs=5, method = None, min_confidence=0.05, spatio_temporal_constraints=None))]
    pub fn new_py(
        shards: i64,
        bbox_history: i64,
        max_idle_epochs: i64,
        method: Option<PyPositionalMetricType>,
        min_confidence: f32,
        spatio_temporal_constraints: Option<SpatioTemporalConstraints>,
    ) -> Self {
        Self::new(
            shards.try_into().expect("Positive number expected"),
            bbox_history.try_into().expect("Positive number expected"),
            max_idle_epochs
                .try_into()
                .expect("Positive number expected"),
            method.unwrap_or(PyPositionalMetricType::maha()).0,
            min_confidence,
            spatio_temporal_constraints,
        )
    }

    #[pyo3(name = "skip_epochs", signature = (n))]
    pub fn skip_epochs_py(&mut self, n: i64) {
        assert!(n > 0);
        self.skip_epochs(n.try_into().unwrap())
    }

    #[pyo3(name = "skip_epochs_for_scene", signature = (scene_id, n))]
    pub fn skip_epochs_for_scene_py(&mut self, scene_id: i64, n: i64) {
        assert!(n > 0 && scene_id >= 0);
        self.skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap())
    }

    /// Get the amount of stored tracks per shard
    ///
    #[pyo3(name = "shard_stats", signature = ())]
    pub fn shard_stats_py(&self) -> Vec<i64> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.store
                    .read()
                    .unwrap()
                    .shard_stats()
                    .into_iter()
                    .map(|e| i64::try_from(e).unwrap())
                    .collect()
            })
        })
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    #[pyo3(name = "current_epoch", signature = ())]
    pub fn current_epoch_py(&self) -> i64 {
        self.current_epoch_with_scene(0).try_into().unwrap()
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    #[pyo3(name = "current_epoch_with_scene", signature = (scene_id))]
    pub fn current_epoch_with_scene_py(&self, scene_id: i64) -> isize {
        assert!(scene_id >= 0);
        self.current_epoch_with_scene(scene_id.try_into().unwrap())
            .try_into()
            .unwrap()
    }

    /// Receive tracking information for observed bboxes of `scene_id` == 0
    ///
    /// # Parameters
    /// * `bboxes` - bounding boxes received from a detector
    ///
    #[pyo3(name = "predict", signature = (bboxes))]
    pub fn predict_py(&mut self, bboxes: Vec<(Universal2DBox, Option<i64>)>) -> Vec<SortTrack> {
        self.predict_with_scene_py(0, bboxes)
    }

    /// Receive tracking information for observed bboxes of `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id provided by a user (class, camera id, etc...)
    /// * `bboxes` - bounding boxes received from a detector
    ///
    #[pyo3(name = "predict_with_scene", signature = (scene_id, bboxes))]
    pub fn predict_with_scene_py(
        &mut self,
        scene_id: i64,
        bboxes: Vec<(Universal2DBox, Option<i64>)>,
    ) -> Vec<SortTrack> {
        assert!(scene_id >= 0);
        Python::with_gil(|py| {
            py.allow_threads(|| self.predict_with_scene(scene_id.try_into().unwrap(), &bboxes))
        })
    }

    /// Fetch and remove all the tracks with expired life
    ///
    #[pyo3(name = "wasted", signature = ())]
    pub fn wasted_py(&mut self) -> Vec<PyWastedSortTrack> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.wasted()
                    .into_iter()
                    .map(PyWastedSortTrack::from)
                    .collect()
            })
        })
    }

    /// Clear all tracks with expired life
    ///
    #[pyo3(name = "clear_wasted", signature = ())]
    pub fn clear_wasted_py(&mut self) {
        Python::with_gil(|py| {
            py.allow_threads(|| self.clear_wasted());
        })
    }

    /// Get idle tracks with not expired life
    ///
    #[pyo3(name = "idle_tracks", signature = ())]
    pub fn idle_tracks_py(&mut self) -> Vec<SortTrack> {
        self.idle_tracks_with_scene_py(0)
    }

    /// Get idle tracks with not expired life
    ///
    #[pyo3(name = "idle_tracks_with_scene", signature = (scene_id))]
    pub fn idle_tracks_with_scene_py(&mut self, scene_id: i64) -> Vec<SortTrack> {
        Python::with_gil(|py| {
            py.allow_threads(|| self.idle_tracks_with_scene(scene_id.try_into().unwrap()))
        })
    }
}
