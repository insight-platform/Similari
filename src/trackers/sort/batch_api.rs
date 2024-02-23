use crate::prelude::{
    NoopNotifier, ObservationBuilder, PositionalMetricType, SortTrack, TrackStoreBuilder,
    Universal2DBox,
};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::track::Track;
use crate::trackers::batch::{PredictionBatchRequest, PredictionBatchResult, SceneTracks};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::metric::SortMetric;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{
    AutoWaste, SortAttributes, SortAttributesOptions, SortAttributesUpdate, SortLookup,
    DEFAULT_AUTO_WASTE_PERIODICITY, MAHALANOBIS_NEW_TRACK_THRESHOLD,
};

use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::trackers::tracker_api::TrackerAPI;
use crate::voting::Voting;
use crossbeam::channel::{Receiver, Sender};
use log::warn;
use rand::Rng;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::{spawn, JoinHandle};

type VotingSenderChannel = Sender<VotingCommands>;
type VotingReceiverChannel = Receiver<VotingCommands>;

type MiddlewareSortTrackStore = TrackStore<SortAttributes, SortMetric, Universal2DBox>;
type MiddlewareSortTrack = Track<SortAttributes, SortMetric, Universal2DBox>;
type BatchBusyMonitor = Arc<(Mutex<usize>, Condvar)>;

enum VotingCommands {
    Distances {
        scene_id: u64,
        distances: TrackDistanceOkIterator<Universal2DBox>,
        channel: Sender<SceneTracks>,
        tracks: Vec<MiddlewareSortTrack>,
        monitor: BatchBusyMonitor,
    },
    Exit,
}

pub struct BatchSort {
    monitor: Option<BatchBusyMonitor>,
    store: Arc<RwLock<MiddlewareSortTrackStore>>,
    wasted_store: RwLock<MiddlewareSortTrackStore>,
    opts: Arc<SortAttributesOptions>,
    voting_threads: Vec<(VotingSenderChannel, JoinHandle<()>)>,
    auto_waste: AutoWaste,
}

impl Drop for BatchSort {
    fn drop(&mut self) {
        let voting_threads = mem::take(&mut self.voting_threads);
        for (tx, t) in voting_threads {
            tx.send(VotingCommands::Exit)
                .expect("Voting thread must be alive.");
            drop(tx);
            t.join()
                .expect("Voting thread is expected to shutdown successfully.");
        }
    }
}

fn voting_thread(
    store: Arc<RwLock<MiddlewareSortTrackStore>>,
    rx: VotingReceiverChannel,
    method: PositionalMetricType,
    track_id: Arc<RwLock<u64>>,
) {
    while let Ok(command) = rx.recv() {
        match command {
            VotingCommands::Distances {
                scene_id,
                distances,
                channel,
                tracks,
                monitor,
            } => {
                let candidates_num = tracks.len();
                let tracks_num = {
                    let store = store.read().expect("Access to store must always succeed");
                    store.shard_stats().iter().sum()
                };

                let voting = SortVoting::new(
                    match method {
                        PositionalMetricType::Mahalanobis => MAHALANOBIS_NEW_TRACK_THRESHOLD,
                        PositionalMetricType::IoU(t) => t,
                    },
                    candidates_num,
                    tracks_num,
                );

                let winners = voting.winners(distances);
                let mut res = Vec::default();
                for mut t in tracks {
                    let source = t.get_track_id();
                    let tid = {
                        let mut track_id = track_id.write().unwrap();
                        *track_id += 1;
                        *track_id
                    };
                    let track_id: u64 = if let Some(dest) = winners.get(&source) {
                        let dest = dest[0];
                        if dest == source {
                            t.set_track_id(tid);
                            store
                                .write()
                                .expect("Access to store must always succeed")
                                .add_track(t)
                                .unwrap();
                            tid
                        } else {
                            store
                                .write()
                                .expect("Access to store must always succeed")
                                .merge_external(dest, &t, Some(&[0]), false)
                                .unwrap();
                            dest
                        }
                    } else {
                        t.set_track_id(tid);
                        store
                            .write()
                            .expect("Access to store must always succeed")
                            .add_track(t)
                            .unwrap();
                        tid
                    };

                    let store = store.read().expect("Access to store must always succeed");
                    let shard = store.get_store(track_id as usize);
                    let track = shard.get(&track_id).unwrap();

                    res.push(SortTrack::from(track))
                }
                let res = channel.send((scene_id, res));
                if let Err(e) = res {
                    warn!("Unable to send results to a caller, likely the caller already closed the channel. Error is: {:?}", e);
                }
                let (lock, cvar) = &*monitor;
                let mut lock = lock.lock().unwrap();
                *lock -= 1;
                cvar.notify_one();
            }
            VotingCommands::Exit => break,
        }
    }
}

impl BatchSort {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        distance_shards: usize,
        voting_shards: usize,
        bbox_history: usize,
        max_idle_epochs: usize,
        method: PositionalMetricType,
        min_confidence: f32,
        spatio_temporal_constraints: Option<SpatioTemporalConstraints>,
        kalman_position_weight: f32,
        kalman_velocity_weight: f32,
    ) -> Self {
        assert!(bbox_history > 0);
        let epoch_db = RwLock::new(HashMap::default());
        let opts = Arc::new(SortAttributesOptions::new(
            Some(epoch_db),
            max_idle_epochs,
            bbox_history,
            spatio_temporal_constraints.unwrap_or_default(),
            kalman_position_weight,
            kalman_velocity_weight,
        ));

        let store = Arc::new(RwLock::new(
            TrackStoreBuilder::new(distance_shards)
                .default_attributes(SortAttributes::new(opts.clone()))
                .metric(SortMetric::new(method, min_confidence))
                .notifier(NoopNotifier)
                .build(),
        ));

        let wasted_store = RwLock::new(
            TrackStoreBuilder::new(distance_shards)
                .default_attributes(SortAttributes::new(opts.clone()))
                .metric(SortMetric::new(method, min_confidence))
                .notifier(NoopNotifier)
                .build(),
        );

        let track_id = Arc::new(RwLock::new(0));

        let voting_threads = (0..voting_shards)
            .map(|_e| {
                let (tx, rx) = crossbeam::channel::unbounded();
                let thread_store = store.clone();
                let thread_track_id = track_id.clone();
                (
                    tx,
                    spawn(move || voting_thread(thread_store, rx, method, thread_track_id)),
                )
            })
            .collect::<Vec<_>>();

        Self {
            monitor: None,
            store,
            wasted_store,
            opts,
            voting_threads,
            auto_waste: AutoWaste {
                periodicity: DEFAULT_AUTO_WASTE_PERIODICITY,
                counter: DEFAULT_AUTO_WASTE_PERIODICITY,
            },
        }
    }

    pub fn predict(
        &mut self,
        batch_request: PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
    ) {
        if self.auto_waste.counter == 0 {
            self.auto_waste();
            self.auto_waste.counter = self.auto_waste.periodicity;
        } else {
            self.auto_waste.counter -= 1;
        }

        if let Some(m) = &self.monitor {
            let (lock, cvar) = &**m;
            let _guard = cvar.wait_while(lock.lock().unwrap(), |v| *v > 0).unwrap();
        }

        self.monitor = Some(Arc::new((
            Mutex::new(batch_request.batch_size()),
            Condvar::new(),
        )));

        for (i, (scene_id, bboxes)) in batch_request.get_batch().iter().enumerate() {
            let mut rng = rand::thread_rng();
            let epoch = self.opts.next_epoch(*scene_id).unwrap();

            let tracks = bboxes
                .iter()
                .map(|(bb, custom_object_id)| {
                    self.store
                        .read()
                        .expect("Access to store must always succeed")
                        .new_track(rng.gen())
                        .observation(
                            ObservationBuilder::new(0)
                                .observation_attributes(bb.clone())
                                .track_attributes_update(SortAttributesUpdate::new_with_scene(
                                    epoch,
                                    *scene_id,
                                    *custom_object_id,
                                ))
                                .build(),
                        )
                        .build()
                        .expect("Track creation must always succeed!")
                })
                .collect::<Vec<_>>();

            let (dists, errs) = {
                let mut store = self
                    .store
                    .write()
                    .expect("Access to store must always succeed");
                store.foreign_track_distances(tracks.clone(), 0, false)
            };

            assert!(errs.all().is_empty());
            let thread_id = i % self.voting_threads.len();
            self.voting_threads[thread_id]
                .0
                .send(VotingCommands::Distances {
                    monitor: self.monitor.as_ref().unwrap().clone(),
                    scene_id: *scene_id,
                    distances: dists.into_iter(),
                    channel: batch_request.get_sender(),
                    tracks,
                })
                .expect("Sending voting request to voting thread must not fail");
        }
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
    for BatchSort
{
    fn get_auto_waste_obj_mut(&mut self) -> &mut AutoWaste {
        &mut self.auto_waste
    }

    fn get_opts(&self) -> &SortAttributesOptions {
        &self.opts
    }

    fn get_main_store_mut(&mut self) -> RwLockWriteGuard<MiddlewareSortTrackStore> {
        self.store.write().unwrap()
    }

    fn get_wasted_store_mut(&mut self) -> RwLockWriteGuard<MiddlewareSortTrackStore> {
        self.wasted_store.write().unwrap()
    }

    fn get_main_store(&self) -> RwLockReadGuard<MiddlewareSortTrackStore> {
        self.store.read().unwrap()
    }

    fn get_wasted_store(&self) -> RwLockReadGuard<MiddlewareSortTrackStore> {
        self.wasted_store.read().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct SortPredictionBatchRequest {
    pub batch: PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
    pub result: Option<PredictionBatchResult>,
}

impl SortPredictionBatchRequest {
    pub fn new() -> Self {
        let (batch, result) = PredictionBatchRequest::new();

        Self {
            batch,
            result: Some(result),
        }
    }

    pub fn add(&mut self, scene_id: u64, bbox: Universal2DBox, custom_object_id: Option<i64>) {
        self.batch.add(scene_id, (bbox, custom_object_id))
    }
}

impl Default for SortPredictionBatchRequest {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::{
        trackers::{
            batch::python::PyPredictionBatchResult,
            sort::{
                python::{PyPositionalMetricType, PySortTrack, PyWastedSortTrack},
                WastedSortTrack,
            },
            spatio_temporal_constraints::python::PySpatioTemporalConstraints,
            tracker_api::TrackerAPI,
        },
        utils::bbox::python::PyUniversal2DBox,
    };

    use super::{BatchSort, SortPredictionBatchRequest};
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "BatchSort")]
    pub struct PyBatchSort(pub(crate) BatchSort);

    #[pymethods]
    impl PyBatchSort {
        #[new]
        #[pyo3(signature = (
        distance_shards = 4,
        voting_shards = 4,
        bbox_history = 1,
        max_idle_epochs = 5,
        method = None,
        min_confidence = 0.05,
        spatio_temporal_constraints = None,
        kalman_position_weight = 1.0 / 20.0,
        kalman_velocity_weight = 1.0 / 160.0
    ))]
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            distance_shards: i64,
            voting_shards: i64,
            bbox_history: i64,
            max_idle_epochs: i64,
            method: Option<PyPositionalMetricType>,
            min_confidence: f32,
            spatio_temporal_constraints: Option<PySpatioTemporalConstraints>,
            kalman_position_weight: f32,
            kalman_velocity_weight: f32,
        ) -> Self {
            Self(BatchSort::new(
                distance_shards
                    .try_into()
                    .expect("Positive number expected"),
                voting_shards.try_into().expect("Positive number expected"),
                bbox_history.try_into().expect("Positive number expected"),
                max_idle_epochs
                    .try_into()
                    .expect("Positive number expected"),
                method.unwrap_or(PyPositionalMetricType::maha()).0,
                min_confidence,
                spatio_temporal_constraints.map(|x| x.0),
                kalman_position_weight,
                kalman_velocity_weight,
            ))
        }

        #[pyo3(signature = (n))]
        fn skip_epochs(&mut self, n: i64) {
            assert!(n > 0);
            self.0.skip_epochs(n.try_into().unwrap())
        }

        #[pyo3(signature = (scene_id, n))]
        fn skip_epochs_for_scene(&mut self, scene_id: i64, n: i64) {
            assert!(n > 0 && scene_id >= 0);
            self.0
                .skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap())
        }

        /// Get the amount of stored tracks per shard
        ///
        #[pyo3(signature = ())]
        fn shard_stats(&self) -> Vec<i64> {
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    self.0
                        .store
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
        #[pyo3(signature = ())]
        fn current_epoch(&self) -> i64 {
            self.0.current_epoch_with_scene(0).try_into().unwrap()
        }

        /// Get the current epoch for `scene_id`
        ///
        /// # Parameters
        /// * `scene_id` - scene id
        ///
        #[pyo3(
        signature = (scene_id)
    )]
        fn current_epoch_with_scene(&self, scene_id: i64) -> isize {
            assert!(scene_id >= 0);
            self.0
                .current_epoch_with_scene(scene_id.try_into().unwrap())
                .try_into()
                .unwrap()
        }

        /// Receive tracking information for observed bboxes of `scene_id` == 0
        ///
        /// # Parameters
        /// * `bboxes` - bounding boxes received from a detector
        ///
        #[pyo3(signature = (batch))]
        fn predict(&mut self, mut batch: PySortPredictionBatchRequest) -> PyPredictionBatchResult {
            self.0.predict(batch.0.batch);
            PyPredictionBatchResult(batch.0.result.take().unwrap())
        }

        /// Remove all the tracks with expired life
        ///
        #[pyo3(signature = ())]
        fn wasted(&mut self) -> Vec<PyWastedSortTrack> {
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    self.0
                        .wasted()
                        .into_iter()
                        .map(WastedSortTrack::from)
                        .map(PyWastedSortTrack)
                        .collect()
                })
            })
        }

        /// Clear all tracks with expired life
        ///
        #[pyo3(signature = ())]
        pub fn clear_wasted(&mut self) {
            Python::with_gil(|py| {
                py.allow_threads(|| self.0.clear_wasted());
            })
        }

        /// Get idle tracks with not expired life
        ///
        #[pyo3(signature = (scene_id))]
        pub fn idle_tracks(&mut self, scene_id: i64) -> Vec<PySortTrack> {
            Python::with_gil(|py| {
                py.allow_threads(|| unsafe {
                    std::mem::transmute(self.0.idle_tracks_with_scene(scene_id.try_into().unwrap()))
                })
            })
        }
    }

    #[pyclass]
    #[pyo3(name = "SortPredictionBatchRequest")]
    #[derive(Debug, Clone)]
    pub struct PySortPredictionBatchRequest(pub(crate) SortPredictionBatchRequest);

    #[pymethods]
    impl PySortPredictionBatchRequest {
        #[new]
        fn new() -> Self {
            Self(SortPredictionBatchRequest::new())
        }

        fn add(&mut self, scene_id: u64, bbox: PyUniversal2DBox, custom_object_id: Option<i64>) {
            self.0.add(scene_id, bbox.0, custom_object_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::BoundingBox;
    use crate::prelude::PositionalMetricType::Mahalanobis;
    use crate::trackers::batch::PredictionBatchRequest;
    use crate::trackers::sort::batch_api::BatchSort;
    use crate::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;

    #[test]
    fn new_drop() {
        let mut bs = BatchSort::new(
            1,
            1,
            1,
            1,
            Mahalanobis,
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
            None,
            1.0 / 20.0,
            1.0 / 160.0,
        );
        let (mut batch, res) = PredictionBatchRequest::new();
        batch.add(0, (BoundingBox::new(0.0, 0.0, 5.0, 10.0).into(), Some(1)));
        batch.add(1, (BoundingBox::new(0.0, 0.0, 5.0, 10.0).into(), Some(2)));

        bs.predict(batch);

        for _ in 0..res.batch_size() {
            let data = res.get();
            dbg!(data);
        }
    }
}
