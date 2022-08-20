use crate::prelude::{
    NoopNotifier, ObservationBuilder, PositionalMetricType, SortTrack, TrackStoreBuilder,
    Universal2DBox,
};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::track::Track;
use crate::trackers::batch::{PredictionBatchRequest, SceneTracks};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::metric::SortMetric;
use crate::trackers::sort::sort_py::PySortPredictionBatchRequest;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{
    AutoWaste, PyPositionalMetricType, PyWastedSortTrack, SortAttributes, SortAttributesOptions,
    SortAttributesUpdate, SortLookup, DEFAULT_AUTO_WASTE_PERIODICITY,
    MAHALANOBIS_NEW_TRACK_THRESHOLD,
};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::trackers::tracker_api::TrackerAPI;
use crate::voting::Voting;
use crossbeam::channel::{Receiver, Sender};
use log::warn;
use pyo3::prelude::*;
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

#[pyclass]
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
                    let track_id: u64 = if let Some(dest) = winners.get(&source) {
                        let dest = dest[0];
                        if dest == source {
                            let tid = {
                                let mut track_id = track_id.write().unwrap();
                                *track_id += 1;
                                *track_id
                            };
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
                        let tid = {
                            let mut track_id = track_id.write().unwrap();
                            *track_id += 1;
                            *track_id
                        };
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
    pub fn new(
        distance_shards: usize,
        voting_shards: usize,
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
            .into_iter()
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

#[pymethods]
impl BatchSort {
    #[new]
    #[args(
        distance_shards = "4",
        voting_shards = "4",
        bbox_history = "1",
        max_idle_epochs = "5",
        spatio_temporal_constraints = "None",
        min_confidence = "0.05"
    )]
    pub fn new_py(
        distance_shards: i64,
        voting_shards: i64,
        bbox_history: i64,
        max_idle_epochs: i64,
        method: PyPositionalMetricType,
        min_confidence: f32,
        spatio_temporal_constraints: Option<SpatioTemporalConstraints>,
    ) -> Self {
        Self::new(
            distance_shards
                .try_into()
                .expect("Positive number expected"),
            voting_shards.try_into().expect("Positive number expected"),
            bbox_history.try_into().expect("Positive number expected"),
            max_idle_epochs
                .try_into()
                .expect("Positive number expected"),
            method.0,
            min_confidence,
            spatio_temporal_constraints,
        )
    }

    #[pyo3(name = "skip_epochs", text_signature = "($self, n)")]
    fn skip_epochs_py(&mut self, n: i64) {
        assert!(n > 0);
        self.skip_epochs(n.try_into().unwrap())
    }

    #[pyo3(
        name = "skip_epochs_for_scene",
        text_signature = "($self, scene_id, n)"
    )]
    fn skip_epochs_for_scene_py(&mut self, scene_id: i64, n: i64) {
        assert!(n > 0 && scene_id >= 0);
        self.skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap())
    }

    /// Get the amount of stored tracks per shard
    ///
    #[pyo3(name = "shard_stats", text_signature = "($self)")]
    fn shard_stats_py(&self) -> Vec<i64> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        py.allow_threads(|| {
            self.store
                .read()
                .unwrap()
                .shard_stats()
                .into_iter()
                .map(|e| i64::try_from(e).unwrap())
                .collect()
        })
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    #[pyo3(name = "current_epoch", text_signature = "($self)")]
    fn current_epoch_py(&self) -> i64 {
        self.current_epoch_with_scene(0).try_into().unwrap()
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    #[pyo3(
        name = "current_epoch_with_scene",
        text_signature = "($self, scene_id)"
    )]
    fn current_epoch_with_scene_py(&self, scene_id: i64) -> isize {
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
    #[pyo3(name = "predict", text_signature = "($self, batch)")]
    fn predict_py(&mut self, batch: PySortPredictionBatchRequest) {
        self.predict(batch.batch);
    }

    /// Remove all the tracks with expired life
    ///
    #[pyo3(name = "wasted", text_signature = "($self)")]
    fn wasted_py(&mut self) -> Vec<PyWastedSortTrack> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        py.allow_threads(|| {
            self.wasted()
                .into_iter()
                .map(PyWastedSortTrack::from)
                .collect()
        })
    }

    /// Clear all tracks with expired life
    ///
    #[pyo3(name = "clear_wasted", text_signature = "($self)")]
    pub fn clear_wasted_py(&mut self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        py.allow_threads(|| self.clear_wasted());
    }

    /// Get idle tracks with not expired life
    ///
    #[pyo3(name = "idle_tracks", text_signature = "($self)")]
    pub fn idle_tracks_py(&mut self) -> Vec<SortTrack> {
        self.idle_tracks_with_scene_py(0)
    }

    /// Get idle tracks with not expired life
    ///
    #[pyo3(name = "idle_tracks_with_scene", text_signature = "($self)")]
    pub fn idle_tracks_with_scene_py(&mut self, scene_id: i64) -> Vec<SortTrack> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        py.allow_threads(|| self.idle_tracks_with_scene(scene_id.try_into().unwrap()))
    }
}
