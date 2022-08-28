use crate::prelude::{
    NoopNotifier, ObservationBuilder, PositionalMetricType, SortTrack, TrackStoreBuilder,
    VisualSortObservation, VisualSortOptions,
};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::track::utils::FromVec;
use crate::track::{Feature, Track};
use crate::trackers::batch::{PredictionBatchRequest, PredictionBatchResult, SceneTracks};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::{
    AutoWaste, SortAttributesOptions, DEFAULT_AUTO_WASTE_PERIODICITY,
    MAHALANOBIS_NEW_TRACK_THRESHOLD,
};
use crate::trackers::tracker_api::TrackerAPI;
use crate::trackers::visual_sort::metric::{VisualMetric, VisualMetricOptions};
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual_sort::track_attributes::{
    VisualAttributes, VisualAttributesUpdate, VisualSortLookup,
};
use crate::trackers::visual_sort::visual_sort_py::PyVisualSortPredictionBatchRequest;
use crate::trackers::visual_sort::voting::VisualVoting;
use crate::trackers::visual_sort::PyWastedVisualSortTrack;
use crate::utils::clipping::bbox_own_areas::{
    exclusively_owned_areas, exclusively_owned_areas_normalized_shares,
};
use crate::voting::Voting;
use crossbeam::channel::{Receiver, Sender};
use log::warn;
use pyo3::prelude::*;
use rand::Rng;
use std::mem;
use std::sync::{Arc, Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::{spawn, JoinHandle};

type VotingSenderChannel = Sender<VotingCommands>;
type VotingReceiverChannel = Receiver<VotingCommands>;

type MiddlewareVisualSortTrackStore =
    TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes>;
type MiddlewareSortTrack = Track<VisualAttributes, VisualMetric, VisualObservationAttributes>;
type BatchBusyMonitor = Arc<(Mutex<usize>, Condvar)>;

enum VotingCommands {
    Distances {
        scene_id: u64,
        distances: TrackDistanceOkIterator<VisualObservationAttributes>,
        channel: Sender<SceneTracks>,
        tracks: Vec<MiddlewareSortTrack>,
        monitor: BatchBusyMonitor,
    },
    Exit,
}

// /// Easy to use Visual SORT tracker implementation
// ///
#[pyclass(text_signature = "(distance_shards, voting_shards, opts)")]
pub struct BatchVisualSort {
    monitor: Option<BatchBusyMonitor>,
    store: Arc<RwLock<MiddlewareVisualSortTrackStore>>,
    wasted_store: RwLock<MiddlewareVisualSortTrackStore>,
    metric_opts: Arc<VisualMetricOptions>,
    track_opts: Arc<SortAttributesOptions>,
    voting_threads: Vec<(VotingSenderChannel, JoinHandle<()>)>,
    auto_waste: AutoWaste,
}

impl Drop for BatchVisualSort {
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
    store: Arc<RwLock<MiddlewareVisualSortTrackStore>>,
    rx: VotingReceiverChannel,
    metric_opts: Arc<VisualMetricOptions>,
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
                let voting = VisualVoting::new(
                    match metric_opts.positional_kind {
                        PositionalMetricType::Mahalanobis => MAHALANOBIS_NEW_TRACK_THRESHOLD,
                        PositionalMetricType::IoU(t) => t,
                    },
                    f32::MAX,
                    metric_opts.visual_min_votes,
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
                        let (dest, vt) = dest[0];
                        if dest == source {
                            t.set_track_id(tid);
                            store.write().unwrap().add_track(t).unwrap();
                            tid
                        } else {
                            t.add_observation(
                                0,
                                None,
                                None,
                                Some(VisualAttributesUpdate::new_voting_type(vt)),
                            )
                            .unwrap();
                            store
                                .write()
                                .unwrap()
                                .merge_external(dest, &t, Some(&[0]), false)
                                .unwrap();
                            dest
                        }
                    } else {
                        t.set_track_id(tid);
                        store.write().unwrap().add_track(t).unwrap();
                        tid
                    };

                    let lock = store.read().unwrap();
                    let store = lock.get_store(track_id as usize);
                    let track = store.get(&track_id).unwrap();

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

impl BatchVisualSort {
    pub fn new(distance_shards: usize, voting_shards: usize, opts: &VisualSortOptions) -> Self {
        let (track_opts, metric) = opts.clone().build();
        let track_opts = Arc::new(track_opts);
        let metric_opts = metric.opts.clone();
        let store = Arc::new(RwLock::new(
            TrackStoreBuilder::new(distance_shards)
                .default_attributes(VisualAttributes::new(track_opts.clone()))
                .metric(metric.clone())
                .notifier(NoopNotifier)
                .build(),
        ));

        let wasted_store = RwLock::new(
            TrackStoreBuilder::new(distance_shards)
                .default_attributes(VisualAttributes::new(track_opts.clone()))
                .metric(metric)
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
                let thread_metric_opts = metric_opts.clone();

                (
                    tx,
                    spawn(move || {
                        voting_thread(thread_store, rx, thread_metric_opts, thread_track_id)
                    }),
                )
            })
            .collect::<Vec<_>>();

        Self {
            monitor: None,
            store,
            wasted_store,
            track_opts,
            metric_opts,
            voting_threads,
            auto_waste: AutoWaste {
                periodicity: DEFAULT_AUTO_WASTE_PERIODICITY,
                counter: DEFAULT_AUTO_WASTE_PERIODICITY,
            },
        }
    }

    pub fn predict(&mut self, batch_request: PredictionBatchRequest<VisualSortObservation>) {
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

        for (i, (scene_id, observations)) in batch_request.get_batch().iter().enumerate() {
            let mut percentages = Vec::default();
            let use_own_area_percentage =
                self.metric_opts.visual_minimal_own_area_percentage_collect
                    + self.metric_opts.visual_minimal_own_area_percentage_use
                    > 0.0;

            if use_own_area_percentage {
                percentages.reserve(observations.len());
                let boxes = observations
                    .iter()
                    .map(|o| &o.bounding_box)
                    .collect::<Vec<_>>();

                percentages = exclusively_owned_areas_normalized_shares(
                    boxes.as_ref(),
                    exclusively_owned_areas(boxes.as_ref()).as_ref(),
                );
            }

            let mut rng = rand::thread_rng();
            let epoch = self.track_opts.next_epoch(*scene_id).unwrap();

            let tracks = observations
                .iter()
                .enumerate()
                .map(|(i, o)| {
                    self.store
                        .read()
                        .expect("Access to store must always succeed")
                        .new_track(rng.gen())
                        .observation({
                            let mut obs = ObservationBuilder::new(0).observation_attributes(
                                if use_own_area_percentage {
                                    VisualObservationAttributes::with_own_area_percentage(
                                        o.feature_quality.unwrap_or(1.0),
                                        o.bounding_box.clone(),
                                        percentages[i],
                                    )
                                } else {
                                    VisualObservationAttributes::new(
                                        o.feature_quality.unwrap_or(1.0),
                                        o.bounding_box.clone(),
                                    )
                                },
                            );

                            if let Some(feature) = o.feature {
                                obs = obs.observation(Feature::from_vec(feature));
                            }

                            obs.track_attributes_update(
                                VisualAttributesUpdate::new_init_with_scene(
                                    epoch,
                                    *scene_id,
                                    o.custom_object_id,
                                ),
                            )
                            .build()
                        })
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
            .lookup(VisualSortLookup::IdleLookup(scene_id))
            .iter()
            .map(|(track_id, _status)| {
                let shard = store.get_store(*track_id as usize);
                let track = shard.get(track_id).unwrap();
                SortTrack::from(track)
            })
            .collect()
    }
}

impl
    TrackerAPI<
        VisualAttributes,
        VisualMetric,
        VisualObservationAttributes,
        SortAttributesOptions,
        NoopNotifier,
    > for BatchVisualSort
{
    fn get_auto_waste_obj_mut(&mut self) -> &mut AutoWaste {
        &mut self.auto_waste
    }

    fn get_opts(&self) -> &SortAttributesOptions {
        &self.track_opts
    }

    fn get_main_store_mut(&mut self) -> RwLockWriteGuard<MiddlewareVisualSortTrackStore> {
        self.store.write().unwrap()
    }

    fn get_wasted_store_mut(&mut self) -> RwLockWriteGuard<MiddlewareVisualSortTrackStore> {
        self.wasted_store.write().unwrap()
    }

    fn get_main_store(&self) -> RwLockReadGuard<MiddlewareVisualSortTrackStore> {
        self.store.read().unwrap()
    }

    fn get_wasted_store(&self) -> RwLockReadGuard<MiddlewareVisualSortTrackStore> {
        self.wasted_store.read().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::{
        BoundingBox, PositionalMetricType, VisualSortMetricType, VisualSortObservation,
        VisualSortOptions,
    };
    use crate::trackers::batch::PredictionBatchRequest;
    use crate::trackers::visual_sort::batch_api::BatchVisualSort;

    #[test]
    fn test() {
        let opts = VisualSortOptions::default()
            .max_idle_epochs(3)
            .kept_history_length(3)
            .visual_metric(VisualSortMetricType::Euclidean(1.0))
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(2)
            .visual_minimal_area(5.0)
            .visual_minimal_quality_use(0.45)
            .visual_minimal_quality_collect(0.7)
            .visual_max_observations(3)
            .visual_min_votes(2);

        let mut tracker = BatchVisualSort::new(1, 1, &opts);
        let (mut batch, predictions) = PredictionBatchRequest::<VisualSortObservation>::new();
        let vec = &vec![1.0, 1.0];
        batch.add(
            1,
            VisualSortObservation::new(
                Some(vec),
                Some(0.9),
                BoundingBox::new(1.0, 1.0, 3.0, 5.0).as_xyaah(),
                Some(13),
            ),
        );
        tracker.predict(batch);
        for _ in 0..predictions.batch_size() {
            let (scene, tracks) = predictions.get();
            assert_eq!(scene, 1);
            assert_eq!(tracks.len(), 1);
            dbg!(tracks);
        }

        let (mut batch, predictions) = PredictionBatchRequest::<VisualSortObservation>::new();
        let vec1 = &vec![1.0, 1.0];
        let vec2 = &vec![0.1, 0.15];
        batch.add(
            1,
            VisualSortObservation::new(
                Some(vec1),
                Some(0.9),
                BoundingBox::new(1.0, 1.0, 3.0, 5.0).as_xyaah(),
                Some(13),
            ),
        );

        batch.add(
            2,
            VisualSortObservation::new(
                Some(vec2),
                Some(0.87),
                BoundingBox::new(5.0, 10.0, 3.0, 5.0).as_xyaah(),
                Some(23),
            ),
        );

        batch.add(
            2,
            VisualSortObservation::new(
                None,
                None,
                BoundingBox::new(25.0, 15.0, 3.0, 5.0).as_xyaah(),
                Some(33),
            ),
        );

        tracker.predict(batch);
        for _ in 0..predictions.batch_size() {
            let (scene, tracks) = predictions.get();
            dbg!(scene, tracks);
        }
    }
}

#[pymethods]
impl BatchVisualSort {
    #[new]
    #[args(distance_shards = "4", voting_shards = "4")]
    pub fn new_py(distance_shards: i64, voting_shards: i64, opts: &VisualSortOptions) -> Self {
        Self::new(
            distance_shards
                .try_into()
                .expect("Positive number expected"),
            voting_shards.try_into().expect("Positive number expected"),
            opts,
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
    fn predict_py(
        &mut self,
        py_batch: PyVisualSortPredictionBatchRequest,
    ) -> PredictionBatchResult {
        let (mut batch, res) = PredictionBatchRequest::<VisualSortObservation>::new();
        for (scene_id, observations) in py_batch.batch.get_batch() {
            for o in observations {
                let f = o.feature.as_ref();
                batch.add(
                    *scene_id,
                    VisualSortObservation::new(
                        f,
                        o.feature_quality,
                        o.bounding_box.clone(),
                        o.custom_object_id,
                    ),
                );
            }
        }
        self.predict(batch);
        res
    }

    /// Remove all the tracks with expired life
    ///
    #[pyo3(name = "wasted", text_signature = "($self)")]
    fn wasted_py(&mut self) -> Vec<PyWastedVisualSortTrack> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        py.allow_threads(|| {
            self.wasted()
                .into_iter()
                .map(PyWastedVisualSortTrack::from)
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
    pub fn idle_tracks_py(&mut self, scene_id: i64) -> Vec<SortTrack> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        py.allow_threads(|| self.idle_tracks_with_scene(scene_id.try_into().unwrap()))
    }
}
