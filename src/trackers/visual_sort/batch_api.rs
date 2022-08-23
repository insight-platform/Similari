use crate::prelude::{
    NoopNotifier, ObservationBuilder, PositionalMetricType, SortTrack, TrackStoreBuilder,
    VisualObservation, VisualSortOptions,
};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::track::utils::FromVec;
use crate::track::{Feature, Track};
use crate::trackers::batch::{PredictionBatchRequest, SceneTracks};
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
use crate::trackers::visual_sort::voting::VisualVoting;
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

    pub fn predict(&mut self, batch_request: PredictionBatchRequest<VisualObservation>) {
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
    #[test]
    fn test() {}
}
