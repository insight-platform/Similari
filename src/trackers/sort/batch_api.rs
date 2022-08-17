use crate::prelude::{
    NoopNotifier, ObservationBuilder, PositionalMetricType, TrackStoreBuilder, Universal2DBox,
};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::track::Track;
use crate::trackers::batch::{PredictionBatchRequest, SceneTracks};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::metric::SortMetric;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{SortAttributes, SortAttributesOptions, SortAttributesUpdate};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::voting::Voting;
use crossbeam::channel::{Receiver, Sender};
use log::warn;
use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Condvar, Mutex, RwLock};
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
    opts: Arc<SortAttributesOptions>,
    voting_threads: Vec<(VotingSenderChannel, JoinHandle<()>)>,
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
                        PositionalMetricType::Mahalanobis => 0.1,
                        PositionalMetricType::IoU(t) => t,
                    },
                    candidates_num,
                    tracks_num,
                );

                let winners = voting.winners(distances);
                let mut res = Vec::default();
                for t in tracks {
                    let source = t.get_track_id();
                    let track_id: u64 = if let Some(dest) = winners.get(&source) {
                        let dest = dest[0];
                        if dest == source {
                            store
                                .write()
                                .expect("Access to store must always succeed")
                                .add_track(t)
                                .unwrap();
                            source
                        } else {
                            store
                                .write()
                                .expect("Access to store must always succeed")
                                .merge_external(dest, &t, Some(&[0]), false)
                                .unwrap();
                            dest
                        }
                    } else {
                        store
                            .write()
                            .expect("Access to store must always succeed")
                            .add_track(t)
                            .unwrap();
                        source
                    };

                    let track = {
                        let store = store.read().expect("Access to store must always succeed");
                        let shard = store.get_store(track_id as usize);
                        shard.get(&track_id).unwrap().clone()
                    };

                    res.push(track.into())
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
                .metric(SortMetric::new(method))
                .notifier(NoopNotifier)
                .build(),
        ));

        let voting_threads = (0..voting_shards)
            .into_iter()
            .map(|_e| {
                let (tx, rx) = crossbeam::channel::unbounded();
                let thread_store = store.clone();
                (tx, spawn(move || voting_thread(thread_store, rx, method)))
            })
            .collect::<Vec<_>>();

        Self {
            monitor: None,
            store,
            opts,
            voting_threads,
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
}

#[cfg(test)]
mod tests {
    use crate::prelude::PositionalMetricType::Mahalanobis;
    use crate::trackers::sort::batch_api::BatchSort;

    #[test]
    fn new_drop() {
        let bs = BatchSort::new(1, 1, 1, 1, Mahalanobis, None);
        drop(bs);
    }
}
