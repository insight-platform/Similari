use crate::prelude::{NoopNotifier, PositionalMetricType, TrackStoreBuilder, Universal2DBox};
use crate::store::track_distance::TrackDistanceOkIterator;
use crate::store::TrackStore;
use crate::trackers::batch::PredictionBatchRequest;
use crate::trackers::sort::metric::SortMetric;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::{SortAttributes, SortAttributesOptions};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crossbeam::channel::{Receiver, Sender};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, RwLock};
use std::thread::{spawn, JoinHandle};

type VotingSenderChannel = Sender<VotingCommands>;
type VotingReceiverChannel = Receiver<VotingCommands>;

enum VotingCommands {
    Distances(
        TrackDistanceOkIterator<Universal2DBox>,
        PredictionBatchRequest<(Universal2DBox, Option<i64>)>,
    ),
    Exit,
}

#[pyclass]
pub struct BatchSort {
    store: TrackStore<SortAttributes, SortMetric, Universal2DBox>,
    method: PositionalMetricType,
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

fn voting_thread(opts: Arc<SortAttributesOptions>, rx: VotingReceiverChannel) {
    while let Ok(command) = rx.recv() {
        match command {
            VotingCommands::Distances(it, request) => {
                drop(it);
                request.send((0, vec![]));
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

        let store = TrackStoreBuilder::new(distance_shards)
            .default_attributes(SortAttributes::new(opts.clone()))
            .metric(SortMetric::new(method))
            .notifier(NoopNotifier)
            .build();

        let voting_threads = (0..voting_shards)
            .into_iter()
            .map(|_e| {
                let (tx, rx) = crossbeam::channel::unbounded();
                let thread_opts = opts.clone();
                (tx, spawn(move || voting_thread(thread_opts, rx)))
            })
            .collect::<Vec<_>>();

        Self {
            store,
            method,
            opts,
            voting_threads,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
