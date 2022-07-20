use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{
    Observation, ObservationAttributes, ObservationMetric, ObservationMetricResult,
    ObservationSpec, Track, TrackAttributes, TrackStatus,
};
use crate::Errors;
use anyhow::Result;
use crossbeam::channel::{Receiver, Sender};
use log::{error, warn};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::{mem, thread};

#[derive(Clone)]
enum Commands<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    Drop(Sender<Results<OA>>),
    FindBaked(Sender<Results<OA>>),
    Distances(Arc<Track<TA, M, OA, N>>, u64, bool, Sender<Results<OA>>),
    Merge(
        u64,
        Track<TA, M, OA, N>,
        Vec<u64>,
        bool,
        Option<Sender<Results<OA>>>,
    ),
}

/// The type that provides lock-ed access to certain shard store
///
pub type StoreMutexGuard<'a, TA, M, FA, N> = MutexGuard<'a, HashMap<u64, Track<TA, M, FA, N>>>;

/// The type that provides the initial track that was in the store before it was merged into
/// target track
///
pub type OwnedMergeResult<TA, M, FA, N> = Result<Option<Track<TA, M, FA, N>>>;

pub type TrackDistances<T> = (Vec<ObservationMetricResult<T>>, Vec<TrackDistanceError<T>>);

#[derive(Debug)]
enum Results<OA>
where
    OA: ObservationAttributes,
{
    Distance(
        Vec<ObservationMetricResult<OA>>,
        Vec<TrackDistanceError<OA>>,
    ),
    BakedStatus(Vec<(u64, Result<TrackStatus>)>),
    Dropped,
    MergeResult(Result<()>),
}

/// Merge future result
///
pub struct FutureMergeResponse<OA>
where
    OA: ObservationAttributes,
{
    receiver: Receiver<Results<OA>>,
    _sender: Sender<Results<OA>>,
}

impl<OA> FutureMergeResponse<OA>
where
    OA: ObservationAttributes,
{
    pub fn get(&self) -> Result<()> {
        let res = self.receiver.recv();
        if res.is_err() {
            res?;
            unreachable!();
        }
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        !self.receiver.is_empty()
    }
}

/// Auxiliary type to express distance calculation errors
pub type TrackDistanceError<M> = Result<Vec<ObservationMetricResult<M>>>;

/// Track store container with accelerated similarity operations.
///
/// TrackStore is implemented for certain attributes (A), attribute update (U), and metric (M), so
/// it handles only such objects. You cannot store tracks with different attributes within the same
/// TrackStore.
///
/// The metric is also defined for TrackStore, however Metric implementation may have various metric
/// calculation options for concrete feature classes. E.g. FEAT1 may be calculated with Euclid distance,
/// while FEAT2 may be calculated with cosine. It is up to Metric implementor how the metric works.
///
/// TrackStore examples can be found at:
/// [examples/*](https://github.com/insight-platform/Similari/blob/main/examples).
///
pub struct TrackStore<TA, M, OA, N = NoopNotifier>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    attributes: TA,
    metric: M,
    notifier: N,
    num_shards: usize,
    #[allow(clippy::type_complexity)]
    stores: Arc<Vec<Mutex<HashMap<u64, Track<TA, M, OA, N>>>>>,
    // receiver: Receiver<Results<FA>>,
    #[allow(clippy::type_complexity)]
    executors: Vec<(Sender<Commands<TA, M, OA, N>>, JoinHandle<()>)>,
}

impl<TA, M, OA, N> Drop for TrackStore<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    fn drop(&mut self) {
        let executors = mem::take(&mut self.executors);
        let (results_sender, results_receiver) = crossbeam::channel::unbounded();
        for (s, j) in executors {
            s.send(Commands::Drop(results_sender.clone())).unwrap();
            let res = results_receiver.recv().unwrap();
            match res {
                Results::Dropped => {
                    j.join().unwrap();
                    drop(s);
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }
}

impl<TA, M, OA, N> Default for TrackStore<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    fn default() -> Self {
        Self::new(None, None, None, 1)
    }
}

/// The basic implementation should fit to most of needs.
///
impl<TA, M, OA, N> TrackStore<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    #[allow(clippy::type_complexity)]
    fn handle_store_ops(
        stores: Arc<Vec<Mutex<HashMap<u64, Track<TA, M, OA, N>>>>>,
        store_id: usize,
        commands_receiver: Receiver<Commands<TA, M, OA, N>>,
    ) {
        let store = stores.get(store_id).unwrap();
        while let Ok(c) = commands_receiver.recv() {
            match c {
                Commands::Drop(channel) => {
                    let _r = channel.send(Results::Dropped);
                    return;
                }
                Commands::FindBaked(channel) => {
                    let baked = store
                        .lock()
                        .unwrap()
                        .iter()
                        .flat_map(|(track_id, track)| {
                            match track.get_attributes().baked(&track.observations) {
                                Ok(status) => match status {
                                    TrackStatus::Pending => None,
                                    other => Some((*track_id, Ok(other))),
                                },
                                Err(e) => Some((*track_id, Err(e))),
                            }
                        })
                        .collect();
                    let r = channel.send(Results::BakedStatus(baked));
                    if let Err(_e) = r {
                        return;
                    }
                }
                Commands::Distances(track, feature_class, only_baked, channel) => {
                    let mut capacity = 0;
                    let res = store
                        .lock()
                        .unwrap()
                        .iter()
                        .flat_map(|(_, other)| {
                            if !only_baked {
                                let dists = track.distances(other, feature_class);
                                match dists {
                                    Ok(dists) => {
                                        capacity += dists.len();
                                        Some(Ok(track.metric.postprocess_distances(dists)))
                                    }
                                    Err(e) => match e.downcast_ref::<Errors>() {
                                        Some(Errors::IncompatibleAttributes) => None,
                                        _ => Some(Err(e)),
                                    },
                                }
                            } else {
                                match other.get_attributes().baked(&other.observations) {
                                    Ok(TrackStatus::Ready) => {
                                        let dists = track.distances(other, feature_class);
                                        match dists {
                                            Ok(dists) => {
                                                capacity += dists.len();
                                                Some(Ok(track.metric.postprocess_distances(dists)))
                                            }
                                            Err(e) => match e.downcast_ref::<Errors>() {
                                                Some(Errors::IncompatibleAttributes) => None,
                                                _ => Some(Err(e)),
                                            },
                                        }
                                    }
                                    _ => None,
                                }
                            }
                        })
                        .collect::<Vec<_>>();

                    let mut distances = Vec::with_capacity(capacity);
                    let mut errors = Vec::new();

                    for r in res {
                        match r {
                            Ok(dists) => {
                                distances.extend_from_slice(&dists);
                            }
                            e => errors.push(e),
                        }
                    }

                    let r = channel.send(Results::Distance(distances, errors));
                    if let Err(_e) = r {
                        return;
                    }
                }
                Commands::Merge(dest_id, src, classes, merge_history, channel_opt) => {
                    let mut store = store.lock().unwrap();
                    let dest = store.get_mut(&dest_id);

                    let res = match dest {
                        Some(dest) => {
                            if dest_id == src.track_id {
                                Err(Errors::SameTrackCalculation(dest_id).into())
                            } else if !classes.is_empty() {
                                dest.merge(&src, &classes, merge_history)
                            } else {
                                dest.merge(&src, &src.get_feature_classes(), merge_history)
                            }
                        }

                        None => Err(Errors::TrackNotFound(dest_id).into()),
                    };

                    if let Some(channel) = channel_opt {
                        if let Err(send_res) = channel.send(Results::MergeResult(res)) {
                            warn!("Receiver channel was dropped before the data sent into it. Error is: {:?}", send_res);
                        }
                    }
                }
            }
        }
    }

    /// Constructor method
    ///
    /// When you construct track store you may pass two initializer objects:
    /// * Metric
    /// * Attributes
    ///
    /// They will be used upon track creation to initialize per-track metric and attributes.
    /// They are cloned when a certain track is created.
    ///
    /// If `None` is passed, `Default` initializers are used.
    ///
    pub fn new(
        metric: Option<M>,
        default_attributes: Option<TA>,
        notifier: Option<N>,
        shards: usize,
    ) -> Self {
        let stores = Arc::new(
            (0..shards)
                .into_iter()
                .map(|_| Mutex::new(HashMap::default()))
                .collect::<Vec<_>>(),
        );
        let my_stores = stores.clone();

        Self {
            //receiver: results_receiver,
            num_shards: shards,
            notifier: if let Some(notifier) = notifier {
                notifier
            } else {
                N::default()
            },
            attributes: if let Some(a) = default_attributes {
                a
            } else {
                TA::default()
            },
            metric: if let Some(m) = metric {
                m
            } else {
                M::default()
            },
            stores: my_stores,
            executors: {
                (0..shards)
                    .into_iter()
                    .map(|s| {
                        let (commands_sender, commands_receiver) = crossbeam::channel::unbounded();
                        let stores = stores.clone();
                        let thread = thread::spawn(move || {
                            Self::handle_store_ops(stores, s, commands_receiver);
                        });
                        (commands_sender, thread)
                    })
                    .collect()
            },
        }
    }

    /// Method is used to find ready to use tracks within the store.
    ///
    /// The search is parallelized with Rayon. The results returned for tracks with
    /// * `TrackBakingStatus::Ready`,
    /// * `TrackBakingStatus::Wasted` or
    /// * `Err(e)`
    ///
    pub fn find_usable(&mut self) -> Vec<(u64, Result<TrackStatus>)> {
        let mut results = Vec::with_capacity(self.shard_stats().iter().sum());
        let (results_sender, results_receiver) = crossbeam::channel::unbounded();
        for (cmd, _) in &mut self.executors {
            cmd.send(Commands::FindBaked(results_sender.clone()))
                .unwrap();
        }
        for (_, _) in &mut self.executors {
            let res = results_receiver.recv().unwrap();
            match res {
                Results::BakedStatus(r) => {
                    results.extend(r);
                }
                _ => {
                    unreachable!();
                }
            }
        }
        results
    }

    /// Counts of objects per every store shard
    ///
    pub fn shard_stats(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for s in self.stores.iter() {
            result.push(s.lock().unwrap().len());
        }
        result
    }

    /// Pulls (and removes) requested tracks from the store.
    ///
    pub fn fetch_tracks(&mut self, tracks: &[u64]) -> Vec<Track<TA, M, OA, N>> {
        let mut res = Vec::default();
        for track_id in tracks {
            let mut tracks_shard = self.get_store(*track_id as usize);
            if let Some(t) = tracks_shard.remove(track_id) {
                res.push(t);
            }
        }
        res
    }

    /// Calculates distances for external track (not in track store) to all tracks in DB which are
    /// allowed.
    ///
    /// # Arguments
    /// * `tracks` - batch external tracks that is used as distance subjects
    /// * `feature_class` - what feature to use for distance calculation
    /// * `only_baked` - calculate distances only across the tracks that have `TrackBakingStatus::Ready` status
    ///
    pub fn foreign_track_distances(
        &mut self,
        tracks: Vec<Track<TA, M, OA, N>>,
        feature_class: u64,
        only_baked: bool,
    ) -> TrackDistances<OA> {
        let tracks_count = tracks.len();

        let (results_sender, results_receiver) = crossbeam::channel::unbounded();
        let mut results = Vec::with_capacity(self.shard_stats().iter().sum());
        let mut errors = Vec::new();

        for track in tracks {
            let track = Arc::new(track);
            for (cmd, _) in &mut self.executors {
                cmd.send(Commands::Distances(
                    track.clone(),
                    feature_class,
                    only_baked,
                    results_sender.clone(),
                ))
                .unwrap();
            }
        }

        for _ in 0..(self.executors.len() * tracks_count) {
            let res = results_receiver.recv().unwrap();
            match res {
                Results::Distance(r, e) => {
                    results.extend_from_slice(&r);
                    errors.extend(e);
                }
                _ => {
                    unreachable!();
                }
            }
        }
        (results, errors)
    }

    /// Calculates track distances for a track within the store
    ///
    /// The distances for (self, self) are not calculated.
    ///
    /// # Arguments
    /// * `tracks` - batch of tracks that are used as distance subjects
    /// * `feature_class` - what feature to use for distance calculation
    /// * `only_baked` - calculate distances only across the tracks that have `TrackBakingStatus::Ready` status
    ///
    pub fn owned_track_distances(
        &mut self,
        tracks: &[u64],
        feature_class: u64,
        only_baked: bool,
    ) -> TrackDistances<OA> {
        let tracks_vec = self.fetch_tracks(tracks);

        if tracks_vec.is_empty() {
            return (vec![], vec![Err(Errors::TracksNotFound.into())]);
        }

        let res = self.foreign_track_distances(tracks_vec.clone(), feature_class, only_baked);

        for t in tracks_vec {
            self.add_track(t).unwrap();
        }

        res
    }

    /// returns the store shard for id
    ///
    pub fn get_store(&self, id: usize) -> StoreMutexGuard<'_, TA, M, OA, N> {
        let store_id = id % self.num_shards;
        self.stores.as_ref().get(store_id).unwrap().lock().unwrap()
    }

    /// returns the store shard for id
    ///
    pub fn get_executor(&self, id: usize) -> usize {
        id % self.num_shards
    }

    /// Adds external track into storage
    ///
    /// # Arguments
    /// * `track` - track compatible with the storage.
    ///
    /// # Returns
    /// * `Ok(track_id)` if added
    /// * `Err(Errors::DuplicateTrackId(track_id))` if failed to add
    ///
    pub fn add_track(&mut self, track: Track<TA, M, OA, N>) -> Result<u64> {
        let track_id = track.track_id;
        let mut store = self.get_store(track_id as usize);
        if store.get(&track_id).is_none() {
            store.insert(track_id, track);
            Ok(track_id)
        } else {
            Err(Errors::DuplicateTrackId(track_id).into())
        }
    }

    /// Injects new feature observation for feature class into track
    ///
    /// # Arguments
    /// * `track_id` - unique Id of the track within the store
    /// * `feature_class` - where the observation will be placed within the track
    /// * `feature_attribute` - feature quality parameter
    /// * `feature` - feature observation
    /// * `attributes_update` - the update to be applied to attributes upon the feature insert
    ///
    pub fn add(
        &mut self,
        track_id: u64,
        feature_class: u64,
        feature_attribute: Option<OA>,
        feature: Option<Observation>,
        attributes_update: Option<TA::Update>,
    ) -> Result<()> {
        let mut tracks = self.get_store(track_id as usize);
        #[allow(clippy::significant_drop_in_scrutinee)]
        match tracks.get_mut(&track_id) {
            None => {
                let mut t = Track {
                    notifier: self.notifier.clone(),
                    attributes: self.attributes.clone(),
                    track_id,
                    observations: HashMap::from([(
                        feature_class,
                        vec![ObservationSpec(feature_attribute, feature)],
                    )]),
                    metric: self.metric.clone(),
                    merge_history: vec![track_id],
                };
                if let Some(attributes_update) = attributes_update {
                    t.update_attributes(attributes_update)?;
                }

                tracks.insert(track_id, t);
            }
            Some(track) => {
                track.add_observation(
                    feature_class,
                    feature_attribute,
                    feature,
                    attributes_update,
                )?;
            }
        }
        Ok(())
    }

    /// Merge store owned tracks
    ///
    /// # Arguments
    /// * `dest_id` - identifier of destination track
    /// * `src_id` - identifier of source track
    /// * `classes` - optional list of classes to merge (otherwise all defined in src are merged into dest)
    /// * `remove_src_if_ok` - whether remove source track from store if merge completed or not
    /// * `merge_history` - configures whether merge history is built upon track merging.
    ///
    /// # Return
    /// * `Ok(Old_Source_Track)` - merge was successful
    /// * `Err(e)` - merge met problems
    ///
    pub fn merge_owned(
        &mut self,
        dest_id: u64,
        src_id: u64,
        classes: Option<&[u64]>,
        remove_src_if_ok: bool,
        merge_history: bool,
    ) -> OwnedMergeResult<TA, M, OA, N> {
        let mut src = self.fetch_tracks(&[src_id]);
        if src.is_empty() {
            return Err(Errors::TrackNotFound(src_id).into());
        }
        let src = src.pop().unwrap();
        match self.merge_external(dest_id, &src, classes, merge_history) {
            Ok(_) => {
                if !remove_src_if_ok {
                    self.add_track(src).unwrap();
                    return Ok(None);
                }
                Ok(Some(src))
            }
            err => {
                self.add_track(src).unwrap();
                err?;
                unreachable!();
            }
        }
    }

    /// Merge external track with destination stored in store without blocking
    ///
    /// # Arguments
    /// * `dest_id` - identifier of destination track
    /// * `src` - source track
    /// * `classes` - optional list of classes to merge (otherwise all defined in src are merged into dest)
    /// * `merge_history` - configures whether merge history is built upon track merging.
    ///
    /// # Return
    /// * `Ok(FutureMergeResponse<FA>)` - future object that contains the receiver channel to gen the result when it is complete
    /// * `Err(e)` - error occurred
    ///
    pub fn merge_external_noblock(
        &mut self,
        dest_id: u64,
        src: Track<TA, M, OA, N>,
        classes: Option<&[u64]>,
        merge_history: bool,
    ) -> Result<FutureMergeResponse<OA>> {
        let (results_sender, results_receiver) = crossbeam::channel::bounded(1);
        let executor_id = self.get_executor(dest_id as usize);
        let (cmd, _) = self.executors.get_mut(executor_id).unwrap();

        let command = Commands::Merge(
            dest_id,
            src,
            if let Some(c) = classes {
                c.to_vec()
            } else {
                vec![]
            },
            merge_history,
            Some(results_sender.clone()),
        );

        let res = cmd.send(command);

        if res.is_err() {
            error!(
                "Executor {} unable to accept the command. Error is: {:?}",
                executor_id, &res
            );
            res?;
            unreachable!();
        }

        Ok(FutureMergeResponse {
            _sender: results_sender,
            receiver: results_receiver,
        })
    }

    /// Merge external track with destination stored in store
    ///
    /// # Arguments
    /// * `dest_id` - identifier of destination track
    /// * `src` - source track
    /// * `classes` - optional list of classes to merge (otherwise all defined in src are merged into dest)
    /// * `merge_history` - configures whether merge history is built upon track merging.
    ///
    /// # Return
    /// * `Ok(())` - merge was successful
    /// * `Err(e)` - merge met problems
    ///
    pub fn merge_external(
        &mut self,
        dest_id: u64,
        src: &Track<TA, M, OA, N>,
        classes: Option<&[u64]>,
        merge_history: bool,
    ) -> Result<()> {
        let res = self.merge_external_noblock(dest_id, src.clone(), classes, merge_history);
        if let Ok(res) = res {
            res.get()
        } else {
            res?;
            unreachable!();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::test_stuff::{current_time_ms, vec2};
    use crate::track::store::TrackStore;
    use crate::track::utils::feature_attributes_sort_dec;
    use crate::track::{
        MetricOutput, NoopNotifier, ObservationAttributes, ObservationMetric, ObservationSpec,
        ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate, TrackStatus,
    };
    use crate::EPS;
    use anyhow::Result;
    use std::thread;
    use std::time::Duration;

    #[derive(Default, Debug, Clone)]
    pub struct TimeAttrs {
        start_time: u128,
        end_time: u128,
        baked_period: u128,
    }

    #[derive(Default, Clone)]
    pub struct TimeAttrUpdates {
        time: u128,
    }

    impl TrackAttributesUpdate<TimeAttrs> for TimeAttrUpdates {
        fn apply(&self, attrs: &mut TimeAttrs) -> Result<()> {
            attrs.end_time = self.time;
            if attrs.start_time == 0 {
                attrs.start_time = self.time;
            }
            Ok(())
        }
    }

    impl TrackAttributes<TimeAttrs, f32> for TimeAttrs {
        type Update = TimeAttrUpdates;

        fn compatible(&self, other: &TimeAttrs) -> bool {
            self.end_time <= other.start_time
        }

        fn merge(&mut self, other: &TimeAttrs) -> Result<()> {
            self.start_time = self.start_time.min(other.start_time);
            self.end_time = self.end_time.max(other.end_time);
            Ok(())
        }

        fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
            if current_time_ms() >= self.baked_period + self.end_time {
                Ok(TrackStatus::Ready)
            } else {
                Ok(TrackStatus::Pending)
            }
        }
    }

    #[derive(Default, Clone)]
    struct TimeMetric {
        max_length: usize,
    }

    impl ObservationMetric<TimeAttrs, f32> for TimeMetric {
        fn metric(
            _feature_class: u64,
            _attrs1: &TimeAttrs,
            _attrs2: &TimeAttrs,
            e1: &ObservationSpec<f32>,
            e2: &ObservationSpec<f32>,
        ) -> MetricOutput<f32> {
            Some((
                f32::calculate_metric_object(&e1.0, &e2.0),
                match (e1.1.as_ref(), e2.1.as_ref()) {
                    (Some(x), Some(y)) => Some(euclidean(x, y)),
                    _ => None,
                },
            ))
        }

        fn optimize(
            &mut self,
            _feature_class: &u64,
            _merge_history: &[u64],
            _attrs: &mut TimeAttrs,
            features: &mut Vec<ObservationSpec<f32>>,
            _prev_length: usize,
            _is_merge: bool,
        ) -> Result<()> {
            features.sort_by(feature_attributes_sort_dec);
            features.truncate(self.max_length);
            Ok(())
        }
    }

    #[test]
    fn new_default_store() -> Result<()> {
        let default_store: TrackStore<TimeAttrs, TimeMetric, f32> = TrackStore::default();
        drop(default_store);
        Ok(())
    }

    #[test]
    fn new_store_10_shards() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            10,
        );
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            Some(TimeAttrUpdates {
                time: current_time_ms(),
            }),
        )?;

        Ok(())
    }

    fn time_attrs_current_ts() -> Option<TimeAttrUpdates> {
        Some(TimeAttrUpdates {
            time: current_time_ms(),
        })
    }

    #[test]
    fn sharding_n_fetch() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            2,
        );

        let stats = store.shard_stats();
        assert_eq!(stats, vec![0, 0]);

        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let stats = store.shard_stats();
        assert_eq!(stats, vec![1, 0]);

        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let stats = store.shard_stats();
        assert_eq!(stats, vec![1, 1]);

        let tracks = store.fetch_tracks(&vec![0, 1]);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].track_id, 0);
        assert_eq!(tracks[1].track_id, 1);

        Ok(())
    }

    #[test]
    fn general_ops() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            1,
        );
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;
        let baked = store.find_usable();
        assert!(baked.is_empty());
        thread::sleep(Duration::from_millis(30));
        let baked = store.find_usable();
        assert_eq!(baked.len(), 1);
        assert_eq!(baked[0].0, 0);

        let vectors = store.fetch_tracks(&baked.into_iter().map(|(t, _)| t).collect::<Vec<_>>());
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].track_id, 0);
        assert_eq!(vectors[0].observations.len(), 1);

        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;
        let (dists, errs) = store.owned_track_distances(&[0], 0, false);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            Some(0.7),
            Some(vec2(1.0, 0.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[0], 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        let (dists, errs) = store.owned_track_distances(&[1], 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 0);

        let mut v = store.fetch_tracks(&vec![0]);

        let v = v.pop().unwrap();
        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        // make it incompatible across the attributes
        thread::sleep(Duration::from_millis(10));
        let mut v = v.clone();
        v.attributes.end_time = current_time_ms();

        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 0);

        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            Some(0.7),
            Some(vec2(1.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let mut v = v.clone();
        v.attributes.end_time = store.get_store(1).get(&1).unwrap().attributes.start_time - 1;
        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((dists[1].feature_distance.as_ref().unwrap() - 1.0).abs() < EPS);
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn baked_similarity() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            2,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
        );

        //thread::sleep(Duration::from_millis(10));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            Some(TimeAttrUpdates {
                time: current_time_ms(),
            }),
        )?;

        let (dists, errs) = store.foreign_track_distances(vec![ext_track.clone()], 0, true);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[1], 0, true);
        assert!(dists.is_empty());
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn all_similarity() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            2,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.foreign_track_distances(vec![ext_track.clone()], 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        thread::sleep(Duration::from_millis(1));
        store.add(
            3,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[1], 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn add_track_ok() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        store.add_track(ext_track)?;
        Ok(())
    }

    #[test]
    fn add_track_dup_id() -> Result<()> {
        let mut ext_track = Track::new(
            0, // duplicate track id
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        assert!(store.add_track(ext_track).is_err());

        Ok(())
    }

    #[test]
    fn merge_ext_tracks() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        ext_track.add_observation(
            1,
            Some(0.8),
            Some(vec2(0.65, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let res = store.merge_external(0, &ext_track, Some(&[0]), true);
        assert!(res.is_ok());
        let classes = store.get_store(0).get(&0).unwrap().get_feature_classes();
        assert_eq!(classes, vec![0]);

        let res = store.merge_external(0, &ext_track, None, true);
        assert!(res.is_ok());
        let mut classes = store.get_store(0).get(&0).unwrap().get_feature_classes();
        classes.sort();
        assert_eq!(classes, vec![0, 1]);

        Ok(())
    }

    #[test]
    fn merge_own_tracks() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            1,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let res = store.merge_owned(0, 1, None, false, true);
        if let Ok(None) = res {
            ();
        } else {
            unreachable!();
        }

        let res = store.merge_owned(0, 1, None, true, true);
        if let Ok(Some(t)) = res {
            assert_eq!(t.track_id, 1);
        } else {
            unreachable!();
        }

        Ok(())
    }
}
