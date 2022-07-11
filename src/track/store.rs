use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{
    AttributeMatch, AttributeUpdate, Feature, Metric, Track, TrackBakingStatus, TrackDistance,
};
use crate::Errors;
use anyhow::Result;
use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::{mem, thread};

#[derive(Clone)]
enum Commands<A, M, U, N>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    M: Metric,
    U: AttributeUpdate<A>,
{
    Drop,
    FindBaked,
    Distances(Arc<Track<A, M, U, N>>, u64, bool),
}

#[derive(Debug)]
enum Results {
    Distance(Vec<TrackDistance>, Vec<TrackDistanceError>),
    BakedStatus(Vec<(u64, Result<TrackBakingStatus>)>),
    Dropped,
}

/// Auxiliary type to express distance calculation errors
pub type TrackDistanceError = Result<Vec<(u64, Result<f32>)>>;

/// Track store container with accelerated similarity operations.
///
/// TrackStore is implemented for certain attributes (A), attribute update (U), and metric (M), so
/// it handles only such objects. You cannot store tracks with different attributes within the same
/// TrackStore.
///
/// The metric is also defined for TrackStore, however Metric implementation may have various metric
/// calculation options for concrete feature classes. E.g. FEAT1 may be calculated with Euclide distance,
/// while FEAT2 may be calculated with cosine. It is up to Metric implementor how the metric works.
///
/// Simple TrackStore example can be found at:
/// [examples/simple.rs](https://github.com/insight-platform/Similari/blob/main/examples/simple.rs).
///
pub struct TrackStore<A, U, M, N = NoopNotifier>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    U: AttributeUpdate<A>,
    M: Metric,
{
    attributes: A,
    metric: M,
    notifier: N,
    num_shards: usize,
    #[allow(clippy::type_complexity)]
    stores: Arc<Vec<Mutex<HashMap<u64, Track<A, M, U, N>>>>>,
    receiver: Receiver<Results>,
    #[allow(clippy::type_complexity)]
    executors: Vec<(Sender<Commands<A, M, U, N>>, JoinHandle<()>)>,
}

impl<A, U, M, N> Drop for TrackStore<A, U, M, N>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    U: AttributeUpdate<A>,
    M: Metric,
{
    fn drop(&mut self) {
        let executors = mem::take(&mut self.executors);
        for (s, j) in executors {
            s.send(Commands::Drop).unwrap();
            let res = self.receiver.recv().unwrap();
            match res {
                Results::Dropped => {
                    j.join().unwrap();
                    drop(s);
                }
                other => {
                    dbg!(other);
                    unreachable!();
                }
            }
        }
    }
}

impl<A, U, M, N> Default for TrackStore<A, U, M, N>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    U: AttributeUpdate<A>,
    M: Metric,
{
    fn default() -> Self {
        Self::new(None, None, None, 1)
    }
}

/// The basic implementation should fit to most of needs.
///
impl<A, U, M, N> TrackStore<A, U, M, N>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    U: AttributeUpdate<A>,
    M: Metric,
{
    #[allow(clippy::type_complexity)]
    fn handle_store_ops(
        stores: Arc<Vec<Mutex<HashMap<u64, Track<A, M, U, N>>>>>,
        store_id: usize,
        commands_receiver: Receiver<Commands<A, M, U, N>>,
        results_sender: Sender<Results>,
    ) {
        let store = stores.get(store_id).unwrap();
        while let Ok(c) = commands_receiver.recv() {
            // if let Err(_e) = results_sender.send(Results::NotImplemented) {
            //     break;
            // }
            match c {
                Commands::Drop => {
                    let _r = results_sender.send(Results::Dropped);
                    return;
                }
                Commands::FindBaked => {
                    let baked = store
                        .lock()
                        .unwrap()
                        .iter()
                        .flat_map(|(track_id, track)| {
                            match track.get_attributes().baked(&track.observations) {
                                Ok(status) => match status {
                                    TrackBakingStatus::Pending => None,
                                    other => Some((*track_id, Ok(other))),
                                },
                                Err(e) => Some((*track_id, Err(e))),
                            }
                        })
                        .collect();
                    let r = results_sender.send(Results::BakedStatus(baked));
                    if let Err(_e) = r {
                        return;
                    }
                }
                Commands::Distances(track, feature_class, only_baked) => {
                    let mut capacity = 0;
                    let res = store
                        .lock()
                        .unwrap()
                        .iter()
                        .flat_map(|(_, other)| {
                            if !only_baked {
                                let dists = track.distances(other, feature_class);
                                if let Ok(d) = &dists {
                                    capacity += d.len()
                                }
                                Some(dists)
                            } else {
                                match other.get_attributes().baked(&other.observations) {
                                    Ok(TrackBakingStatus::Ready) => {
                                        let dists = track.distances(other, feature_class);
                                        if let Ok(d) = &dists {
                                            capacity += d.len()
                                        }
                                        Some(dists)
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
                            Ok(dists) => distances.extend(dists),
                            e => errors.push(e),
                        }
                    }

                    let r = results_sender.send(Results::Distance(distances, errors));
                    if let Err(_e) = r {
                        return;
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
        default_attributes: Option<A>,
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
        let (results_sender, results_receiver) = crossbeam::channel::unbounded();

        Self {
            receiver: results_receiver,
            num_shards: shards,
            notifier: if let Some(notifier) = notifier {
                notifier
            } else {
                N::default()
            },
            attributes: if let Some(a) = default_attributes {
                a
            } else {
                A::default()
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
                        let (commands_sender, commands_receiver) = crossbeam::channel::bounded(1);
                        let stores = stores.clone();
                        let thread_results_sender = results_sender.clone();
                        let thread = thread::spawn(move || {
                            Self::handle_store_ops(
                                stores,
                                s,
                                commands_receiver,
                                thread_results_sender,
                            );
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
    pub fn find_baked(&mut self) -> Vec<(u64, Result<TrackBakingStatus>)> {
        let mut results = Vec::with_capacity(self.shard_stats().iter().sum());
        for (cmd, _) in &mut self.executors {
            cmd.send(Commands::FindBaked).unwrap();
        }
        for (_, _) in &mut self.executors {
            let res = self.receiver.recv().unwrap();
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

    pub fn shard_stats(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for s in self.stores.iter() {
            result.push(s.lock().unwrap().len());
        }
        result
    }

    /// Pulls (and removes) requested tracks from the store.
    ///
    pub fn fetch_tracks(&mut self, tracks: &Vec<u64>) -> Vec<Track<A, M, U, N>> {
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
    /// * `track` - external track that is used as a distance subject
    /// * `feature_class` - what feature to use for distance calculation
    /// * `only_baked` - calculate distances only across the tracks that have `TrackBakingStatus::Ready` status
    ///
    pub fn foreign_track_distances(
        &mut self,
        track: Arc<Track<A, M, U, N>>,
        feature_class: u64,
        only_baked: bool,
    ) -> (Vec<TrackDistance>, Vec<TrackDistanceError>) {
        let mut results = Vec::with_capacity(self.shard_stats().iter().sum());
        let mut errors = Vec::new();
        for (cmd, _) in &mut self.executors {
            cmd.send(Commands::Distances(
                track.clone(),
                feature_class,
                only_baked,
            ))
            .unwrap();
        }
        for (_, _) in &mut self.executors {
            let res = self.receiver.recv().unwrap();
            match res {
                Results::Distance(r, e) => {
                    results.extend(r);
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
    /// * `track` - external track that is used as a distance subject
    /// * `feature_class` - what feature to use for distance calculation
    /// * `only_baked` - calculate distances only across the tracks that have `TrackBakingStatus::Ready` status
    ///
    pub fn owned_track_distances(
        &mut self,
        track_id: u64,
        feature_class: u64,
        only_baked: bool,
    ) -> (Vec<TrackDistance>, Vec<TrackDistanceError>) {
        let track = self.fetch_tracks(&vec![track_id]).pop();
        if track.is_none() {
            return (vec![], vec![Err(Errors::TrackNotFound(track_id).into())]);
        }
        let track = Arc::new(track.unwrap());

        let res = self.foreign_track_distances(track.clone(), feature_class, only_baked);
        let track = (*track).clone();
        self.add_track(track).unwrap();
        res
    }

    pub fn get_store(&self, id: usize) -> MutexGuard<HashMap<u64, Track<A, M, U, N>>> {
        let store_id = (id % self.num_shards) as usize;
        self.stores.as_ref().get(store_id).unwrap().lock().unwrap()
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
    pub fn add_track(&mut self, track: Track<A, M, U, N>) -> Result<u64> {
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
    /// * `feature_q` - feature quality parameter
    /// * `feature` - feature observation
    /// * `attributes_update` - the update to be applied to attributes upon the feature insert
    ///
    pub fn add(
        &mut self,
        track_id: u64,
        feature_class: u64,
        feature_q: f32,
        feature: Feature,
        attributes_update: U,
    ) -> Result<()> {
        let mut tracks = self.get_store(track_id as usize);
        #[allow(clippy::significant_drop_in_scrutinee)]
        match tracks.get_mut(&track_id) {
            None => {
                let mut t = Track {
                    notifier: self.notifier.clone(),
                    attributes: self.attributes.clone(),
                    track_id,
                    observations: HashMap::from([(feature_class, vec![(feature_q, feature)])]),
                    metric: self.metric.clone(),
                    phantom_attribute_update: PhantomData,
                    merge_history: vec![track_id],
                };
                t.update_attributes(attributes_update)?;
                tracks.insert(track_id, t);
            }
            Some(track) => {
                track.add_observation(feature_class, feature_q, feature, attributes_update)?;
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
    pub fn merge_owned(
        &mut self,
        dest_id: u64,
        src_id: u64,
        classes: Option<&[u64]>,
        remove_src_if_ok: bool,
    ) -> Result<Option<Track<A, M, U, N>>> {
        let mut src = self.fetch_tracks(&vec![src_id]);
        if src.is_empty() {
            return Err(Errors::TrackNotFound(src_id).into());
        }
        let src = src.pop().unwrap();
        match self.merge_external(dest_id, &src, classes) {
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

    /// Merge external track with destination stored in store
    ///
    /// * `dest_id` - identifier of destination track
    /// * `src` - source track
    /// * `classes` - optional list of classes to merge (otherwise all defined in src are merged into dest)
    ///
    pub fn merge_external(
        &mut self,
        dest_id: u64,
        src: &Track<A, M, U, N>,
        classes: Option<&[u64]>,
    ) -> Result<()> {
        let mut tracks = self.get_store(dest_id as usize);
        let dest = tracks.get_mut(&dest_id);

        match dest {
            Some(dest) => {
                if dest_id == src.track_id {
                    return Err(Errors::SameTrackCalculation(dest_id).into());
                }
                let res = if let Some(classes) = classes {
                    dest.merge(src, classes)
                } else {
                    dest.merge(src, &src.get_feature_classes())
                };
                res?;
                Ok(())
            }
            None => Err(Errors::TrackNotFound(dest_id).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::track::store::TrackStore;
    use crate::track::{
        feat_confidence_cmp, AttributeMatch, AttributeUpdate, Feature, FeatureObservationsGroups,
        FeatureSpec, FromVec, Metric, NoopNotifier, Track, TrackBakingStatus,
    };
    use crate::{Errors, EPS};
    use anyhow::Result;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

    impl AttributeUpdate<TimeAttrs> for TimeAttrUpdates {
        fn apply(&self, attrs: &mut TimeAttrs) -> Result<()> {
            attrs.end_time = self.time;
            if attrs.start_time == 0 {
                attrs.start_time = self.time;
            }
            Ok(())
        }
    }

    impl AttributeMatch<TimeAttrs> for TimeAttrs {
        fn compatible(&self, other: &TimeAttrs) -> bool {
            self.end_time <= other.start_time
        }

        fn merge(&mut self, other: &TimeAttrs) -> Result<()> {
            self.start_time = self.start_time.min(other.start_time);
            self.end_time = self.end_time.max(other.end_time);
            Ok(())
        }

        fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
            if SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
                - self.end_time
                > self.baked_period
            {
                Ok(TrackBakingStatus::Ready)
            } else {
                Ok(TrackBakingStatus::Pending)
            }
        }
    }

    #[derive(Default, Clone)]
    struct TimeMetric {
        max_length: usize,
    }

    impl Metric for TimeMetric {
        fn distance(_feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32> {
            Ok(euclidean(&e1.1, &e2.1))
        }

        fn optimize(
            &mut self,
            _feature_class: &u64,
            _merge_history: &[u64],
            features: &mut Vec<FeatureSpec>,
            _prev_length: usize,
        ) -> Result<()> {
            features.sort_by(feat_confidence_cmp);
            features.truncate(self.max_length);
            Ok(())
        }
    }

    fn vec2(x: f32, y: f32) -> Feature {
        Feature::from_vec(vec![x, y])
    }

    fn current_time_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    #[test]
    fn new_default_store() -> Result<()> {
        let default_store: TrackStore<TimeAttrs, TimeAttrUpdates, TimeMetric> =
            TrackStore::default();
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        Ok(())
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let stats = store.shard_stats();
        assert_eq!(stats, vec![1, 0]);

        store.add(
            1,
            0,
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;
        let baked = store.find_baked();
        assert!(baked.is_empty());
        thread::sleep(Duration::from_millis(30));
        let baked = store.find_baked();
        assert_eq!(baked.len(), 1);
        assert_eq!(baked[0].0, 0);

        let vectors = store.fetch_tracks(&baked.into_iter().map(|(t, _)| t).collect());
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].track_id, 0);
        assert_eq!(vectors[0].observations.len(), 1);

        store.add(
            0,
            0,
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
        )?;
        let (dists, errs) = store.owned_track_distances(0, 0, false);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            0.7,
            vec2(1.0, 0.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let (dists, errs) = store.owned_track_distances(0, 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        let (dists, errs) = store.owned_track_distances(1, 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 1);
        match errs[0].as_ref() {
            Ok(_) => {
                unreachable!();
            }
            Err(e) => {
                let errs = e.downcast_ref::<Errors>().unwrap();
                match errs {
                    Errors::IncompatibleAttributes => {}
                    Errors::ObservationForClassNotFound(_t1, _t2, _c) => {
                        unreachable!();
                    }
                    Errors::TrackNotFound(_t)
                    | Errors::DuplicateTrackId(_t)
                    | Errors::SameTrackCalculation(_t) => {
                        unreachable!();
                    }
                }
            }
        }

        let mut v = store.fetch_tracks(&vec![0]);

        let v = Arc::new(v.pop().unwrap());
        let (dists, errs) = store.foreign_track_distances(v.clone(), 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        // make it incompatible across the attributes
        thread::sleep(Duration::from_millis(10));
        let mut v = (*v).clone();
        v.attributes.end_time = current_time_ms();
        let v = Arc::new(v);

        let (dists, errs) = store.foreign_track_distances(v.clone(), 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 1);
        match errs[0].as_ref() {
            Ok(_) => {
                unreachable!();
            }
            Err(e) => {
                let errs = e.downcast_ref::<Errors>().unwrap();
                match errs {
                    Errors::IncompatibleAttributes => {}
                    Errors::ObservationForClassNotFound(_t1, _t2, _c) => {
                        unreachable!();
                    }
                    Errors::TrackNotFound(_t)
                    | Errors::DuplicateTrackId(_t)
                    | Errors::SameTrackCalculation(_t) => {
                        unreachable!();
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            0.7,
            vec2(1.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let mut v = (*v).clone();
        v.attributes.end_time = store.get_store(1).get(&1).unwrap().attributes.start_time - 1;
        let v = Arc::new(v);
        let (dists, errs) = store.foreign_track_distances(v.clone(), 0, false);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((dists[1].1.as_ref().unwrap() - 1.0).abs() < EPS);
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn only_baked_similarity() -> Result<()> {
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
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

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            0.8,
            vec2(0.66, 0.33),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let ext_track = Arc::new(ext_track);
        let (dists, errs) = store.foreign_track_distances(ext_track.clone(), 0, true);
        assert!(dists.is_empty());
        assert!(errs.is_empty());

        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let (dists, errs) = store.owned_track_distances(0, 0, true);
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
            0.8,
            vec2(0.66, 0.33),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let ext_track = Arc::new(ext_track);
        let (dists, errs) = store.foreign_track_distances(ext_track.clone(), 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        let (dists, errs) = store.foreign_track_distances(ext_track.clone(), 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        thread::sleep(Duration::from_millis(1));
        store.add(
            3,
            0,
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let (dists, errs) = store.owned_track_distances(1, 0, false);
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
            0.8,
            vec2(0.66, 0.33),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
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
            0.8,
            vec2(0.66, 0.33),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
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
            0.8,
            vec2(0.66, 0.33),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
        )?;

        ext_track.add_observation(
            1,
            0.8,
            vec2(0.65, 0.33),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
        )?;

        let res = store.merge_external(0, &ext_track, Some(&[0]));
        assert!(res.is_ok());
        let classes = store.get_store(0).get(&0).unwrap().get_feature_classes();
        assert_eq!(classes, vec![0]);

        let res = store.merge_external(0, &ext_track, None);
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
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            1,
            0.9,
            vec2(0.0, 1.0),
            TimeAttrUpdates {
                time: current_time_ms(),
            },
        )?;

        let res = store.merge_owned(0, 1, None, false);
        if let Ok(None) = res {
            ();
        } else {
            unreachable!();
        }

        let res = store.merge_owned(0, 1, None, true);
        if let Ok(Some(t)) = res {
            assert_eq!(t.track_id, 1);
        } else {
            unreachable!();
        }

        Ok(())
    }
}
