use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::Errors;
use anyhow::Result;
use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::take;
use ultraviolet::f32x8;

pub mod builder;
pub mod notify;
pub mod store;
pub mod utils;
pub mod voting;

/// Return type for distance between the current track's and other track observation pair
///
#[derive(Debug, Clone)]
pub struct ObservationMetricOk<OA>
where
    OA: ObservationAttributes,
{
    /// source track ID
    pub from: u64,
    /// compared track ID
    pub to: u64,
    /// custom feature attribute metric object calculated for pairwise feature attributes
    pub attribute_metric: Option<OA::MetricObject>,
    /// distance calculated for pairwise feature vectors
    pub feature_distance: Option<f32>,
}

impl<OA> ObservationMetricOk<OA>
where
    OA: ObservationAttributes,
{
    pub fn new(
        from: u64,
        to: u64,
        attribute_metric: Option<OA::MetricObject>,
        feature_distance: Option<f32>,
    ) -> Self {
        Self {
            from,
            to,
            attribute_metric,
            feature_distance,
        }
    }
}

/// Internal feature vector representation.
///
pub type Feature = Vec<f32x8>;

/// Number of SIMD lanes used to store observation parts internally
const FEATURE_LANES_SIZE: usize = 8;

/// Observation specification.
///
/// It is a tuple struct of optional observation attributes (T) and optional feature vector itself.
/// Observations are collected from the real world and placed into tracks. Later the observations are used
/// to calculate the distances between tracks to make merging.
///
#[derive(Default, Clone)]
pub struct Observation<T>(pub(crate) Option<T>, pub(crate) Option<Feature>)
where
    T: Send + Sync + Clone + 'static;

impl<T> Observation<T>
where
    T: Send + Sync + Clone + 'static,
{
    pub fn new(attrs: Option<T>, feature: Option<Feature>) -> Self {
        Self(attrs, feature)
    }

    /// Access to observation attributes
    ///
    pub fn attr(&self) -> &Option<T> {
        &self.0
    }

    /// Access to observation attributes for modification purposes
    ///
    pub fn attr_mut(&mut self) -> &mut Option<T> {
        &mut self.0
    }

    /// Access to observation feature
    ///
    pub fn feature(&self) -> &Option<Feature> {
        &self.1
    }

    /// Access to observation feature for modification purposes
    ///
    pub fn feature_mut(&mut self) -> &mut Option<Feature> {
        &mut self.1
    }
}

/// HashTable that accumulates observations within the track.
///
/// The key is the feature class the value is the vector of observations collected.
///
pub type ObservationsDb<T> = HashMap<u64, Vec<Observation<T>>>;

/// Custom observation attributes object that is the part of the observation together with the feature vector.
///
pub trait ObservationAttributes: Send + Sync + Clone + 'static {
    type MetricObject: Debug + Send + Sync + Clone + 'static;
    fn calculate_metric_object(l: &Option<&Self>, r: &Option<&Self>) -> Option<Self::MetricObject>;
}

/// Output result type used by metric when pairwise metric is calculated
///
/// `None` - no metric for that pair - the result will be dropped (optimization technique)
/// `Some(Option<X>, Option<Y>)` - metric is calculated, values are inside.
///  where
///   * `Option<X>` is the metric object computed for observation attributes;
///   * `Option<Y>` is the distance computed for feature vectors of the observation.
///
pub type MetricOutput<T> = Option<(Option<T>, Option<f32>)>;

/// Query object that is a parameter of the ``ObservationMetric::metric` method.
///
/// The query is used to make pairwise comparison of observations for two tracks.
/// There is a
///  * `candidate` track - the one, that is selected as a comparison subject
///  * `track` track - the one, that is iterated over those kept in the store
///
pub struct MetricQuery<'a, TA, OA: ObservationAttributes> {
    /// * `feature_class` - class of currently used feature
    pub feature_class: u64,
    /// * `candidate_attrs` - candidate track attributes
    pub candidate_attrs: &'a TA,
    /// * `candidate_observation` - candidate track observation
    pub candidate_observation: &'a Observation<OA>,
    /// * `track_attrs` - track attributes
    pub track_attrs: &'a TA,
    /// * `track_observation` - track observation
    pub track_observation: &'a Observation<OA>,
}

/// The trait that implements the methods for observations comparison, optimization and filtering.
///
/// This is the one of the most important elements of the track. It defines how track distances are
/// computed, how track observations are compacted and transformed upon merging.
///
pub trait ObservationMetric<TA, OA: ObservationAttributes>: Send + Sync + Clone + 'static {
    /// calculates the distance between two features.
    ///
    /// # Parameters
    /// * `mq` - query to calculate metric
    ///
    fn metric(&self, mq: &'_ MetricQuery<'_, TA, OA>) -> MetricOutput<OA::MetricObject>;

    /// the method is used every time, when a new observation is added to the feature storage as well as when
    /// two tracks are merged.
    ///
    /// # Arguments
    ///
    /// * `feature_class` - the feature class
    /// * `merge_history` - the vector of track identifiers collected upon every merge
    /// * `attributes` - mutable track attributes that can be updated or read during optimization
    /// * `observations` - observations to optimize
    /// * `prev_length` - previous length of observations (before the current observation was added or merge occurred)
    /// * `is_merge` - true, when the op is for track merging, false when the observation is added to the track
    ///
    /// # Returns
    /// * `Ok(())` if the optimization is successful
    /// * `Err(e)` if the optimization failed
    ///
    fn optimize(
        &mut self,
        feature_class: u64,
        merge_history: &[u64],
        attributes: &mut TA,
        observations: &mut Vec<Observation<OA>>,
        prev_length: usize,
        is_merge: bool,
    ) -> Result<()>;

    /// The postprocessing is run just before the executor returns calculated distances.
    ///
    /// The postprocessing is aimed to remove non-viable, invalid distances that can be skipped
    /// to improve the performance or the quality of further track voting process.
    ///
    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<OA>>,
    ) -> Vec<ObservationMetricOk<OA>> {
        unfiltered
    }
}

/// Enum which specifies the status of feature tracks in storage. When the feature tracks are collected,
/// eventually the track must be complete so it can be used for
/// database search and later merge operations.
///
#[derive(Clone, Debug)]
pub enum TrackStatus {
    /// The track is ready and can be used to find similar tracks for merge.
    Ready,
    /// The track is not ready and still being collected.
    Pending,
    /// The track is invalid because somehow became incorrect or outdated along the way.
    Wasted,
}

/// The trait that must be implemented by a search query object to run searches over the store
///
pub trait LookupRequest<TA, OA>: Send + Sync + Clone + 'static
where
    TA: TrackAttributes<TA, OA>,
    OA: ObservationAttributes,
{
    fn lookup(
        &self,
        _attributes: &TA,
        _observations: &ObservationsDb<OA>,
        _merge_history: &[u64],
    ) -> bool {
        false
    }
}

/// Do nothing lookup implementation that can be put anywhere lookup is required.
///
/// It is compatible with all TA, OA. Const parameter defines what lookup returns:
/// * `false` - all lookup elements are ignored
/// * `true` - all lookup elements are returned
///
pub struct NoopLookup<TA, OA, const RES: bool = false>
where
    TA: TrackAttributes<TA, OA>,
    OA: ObservationAttributes,
{
    _ta: PhantomData<TA>,
    _oa: PhantomData<OA>,
}

impl<TA, OA, const RES: bool> Clone for NoopLookup<TA, OA, RES>
where
    TA: TrackAttributes<TA, OA>,
    OA: ObservationAttributes,
{
    fn clone(&self) -> Self {
        NoopLookup {
            _ta: PhantomData,
            _oa: PhantomData,
        }
    }
}

impl<TA, OA, const RES: bool> Default for NoopLookup<TA, OA, RES>
where
    TA: TrackAttributes<TA, OA>,
    OA: ObservationAttributes,
{
    fn default() -> Self {
        NoopLookup {
            _ta: PhantomData,
            _oa: PhantomData,
        }
    }
}

impl<TA, OA, const RES: bool> LookupRequest<TA, OA> for NoopLookup<TA, OA, RES>
where
    TA: TrackAttributes<TA, OA>,
    OA: ObservationAttributes,
{
    fn lookup(
        &self,
        _attributes: &TA,
        _observations: &ObservationsDb<OA>,
        _merge_history: &[u64],
    ) -> bool {
        RES
    }
}

/// The trait represents user defined Track Attributes. It is used to define custom attributes that
/// fit a domain field where tracking implemented.
///
/// When the user implements track attributes they has to implement this trait to create a valid attributes object.
///
pub trait TrackAttributes<TA: TrackAttributes<TA, OA>, OA: ObservationAttributes>:
    Send + Sync + Clone + 'static
{
    type Update: TrackAttributesUpdate<TA>;
    type Lookup: LookupRequest<TA, OA>;
    /// The method is used to evaluate attributes of two tracks to determine whether tracks are compatible
    /// for distance calculation. When the attributes are compatible, the method returns `true`.
    ///
    /// E.g.
    ///     Let's imagine the case when the track includes the attributes for track begin and end timestamps.
    ///     The tracks are compatible their timeframes don't intersect between each other. The method `compatible`
    ///     can decide that.
    ///
    fn compatible(&self, other: &TA) -> bool;

    /// When the tracks are merged, their attributes are merged as well. The method defines the approach to merge attributes.
    ///
    /// E.g.
    ///     Let's imagine the case when the track includes the attributes for track begin and end timestamps.
    ///     Merge operation may look like `[b1; e1] + [b2; e2] -> [min(b1, b2); max(e1, e2)]`.
    ///
    fn merge(&mut self, other: &TA) -> Result<()>;

    /// The method is used by storage to determine when track is ready/not ready/wasted. Look at [TrackStatus](TrackStatus).
    ///
    /// It uses attribute information collected across the track config.toml and features information.
    ///
    /// E.g.
    ///     track is ready when
    ///          `now - end_timestamp > 30s` (no features collected during the last 30 seconds).
    ///
    fn baked(&self, observations: &ObservationsDb<OA>) -> Result<TrackStatus>;
}

/// The attribute update information that is sent with new features to the track is represented by the trait.
///
/// The trait must be implemented for update struct for specific attributes struct implementation.
///
pub trait TrackAttributesUpdate<TA>: Clone + Send + Sync + 'static {
    /// Method is used to update track attributes from update structure.
    ///
    fn apply(&self, attrs: &mut TA) -> Result<()>;
}

/// Represents track of observations - it's a core concept of the library.
///
/// The track is created for specific attributes(A), Metric(M) and AttributeUpdate(U).
/// * Attributes hold track meta information specific for certain domain.
/// * Metric defines how to compare track features and optimize features when tracks are
///   merged or collected
/// * AttributeUpdate specifies how attributes are update from external sources.
///
#[derive(Default, Clone)]
pub struct Track<TA, M, OA, N = NoopNotifier>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    attributes: TA,
    track_id: u64,
    observations: ObservationsDb<OA>,
    metric: M,
    merge_history: Vec<u64>,
    notifier: N,
}

/// One and only parametrized track implementation.
///
impl<TA, M, OA, N> Track<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    /// Creates a new track with id `track_id` with `metric` initializer object and `attributes` initializer object.
    ///
    /// The `metric` and `attributes` are optional, if `None` is specified, then `Default` initializer is used.
    ///
    pub fn new(track_id: u64, metric: M, attributes: TA, notifier: N) -> Self {
        let mut v = Self {
            notifier,
            attributes,
            track_id,
            metric,
            observations: ObservationsDb::default(),
            merge_history: vec![track_id],
        };
        v.notifier.send(track_id);
        v
    }

    /// Returns track_id.
    ///
    pub fn get_track_id(&self) -> u64 {
        self.track_id
    }

    /// Sets track_id.
    ///
    pub fn set_track_id(&mut self, track_id: u64) -> u64 {
        let old = self.track_id;
        self.track_id = track_id;
        old
    }

    /// Returns current track attributes.
    ///
    pub fn get_attributes(&self) -> &TA {
        &self.attributes
    }

    pub fn get_observations(&self, feature_class: u64) -> Option<&Vec<Observation<OA>>> {
        self.observations.get(&feature_class)
    }

    /// Returns the current track merge history for the track
    ///
    pub fn get_merge_history(&self) -> &Vec<u64> {
        &self.merge_history
    }

    /// Returns all classes present
    ///
    pub fn get_feature_classes(&self) -> Vec<u64> {
        self.observations.keys().cloned().collect()
    }

    fn update_attributes(&mut self, update: &TA::Update) -> Result<()> {
        update.apply(&mut self.attributes)
    }

    /// Adds new observation to track.
    ///
    /// When the method is called, the track attributes are updated according to `update` argument, and the feature
    /// is placed into features for a specified feature class.
    ///
    /// # Arguments
    /// * `feature_class` - class of observation
    /// * `feature_attributes` - quality of the feature (confidence, or another parameter that defines how the observation is valuable across the observations).
    /// * `feature` - observation to add to the track for specified `feature_class`.
    /// * `track_attributes_update` - attribute update message
    ///
    /// # Returns
    /// Returns `Result<()>` where `Ok(())` if attributes are updated without errors AND observation is added AND observations optimized without errors.
    ///
    ///
    pub fn add_observation(
        &mut self,
        feature_class: u64,
        feature_attributes: Option<OA>,
        feature: Option<Feature>,
        track_attributes_update: Option<TA::Update>,
    ) -> Result<()> {
        let last_attributes = self.attributes.clone();
        let last_observations = self.observations.clone();
        let last_metric = self.metric.clone();

        if let Some(track_attributes_update) = &track_attributes_update {
            let res = self.update_attributes(track_attributes_update);
            if res.is_err() {
                self.attributes = last_attributes;
                res?;
                unreachable!();
            }
        }

        if feature.is_none() && feature_attributes.is_none() {
            self.notifier.send(self.track_id);
            return Ok(());
        }

        match self.observations.get_mut(&feature_class) {
            None => {
                self.observations.insert(
                    feature_class,
                    vec![Observation(feature_attributes, feature)],
                );
            }
            Some(observations) => {
                observations.push(Observation(feature_attributes, feature));
            }
        }
        let observations = self.observations.get_mut(&feature_class).unwrap();
        let prev_length = observations.len() - 1;

        let res = self.metric.optimize(
            feature_class,
            &self.merge_history,
            &mut self.attributes,
            observations,
            prev_length,
            false,
        );
        if res.is_err() {
            self.attributes = last_attributes;
            self.observations = last_observations;
            self.metric = last_metric;
            res?;
            unreachable!();
        }
        self.notifier.send(self.track_id);
        Ok(())
    }

    /// Merges vector into current track across specified feature classes.
    ///
    /// The merge works across specified feature classes:
    /// * step 1: attributes are merged
    /// * step 2.0: features are merged for classes
    /// * step 2.1: features are optimized for every class
    ///
    /// If feature class doesn't exist any of tracks it's skipped, otherwise:
    ///
    /// * both: `{S[class]} U {OTHER[class]}`
    /// * self: `{S[class]}`
    /// * other: `{OTHER[class]}`
    ///
    /// # Parameters
    /// * `other` - track to merge into self
    /// * `merge_history` - defines add merged track id into self merge history or not
    ///
    pub fn merge(&mut self, other: &Self, classes: &[u64], merge_history: bool) -> Result<()> {
        let last_attributes = self.attributes.clone();
        let res = self.attributes.merge(&other.attributes);
        if res.is_err() {
            self.attributes = last_attributes;
            res?;
            unreachable!();
        }

        let last_observations = self.observations.clone();
        let last_metric = self.metric.clone();

        for cls in classes {
            let dest = self.observations.get_mut(cls);
            let src = other.observations.get(cls);
            let prev_length = match (dest, src) {
                (Some(dest_observations), Some(src_observations)) => {
                    let prev_length = dest_observations.len();
                    dest_observations.extend(src_observations.iter().cloned());
                    Some(prev_length)
                }
                (None, Some(src_observations)) => {
                    self.observations.insert(*cls, src_observations.clone());
                    Some(0)
                }

                (Some(dest_observations), None) => {
                    let prev_length = dest_observations.len();
                    Some(prev_length)
                }

                _ => None,
            };
            let merge_history = if merge_history {
                self.merge_history
                    .iter()
                    .chain(other.merge_history.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            } else {
                take(&mut self.merge_history)
            };

            if let Some(prev_length) = prev_length {
                let res = self.metric.optimize(
                    *cls,
                    &merge_history,
                    &mut self.attributes,
                    self.observations.get_mut(cls).unwrap(),
                    prev_length,
                    true,
                );

                if res.is_err() {
                    self.attributes = last_attributes;
                    self.observations = last_observations;
                    self.metric = last_metric;
                    res?;
                    unreachable!();
                }
                self.merge_history = merge_history;
            }
        }

        self.notifier.send(self.track_id);
        Ok(())
    }

    /// Calculates distances between all features for two tracks for a class.
    ///
    /// First it calculates cartesian product `S X O` and calculates the distance for every pair.
    ///
    /// Before it calculates the distance, it checks that attributes are compatible. If no,
    /// [`Err(Errors::IncompatibleAttributes)`](Errors::IncompatibleAttributes) returned. Otherwise,
    /// the vector of distances returned that holds `(other.track_id, Result<f32>)` pairs. `Track_id` is
    /// the same for all results and used in higher level operations. `Result<f32>` is `Ok(f32)` when
    /// the distance calculated by `Metric` well, `Err(e)` when `Metric` is unable to calculate the distance.
    ///
    /// # Parameters
    /// * `other` - track to find distances to
    /// * `feature_class` - what feature class to use to calculate distances
    /// * `filter` - defines either results are filtered by distance before the output or not
    pub fn distances(
        &self,
        other: &Self,
        feature_class: u64,
    ) -> Result<Vec<ObservationMetricOk<OA>>> {
        if !self.attributes.compatible(&other.attributes) {
            Err(Errors::IncompatibleAttributes.into())
        } else {
            match (
                self.observations.get(&feature_class),
                other.observations.get(&feature_class),
            ) {
                (Some(left), Some(right)) => Ok(left
                    .iter()
                    .cartesian_product(right.iter())
                    .flat_map(|(l, r)| {
                        let mq = MetricQuery {
                            feature_class,
                            candidate_attrs: self.get_attributes(),
                            candidate_observation: l,
                            track_attrs: other.get_attributes(),
                            track_observation: r,
                        };

                        // let (attribute_metric, feature_distance) = self.metric.new_metric(&mq)?;
                        let (attribute_metric, feature_distance) = self.metric.metric(
                            &mq, // feature_class,
                                // self.get_attributes(),
                                // other.get_attributes(),
                                // l,
                                // r,
                        )?;
                        Some(ObservationMetricOk {
                            from: self.track_id,
                            to: other.track_id,
                            attribute_metric,
                            feature_distance,
                        })
                    })
                    .collect()),
                _ => Err(Errors::ObservationForClassNotFound(
                    self.track_id,
                    other.track_id,
                    feature_class,
                )
                .into()),
            }
        }
    }

    pub fn lookup(&self, query: &TA::Lookup) -> bool {
        query.lookup(&self.attributes, &self.observations, &self.merge_history)
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::examples::current_time_sec;
    use crate::prelude::{NoopNotifier, TrackBuilder};
    use crate::track::utils::{feature_attributes_sort_dec, FromVec};
    use crate::track::{
        Feature, LookupRequest, MetricOutput, MetricQuery, NoopLookup, Observation,
        ObservationAttributes, ObservationMetric, ObservationsDb, Track, TrackAttributes,
        TrackAttributesUpdate, TrackStatus,
    };
    use crate::EPS;
    use anyhow::Result;

    #[derive(Clone)]
    pub struct DefaultAttrs;

    #[derive(Clone)]
    pub struct DefaultAttrUpdates;

    impl TrackAttributesUpdate<DefaultAttrs> for DefaultAttrUpdates {
        fn apply(&self, _attrs: &mut DefaultAttrs) -> Result<()> {
            Ok(())
        }
    }

    impl TrackAttributes<DefaultAttrs, f32> for DefaultAttrs {
        type Update = DefaultAttrUpdates;
        type Lookup = NoopLookup<DefaultAttrs, f32>;

        fn compatible(&self, _other: &DefaultAttrs) -> bool {
            true
        }

        fn merge(&mut self, _other: &DefaultAttrs) -> Result<()> {
            Ok(())
        }

        fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
            Ok(TrackStatus::Pending)
        }
    }

    #[derive(Clone)]
    struct DefaultMetric;
    impl ObservationMetric<DefaultAttrs, f32> for DefaultMetric {
        fn metric(&self, mq: &MetricQuery<'_, DefaultAttrs, f32>) -> MetricOutput<f32> {
            let (e1, e2) = (mq.candidate_observation, mq.track_observation);
            Some((
                f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
                match (e1.feature().as_ref(), e2.feature().as_ref()) {
                    (Some(x), Some(y)) => Some(euclidean(x, y)),
                    _ => None,
                },
            ))
        }

        fn optimize(
            &mut self,
            _feature_class: u64,
            _merge_history: &[u64],
            _attributes: &mut DefaultAttrs,
            features: &mut Vec<Observation<f32>>,
            _prev_length: usize,
            _is_merge: bool,
        ) -> Result<()> {
            features.sort_by(feature_attributes_sort_dec);
            features.truncate(20);
            Ok(())
        }
    }

    #[test]
    fn init() {
        let t1 = Track::new(3, DefaultMetric, DefaultAttrs, NoopNotifier);
        assert_eq!(t1.get_track_id(), 3);
    }

    #[test]
    fn track_distances() -> Result<()> {
        let mut t1 = Track::new(1, DefaultMetric, DefaultAttrs, NoopNotifier);
        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::new(2, DefaultMetric, DefaultAttrs, NoopNotifier);
        t2.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;

        let dists = t1.distances(&t1, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!(*dists[0].feature_distance.as_ref().unwrap() < EPS);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((*dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);

        t2.add_observation(
            0,
            Some(0.2),
            Some(Feature::from_vec(vec![1f32, 1.0f32, 0.0])),
            None,
        )?;

        assert_eq!(t2.observations.get(&0).unwrap().len(), 2);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 2);
        assert!((*dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((*dists[1].feature_distance.as_ref().unwrap() - 1.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn merge_same() -> Result<()> {
        let mut t1 = Track::new(1, DefaultMetric, DefaultAttrs, NoopNotifier);
        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::new(2, DefaultMetric, DefaultAttrs, NoopNotifier);
        t2.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;
        let r = t1.merge(&t2, &[0], true);
        assert!(r.is_ok());
        assert_eq!(t1.observations.get(&0).unwrap().len(), 2);
        Ok(())
    }

    #[test]
    fn merge_other_feature_class() -> Result<()> {
        let mut t1 = Track::new(1, DefaultMetric, DefaultAttrs, NoopNotifier);
        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::new(2, DefaultMetric, DefaultAttrs, NoopNotifier);
        t2.add_observation(
            1,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;
        let r = t1.merge(&t2, &[1], true);
        assert!(r.is_ok());
        assert_eq!(t1.observations.get(&0).unwrap().len(), 1);
        assert_eq!(t1.observations.get(&1).unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn attribute_compatible_match() -> Result<()> {
        #[derive(Default, Debug, Clone)]
        pub struct TimeAttrs {
            start_time: u64,
            end_time: u64,
        }

        #[derive(Default, Clone)]
        pub struct TimeAttrUpdates {
            time: u64,
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
            type Lookup = NoopLookup<TimeAttrs, f32>;

            fn compatible(&self, other: &TimeAttrs) -> bool {
                self.end_time <= other.start_time
            }

            fn merge(&mut self, other: &TimeAttrs) -> Result<()> {
                self.start_time = self.start_time.min(other.start_time);
                self.end_time = self.end_time.max(other.end_time);
                Ok(())
            }

            fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
                if current_time_sec() - self.end_time > 30 {
                    Ok(TrackStatus::Ready)
                } else {
                    Ok(TrackStatus::Pending)
                }
            }
        }

        #[derive(Default, Clone)]
        struct TimeMetric;
        impl ObservationMetric<TimeAttrs, f32> for TimeMetric {
            fn metric(&self, mq: &MetricQuery<'_, TimeAttrs, f32>) -> MetricOutput<f32> {
                let (e1, e2) = (mq.candidate_observation, mq.track_observation);
                Some((
                    f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
                    match (e1.feature().as_ref(), e2.feature().as_ref()) {
                        (Some(x), Some(y)) => Some(euclidean(x, y)),
                        _ => None,
                    },
                ))
            }

            fn optimize(
                &mut self,
                _feature_class: u64,
                _merge_history: &[u64],
                _attributes: &mut TimeAttrs,
                features: &mut Vec<Observation<f32>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                features.sort_by(feature_attributes_sort_dec);
                features.truncate(20);
                Ok(())
            }
        }

        let mut t1 = Track::new(1, TimeMetric::default(), TimeAttrs::default(), NoopNotifier);
        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            Some(TimeAttrUpdates { time: 2 }),
        )?;

        let mut t2 = Track::new(2, TimeMetric::default(), TimeAttrs::default(), NoopNotifier);
        t2.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            Some(TimeAttrUpdates { time: 3 }),
        )?;

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((*dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert_eq!(dists[0].to, 2);

        let mut t3 = Track::new(3, TimeMetric::default(), TimeAttrs::default(), NoopNotifier);
        t3.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            Some(TimeAttrUpdates { time: 1 }),
        )?;

        let dists = t1.distances(&t3, 0);
        assert!(dists.is_err());
        Ok(())
    }

    #[test]
    fn get_classes() -> Result<()> {
        let mut t1 = Track::new(1, DefaultMetric, DefaultAttrs, NoopNotifier);
        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        t1.add_observation(
            1,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;
        let mut classes = t1.get_feature_classes();
        classes.sort();

        assert_eq!(classes, vec![0, 1]);

        Ok(())
    }

    #[test]
    fn attr_metric_update_recover() {
        use thiserror::Error;

        #[derive(Error, Debug)]
        enum TestError {
            #[error("Update Error")]
            Update,
            #[error("Unable to Merge")]
            Merge,
            #[error("Unable to Optimize")]
            Optimize,
        }

        #[derive(Default, Clone, PartialEq, Eq, Debug)]
        pub struct LocalAttrs {
            pub count: u32,
        }

        #[derive(Clone)]
        pub struct LocalAttrsUpdates {
            ignore: bool,
        }

        impl TrackAttributesUpdate<LocalAttrs> for LocalAttrsUpdates {
            fn apply(&self, attrs: &mut LocalAttrs) -> Result<()> {
                if !self.ignore {
                    attrs.count += 1;
                    if attrs.count > 1 {
                        Err(TestError::Update.into())
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
        }

        impl TrackAttributes<LocalAttrs, f32> for LocalAttrs {
            type Update = LocalAttrsUpdates;
            type Lookup = NoopLookup<LocalAttrs, f32>;

            fn compatible(&self, _other: &LocalAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &LocalAttrs) -> Result<()> {
                Err(TestError::Merge.into())
            }

            fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
                Ok(TrackStatus::Pending)
            }
        }

        #[derive(Clone)]
        struct LocalMetric;
        impl ObservationMetric<LocalAttrs, f32> for LocalMetric {
            fn metric(&self, mq: &MetricQuery<LocalAttrs, f32>) -> MetricOutput<f32> {
                let (e1, e2) = (mq.candidate_observation, mq.track_observation);
                Some((
                    f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
                    match (e1.feature().as_ref(), e2.feature().as_ref()) {
                        (Some(x), Some(y)) => Some(euclidean(x, y)),
                        _ => None,
                    },
                ))
            }

            fn optimize(
                &mut self,
                _feature_class: u64,
                _merge_history: &[u64],
                _attributes: &mut LocalAttrs,
                _features: &mut Vec<Observation<f32>>,
                prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                if prev_length == 1 {
                    Err(TestError::Optimize.into())
                } else {
                    Ok(())
                }
            }
        }

        let mut t1 = Track::new(1, LocalMetric, LocalAttrs::default(), NoopNotifier);
        assert_eq!(t1.attributes, LocalAttrs { count: 0 });
        let res = t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            Some(LocalAttrsUpdates { ignore: false }),
        );
        assert!(res.is_ok());
        assert_eq!(t1.attributes, LocalAttrs { count: 1 });

        let res = t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            Some(LocalAttrsUpdates { ignore: true }),
        );
        assert!(res.is_err());
        if let Err(e) = res {
            match e.root_cause().downcast_ref::<TestError>().unwrap() {
                TestError::Update | TestError::Merge => {
                    unreachable!();
                }
                TestError::Optimize => {}
            }
        } else {
            unreachable!();
        }

        assert_eq!(t1.attributes, LocalAttrs { count: 1 });

        let mut t2 = Track::new(2, LocalMetric, LocalAttrs::default(), NoopNotifier);
        assert_eq!(t2.attributes, LocalAttrs { count: 0 });
        let res = t2.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            Some(LocalAttrsUpdates { ignore: false }),
        );
        assert!(res.is_ok());
        assert_eq!(t2.attributes, LocalAttrs { count: 1 });

        let res = t1.merge(&t2, &[0], true);
        if let Err(e) = res {
            match e.root_cause().downcast_ref::<TestError>().unwrap() {
                TestError::Update | TestError::Optimize => {
                    unreachable!();
                }
                TestError::Merge => {}
            }
        } else {
            unreachable!();
        }
        assert_eq!(t1.attributes, LocalAttrs { count: 1 });
    }

    #[test]
    fn merge_history() -> Result<()> {
        let mut t1 = Track::new(0, DefaultMetric, DefaultAttrs, NoopNotifier);
        let mut t2 = Track::new(1, DefaultMetric, DefaultAttrs, NoopNotifier);

        t1.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        t2.add_observation(
            0,
            Some(0.3),
            Some(Feature::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;

        let mut track_with_merge_history = t1.clone();
        let _r = track_with_merge_history.merge(&t2, &[0], true);
        assert_eq!(track_with_merge_history.merge_history, vec![0, 1]);

        let _r = t1.merge(&t2, &[0], false);
        assert_eq!(t1.merge_history, vec![0]);

        Ok(())
    }

    #[test]
    fn unit_track() {
        #[derive(Clone)]
        pub struct UnitAttrs;

        #[derive(Clone)]
        pub struct UnitAttrUpdates;

        impl TrackAttributesUpdate<UnitAttrs> for UnitAttrUpdates {
            fn apply(&self, _attrs: &mut UnitAttrs) -> Result<()> {
                Ok(())
            }
        }

        impl TrackAttributes<UnitAttrs, ()> for UnitAttrs {
            type Update = UnitAttrUpdates;
            type Lookup = NoopLookup<UnitAttrs, ()>;

            fn compatible(&self, _other: &UnitAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &UnitAttrs) -> Result<()> {
                Ok(())
            }

            fn baked(&self, _observations: &ObservationsDb<()>) -> Result<TrackStatus> {
                Ok(TrackStatus::Pending)
            }
        }

        #[derive(Clone)]
        struct UnitMetric;
        impl ObservationMetric<UnitAttrs, ()> for UnitMetric {
            fn metric(&self, mq: &MetricQuery<UnitAttrs, ()>) -> MetricOutput<()> {
                let (e1, e2) = (mq.candidate_observation, mq.track_observation);
                Some((
                    None,
                    match (e1.1.as_ref(), e2.1.as_ref()) {
                        (Some(x), Some(y)) => Some(euclidean(x, y)),
                        _ => None,
                    },
                ))
            }

            fn optimize(
                &mut self,
                _feature_class: u64,
                _merge_history: &[u64],
                _attributes: &mut UnitAttrs,
                features: &mut Vec<Observation<()>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                features.sort_by(feature_attributes_sort_dec);
                features.truncate(20);
                Ok(())
            }
        }

        let _t1 = Track::new(1, UnitMetric, UnitAttrs, NoopNotifier);
    }

    #[test]
    fn lookup() {
        #[derive(Default, Clone)]
        struct Lookup;
        impl LookupRequest<LookupAttrs, f32> for Lookup {
            fn lookup(
                &self,
                _attributes: &LookupAttrs,
                _observations: &ObservationsDb<f32>,
                _merge_history: &[u64],
            ) -> bool {
                true
            }
        }

        #[derive(Clone, Default)]
        struct LookupAttrs;

        #[derive(Clone)]
        pub struct LookupAttributeUpdate;

        impl TrackAttributesUpdate<LookupAttrs> for LookupAttributeUpdate {
            fn apply(&self, _attrs: &mut LookupAttrs) -> Result<()> {
                Ok(())
            }
        }

        impl TrackAttributes<LookupAttrs, f32> for LookupAttrs {
            type Update = LookupAttributeUpdate;
            type Lookup = Lookup;

            fn compatible(&self, _other: &LookupAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &LookupAttrs) -> Result<()> {
                Ok(())
            }

            fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
                Ok(TrackStatus::Ready)
            }
        }

        #[derive(Clone)]
        pub struct LookupMetric;

        impl ObservationMetric<LookupAttrs, f32> for LookupMetric {
            fn metric(&self, mq: &MetricQuery<LookupAttrs, f32>) -> MetricOutput<f32> {
                let (e1, e2) = (mq.candidate_observation, mq.track_observation);
                Some((
                    f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
                    match (e1.feature().as_ref(), e2.feature().as_ref()) {
                        (Some(x), Some(y)) => Some(euclidean(x, y)),
                        _ => None,
                    },
                ))
            }

            fn optimize(
                &mut self,
                _feature_class: u64,
                _merge_history: &[u64],
                _attrs: &mut LookupAttrs,
                _features: &mut Vec<Observation<f32>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                Ok(())
            }
        }

        let t: Track<LookupAttrs, LookupMetric, f32> = TrackBuilder::default()
            .metric(LookupMetric)
            .attributes(LookupAttrs)
            .notifier(NoopNotifier)
            .build()
            .unwrap();
        assert!(t.lookup(&Lookup));
    }
}
