use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::Errors;
use anyhow::Result;
use itertools::Itertools;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::take;
use ultraviolet::f32x8;

pub mod builder;
pub mod notify;
pub mod store;
pub mod utils;
pub mod voting;

/// Return type item for distances between the current track and other track.
///
#[derive(Debug, Clone)]
pub struct ObservationMetricResult<M>
where
    M: ObservationAttributes,
{
    /// source track ID
    pub from: u64,
    /// compared track ID
    pub to: u64,
    /// custom feature attribute metric object calculated for pairwise feature attributes
    pub attribute_metric: Option<M::MetricObject>,
    /// distance calculated for pairwise features
    pub feature_distance: Option<f32>,
}

impl<M> ObservationMetricResult<M>
where
    M: ObservationAttributes,
{
    pub fn new(
        from: u64,
        to: u64,
        attribute_metric: Option<M::MetricObject>,
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
pub type Observation = Vec<f32x8>;

/// Number of SIMD lanes used to store observation parts internally
const FEATURE_LANES_SIZE: usize = 8;

/// Feature specification. It is a tuple of observation attributes (T) and Observation itself. Such a representation
/// is used to support the estimation for every observation during the collecting. If the model doesn't provide the feature attributes
/// `()` may be used.
#[derive(Default, Clone)]
pub struct ObservationSpec<T>(pub Option<T>, pub Option<Observation>)
where
    T: Default + Send + Sync + Clone + 'static + PartialOrd;

/// Table that accumulates observed features across the tracks (or objects)
pub type ObservationsDb<T> = HashMap<u64, Vec<ObservationSpec<T>>>;

/// Custom feature attributes object that accompanies the observation itself
pub trait ObservationAttributes: Default + Send + Sync + Clone + PartialOrd + 'static {
    type MetricObject: Debug + Default + Send + Sync + Clone + PartialOrd + 'static;
    fn calculate_metric_object(l: &Option<Self>, r: &Option<Self>) -> Option<Self::MetricObject>;
}

/// Output result type used by metric when pairwise metric is calculated
///
/// `None` - no metric for that pair - the result will be dropped (optimization technique)
/// `Some(X, Y)` - metric is calculated, values are inside
///
pub type MetricOutput<T> = Option<(Option<T>, Option<f32>)>;

/// The trait that implements the methods for features comparison and filtering
pub trait ObservationMetric<TA, OA: ObservationAttributes>:
    Default + Send + Sync + Clone + 'static
{
    /// calculates the distance between two features.
    /// The output is `Result<f32>` because the method may return distance calculation error if the distance
    /// cannot be computed for two features. E.g. when one of them has low confidence.
    ///
    /// # Parameters
    /// * `feature_class` - class of currently used feature
    /// * `left_attrs` - left track attributes
    /// * `right_attrs` - right track attributes
    /// * `left_observation` - left track observation
    /// * `right_observation` - right track observation
    ///
    fn metric(
        feature_class: u64,
        left_attrs: &TA,
        right_attrs: &TA,
        left_observation: &ObservationSpec<OA>,
        right_observation: &ObservationSpec<OA>,
    ) -> MetricOutput<OA::MetricObject>;

    /// the method is used every time, when a new observation is added to the feature storage as well as when
    /// two tracks are merged.
    ///
    /// # Arguments
    ///
    /// * `feature_class` - the feature class
    /// * `merge_history` - how many times the track was merged already (it may be used to calculate maximum amount of observation for the feature)
    /// * `attributes` - mutable attributes that can be updated or read during optimization
    /// * `observations` - features to optimize
    /// * `prev_length` - previous length of observations (before the current observation was added or merge occurred)
    /// * `is_merge` - true, when merge op, false when the feature added to the object
    ///
    /// # Returns
    /// * `Ok(())` if the optimization is successful
    /// * `Err(e)` if the optimization failed
    ///
    fn optimize(
        &mut self,
        feature_class: &u64,
        merge_history: &[u64],
        attributes: &mut TA,
        observations: &mut Vec<ObservationSpec<OA>>,
        prev_length: usize,
        is_merge: bool,
    ) -> Result<()>;

    /// The postprocessing is run just before the executor returns calculated distances. `self` is a metric object
    /// kept in track that passed to distance calculation methods.
    ///
    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricResult<OA>>,
    ) -> Vec<ObservationMetricResult<OA>> {
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
    /// The track is invalid because somehow became incorrect during the collection.
    Wasted,
}

/// The trait represents user defined Track Attributes. It is used to define custom attributes that
/// fit a domain field where tracking implemented.
///
/// When the user implements track attributes they has to implement this trait to create a valid attributes object.
///
pub trait TrackAttributes<TA, OA: ObservationAttributes>:
    Default + Send + Sync + Clone + 'static
{
    type Update: TrackAttributesUpdate<TA>;
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
    /// It uses attribute information collected across the track config and features information.
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
    pub fn new(
        track_id: u64,
        metric: Option<M>,
        attributes: Option<TA>,
        notifier: Option<N>,
    ) -> Self {
        let mut v = Self {
            notifier: if let Some(notifier) = notifier {
                notifier
            } else {
                N::default()
            },
            attributes: if let Some(attributes) = attributes {
                attributes
            } else {
                TA::default()
            },
            track_id,
            observations: Default::default(),
            metric: if let Some(m) = metric {
                m
            } else {
                M::default()
            },
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

    pub fn get_observations(&self, feature_class: u64) -> Option<&Vec<ObservationSpec<OA>>> {
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

    fn update_attributes(&mut self, update: TA::Update) -> Result<()> {
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
        feature: Option<Observation>,
        track_attributes_update: Option<TA::Update>,
    ) -> Result<()> {
        let last_attributes = self.attributes.clone();
        let last_observations = self.observations.clone();
        let last_metric = self.metric.clone();

        if let Some(track_attributes_update) = track_attributes_update {
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
                    vec![ObservationSpec(feature_attributes, feature)],
                );
            }
            Some(observations) => {
                observations.push(ObservationSpec(feature_attributes, feature));
            }
        }
        let observations = self.observations.get_mut(&feature_class).unwrap();
        let prev_length = observations.len() - 1;

        let res = self.metric.optimize(
            &feature_class,
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
                vec![self.merge_history.clone(), other.merge_history.clone()]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
            } else {
                take(&mut self.merge_history)
            };

            if let Some(prev_length) = prev_length {
                let res = self.metric.optimize(
                    cls,
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
    ) -> Result<Vec<ObservationMetricResult<OA>>> {
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
                        let (attribute_metric, feature_distance) = M::metric(
                            feature_class,
                            self.get_attributes(),
                            other.get_attributes(),
                            l,
                            r,
                        )?;
                        Some(ObservationMetricResult {
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
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::test_stuff::current_time_sec;
    use crate::track::utils::{feature_attributes_sort_dec, FromVec};
    use crate::track::{
        MetricOutput, Observation, ObservationAttributes, ObservationMetric, ObservationSpec,
        ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate, TrackStatus,
    };
    use crate::EPS;
    use anyhow::Result;

    #[derive(Default, Clone)]
    pub struct DefaultAttrs;

    #[derive(Default, Clone)]
    pub struct DefaultAttrUpdates;

    impl TrackAttributesUpdate<DefaultAttrs> for DefaultAttrUpdates {
        fn apply(&self, _attrs: &mut DefaultAttrs) -> Result<()> {
            Ok(())
        }
    }

    impl TrackAttributes<DefaultAttrs, f32> for DefaultAttrs {
        type Update = DefaultAttrUpdates;

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

    #[derive(Default, Clone)]
    struct DefaultMetric;
    impl ObservationMetric<DefaultAttrs, f32> for DefaultMetric {
        fn metric(
            _feature_class: u64,
            _attrs1: &DefaultAttrs,
            _attrs2: &DefaultAttrs,
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
            _attributes: &mut DefaultAttrs,
            features: &mut Vec<ObservationSpec<f32>>,
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
        let t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::new(3, None, None, None);
        assert_eq!(t1.get_track_id(), 3);
    }

    #[test]
    fn track_distances() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
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
            Some(Observation::from_vec(vec![1f32, 1.0f32, 0.0])),
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
        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;
        let r = t1.merge(&t2, &vec![0], true);
        assert!(r.is_ok());
        assert_eq!(t1.observations.get(&0).unwrap().len(), 2);
        Ok(())
    }

    #[test]
    fn merge_other_feature_class() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            1,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;
        let r = t1.merge(&t2, &vec![1], true);
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
                _attributes: &mut TimeAttrs,
                features: &mut Vec<ObservationSpec<f32>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                features.sort_by(feature_attributes_sort_dec);
                features.truncate(20);
                Ok(())
            }
        }

        let mut t1: Track<TimeAttrs, TimeMetric, f32> = Track::default();
        t1.track_id = 1;
        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            Some(TimeAttrUpdates { time: 2 }),
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
            Some(TimeAttrUpdates { time: 3 }),
        )?;
        t2.track_id = 2;

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((*dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert_eq!(dists[0].to, 2);

        let mut t3 = Track::default();
        t3.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
            Some(TimeAttrUpdates { time: 1 }),
        )?;

        let dists = t1.distances(&t3, 0);
        assert!(dists.is_err());
        Ok(())
    }

    #[test]
    fn get_classes() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        t1.add_observation(
            1,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
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
            UpdateError,
            #[error("MergeError")]
            MergeError,
            #[error("OptimizeError")]
            OptimizeError,
        }

        #[derive(Default, Clone, PartialEq, Debug)]
        pub struct DefaultAttrs {
            pub count: u32,
        }

        #[derive(Default, Clone)]
        pub struct DefaultAttrUpdates {
            ignore: bool,
        }

        impl TrackAttributesUpdate<DefaultAttrs> for DefaultAttrUpdates {
            fn apply(&self, attrs: &mut DefaultAttrs) -> Result<()> {
                if !self.ignore {
                    attrs.count += 1;
                    if attrs.count > 1 {
                        Err(TestError::UpdateError.into())
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
        }

        impl TrackAttributes<DefaultAttrs, f32> for DefaultAttrs {
            type Update = DefaultAttrUpdates;

            fn compatible(&self, _other: &DefaultAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &DefaultAttrs) -> Result<()> {
                Err(TestError::MergeError.into())
            }

            fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
                Ok(TrackStatus::Pending)
            }
        }

        #[derive(Default, Clone)]
        struct DefaultMetric;
        impl ObservationMetric<DefaultAttrs, f32> for DefaultMetric {
            fn metric(
                _feature_class: u64,
                _attrs1: &DefaultAttrs,
                _attrs2: &DefaultAttrs,
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
                _attributes: &mut DefaultAttrs,
                _features: &mut Vec<ObservationSpec<f32>>,
                prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                if prev_length == 1 {
                    Err(TestError::OptimizeError.into())
                } else {
                    Ok(())
                }
            }
        }

        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        assert_eq!(t1.attributes, DefaultAttrs { count: 0 });
        let res = t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            Some(DefaultAttrUpdates { ignore: false }),
        );
        assert!(res.is_ok());
        assert_eq!(t1.attributes, DefaultAttrs { count: 1 });

        let res = t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            Some(DefaultAttrUpdates { ignore: true }),
        );
        assert!(res.is_err());
        if let Err(e) = res {
            match e.root_cause().downcast_ref::<TestError>().unwrap() {
                TestError::UpdateError | TestError::MergeError => {
                    unreachable!();
                }
                TestError::OptimizeError => {}
            }
        } else {
            unreachable!();
        }

        assert_eq!(t1.attributes, DefaultAttrs { count: 1 });

        let mut t2: Track<DefaultAttrs, DefaultMetric, f32> = Track::default();
        assert_eq!(t2.attributes, DefaultAttrs { count: 0 });
        let res = t2.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            Some(DefaultAttrUpdates { ignore: false }),
        );
        assert!(res.is_ok());
        assert_eq!(t2.attributes, DefaultAttrs { count: 1 });

        let res = t1.merge(&t2, &vec![0], true);
        if let Err(e) = res {
            match e.root_cause().downcast_ref::<TestError>().unwrap() {
                TestError::UpdateError | TestError::OptimizeError => {
                    unreachable!();
                }
                TestError::MergeError => {}
            }
        } else {
            unreachable!();
        }
        assert_eq!(t1.attributes, DefaultAttrs { count: 1 });
    }

    #[test]
    fn merge_history() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, f32> = Track::new(0, None, None, None);
        let mut t2 = Track::new(1, None, None, None);

        t1.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![1f32, 0.0, 0.0])),
            None,
        )?;

        t2.add_observation(
            0,
            Some(0.3),
            Some(Observation::from_vec(vec![0f32, 1.0f32, 0.0])),
            None,
        )?;

        let mut track_with_merge_history = t1.clone();
        let _r = track_with_merge_history.merge(&t2, &vec![0], true);
        assert_eq!(track_with_merge_history.merge_history, vec![0, 1]);

        let _r = t1.merge(&t2, &vec![0], false);
        assert_eq!(t1.merge_history, vec![0]);

        Ok(())
    }

    #[test]
    fn unit_track() {
        #[derive(Default, Clone)]
        pub struct UnitAttrs;

        #[derive(Default, Clone)]
        pub struct UnitAttrUpdates;

        impl TrackAttributesUpdate<UnitAttrs> for UnitAttrUpdates {
            fn apply(&self, _attrs: &mut UnitAttrs) -> Result<()> {
                Ok(())
            }
        }

        impl TrackAttributes<UnitAttrs, ()> for UnitAttrs {
            type Update = UnitAttrUpdates;

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

        #[derive(Default, Clone)]
        struct UnitMetric;
        impl ObservationMetric<UnitAttrs, ()> for UnitMetric {
            fn metric(
                _feature_class: u64,
                _attrs1: &UnitAttrs,
                _attrs2: &UnitAttrs,
                e1: &ObservationSpec<()>,
                e2: &ObservationSpec<()>,
            ) -> MetricOutput<()> {
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
                _feature_class: &u64,
                _merge_history: &[u64],
                _attributes: &mut UnitAttrs,
                features: &mut Vec<ObservationSpec<()>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                features.sort_by(feature_attributes_sort_dec);
                features.truncate(20);
                Ok(())
            }
        }

        let _t1: Track<UnitAttrs, UnitMetric, ()> = Track::new(0, None, None, None);
    }
}
