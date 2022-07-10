use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::Errors;
use anyhow::Result;
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;
use ultraviolet::f32x8;

pub mod notify;
pub mod store;
pub mod voting;

pub type TrackDistance = (u64, Result<f32>);

/// Feature vector representation. It is a valid Nalgebra dynamic matrix
pub type Feature = Vec<f32x8>;

pub trait FromVec<V> {
    fn from_vec(vec: V) -> Feature;
}

impl FromVec<Vec<f32>> for Feature {
    fn from_vec(vec: Vec<f32>) -> Feature {
        let mut feature = {
            let one_more = if vec.len() % INT_FEATURE_SIZE > 0 {
                1
            } else {
                0
            };
            Feature::with_capacity(vec.len() / INT_FEATURE_SIZE + one_more)
        };

        let mut acc: [f32; 8] = [0.0; 8];
        let mut part = 0;
        for (counter, i) in vec.into_iter().enumerate() {
            part = counter % INT_FEATURE_SIZE;
            if part == 0 {
                acc = [0.0; 8];
            }
            acc[part] = i;
            if part == INT_FEATURE_SIZE - 1 {
                feature.push(f32x8::new(acc));
                part = 8;
            }
        }

        if part < 8 {
            feature.push(f32x8::new(acc));
        }
        feature
    }
}

//impl From<> for Feature {
//    fn from(f: OMatrix<f32, Dynamic, Dynamic>) -> Self {}
//}

const INT_FEATURE_SIZE: usize = 8;

/// Feature specification. It is a tuple of confidence (f32) and Feature itself. Such a representation
/// is used to filter low quality features during the collecting. If the model doesn't provide the confidence
/// arbitrary confidence may be used and filtering implemented accordingly.
pub type FeatureSpec = (f32, Feature);

/// Table that accumulates observed features across the tracks (or objects)
pub type FeatureObservationsGroups = HashMap<u64, Vec<FeatureSpec>>;

/// The trait that implements the methods for features comparison and filtering
pub trait Metric: Default + Send + Sync + Clone + 'static {
    /// calculates the distance between two features.
    /// The output is `Result<f32>` because the method may return distance calculation error if the distance
    /// cannot be computed for two features. E.g. when one of them has low confidence.
    fn distance(feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32>;

    /// the method is used every time, when a new observation is added to the feature storage as well as when
    /// two tracks are merged.
    ///
    /// # Arguments
    ///
    /// * `feature_class` - the feature class
    /// * `merge_history` - how many times the track was merged already (it may be used to calculate maximum amount of observation for the feature)
    /// * `observations` - features to optimize
    /// * `prev_length` - previous length of observations (before the current observation was added or merge occurred)
    ///
    /// # Returns
    /// * `Ok(())` if the optimization is successful
    /// * `Err(e)` if the optimization failed
    ///
    fn optimize(
        &mut self,
        feature_class: &u64,
        merge_history: &[u64],
        observations: &mut Vec<FeatureSpec>,
        prev_length: usize,
    ) -> Result<()>;
}

/// Enum which specifies the status of feature tracks in storage. When the feature tracks are collected,
/// eventually the track must be complete so it can be used for
/// database search and later merge operations.
///
#[derive(Clone, Debug)]
pub enum TrackBakingStatus {
    /// The track is ready and can be used to find similar tracks for merge.
    Ready,
    /// The track is not ready and still being collected.
    Pending,
    /// The track is invalid because somehow became incorrect during the collection.
    Wasted,
}

/// The trait represents user defined Attributes of the track and is used to define custom attributes that
/// fit a domain field
///
/// When the user implements attributes they has to implement this trait to create a valid attributes object.
///
pub trait AttributeMatch<A>: Default + Send + Sync + Clone + 'static {
    /// The method is used to evaluate attributes of two tracks to determine whether tracks are compatible
    /// for distance calculation. When the attributes are compatible, the method returns `true`.
    ///
    /// E.g.
    ///     Let's imagine the case when the track includes the attributes for track begin and end timestamps.
    ///     The tracks are compatible their timeframes don't intersect between each other. The method `compatible`
    ///     can decide that.
    ///
    fn compatible(&self, other: &A) -> bool;

    /// When the tracks are merged, their attributes are merged as well. The method defines the approach to merge attributes.
    ///
    /// E.g.
    ///     Let's imagine the case when the track includes the attributes for track begin and end timestamps.
    ///     Merge operation may look like `[b1; e1] + [b2; e2] -> [min(b1, b2); max(e1, e2)]`.
    ///
    fn merge(&mut self, other: &A) -> Result<()>;

    /// The method is used by storage to determine when track is ready/not ready/wasted. Look at [TrackBakingStatus](TrackBakingStatus).
    ///
    /// It uses attribute information collected across the track config and features information.
    ///
    /// E.g.
    ///     track is ready when
    ///          `now - end_timestamp > 30s` (no features collected during the last 30 seconds).
    ///
    fn baked(&self, observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus>;
}

/// The attribute update information that is sent with new features to the track is represented by the trait.
///
/// The trait must be implemented for update struct for specific attributes struct implementation.
///
pub trait AttributeUpdate<A>: Clone + Send + Sync + 'static {
    /// Method is used to update track attributes from update structure.
    ///
    fn apply(&self, attrs: &mut A) -> Result<()>;
}

/// Utility function that can be used by [Metric](Metric::optimize) implementors to sort
/// features by confidence parameter decreasingly to purge low confidence features.
///
pub fn feat_confidence_cmp(e1: &FeatureSpec, e2: &FeatureSpec) -> Ordering {
    e2.0.partial_cmp(&e1.0).unwrap()
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
pub struct Track<A, M, U, N = NoopNotifier>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    M: Metric,
    U: AttributeUpdate<A>,
{
    attributes: A,
    track_id: u64,
    observations: FeatureObservationsGroups,
    metric: M,
    phantom_attribute_update: PhantomData<U>,
    merge_history: Vec<u64>,
    notifier: N,
}

/// One and only parametrized track implementation.
///
impl<A, M, U, N> Track<A, M, U, N>
where
    N: ChangeNotifier,
    A: AttributeMatch<A>,
    M: Metric,
    U: AttributeUpdate<A>,
{
    /// Creates a new track with id `track_id` with `metric` initializer object and `attributes` initializer object.
    ///
    /// The `metric` and `attributes` are optional, if `None` is specified, then `Default` initializer is used.
    ///
    pub fn new(
        track_id: u64,
        metric: Option<M>,
        attributes: Option<A>,
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
                A::default()
            },
            track_id,
            observations: Default::default(),
            metric: if let Some(m) = metric {
                m
            } else {
                M::default()
            },
            phantom_attribute_update: Default::default(),
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

    /// Returns current track attributes.
    ///
    pub fn get_attributes(&self) -> &A {
        &self.attributes
    }

    /// Returns all classes
    ///
    pub fn get_feature_classes(&self) -> Vec<u64> {
        self.observations.keys().cloned().collect()
    }

    fn update_attributes(&mut self, update: U) -> Result<()> {
        update.apply(&mut self.attributes)
    }

    /// Adds new observation to track.
    ///
    /// When the method is called, the track attributes are updated according to `update` argument, and the feature
    /// is placed into features for a specified feature class.
    ///
    /// # Arguments
    /// * `feature_class` - class of observation
    /// * `feature_q` - quality of the feature (confidence, or another parameter that defines how the observation is valuable across the observations).
    /// * `feature` - observation to add to the track for specified `feature_class`.
    /// * `attribute_update` - attribute update message
    ///
    /// # Returns
    /// Returns `Result<()>` where `Ok(())` if attributes are updated without errors AND observation is added AND observations optimized without errors.
    ///
    ///
    pub fn add_observation(
        &mut self,
        feature_class: u64,
        feature_q: f32,
        feature: Feature,
        attribute_update: U,
    ) -> Result<()> {
        let last_attributes = self.attributes.clone();
        let last_observations = self.observations.clone();
        let last_metric = self.metric.clone();

        let res = self.update_attributes(attribute_update);
        if res.is_err() {
            self.attributes = last_attributes;
            res?;
            unreachable!();
        }

        match self.observations.get_mut(&feature_class) {
            None => {
                self.observations
                    .insert(feature_class, vec![(feature_q, feature)]);
            }
            Some(observations) => {
                observations.push((feature_q, feature));
            }
        }
        let observations = self.observations.get_mut(&feature_class).unwrap();
        let prev_length = observations.len() - 1;

        let res = self.metric.optimize(
            &feature_class,
            &self.merge_history,
            observations,
            prev_length,
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
    pub fn merge(&mut self, other: &Self, classes: &[u64]) -> Result<()> {
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

            if let Some(prev_length) = prev_length {
                let res = self.metric.optimize(
                    cls,
                    &self.merge_history,
                    self.observations.get_mut(cls).unwrap(),
                    prev_length,
                );

                if res.is_err() {
                    self.attributes = last_attributes;
                    self.observations = last_observations;
                    self.metric = last_metric;
                    res?;
                    unreachable!();
                }
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
    pub fn distances(&self, other: &Self, feature_class: u64) -> Result<Vec<TrackDistance>> {
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
                    .map(|(l, r)| (other.track_id, M::distance(feature_class, l, r)))
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
    use crate::track::{
        feat_confidence_cmp, AttributeMatch, AttributeUpdate, Feature, FeatureObservationsGroups,
        FeatureSpec, FromVec, Metric, Track, TrackBakingStatus,
    };
    use crate::EPS;
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Default, Clone)]
    pub struct DefaultAttrs;

    #[derive(Default, Clone)]
    pub struct DefaultAttrUpdates;

    impl AttributeUpdate<DefaultAttrs> for DefaultAttrUpdates {
        fn apply(&self, _attrs: &mut DefaultAttrs) -> Result<()> {
            Ok(())
        }
    }

    impl AttributeMatch<DefaultAttrs> for DefaultAttrs {
        fn compatible(&self, _other: &DefaultAttrs) -> bool {
            true
        }

        fn merge(&mut self, _other: &DefaultAttrs) -> Result<()> {
            Ok(())
        }

        fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
            Ok(TrackBakingStatus::Pending)
        }
    }

    #[derive(Default, Clone)]
    struct DefaultMetric;
    impl Metric for DefaultMetric {
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
            features.truncate(20);
            Ok(())
        }
    }

    #[test]
    fn init() {
        let t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> =
            Track::new(3, None, None, None);
        assert_eq!(t1.get_track_id(), 3);
    }

    #[test]
    fn track_distances() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let dists = t1.distances(&t1, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!(*dists[0].1.as_ref().unwrap() < EPS);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((*dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);

        t2.add_observation(
            0,
            0.2,
            Feature::from_vec(vec![1f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        )?;

        assert_eq!(t2.observations.get(&0).unwrap().len(), 2);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 2);
        assert!((*dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((*dists[1].1.as_ref().unwrap() - 1.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn merge_same() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        )?;
        let r = t1.merge(&t2, &vec![0]);
        assert!(r.is_ok());
        assert_eq!(t1.observations.get(&0).unwrap().len(), 2);
        Ok(())
    }

    #[test]
    fn merge_other_feature_class() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            1,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        )?;
        let r = t1.merge(&t2, &vec![1]);
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

            fn baked(
                &self,
                _observations: &FeatureObservationsGroups,
            ) -> Result<TrackBakingStatus> {
                if SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    - self.end_time
                    > 30
                {
                    Ok(TrackBakingStatus::Ready)
                } else {
                    Ok(TrackBakingStatus::Pending)
                }
            }
        }

        #[derive(Default, Clone)]
        struct TimeMetric;
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
                features.truncate(20);
                Ok(())
            }
        }

        let mut t1: Track<TimeAttrs, TimeMetric, TimeAttrUpdates> = Track::default();
        t1.track_id = 1;
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            TimeAttrUpdates { time: 2 },
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            TimeAttrUpdates { time: 3 },
        )?;
        t2.track_id = 2;

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((*dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert_eq!(dists[0].0, 2);

        let mut t3 = Track::default();
        t3.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            TimeAttrUpdates { time: 1 },
        )?;

        let dists = t1.distances(&t3, 0);
        assert!(dists.is_err());
        Ok(())
    }

    #[test]
    fn get_classes() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        t1.add_observation(
            1,
            0.3,
            Feature::from_vec(vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
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

        impl AttributeUpdate<DefaultAttrs> for DefaultAttrUpdates {
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

        impl AttributeMatch<DefaultAttrs> for DefaultAttrs {
            fn compatible(&self, _other: &DefaultAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &DefaultAttrs) -> Result<()> {
                Err(TestError::MergeError.into())
            }

            fn baked(
                &self,
                _observations: &FeatureObservationsGroups,
            ) -> Result<TrackBakingStatus> {
                Ok(TrackBakingStatus::Pending)
            }
        }

        #[derive(Default, Clone)]
        struct DefaultMetric;
        impl Metric for DefaultMetric {
            fn distance(_feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32> {
                Ok(euclidean(&e1.1, &e2.1))
            }

            fn optimize(
                &mut self,
                _feature_class: &u64,
                _merge_history: &[u64],
                _features: &mut Vec<FeatureSpec>,
                prev_length: usize,
            ) -> Result<()> {
                if prev_length == 1 {
                    Err(TestError::OptimizeError.into())
                } else {
                    Ok(())
                }
            }
        }

        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        assert_eq!(t1.attributes, DefaultAttrs { count: 0 });
        let res = t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates { ignore: false },
        );
        assert!(res.is_ok());
        assert_eq!(t1.attributes, DefaultAttrs { count: 1 });

        let res = t1.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates { ignore: true },
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

        let mut t2: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        assert_eq!(t2.attributes, DefaultAttrs { count: 0 });
        let res = t2.add_observation(
            0,
            0.3,
            Feature::from_vec(vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates { ignore: false },
        );
        assert!(res.is_ok());
        assert_eq!(t2.attributes, DefaultAttrs { count: 1 });

        let res = t1.merge(&t2, &vec![0]);
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
}
