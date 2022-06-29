/// Module doc
use crate::Errors;
use anyhow::Result;
use itertools::Itertools;
use nalgebra::{Dynamic, OMatrix};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;

pub mod store;
pub mod voting;

/// Feature vector representation. It is a valid Nalgebra dynamic matrix
pub type Feature = OMatrix<f32, Dynamic, Dynamic>;

/// Feature specification. It is a tuple of confidence (f32) and Feature itself. Such a representation
/// is used to filter low quality features during the collecting. If the model doesn't provide the confidence
/// arbitrary confidence may be used and filtering implemented accordingly.
pub type FeatureSpec = (f32, Feature);

/// Table that accumulates observed features across the tracks (or objects)
pub type FeatureObservationsGroups = HashMap<u64, Vec<FeatureSpec>>;

/// The trait that implements the methods for features comparison and filtering
pub trait Metric {
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
#[derive(Clone)]
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
pub trait AttributeMatch<A> {
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
    /// It uses attribute information collected across the track build and features information.
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
pub trait AttributeUpdate<A> {
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
#[derive(Default)]
pub struct Track<A, M, U>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
{
    attributes: A,
    track_id: u64,
    observations: FeatureObservationsGroups,
    metric: M,
    phantom_attribute_update: PhantomData<U>,
    merge_history: Vec<u64>,
}

/// One and only parametrized track implementation.
///
impl<A, M, U> Track<A, M, U>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
{
    /// Creates a new track with id `track_id` with `metric` initializer object and `attributes` initializer object.
    ///
    /// The `metric` and `attributes` are optional, if `None` is specified, then `Default` initializer is used.
    ///
    pub fn new(track_id: u64, metric: Option<M>, attributes: Option<A>) -> Self {
        Self {
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
        }
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
    /// When the method returns Err, the track is likely incorrect and must be either removed from store or validated by user somehow. That's because
    /// all operations mentioned above are not transactional to avoid memory copies.
    ///
    pub fn add_observation(
        &mut self,
        feature_class: u64,
        feature_q: f32,
        feature: Feature,
        attribute_update: U,
    ) -> Result<()> {
        self.update_attributes(attribute_update)?;
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

        self.metric.optimize(
            &feature_class,
            &self.merge_history,
            observations,
            prev_length,
        )?;

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
    /// * both: `{S[class]} U {OTHER[class]}`
    /// * self: `{S[class]}`
    /// * other: `{OTHER[class]}`
    ///
    pub fn merge(&mut self, other: &Self, classes: &Vec<u64>) -> Result<()> {
        self.attributes.merge(&other.attributes)?;
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
                self.metric.optimize(
                    &cls,
                    &self.merge_history,
                    self.observations.get_mut(cls).unwrap(),
                    prev_length,
                )?;
            }
        }
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
    pub fn distances(&self, other: &Self, feature_class: u64) -> Result<Vec<(u64, Result<f32>)>> {
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
                _ => Err(Errors::MissingObservation.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::track::{
        feat_confidence_cmp, AttributeMatch, AttributeUpdate, Feature, FeatureObservationsGroups,
        FeatureSpec, Metric, Track, TrackBakingStatus,
    };
    use crate::EPS;
    use anyhow::Result;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Default)]
    pub struct DefaultAttrs;

    #[derive(Default)]
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

    #[derive(Default)]
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
    fn basic_methods() {
        let t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::new(3, None, None);
        assert_eq!(t1.get_track_id(), 3);
    }

    #[test]
    fn track_distances() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
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
            Feature::from_vec(1, 3, vec![1f32, 1.0f32, 0.0]),
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
    fn merge() -> Result<()> {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add_observation(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        )?;
        let r = t1.merge(&t2, &vec![0]);
        assert!(r.is_ok());
        assert_eq!(t1.observations.get(&0).unwrap().len(), 2);
        Ok(())
    }

    #[test]
    fn attribute_compatible_match() -> Result<()> {
        #[derive(Default, Debug)]
        pub struct TimeAttrs {
            start_time: u64,
            end_time: u64,
        }

        #[derive(Default)]
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

        #[derive(Default)]
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
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            TimeAttrUpdates { time: 2 },
        )?;

        let mut t2 = Track::default();
        t2.add_observation(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
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
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            TimeAttrUpdates { time: 1 },
        )?;

        let dists = t1.distances(&t3, 0);
        assert!(dists.is_err());
        Ok(())
    }
}
