use itertools::Itertools;
use nalgebra::{Dynamic, OMatrix};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Sub;

pub mod store;

pub type Feature = OMatrix<f32, Dynamic, Dynamic>;
pub type FeatureSpec = (f32, Feature);
pub type FeatureObservationsGroups = HashMap<u64, Vec<FeatureSpec>>;

pub fn standard_vector_distance(f1: &Feature, f2: &Feature) -> f32 {
    f1.sub(f2).map(|component| component * component).sum()
}

pub trait Metric {
    fn distance(feature_id: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> f32;
    fn optimize(
        &mut self,
        feature_id: &u64,
        merge_history: &[u64],
        observations: &mut Vec<FeatureSpec>,
        prev_length: usize,
    );
    // features.sort_by(feat_sort_cmp);
    // features.truncate(M::filter(feature_id, &self.merge_history));
}

pub trait AttributeMatch<A> {
    fn compatible(&self, other: &A) -> bool;
    fn merge(&mut self, other: &A);
    fn baked(&self, observations: &FeatureObservationsGroups) -> bool;
}

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

pub trait AttributeUpdate<A> {
    fn apply(&self, attrs: &mut A);
}

pub fn feat_sort_cmp(e1: &FeatureSpec, e2: &FeatureSpec) -> Ordering {
    e2.0.partial_cmp(&e1.0).unwrap()
}

impl<A, M, U> Track<A, M, U>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
{
    pub fn new(track_id: u64) -> Self {
        Self {
            attributes: Default::default(),
            track_id,
            observations: Default::default(),
            metric: M::default(),
            phantom_attribute_update: Default::default(),
            merge_history: vec![track_id],
        }
    }

    pub fn get_track_id(&self) -> u64 {
        self.track_id
    }

    pub fn get_attributes(&self) -> &A {
        &self.attributes
    }

    fn update_attributes(&mut self, update: U) {
        update.apply(&mut self.attributes);
    }

    fn add(&mut self, feature_id: u64, reid_q: f32, reid_v: Feature, update: U) {
        self.update_attributes(update);
        match self.observations.get_mut(&feature_id) {
            None => {
                self.observations.insert(feature_id, vec![(reid_q, reid_v)]);
            }
            Some(observations) => {
                observations.push((reid_q, reid_v));
            }
        }
        let observations = self.observations.get_mut(&feature_id).unwrap();
        let prev_length = observations.len() - 1;

        self.metric
            .optimize(&feature_id, &self.merge_history, observations, prev_length);
    }

    pub fn merge(&mut self, other: &Self, features: &Vec<u64>) {
        self.attributes.merge(&other.attributes);
        for feature_id in features {
            let dest = self.observations.get_mut(feature_id);
            let src = other.observations.get(feature_id);
            if let (Some(dest_observations), Some(src_observations)) = (dest, src) {
                let prev_length = dest_observations.len();
                dest_observations.extend(src_observations.iter().cloned());
                self.metric.optimize(
                    &feature_id,
                    &self.merge_history,
                    dest_observations,
                    prev_length,
                );
            }
        }
    }

    pub(crate) fn distances(&self, other: &Self, feature_id: u64) -> Option<Vec<(u64, f32)>> {
        if !self.attributes.compatible(&other.attributes) {
            None
        } else {
            match (
                self.observations.get(&feature_id),
                other.observations.get(&feature_id),
            ) {
                (Some(left), Some(right)) => Some(
                    left.iter()
                        .cartesian_product(right.iter())
                        .map(|(l, r)| (other.track_id, M::distance(feature_id, l, r)))
                        .collect(),
                ),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::track::{
        feat_sort_cmp, standard_vector_distance, AttributeMatch, AttributeUpdate, Feature,
        FeatureObservationsGroups, FeatureSpec, Metric, Track,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    const EPS: f32 = 0.00001;

    #[derive(Default)]
    pub struct DefaultAttrs;

    #[derive(Default)]
    pub struct DefaultAttrUpdates;

    impl AttributeUpdate<DefaultAttrs> for DefaultAttrUpdates {
        fn apply(&self, _attrs: &mut DefaultAttrs) {}
    }

    impl AttributeMatch<DefaultAttrs> for DefaultAttrs {
        fn compatible(&self, _other: &DefaultAttrs) -> bool {
            true
        }

        fn merge(&mut self, _other: &DefaultAttrs) {}

        fn baked(&self, _observations: &FeatureObservationsGroups) -> bool {
            false
        }
    }

    #[derive(Default)]
    struct DefaultMetric;
    impl Metric for DefaultMetric {
        fn distance(_feature_id: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> f32 {
            standard_vector_distance(&e1.1, &e2.1)
        }

        fn optimize(
            &mut self,
            _feature_id: &u64,
            _merge_history: &[u64],
            features: &mut Vec<FeatureSpec>,
            _prev_length: usize,
        ) {
            features.sort_by(feat_sort_cmp);
            features.truncate(20);
        }
    }

    #[test]
    fn vec_distances() {
        let v1 = Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]);
        let v2 = Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]);
        let d = standard_vector_distance(&v1, &v1);
        assert!(d < EPS);

        let d = standard_vector_distance(&v1, &v2);
        assert!((d - 2.0f32).abs() < EPS);
    }

    #[test]
    fn basic_methods() {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::new(3);
        assert_eq!(t1.get_track_id(), 3);
    }

    #[test]
    fn track_distances() {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        );

        let mut t2 = Track::default();
        t2.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        );

        let dists = t1.distances(&t1, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!(dists[0].1 < EPS);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((dists[0].1 - 2.0).abs() < EPS);

        t2.add(
            0,
            0.2,
            Feature::from_vec(1, 3, vec![1f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        );

        assert_eq!(t2.observations.get(&0).unwrap().len(), 2);

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 2);
        assert!((dists[0].1 - 2.0).abs() < EPS);
        assert!((dists[1].1 - 1.0).abs() < EPS);
    }

    #[test]
    fn merge() {
        let mut t1: Track<DefaultAttrs, DefaultMetric, DefaultAttrUpdates> = Track::default();
        t1.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            DefaultAttrUpdates {},
        );

        let mut t2 = Track::default();
        t2.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            DefaultAttrUpdates {},
        );
        t1.merge(&t2, &vec![0]);
        assert_eq!(t1.observations.get(&0).unwrap().len(), 2);
    }

    #[test]
    fn attribute_compatible_match() {
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
            fn apply(&self, attrs: &mut TimeAttrs) {
                attrs.end_time = self.time;
                if attrs.start_time == 0 {
                    attrs.start_time = self.time;
                }
            }
        }

        impl AttributeMatch<TimeAttrs> for TimeAttrs {
            fn compatible(&self, other: &TimeAttrs) -> bool {
                self.end_time <= other.start_time
            }

            fn merge(&mut self, other: &TimeAttrs) {
                self.start_time = self.start_time.min(other.start_time);
                self.end_time = self.end_time.max(other.end_time);
            }

            fn baked(&self, _observations: &FeatureObservationsGroups) -> bool {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    - self.end_time
                    > 30
            }
        }

        #[derive(Default)]
        struct TimeMetric;
        impl Metric for TimeMetric {
            fn distance(_feature_id: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> f32 {
                standard_vector_distance(&e1.1, &e2.1)
            }

            fn optimize(
                &mut self,
                _feature_id: &u64,
                _merge_history: &[u64],
                features: &mut Vec<FeatureSpec>,
                _prev_length: usize,
            ) {
                features.sort_by(feat_sort_cmp);
                features.truncate(20);
            }
        }

        let mut t1: Track<TimeAttrs, TimeMetric, TimeAttrUpdates> = Track::default();
        t1.track_id = 1;
        t1.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]),
            TimeAttrUpdates { time: 2 },
        );

        let mut t2 = Track::default();
        t2.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            TimeAttrUpdates { time: 3 },
        );
        t2.track_id = 2;

        let dists = t1.distances(&t2, 0);
        let dists = dists.unwrap();
        assert_eq!(dists.len(), 1);
        assert!((dists[0].1 - 2.0).abs() < EPS);
        assert_eq!(dists[0].0, 2);

        let mut t3 = Track::default();
        t3.add(
            0,
            0.3,
            Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]),
            TimeAttrUpdates { time: 1 },
        );

        let dists = t1.distances(&t3, 0);
        assert!(dists.is_none());
    }
}
