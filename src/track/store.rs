use crate::track::{AttributeMatch, AttributeUpdate, Feature, Metric, Track};
use crate::Errors;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync + Clone,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
{
    attributes: A,
    metric: M,
    tracks: HashMap<u64, Track<A, M, U>>,
}

impl<A, U, M> Default for TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync + Clone,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync + Clone,
{
    fn default() -> Self {
        Self::new(None, None)
    }
}

impl<A, U, M> TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync + Clone,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync + Clone,
{
    pub fn new(metric: Option<M>, default_attributes: Option<A>) -> Self {
        Self {
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
            tracks: HashMap::default(),
        }
    }

    pub fn find_baked(&self) -> Vec<u64> {
        self.tracks
            .par_iter()
            .flat_map(|(track_id, track)| {
                if track.get_attributes().baked(&track.observations) {
                    Some(*track_id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn fetch_tracks(&mut self, tracks: &Vec<u64>) -> Vec<Track<A, M, U>> {
        let mut res = Vec::default();
        for track_id in tracks {
            if let Some(t) = self.tracks.remove(track_id) {
                res.push(t);
            }
        }
        res
    }

    pub fn foreign_track_distances(
        &self,
        track: &Track<A, M, U>,
        feature_id: u64,
    ) -> (
        Vec<(u64, Result<f32>)>,
        Vec<Result<Vec<(u64, Result<f32>)>>>,
    ) {
        let res: Vec<_> = self
            .tracks
            .par_iter()
            .map(|(_, other)| track.distances(other, feature_id))
            .collect();

        let mut distances = Vec::default();
        let mut errors = Vec::default();

        for r in res {
            match r {
                Ok(dists) => distances.extend(dists),
                e => errors.push(e),
            }
        }

        (distances, errors)
    }

    pub fn owned_track_distances(
        &self,
        track_id: u64,
        feature_id: u64,
    ) -> (
        Vec<(u64, Result<f32>)>,
        Vec<Result<Vec<(u64, Result<f32>)>>>,
    ) {
        let track = self.tracks.get(&track_id);
        if track.is_none() {
            return (vec![], vec![Err(Errors::MissingTrack.into())]);
        }
        let track = track.unwrap();
        let res: Vec<_> = self
            .tracks
            .par_iter()
            .filter(|(other_track_id, _)| **other_track_id != track_id)
            .map(|(_, other)| track.distances(other, feature_id))
            .collect();

        let mut distances = Vec::default();
        let mut errors = Vec::default();

        for r in res {
            match r {
                Ok(dists) => distances.extend(dists),
                e => errors.push(e),
            }
        }

        (distances, errors)
    }

    pub fn add(
        &mut self,
        track_id: u64,
        feature_id: u64,
        reid_q: f32,
        reid_v: Feature,
        attribute_update: U,
    ) {
        match self.tracks.get_mut(&track_id) {
            None => {
                let mut t = Track {
                    attributes: self.attributes.clone(),
                    track_id,
                    observations: HashMap::from([(feature_id, vec![(reid_q, reid_v)])]),
                    metric: self.metric.clone(),
                    phantom_attribute_update: PhantomData,
                    merge_history: vec![track_id],
                };
                t.update_attributes(attribute_update);
                self.tracks.insert(track_id, t);
            }
            Some(track) => {
                track.add(feature_id, reid_q, reid_v, attribute_update);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::track::store::TrackStore;
    use crate::track::{
        feat_sort_cmp, standard_vector_distance, AttributeMatch, AttributeUpdate, Feature,
        FeatureObservationsGroups, FeatureSpec, Metric,
    };
    use crate::{Errors, EPS};
    use anyhow::Result;
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[derive(Default, Debug, Clone)]
    pub struct TimeAttrs {
        start_time: u128,
        end_time: u128,
        baked_period: u128,
    }

    #[derive(Default)]
    pub struct TimeAttrUpdates {
        time: u128,
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

        fn merge(&mut self, other: &TimeAttrs) -> Result<()> {
            self.start_time = self.start_time.min(other.start_time);
            self.end_time = self.end_time.max(other.end_time);
            Ok(())
        }

        fn baked(&self, _observations: &FeatureObservationsGroups) -> bool {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
                - self.end_time
                > self.baked_period
        }
    }

    #[derive(Default, Clone)]
    struct TimeMetric {
        max_length: usize,
    }

    impl Metric for TimeMetric {
        fn distance(_feature_id: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32> {
            Ok(standard_vector_distance(&e1.1, &e2.1))
        }

        fn optimize(
            &mut self,
            _feature_id: &u64,
            _merge_history: &[u64],
            features: &mut Vec<FeatureSpec>,
            _prev_length: usize,
        ) {
            features.sort_by(feat_sort_cmp);
            features.truncate(self.max_length);
        }
    }

    fn vec2(x: f32, y: f32) -> Feature {
        Feature::from_vec(1, 2, vec![x, y])
    }

    #[test]
    fn test_storage() {
        let _default_store: TrackStore<TimeAttrs, TimeAttrUpdates, TimeMetric> =
            TrackStore::default();

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
        );
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
        );
        let baked = store.find_baked();
        assert!(baked.is_empty());
        thread::sleep(Duration::from_millis(30));
        let baked = store.find_baked();
        assert_eq!(baked.len(), 1);
        assert_eq!(baked[0], 0);

        let vectors = store.fetch_tracks(&baked);
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
        );
        let (dists, errs) = store.owned_track_distances(0, 0);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            0.7,
            vec2(1.0, 0.0),
            TimeAttrUpdates {
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
        );

        let (dists, errs) = store.owned_track_distances(0, 0);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0).abs() < EPS);
        assert!(errs.is_empty());

        let (dists, errs) = store.owned_track_distances(1, 0);
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
                    Errors::MissingObservation
                    | Errors::MissingTrack
                    | Errors::SelfDistanceCalculation => {
                        unreachable!();
                    }
                }
            }
        }

        let mut v = store.fetch_tracks(&vec![0]);

        let (dists, errs) = store.foreign_track_distances(&v[0], 0);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0).abs() < EPS);
        assert!(errs.is_empty());

        // make it incompatible across the attributes
        thread::sleep(Duration::from_millis(10));
        v[0].attributes.end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let (dists, errs) = store.foreign_track_distances(&v[0], 0);
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
                    Errors::MissingObservation
                    | Errors::MissingTrack
                    | Errors::SelfDistanceCalculation => {
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
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            },
        );

        v[0].attributes.end_time = store.tracks.get(&1).unwrap().attributes.start_time - 1;
        let (dists, errs) = store.foreign_track_distances(&v[0], 0);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0).abs() < EPS);
        assert!((dists[1].1.as_ref().unwrap() - 1.0).abs() < EPS);
        assert!(errs.is_empty());
    }
}
