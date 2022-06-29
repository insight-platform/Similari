use crate::track::{
    AttributeMatch, AttributeUpdate, Feature, Metric, Track, TrackBakingStatus, TrackDistance,
};
use crate::Errors;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;

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

/// The basic implementation should fit to most of needs.
///
impl<A, U, M> TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync + Clone,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync + Clone,
{
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

    /// Method is used to find ready to use tracks within the store.
    ///
    /// The search is parallelized with Rayon.
    ///
    pub fn find_baked(&self) -> Vec<(u64, Result<TrackBakingStatus>)> {
        self.tracks
            .par_iter()
            .flat_map(
                |(track_id, track)| match track.get_attributes().baked(&track.observations) {
                    Ok(status) => match status {
                        TrackBakingStatus::Pending => None,
                        other => Some((*track_id, Ok(other))),
                    },
                    Err(e) => Some((*track_id, Err(e))),
                },
            )
            .collect()
    }

    /// Access track in read-only mode
    ///
    pub fn get(&self, track_id: u64) -> Option<&Track<A, M, U>> {
        self.tracks.get(&track_id)
    }

    /// Access track in read-write mode
    ///
    pub fn get_mut(&mut self, track_id: u64) -> Option<&mut Track<A, M, U>> {
        self.tracks.get_mut(&track_id)
    }

    /// Pulls (and removes) requested tracks from the store.
    ///
    pub fn fetch_tracks(&mut self, tracks: &Vec<u64>) -> Vec<Track<A, M, U>> {
        let mut res = Vec::default();
        for track_id in tracks {
            if let Some(t) = self.tracks.remove(track_id) {
                res.push(t);
            }
        }
        res
    }

    /// Calculates distances for external track (not in track store) to all tracks in DB which are
    /// allowed.
    ///
    pub fn foreign_track_distances(
        &self,
        track: &Track<A, M, U>,
        feature_class: u64,
        only_baked: bool,
    ) -> (Vec<TrackDistance>, Vec<TrackDistanceError>) {
        let res: Vec<_> = self
            .tracks
            .par_iter()
            .flat_map(|(_, other)| {
                if !only_baked {
                    Some(track.distances(other, feature_class))
                } else {
                    match other.get_attributes().baked(&other.observations) {
                        Ok(TrackBakingStatus::Ready) => Some(track.distances(other, feature_class)),
                        _ => None,
                    }
                }
            })
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

    /// Calculates track distances for a track inside store
    ///
    pub fn owned_track_distances(
        &self,
        track_id: u64,
        feature_class: u64,
        only_baked: bool,
    ) -> (Vec<TrackDistance>, Vec<TrackDistanceError>) {
        let track = self.tracks.get(&track_id);
        if track.is_none() {
            return (vec![], vec![Err(Errors::TrackNotFound(track_id).into())]);
        }
        let track = track.unwrap();
        let res: Vec<_> = self
            .tracks
            .par_iter()
            .filter(|(other_track_id, _)| **other_track_id != track_id)
            .flat_map(|(_, other)| {
                if !only_baked {
                    Some(track.distances(other, feature_class))
                } else {
                    match other.get_attributes().baked(&other.observations) {
                        Ok(TrackBakingStatus::Ready) => Some(track.distances(other, feature_class)),
                        _ => None,
                    }
                }
            })
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

    /// Injects new feature observation for feature class into track
    ///
    pub fn add(
        &mut self,
        track_id: u64,
        feature_class: u64,
        reid_q: f32,
        reid_v: Feature,
        attribute_update: U,
    ) -> Result<()> {
        match self.tracks.get_mut(&track_id) {
            None => {
                let mut t = Track {
                    attributes: self.attributes.clone(),
                    track_id,
                    observations: HashMap::from([(feature_class, vec![(reid_q, reid_v)])]),
                    metric: self.metric.clone(),
                    phantom_attribute_update: PhantomData,
                    merge_history: vec![track_id],
                };
                t.update_attributes(attribute_update)?;
                self.tracks.insert(track_id, t);
            }
            Some(track) => {
                track.add_observation(feature_class, reid_q, reid_v, attribute_update)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::track::store::TrackStore;
    use crate::track::{
        feat_confidence_cmp, AttributeMatch, AttributeUpdate, Feature, FeatureObservationsGroups,
        FeatureSpec, Metric, TrackBakingStatus,
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
        Feature::from_vec(1, 2, vec![x, y])
    }

    #[test]
    fn test_store() -> Result<()> {
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
                time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
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
                    Errors::TrackNotFound(_t) => {
                        unreachable!();
                    }
                    Errors::SelfDistanceCalculation => {
                        unreachable!();
                    }
                }
            }
        }

        let mut v = store.fetch_tracks(&vec![0]);

        let (dists, errs) = store.foreign_track_distances(&v[0], 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        // make it incompatible across the attributes
        thread::sleep(Duration::from_millis(10));
        v[0].attributes.end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let (dists, errs) = store.foreign_track_distances(&v[0], 0, false);
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
                    Errors::TrackNotFound(_t) => {
                        unreachable!();
                    }
                    Errors::SelfDistanceCalculation => {
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
        )?;

        v[0].attributes.end_time = store.tracks.get(&1).unwrap().attributes.start_time - 1;
        let (dists, errs) = store.foreign_track_distances(&v[0], 0, false);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].0, 1);
        assert!(dists[0].1.is_ok());
        assert!((dists[0].1.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((dists[1].1.as_ref().unwrap() - 1.0).abs() < EPS);
        assert!(errs.is_empty());

        Ok(())
    }
}
