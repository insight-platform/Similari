use anyhow::Result;
use similari::db;
use similari::distance::euclidean;
use similari::track::voting::{TopNVoting, Voting};
use similari::track::{
    feat_confidence_cmp, AttributeMatch, AttributeUpdate, Feature, FeatureObservationsGroups,
    FeatureSpec, Metric, Track, TrackBakingStatus,
};
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    #[error("Only single feature vector can be set")]
    SetError,
    #[error("Incompatible attributes")]
    Incompatible,
}

#[derive(Debug, Clone)]
pub struct SimpleAttrs {
    set: bool,
}

impl Default for SimpleAttrs {
    fn default() -> Self {
        Self { set: false }
    }
}

#[derive(Default)]
pub struct SimpleAttributeUpdate;

impl AttributeUpdate<SimpleAttrs> for SimpleAttributeUpdate {
    fn apply(&self, attrs: &mut SimpleAttrs) -> Result<()> {
        if attrs.set {
            return Err(AppError::SetError.into());
        }
        attrs.set = true;
        Ok(())
    }
}

impl AttributeMatch<SimpleAttrs> for SimpleAttrs {
    fn compatible(&self, other: &SimpleAttrs) -> bool {
        self.set && other.set
    }

    fn merge(&mut self, other: &SimpleAttrs) -> Result<()> {
        if self.compatible(&other) {
            Ok(())
        } else {
            Err(AppError::Incompatible.into())
        }
    }

    fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
        if self.set {
            Ok(TrackBakingStatus::Ready)
        } else {
            Ok(TrackBakingStatus::Pending)
        }
    }
}

#[derive(Default, Clone)]
struct SimpleMetric;

impl Metric for SimpleMetric {
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
        features.truncate(1);
        Ok(())
    }
}

fn vec2(x: f32, y: f32) -> Feature {
    Feature::from_vec(1, 2, vec![x, y])
}

fn main() {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = db::TrackStore::new(Some(SimpleMetric::default()), Some(SimpleAttrs::default()));
    let res = db.add(
        0,
        DEFAULT_FEATURE,
        1.0,
        vec2(1.0, 0.0),
        SimpleAttributeUpdate {},
    );
    assert!(res.is_ok());

    let res = db.add(
        0,
        DEFAULT_FEATURE,
        1.0,
        vec2(1.0, 0.0),
        SimpleAttributeUpdate {},
    );
    // attribute implementation prevents secondary observations to be added to the same track
    assert!(res.is_err());

    let baked = db.find_baked();
    assert_eq!(
        baked
            .into_iter()
            .filter(|(_b, r)| r.is_ok())
            .map(|(x, _)| x)
            .collect::<Vec<_>>(),
        vec![0u64]
    );

    let res = db.add(
        1,
        DEFAULT_FEATURE,
        0.9,
        vec2(0.9, 0.1),
        SimpleAttributeUpdate {},
    );
    assert!(res.is_ok());

    let mut ext_track = Track::new(
        2,
        Some(SimpleMetric::default()),
        Some(SimpleAttrs::default()),
    );

    let res = ext_track.add_observation(
        DEFAULT_FEATURE,
        0.8,
        vec2(0.66, 0.33),
        SimpleAttributeUpdate {},
    );

    assert!(res.is_ok());

    let (dists, errs) = db.foreign_track_distances(&ext_track, 0, true);

    assert_eq!(dists.len(), 2);
    assert_eq!(errs.len(), 0);

    eprintln!("Distances: {:?}", &dists);
    eprintln!("Errs: {:?}", &errs);

    let top1_voting_engine = TopNVoting::new(1, 1.0, 1);
    let results = top1_voting_engine.find_merge_candidates(dists);
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );

    let (dists, _errs) = db.foreign_track_distances(&ext_track, 0, true);

    // max distance filter set to 0.4
    let top1_voting_engine_filter = TopNVoting::new(2, 0.4, 1);
    let results = top1_voting_engine_filter.find_merge_candidates(dists);
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );
}
