use similari::store;
use similari::test_stuff::{vec2, SimpleAttributeUpdate, SimpleAttrs, SimpleMetric};
use similari::track::notify::NoopNotifier;
use similari::track::Track;
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;
use std::sync::Arc;

fn main() {
    if cfg!(target_feature = "avx2") {
        eprintln!("AVX2 is on");
    }

    const DEFAULT_FEATURE: u64 = 0;
    let mut db = store::TrackStore::new(
        Some(SimpleMetric::default()),
        Some(SimpleAttrs::default()),
        None,
        num_cpus::get(),
    );
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

    let baked = db.find_usable();
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
        Some(NoopNotifier::default()),
    );

    let res = ext_track.add_observation(
        DEFAULT_FEATURE,
        0.8,
        vec2(0.66, 0.33),
        SimpleAttributeUpdate {},
    );

    assert!(res.is_ok());

    let (dists, errs) = db.foreign_track_distances(Arc::new(ext_track), 0, true, None);
    assert_eq!(errs.len(), 0);

    eprintln!("Distances: {:?}", &dists);
    eprintln!("Errs: {:?}", &errs);

    let top1_voting_engine = TopNVoting::new(1, 1.0, 1);
    let results = top1_voting_engine.winners(&dists);
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );

    // max distance filter set to 0.4
    let top1_voting_engine_filter = TopNVoting::new(2, 0.4, 1);
    let results = top1_voting_engine_filter.winners(&dists);
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );
}
