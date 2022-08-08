use similari::examples::{vec2, SimpleAttributeUpdate, SimpleAttrs, SimpleMetric};
use similari::store;
use similari::track::notify::NoopNotifier;
use similari::track::Track;
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;

fn main() {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = store::TrackStore::new(
        SimpleMetric::default(),
        SimpleAttrs::default(),
        NoopNotifier,
        num_cpus::get(),
    );
    let res = db.add(
        0,
        DEFAULT_FEATURE,
        Some(1.0),
        Some(vec2(1.0, 0.0)),
        Some(SimpleAttributeUpdate {}),
    );
    assert!(res.is_ok());

    let res = db.add(
        0,
        DEFAULT_FEATURE,
        Some(1.0),
        Some(vec2(1.0, 0.0)),
        Some(SimpleAttributeUpdate {}),
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
        Some(0.9),
        Some(vec2(0.9, 0.1)),
        Some(SimpleAttributeUpdate {}),
    );
    assert!(res.is_ok());

    let mut ext_track = Track::new(
        2,
        SimpleMetric::default(),
        SimpleAttrs::default(),
        NoopNotifier,
    );

    let res = ext_track.add_observation(
        DEFAULT_FEATURE,
        Some(0.8),
        Some(vec2(0.66, 0.33)),
        Some(SimpleAttributeUpdate {}),
    );

    assert!(res.is_ok());

    let (dists, errs) = db.foreign_track_distances(vec![ext_track], 0, true);
    let dists = dists.all();
    let errs = errs.all();
    assert_eq!(errs.len(), 0);

    eprintln!("Distances: {:?}", &dists);
    eprintln!("Errs: {:?}", &errs);

    let top1_voting_engine: TopNVoting<f32> = TopNVoting::new(2, 1.0, 1);
    let results = top1_voting_engine.winners(dists.clone());
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );

    // max distance filter set to 0.4
    let top1_voting_engine_filter: TopNVoting<f32> = TopNVoting::new(2, 0.4, 1);
    let results = top1_voting_engine_filter.winners(dists);
    eprintln!(
        "Voting results (the less distance, the better result): {:?}",
        &results
    );
}
