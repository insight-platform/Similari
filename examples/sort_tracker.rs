use similari::examples::{current_time_ms, BoxGen2};
use similari::prelude::{NoopNotifier, ObservationBuilder, TrackBuilder};
use similari::store::TrackStore;
use similari::trackers::sort::{
    SortAttributes, SortMetric, SortVoting, DEFAULT_SORT_IOU_THRESHOLD,
};
use similari::utils::bbox::BBox;
use similari::voting::Voting;
use std::thread;
use std::time::Duration;

const FEAT0: u64 = 0;

fn main() {
    let mut store: TrackStore<SortAttributes, SortMetric, BBox> = TrackStore::default();

    let pos_drift = 1.0;
    let box_drift = 0.2;
    let mut b1 = BoxGen2::new_directional_movement(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);

    let mut b2 = BoxGen2::new_directional_movement(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..100 {
        let obj1b = b1.next().unwrap();
        dbg!(&obj1b);
        let obj2b = b2.next().unwrap();

        let obj1t = TrackBuilder::new(u64::try_from(current_time_ms()).unwrap())
            .notifier(NoopNotifier)
            .track_attrs(SortAttributes::default())
            .metric(SortMetric::default())
            .observation(
                ObservationBuilder::new(FEAT0)
                    .observation_attributes(obj1b)
                    .build(),
            )
            .build()
            .unwrap();

        let obj2t = TrackBuilder::new(u64::try_from(current_time_ms()).unwrap() + 1)
            .notifier(NoopNotifier)
            .track_attrs(SortAttributes::default())
            .metric(SortMetric::default())
            .observation(
                ObservationBuilder::new(FEAT0)
                    .observation_attributes(obj2b)
                    .build(),
            )
            .build()
            .unwrap();

        thread::sleep(Duration::from_millis(2));

        for t in [obj1t, obj2t] {
            let search_track = t.clone();
            let (dists, errs) = store.foreign_track_distances(vec![search_track], FEAT0, false);
            assert!(errs.all().is_empty());
            let voting = SortVoting::new(
                DEFAULT_SORT_IOU_THRESHOLD,
                1,
                store.shard_stats().iter().sum(),
            );
            let dists = dists.all();
            let mut winners = voting.winners(dists);
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                let winner = winners.get_mut(&t.get_track_id()).unwrap().pop().unwrap();
                if winner == t.get_track_id() {
                    store.add_track(t).unwrap();
                } else {
                    store.merge_external(winner, &t, None, false).unwrap();
                }
            }
        }
    }

    let tracks = store.find_usable();
    for (t, _) in tracks {
        let t = store.fetch_tracks(&vec![t]);
        eprintln!("Track id: {}", t[0].get_track_id());
        eprintln!("Boxes: {:#?}", t[0].get_attributes().bboxes);
    }
}
