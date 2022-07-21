use similari::examples::iou::{BBoxAttributes, BBoxAttributesUpdate, IOUMetric};
use similari::examples::{current_time_ms, BoxGen2};
use similari::store::TrackStore;
use similari::track::Track;
use similari::utils::bbox::{BBox, IOUTopNVoting};
use similari::voting::Voting;
use std::thread;
use std::time::Duration;

const FEAT0: u64 = 0;

fn main() {
    let mut store: TrackStore<BBoxAttributes, IOUMetric, BBox> = TrackStore::default();

    let voting = IOUTopNVoting {
        topn: 1,
        min_distance: 0.5,
        min_votes: 1,
    };

    let pos_drift = 1.0;
    let box_drift = 1.0;
    let mut b1 = BoxGen2::new(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);

    let mut b2 = BoxGen2::new(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = b1.next();
        let obj2b = b2.next();

        let mut obj1t: Track<BBoxAttributes, IOUMetric, BBox> =
            Track::new(u64::try_from(current_time_ms()).unwrap(), None, None, None);

        obj1t
            .add_observation(FEAT0, obj1b, None, Some(BBoxAttributesUpdate))
            .unwrap();

        let mut obj2t: Track<BBoxAttributes, IOUMetric, BBox> = Track::new(
            u64::try_from(current_time_ms()).unwrap() + 1,
            None,
            None,
            None,
        );

        obj2t
            .add_observation(FEAT0, obj2b, None, Some(BBoxAttributesUpdate))
            .unwrap();

        thread::sleep(Duration::from_millis(2));

        for t in [obj1t, obj2t] {
            let search_track = t.clone();
            let (dists, errs) = store.foreign_track_distances(vec![search_track], FEAT0, false);
            assert!(errs.all().is_empty());
            let mut winners = voting.winners(dists.all());
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                let winner = winners.get_mut(&t.get_track_id()).unwrap().pop().unwrap();

                store
                    .merge_external(winner.winner_track, &t, None, false)
                    .unwrap();
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
