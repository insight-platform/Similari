use similari::examples::BoxGen2;
use similari::prelude::Sort;
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::trackers::sort::PositionalMetricType::Mahalanobis;
use similari::trackers::tracker_api::TrackerAPI;
use similari::utils::bbox::BoundingBox;

fn main() {
    let mut tracker = Sort::new(
        1,
        10,
        1,
        Mahalanobis,
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        None,
        1.0 / 20.0,
        1.0 / 160.0,
    );

    let pos_drift = 1.0;
    let box_drift = 0.2;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = b1.next().unwrap();
        let obj2b = b2.next().unwrap();
        let _tracks = tracker.predict(&[(obj1b.into(), None), (obj2b.into(), None)]);
        //eprintln!("Tracked objects: {:#?}", _tracks);
    }

    tracker.skip_epochs(2);

    let tracks = tracker.wasted();
    for t in tracks {
        eprintln!("Track id: {}", t.get_track_id());
        eprintln!(
            "Boxes: {:#?}",
            t.get_attributes()
                .predicted_boxes
                .iter()
                .map(|x| BoundingBox::try_from(x))
                .collect::<Vec<_>>()
        );
    }
}
