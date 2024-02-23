use similari::examples::BoxGen2;
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::trackers::sort::simple_api::Sort;
use similari::trackers::sort::PositionalMetricType::IoU;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::trackers::tracker_api::TrackerAPI;
use similari::utils::bbox::Universal2DBox;

fn main() {
    let mut tracker = Sort::new(
        1,
        10,
        1,
        IoU(DEFAULT_SORT_IOU_THRESHOLD),
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        None,
        1.0 / 20.0,
        1.0 / 160.0,
    );

    let pos_drift = 1.0;
    let box_drift = 0.1;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for i in 0..30 {
        let obj1b = Universal2DBox::from(b1.next().unwrap()).rotate(0.35 + (i as f32 / 10.0));
        let obj2b = Universal2DBox::from(b2.next().unwrap()).rotate(0.55 + (i as f32 / 10.0));
        let _tracks = tracker.predict(&[(obj1b, None), (obj2b, None)]);
    }

    tracker.skip_epochs(2);

    let tracks = tracker.wasted();
    for t in tracks {
        eprintln!("Track id: {}", t.get_track_id());
        eprintln!("Boxes: {:#?}", t.get_attributes().predicted_boxes);
    }
}
