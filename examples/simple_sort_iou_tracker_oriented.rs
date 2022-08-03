use similari::examples::BoxGen2;
use similari::trackers::sort::simple_iou::SORT;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::utils::bbox::GenericBBox;

fn main() {
    let mut tracker = SORT::new(1, 10, 1, DEFAULT_SORT_IOU_THRESHOLD);

    let pos_drift = 1.0;
    let box_drift = 0.1;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for i in 0..30 {
        let obj1b = GenericBBox::from(b1.next().unwrap()).rotate(0.35 + (i as f32 / 10.0));
        let obj2b = GenericBBox::from(b2.next().unwrap()).rotate(0.55 + (i as f32 / 10.0));
        let _tracks = tracker.predict(&[obj1b, obj2b]);
    }

    tracker.skip_epochs(2);

    let tracks = tracker.wasted();
    for t in tracks {
        eprintln!("Track id: {}", t.get_track_id());
        eprintln!("Boxes: {:#?}", t.get_attributes().predicted_boxes);
    }
}
