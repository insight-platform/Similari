use anyhow::Result;
use similari::examples::BoxGen2;
use similari::trackers::sort::simple_iou::IoUSort;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::utils::bbox::BoundingBox;

fn main() {
    let mut tracker = IoUSort::new(1, 10, 1, DEFAULT_SORT_IOU_THRESHOLD, None);

    let pos_drift = 1.0;
    let box_drift = 0.2;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = b1.next().unwrap();
        let obj2b = b2.next().unwrap();
        let _tracks = tracker.predict(&[obj1b.into(), obj2b.into()]);
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
                .clone()
                .into_iter()
                .map(|x| {
                    let r: Result<BoundingBox> = x.into();
                    r.unwrap()
                })
                .collect::<Vec<_>>()
        );
    }
}
