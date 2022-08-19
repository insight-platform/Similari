use anyhow::Result;
use similari::examples::BoxGen2;
use similari::trackers::batch::PredictionBatchRequest;
use similari::trackers::sort::batch_api::BatchSort;
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::trackers::sort::PositionalMetricType::IoU;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::trackers::tracker_api::TrackerAPI;
use similari::utils::bbox::BoundingBox;

fn main() {
    let mut tracker = BatchSort::new(
        1,
        1,
        10,
        1,
        IoU(DEFAULT_SORT_IOU_THRESHOLD),
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        None,
    );

    let pos_drift = 1.0;
    let box_drift = 0.01;
    let mut b1 = BoxGen2::new_monotonous(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);
    let mut b2 = BoxGen2::new_monotonous(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = b1.next().unwrap();
        let obj2b = b2.next().unwrap();
        let (mut batch, res) = PredictionBatchRequest::new();
        batch.add(0, (obj1b.into(), None));
        batch.add(0, (obj2b.into(), None));
        tracker.predict(batch);
        for _ in 0..res.batch_size() {
            let predictions = res.get();
            //eprintln!("Scene Tracks: {:?}", &predictions);
            drop(predictions);
        }
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