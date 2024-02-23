#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::trackers::batch::PredictionBatchRequest;
use similari::trackers::sort::batch_api::BatchSort;
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::trackers::sort::PositionalMetricType::IoU;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use similari::trackers::tracker_api::TrackerAPI;
use test::Bencher;

#[bench]
fn batch_sort_iou_00010(b: &mut Bencher) {
    bench_batch_sort(10, b);
}

#[bench]
fn batch_sort_iou_00100(b: &mut Bencher) {
    bench_batch_sort(100, b);
}

#[bench]
fn batch_sort_iou_00500(b: &mut Bencher) {
    bench_batch_sort(500, b);
}

fn bench_batch_sort(objects: usize, b: &mut Bencher) {
    let pos_drift = 1.0;
    let box_drift = 0.001;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(BoxGen2::new(
            1000.0 * i as f32,
            1000.0 * i as f32,
            50.0,
            50.0,
            pos_drift,
            box_drift,
        ))
    }

    let mut iteration = 0;
    let ncores = match objects {
        10 => 1,
        100 => 2,
        _ => num_cpus::get(),
    };

    let mut tracker = BatchSort::new(
        ncores,
        ncores,
        10,
        1,
        IoU(DEFAULT_SORT_IOU_THRESHOLD),
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        Some(SpatioTemporalConstraints::default().constraints(&[(1, 1.0)])),
        1.0 / 20.0,
        1.0 / 160.0,
    );

    let mut count = 0;
    b.iter(|| {
        count += 1;
        let (mut batch, res) = PredictionBatchRequest::new();
        for i in &mut iterators {
            iteration += 1;
            let b = i.next();
            batch.add(0, (b.unwrap().into(), Some(1)));
        }
        tracker.predict(batch);
        for _ in 0..res.batch_size() {
            let (_scene, _tracks) = res.get();
        }
    });

    eprintln!("Store stats: {:?}", tracker.active_shard_stats());
    assert_eq!(
        tracker.active_shard_stats().into_iter().sum::<usize>(),
        objects
    );

    let wasted = tracker.wasted();
    assert!(wasted.is_empty());

    tracker.skip_epochs(2);
    let wasted = tracker.wasted();
    assert_eq!(wasted.len(), objects);
    for w in wasted {
        assert_eq!(w.get_attributes().track_length, count);
    }
    eprintln!("Benchmark complete");
}
