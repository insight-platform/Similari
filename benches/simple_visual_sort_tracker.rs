#![feature(test)]

extern crate test;

use rand::distributions::Uniform;
use rand::Rng;
use similari::examples::BoxGen2;
use similari::prelude::{VisualSort, VisualSortObservation, VisualSortOptions};
use similari::trackers::sort::PositionalMetricType;
use similari::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use similari::trackers::tracker_api::TrackerAPI;
use similari::trackers::visual_sort::metric::VisualSortMetricType;
use test::Bencher;

#[bench]
fn visual_sort_iou_00010x3x0128(b: &mut Bencher) {
    bench_visual_sort(10, 128, b);
}

#[bench]
fn visual_sort_iou_00050x3x0128(b: &mut Bencher) {
    bench_visual_sort(50, 128, b);
}

#[bench]
fn visual_sort_iou_00100x3x0128(b: &mut Bencher) {
    bench_visual_sort(100, 128, b);
}

#[bench]
fn visual_sort_iou_00010x3x0256(b: &mut Bencher) {
    bench_visual_sort(10, 256, b);
}

#[bench]
fn visual_sort_iou_00050x3x0256(b: &mut Bencher) {
    bench_visual_sort(50, 256, b);
}

#[bench]
fn visual_sort_iou_00100x3x0256(b: &mut Bencher) {
    bench_visual_sort(100, 256, b);
}

#[bench]
fn visual_sort_iou_00010x3x0512(b: &mut Bencher) {
    bench_visual_sort(10, 512, b);
}

#[bench]
fn visual_sort_iou_00050x3x0512(b: &mut Bencher) {
    bench_visual_sort(50, 512, b);
}

#[bench]
fn visual_sort_iou_00100x3x0512(b: &mut Bencher) {
    bench_visual_sort(100, 512, b);
}

#[bench]
fn visual_sort_iou_00010x3x1024(b: &mut Bencher) {
    bench_visual_sort(10, 1024, b);
}

#[bench]
fn visual_sort_iou_00050x3x1024(b: &mut Bencher) {
    bench_visual_sort(50, 1024, b);
}

#[bench]
fn visual_sort_iou_00100x3x1024(b: &mut Bencher) {
    bench_visual_sort(100, 1024, b);
}

#[bench]
fn visual_sort_iou_00010x3x2048(b: &mut Bencher) {
    bench_visual_sort(10, 2048, b);
}

#[bench]
fn visual_sort_iou_00050x3x2048(b: &mut Bencher) {
    bench_visual_sort(50, 2048, b);
}

#[bench]
fn visual_sort_iou_00100x3x2048(b: &mut Bencher) {
    bench_visual_sort(100, 2048, b);
}

fn bench_visual_sort(objects: usize, len: usize, b: &mut Bencher) {
    let pos_drift = 1.0;
    let box_drift = 0.001;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(BoxGen2::new(
            1000.0 * i as f32,
            1000.0 * i as f32,
            20.0,
            50.0,
            pos_drift,
            box_drift,
        ))
    }

    let mut iteration = 0;
    //let ncores = num_cpus::get();

    let opts = VisualSortOptions::default()
        .positional_metric(PositionalMetricType::IoU(0.3))
        .visual_metric(VisualSortMetricType::Euclidean(10.0))
        .visual_max_observations(3)
        .spatio_temporal_constraints(SpatioTemporalConstraints::default().constraints(&[(1, 1.0)]))
        .visual_minimal_own_area_percentage_use(0.5)
        .visual_minimal_own_area_percentage_collect(0.6)
        .kalman_position_weight(1.0 / 20.0)
        .kalman_velocity_weight(1.0 / 160.0)
        .visual_min_votes(2);

    let ncores = match objects {
        10 => 1,
        50 => 2,
        _ => num_cpus::get(),
    };

    let mut tracker = VisualSort::new(ncores, &opts);

    let mut count = 0;
    b.iter(|| {
        count += 1;
        let mut observations = Vec::new();

        let mut features = Vec::new();
        let mut rng = rand::thread_rng();
        let gen = Uniform::new(-0.01, 0.01);

        for (index, _) in iterators.iter().enumerate() {
            let f = (0..len)
                .map(|_| rng.sample(&gen) + index as f32 * 10.0)
                .collect::<Vec<f32>>();
            features.push(f);
        }

        for (index, bi) in iterators.iter_mut().enumerate() {
            iteration += 1;
            let b = bi.next();
            let f = &features[index];
            observations.push(VisualSortObservation::new(
                Some(f),
                Some(1.0),
                b.unwrap().into(),
                Some(0),
            ));
        }
        let tracks = tracker.predict(&observations);
        assert_eq!(tracks.len(), objects);
    });
    eprintln!("Store stats: {:?}", tracker.active_shard_stats());
    assert_eq!(
        tracker.active_shard_stats().into_iter().sum::<usize>(),
        objects
    );

    let wasted = tracker.wasted();
    assert!(wasted.is_empty());

    tracker.skip_epochs(10);
    let wasted = tracker.wasted();
    assert_eq!(wasted.len(), objects);
    for w in wasted {
        assert_eq!(w.get_attributes().track_length, count);
    }
}
