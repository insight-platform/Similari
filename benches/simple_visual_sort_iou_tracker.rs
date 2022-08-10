#![feature(test)]

extern crate test;

use rand::distributions::Uniform;
use rand::Rng;
use similari::examples::BoxGen2;
use similari::prelude::{VisualObservation, VisualSort, VisualSortOptions};
use similari::trackers::visual::metric::{PositionalMetricType, VisualMetricType};
use test::Bencher;

#[bench]
fn visual_sort_iou_00010x3x128(b: &mut Bencher) {
    bench_visual_sort(10, b);
}

#[bench]
fn visual_sort_iou_00100x3x128(b: &mut Bencher) {
    bench_visual_sort(100, b);
}

#[bench]
fn visual_sort_iou_00500x3x128(b: &mut Bencher) {
    bench_visual_sort(500, b);
}

fn bench_visual_sort(objects: usize, b: &mut Bencher) {
    let pos_drift = 1.0;
    let box_drift = 0.01;
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
    let ncores = num_cpus::get();

    let opts = VisualSortOptions::default()
        .positional_metric(PositionalMetricType::IoU(0.3))
        .visual_metric(VisualMetricType::Euclidean(10.0))
        .visual_max_observations(3)
        .visual_max_distance(30.0)
        .visual_min_votes(2);

    let mut tracker = VisualSort::new(ncores, &opts);

    let mut count = 0;
    b.iter(|| {
        count += 1;
        let mut observations = Vec::new();

        let mut features = Vec::new();
        let mut rng = rand::thread_rng();
        let gen = Uniform::new(-0.01, 0.01);

        for (index, _) in iterators.iter().enumerate() {
            let f = (0..128)
                .map(|_| rng.sample(&gen) + index as f32 * 10.0)
                .collect::<Vec<f32>>();
            features.push(f);
        }

        for (index, bi) in iterators.iter_mut().enumerate() {
            iteration += 1;
            let b = bi.next();
            let f = &features[index];
            observations.push(VisualObservation::new(
                Some(f),
                Some(1.0),
                b.unwrap().into(),
                Some(0),
            ));
        }
        let tracks = tracker.predict(&observations);
        assert_eq!(tracks.len(), objects);
    });
    eprintln!("Store stats: {:?}", tracker.shard_stats());
    assert_eq!(tracker.shard_stats().into_iter().sum::<usize>(), objects);

    let wasted = tracker.wasted();
    assert!(wasted.is_empty());

    tracker.skip_epochs(10);
    let wasted = tracker.wasted();
    assert_eq!(wasted.len(), objects);
    for w in wasted {
        assert_eq!(w.get_attributes().track_length, count);
    }
}
