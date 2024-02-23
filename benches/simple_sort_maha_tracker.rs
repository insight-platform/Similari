#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::prelude::Sort;
use similari::trackers::sort::metric::DEFAULT_MINIMAL_SORT_CONFIDENCE;
use similari::trackers::sort::PositionalMetricType::Mahalanobis;
use similari::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use similari::trackers::tracker_api::TrackerAPI;
use test::Bencher;

#[bench]
fn sort_maha_00010(b: &mut Bencher) {
    bench_sort(10, b);
}

#[bench]
fn sort_maha_00100(b: &mut Bencher) {
    bench_sort(100, b);
}

#[bench]
fn sort_maha_00500(b: &mut Bencher) {
    bench_sort(500, b);
}

fn bench_sort(objects: usize, b: &mut Bencher) {
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

    let mut tracker = Sort::new(
        ncores,
        10,
        1,
        Mahalanobis,
        DEFAULT_MINIMAL_SORT_CONFIDENCE,
        Some(SpatioTemporalConstraints::default().constraints(&[(1, 1.0)])),
        1.0 / 20.0,
        1.0 / 160.0,
    );

    let mut count = 0;
    b.iter(|| {
        count += 1;
        let mut observations = Vec::new();
        for i in &mut iterators {
            iteration += 1;
            let b = i.next();
            observations.push((b.unwrap().into(), None));
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

    tracker.skip_epochs(2);
    let wasted = tracker.wasted();
    assert_eq!(wasted.len(), objects);
    for w in wasted {
        assert_eq!(w.get_attributes().track_length, count);
    }
}
