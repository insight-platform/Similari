#![feature(test)]

extern crate test;

use rand::{distributions::Uniform, Rng};
use similari::examples::{UnboundAttributeUpdate, UnboundAttrs, UnboundMetric};
use similari::prelude::{ObservationBuilder, TrackStoreBuilder};
use similari::track::notify::NoopNotifier;
use similari::track::utils::FromVec;
use similari::track::Feature;
use test::Bencher;

#[bench]
fn simple_0512_0001k(b: &mut Bencher) {
    bench_capacity_len(512, 1000, b);
}

#[bench]
fn simple_0512_0010k(b: &mut Bencher) {
    bench_capacity_len(512, 10000, b);
}

#[bench]
fn simple_0128_001k(b: &mut Bencher) {
    bench_capacity_len(128, 1000, b);
}

#[bench]
fn simple_0128_010k(b: &mut Bencher) {
    bench_capacity_len(128, 10000, b);
}

fn bench_capacity_len(vec_len: usize, count: usize, b: &mut Bencher) {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = TrackStoreBuilder::new(num_cpus::get())
        .metric(UnboundMetric::default())
        .default_attributes(UnboundAttrs::default())
        .notifier(NoopNotifier)
        .build();
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(0.0, 1.0);

    for i in 0..count {
        let res = db.add(
            i as u64,
            DEFAULT_FEATURE,
            Some(1.0),
            Some(Feature::from_vec(
                (0..vec_len).map(|_| rng.sample(&gen)).collect::<Vec<_>>(),
            )),
            Some(UnboundAttributeUpdate {}),
        );
        assert!(res.is_ok());
    }

    b.iter(|| {
        let t = db
            .new_track(count as u64 + 1)
            .observation(
                ObservationBuilder::new(DEFAULT_FEATURE)
                    .observation_attributes(1.0)
                    .observation(Feature::from_vec(
                        (0..vec_len).map(|_| rng.sample(&gen)).collect::<Vec<_>>(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let (dists, errs) = db.foreign_track_distances(vec![t], DEFAULT_FEATURE, true);
        assert_eq!(dists.all().len(), count);
        assert!(errs.all().is_empty());
    });
}
