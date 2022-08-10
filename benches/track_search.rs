#![feature(test)]

extern crate test;

use rand::{distributions::Uniform, Rng};
use similari::examples::{UnboundAttributeUpdate, UnboundAttrs, UnboundMetric};
use similari::track::Feature;

use similari::prelude::TrackStoreBuilder;
use similari::track::notify::NoopNotifier;
use similari::track::utils::FromVec;
use test::Bencher;

fn bench_capacity_len(vec_len: usize, track_len: usize, count: usize, b: &mut Bencher) {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = TrackStoreBuilder::new(num_cpus::get())
        .metric(UnboundMetric::default())
        .default_attributes(UnboundAttrs::default())
        .notifier(NoopNotifier)
        .build();
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(0.0, 1.0);

    for i in 0..count {
        for _j in 0..track_len {
            let res = db.add(
                i as u64,
                DEFAULT_FEATURE,
                Some(1.0),
                Some(Feature::from_vec(
                    (0..vec_len).map(|_| rng.sample(&gen)).collect::<Vec<_>>(),
                )),
                None,
            );
            assert!(res.is_ok());
        }
    }

    let mut t = db.new_track(count as u64 + 1).build().unwrap();
    for _j in 0..track_len {
        let _ = t.add_observation(
            DEFAULT_FEATURE,
            Some(1.0),
            Some(Feature::from_vec(
                (0..vec_len).map(|_| rng.sample(&gen)).collect::<Vec<_>>(),
            )),
            Some(UnboundAttributeUpdate),
        );
    }

    b.iter(move || {
        let (dists, errs) = db.foreign_track_distances(vec![t.clone()], DEFAULT_FEATURE, true);
        dists.all();
        errs.all();
    });
}

#[bench]
fn track_0256_030_100(b: &mut Bencher) {
    bench_capacity_len(256, 30, 100, b);
}

#[bench]
fn track_0512_030_100(b: &mut Bencher) {
    bench_capacity_len(512, 30, 100, b);
}

#[bench]
fn track_1024_030_100(b: &mut Bencher) {
    bench_capacity_len(1024, 30, 100, b);
}

#[bench]
fn track_0256_030_01k(b: &mut Bencher) {
    bench_capacity_len(256, 30, 1000, b);
}
