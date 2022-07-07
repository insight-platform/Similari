#![feature(test)]

extern crate test;

use rand::{distributions::Uniform, Rng};
use similari::store;
use similari::test_stuff::{UnboundAttributeUpdate, UnboundAttrs, UnboundMetric};
use similari::track::{Feature, Track};
use std::sync::Arc;

use similari::track::notify::NoopNotifier;
use test::Bencher;

fn bench_capacity_len(vec_len: usize, track_len: usize, count: usize, b: &mut Bencher) {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = store::TrackStore::new(
        Some(UnboundMetric::default()),
        Some(UnboundAttrs::default()),
        None,
        num_cpus::get(),
    );
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(0.0, 1.0);

    for i in 0..count {
        for _j in 0..track_len {
            let res = db.add(
                i as u64,
                DEFAULT_FEATURE,
                1.0,
                Feature::from_vec(1, vec_len, (0..vec_len).map(|_| rng.sample(&gen)).collect()),
                UnboundAttributeUpdate {},
            );
            assert!(res.is_ok());
        }
    }
    b.iter(|| {
        let mut t = Track::new(
            count as u64 + 1,
            Some(UnboundMetric::default()),
            Some(UnboundAttrs::default()),
            Some(NoopNotifier::default()),
        );
        for _j in 0..track_len {
            let _ = t.add_observation(
                DEFAULT_FEATURE,
                1.0,
                Feature::from_vec(1, vec_len, (0..vec_len).map(|_| rng.sample(&gen)).collect()),
                UnboundAttributeUpdate {},
            );
        }

        db.foreign_track_distances(Arc::new(t), DEFAULT_FEATURE, true);
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
