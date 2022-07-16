#![feature(test)]

extern crate test;

use rand::{distributions::Uniform, Rng};
use similari::store;
use similari::test_stuff::{SimpleAttributeUpdate, SimpleAttrs, SimpleMetric};
use similari::track::notify::NoopNotifier;
use similari::track::utils::FromVec;
use similari::track::{Observation, Track};
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
fn simple_0512_0100k(b: &mut Bencher) {
    bench_capacity_len(512, 100000, b);
}

#[bench]
fn simple_0512_1000k(b: &mut Bencher) {
    bench_capacity_len(512, 1000000, b);
}

#[bench]
fn simple_0128_001k(b: &mut Bencher) {
    bench_capacity_len(128, 1000, b);
}

#[bench]
fn simple_0128_010k(b: &mut Bencher) {
    bench_capacity_len(128, 10000, b);
}

#[bench]
fn simple_0128_100k(b: &mut Bencher) {
    bench_capacity_len(128, 100000, b);
}

#[bench]
fn simple_1024_001k(b: &mut Bencher) {
    bench_capacity_len(1024, 1000, b);
}

#[bench]
fn simple_1024_010k(b: &mut Bencher) {
    bench_capacity_len(1024, 10000, b);
}

#[bench]
fn simple_1024_100k(b: &mut Bencher) {
    bench_capacity_len(1024, 100000, b);
}

#[bench]
fn simple_2048_001k(b: &mut Bencher) {
    bench_capacity_len(2048, 1000, b);
}

#[bench]
fn simple_2048_010k(b: &mut Bencher) {
    bench_capacity_len(2048, 10000, b);
}

#[bench]
fn simple_2048_100k(b: &mut Bencher) {
    bench_capacity_len(2048, 100000, b);
}

fn bench_capacity_len(vec_len: usize, count: usize, b: &mut Bencher) {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = store::TrackStore::new(
        Some(SimpleMetric::default()),
        Some(SimpleAttrs::default()),
        None,
        num_cpus::get(),
    );
    let mut rng = rand::thread_rng();
    let gen = Uniform::new(0.0, 1.0);

    for i in 0..count {
        let res = db.add(
            i as u64,
            DEFAULT_FEATURE,
            Some(1.0),
            Some(Observation::from_vec(
                (0..vec_len).map(|_| rng.sample(&gen)).collect(),
            )),
            Some(SimpleAttributeUpdate {}),
        );
        assert!(res.is_ok());
    }

    let mut t = Track::new(
        count as u64 + 1,
        Some(SimpleMetric::default()),
        Some(SimpleAttrs::default()),
        Some(NoopNotifier::default()),
    );

    let _ = t.add_observation(
        DEFAULT_FEATURE,
        Some(1.0),
        Some(Observation::from_vec(
            (0..vec_len).map(|_| rng.sample(&gen)).collect(),
        )),
        None,
    );

    b.iter(|| {
        db.foreign_track_distances(t.clone(), DEFAULT_FEATURE, true, None);
    });
}
