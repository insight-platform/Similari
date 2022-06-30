#![feature(test)]

extern crate test;
use rand::{distributions::Uniform, Rng};
use similari::store;
use similari::test_stuff::{UnboundAttributeUpdate, UnboundAttrs, UnboundMetric};
use similari::track::{Feature, Track};

mod t_30 {
    use crate::bench_capacity_len;
    use test::Bencher;

    #[bench]
    fn track_0512_30_001k(b: &mut Bencher) {
        bench_capacity_len(512, 30, 1000, b);
    }

    #[bench]
    fn track_0512_30_010k(b: &mut Bencher) {
        bench_capacity_len(512, 30, 10000, b);
    }

    #[bench]
    fn track_0512_30_100k(b: &mut Bencher) {
        bench_capacity_len(512, 30, 100000, b);
    }

    #[bench]
    fn track_1024_30_001k(b: &mut Bencher) {
        bench_capacity_len(1024, 30, 1000, b);
    }

    #[bench]
    fn track_1024_30_010k(b: &mut Bencher) {
        bench_capacity_len(1024, 30, 10000, b);
    }
    #[bench]
    fn track_1024_30_100k(b: &mut Bencher) {
        bench_capacity_len(1024, 30, 100000, b);
    }
}

mod t_60 {
    use crate::bench_capacity_len;
    use test::Bencher;

    #[bench]
    fn track_0512_60_001k(b: &mut Bencher) {
        bench_capacity_len(512, 60, 1000, b);
    }

    #[bench]
    fn track_0512_60_010k(b: &mut Bencher) {
        bench_capacity_len(512, 60, 10000, b);
    }

    #[bench]
    fn track_0512_60_100k(b: &mut Bencher) {
        bench_capacity_len(512, 60, 100000, b);
    }

    #[bench]
    fn track_1024_60_001k(b: &mut Bencher) {
        bench_capacity_len(1024, 60, 1000, b);
    }

    #[bench]
    fn track_1024_60_010k(b: &mut Bencher) {
        bench_capacity_len(1024, 60, 10000, b);
    }
    #[bench]
    fn track_1024_60_100k(b: &mut Bencher) {
        bench_capacity_len(1024, 60, 100000, b);
    }
}

mod t_120 {
    use crate::bench_capacity_len;
    use test::Bencher;

    #[bench]
    fn track_0512_120_001k(b: &mut Bencher) {
        bench_capacity_len(512, 120, 1000, b);
    }

    #[bench]
    fn track_0512_120_010k(b: &mut Bencher) {
        bench_capacity_len(512, 120, 10000, b);
    }

    #[bench]
    fn track_0512_120_100k(b: &mut Bencher) {
        bench_capacity_len(512, 120, 100000, b);
    }

    #[bench]
    fn track_1024_120_001k(b: &mut Bencher) {
        bench_capacity_len(1024, 120, 1000, b);
    }

    #[bench]
    fn track_1024_120_010k(b: &mut Bencher) {
        bench_capacity_len(1024, 120, 10000, b);
    }
    #[bench]
    fn track_1024_120_100k(b: &mut Bencher) {
        bench_capacity_len(1024, 120, 100000, b);
    }
}

use test::Bencher;

fn bench_capacity_len(vec_len: usize, track_len: usize, count: usize, b: &mut Bencher) {
    const DEFAULT_FEATURE: u64 = 0;
    let mut db = store::TrackStore::new(
        Some(UnboundMetric::default()),
        Some(UnboundAttrs::default()),
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
        );
        for _j in 0..track_len {
            let _ = t.add_observation(
                DEFAULT_FEATURE,
                1.0,
                Feature::from_vec(1, vec_len, (0..vec_len).map(|_| rng.sample(&gen)).collect()),
                UnboundAttributeUpdate {},
            );
        }

        db.foreign_track_distances(&t, DEFAULT_FEATURE, true);
    });
}
