#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::utils::bbox::Universal2DBox;
use similari::utils::nms::nms;
use test::Bencher;

#[bench]
fn nms_00010(b: &mut Bencher) {
    bench_nms(10, b, nms);
}

#[bench]
fn nms_00100(b: &mut Bencher) {
    bench_nms(100, b, nms);
}

#[bench]
fn nms_00300(b: &mut Bencher) {
    bench_nms(300, b, nms);
}

#[bench]
fn nms_00500(b: &mut Bencher) {
    bench_nms(500, b, nms);
}

#[bench]
fn nms_01000(b: &mut Bencher) {
    bench_nms(1000, b, nms);
}

fn bench_nms(
    objects: usize,
    b: &mut Bencher,
    f: fn(&[(Universal2DBox, Option<f32>)], f32, Option<f32>) -> Vec<&Universal2DBox>,
) {
    let pos_drift = 10.0;
    let box_drift = 1.0;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(BoxGen2::new(
            i as f32, i as f32, 50.0, 50.0, pos_drift, box_drift,
        ));
    }

    b.iter(|| {
        let mut observations = Vec::new();
        for i in &mut iterators {
            let b = i.next();
            observations.push((b.unwrap().into(), None));
        }
        f(&observations, 0.8, None);
    });
}
