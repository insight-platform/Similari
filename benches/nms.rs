#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::utils::bbox::GenericBBox;
use similari::utils::nms::{nms, parallel_nms};
use test::Bencher;

#[bench]
fn bench_nms_00010(b: &mut Bencher) {
    bench_sort(10, b, nms);
}

#[bench]
fn bench_nms_00100(b: &mut Bencher) {
    bench_sort(100, b, nms);
}

#[bench]
fn bench_nms_00200(b: &mut Bencher) {
    bench_sort(200, b, nms);
}

#[bench]
fn bench_nms_00300(b: &mut Bencher) {
    bench_sort(300, b, nms);
}

#[bench]
fn bench_nms_00400(b: &mut Bencher) {
    bench_sort(400, b, nms);
}

#[bench]
fn bench_nms_00500(b: &mut Bencher) {
    bench_sort(500, b, nms);
}

#[bench]
fn bench_nms_01000(b: &mut Bencher) {
    bench_sort(1000, b, nms);
}

#[bench]
fn bench_parallel_nms_00010(b: &mut Bencher) {
    bench_sort(10, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_00100(b: &mut Bencher) {
    bench_sort(100, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_00200(b: &mut Bencher) {
    bench_sort(200, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_00300(b: &mut Bencher) {
    bench_sort(300, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_00400(b: &mut Bencher) {
    bench_sort(400, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_00500(b: &mut Bencher) {
    bench_sort(500, b, parallel_nms);
}

#[bench]
fn bench_parallel_nms_01000(b: &mut Bencher) {
    bench_sort(1000, b, parallel_nms);
}

fn bench_sort(
    objects: usize,
    b: &mut Bencher,
    f: fn(&[(GenericBBox, Option<f32>)], f32, Option<f32>) -> Vec<&GenericBBox>,
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
