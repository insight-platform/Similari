#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::utils::bbox::GenericBBox;
use similari::utils::nms::nms;
use test::Bencher;

#[bench]
fn bench_nms_oriented_00010(b: &mut Bencher) {
    bench_sort(10, b);
}

#[bench]
fn bench_nms_oriented_00100(b: &mut Bencher) {
    bench_sort(100, b);
}

#[bench]
fn bench_nms_oriented_00200(b: &mut Bencher) {
    bench_sort(200, b);
}

#[bench]
fn bench_nms_oriented_00300(b: &mut Bencher) {
    bench_sort(300, b);
}

#[bench]
fn bench_nms_oriented_00400(b: &mut Bencher) {
    bench_sort(400, b);
}

#[bench]
fn bench_nms_oriented_00500(b: &mut Bencher) {
    bench_sort(500, b);
}

fn bench_sort(objects: usize, b: &mut Bencher) {
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
        for (indx, i) in iterators.iter_mut().enumerate() {
            let b = i.next();
            let bb: GenericBBox = b.unwrap().into();
            observations.push((bb.rotate(indx as f32 / 10.0), None));
        }
        nms(&observations, 0.8, None);
    });
}
