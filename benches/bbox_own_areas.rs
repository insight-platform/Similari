#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::utils::clipping::bbox_own_areas::{
    compute_own_areas_percentage, compute_own_polygons,
};
use test::Bencher;

#[bench]
fn bbox_own_areas_00010(b: &mut Bencher) {
    bench_bbox_own_areas(10, b);
}

#[bench]
fn bbox_own_areas_00025(b: &mut Bencher) {
    bench_bbox_own_areas(25, b);
}

#[bench]
fn bbox_own_areas_00050(b: &mut Bencher) {
    bench_bbox_own_areas(50, b);
}

#[bench]
fn bbox_own_areas_00100(b: &mut Bencher) {
    bench_bbox_own_areas(100, b);
}

fn bench_bbox_own_areas(objects: usize, b: &mut Bencher) {
    let pos_drift = 20.0;
    let box_drift = 5.0;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(BoxGen2::new(
            i as f32, i as f32, 10.0, 10.0, pos_drift, box_drift,
        ));
    }

    b.iter(|| {
        let mut observations = Vec::new();
        for i in &mut iterators {
            let b = i.next();
            observations.push(b.unwrap().into());
        }
        let input = observations.iter().collect::<Vec<_>>();
        let polygons = compute_own_polygons(input.as_slice());
        let areas = compute_own_areas_percentage(input.as_slice(), polygons.as_slice());
        assert_eq!(areas.len(), objects);
    });
}
