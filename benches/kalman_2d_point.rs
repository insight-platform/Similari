#![feature(test)]

extern crate test;

use nalgebra::Point2;
use similari::examples::FeatGen2;
use similari::utils::kalman::kalman_2d_point::Point2DKalmanFilter;
use test::Bencher;

#[bench]
fn kalman_2d_point_100k(b: &mut Bencher) {
    const N: usize = 100_000;
    let f = Point2DKalmanFilter::default();
    let mut pt = FeatGen2::new(-10.0, 2.0, 0.2);

    b.iter(|| {
        let v = pt.next().unwrap().feature().as_ref().unwrap().clone();
        let n = v[0].as_array_ref();

        let mut state = f.initiate(&Point2::from([n[0], n[1]]));
        for _i in 0..N {
            let v = pt.next().unwrap().feature().as_ref().unwrap().clone();
            let n = v[0].as_array_ref();
            state = f.predict(&state);

            let p = Point2::from([n[0], n[1]]);
            state = f.update(&state, &p);
        }
    });
}
