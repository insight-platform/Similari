#![feature(test)]

extern crate test;

use similari::examples::FeatGen2;
use similari::utils::bbox::Universal2DBox;
use similari::utils::kalman::kalman_bbox::Universal2DBoxKalmanFilter;
use test::Bencher;

#[bench]
fn kalman_100k(b: &mut Bencher) {
    const N: usize = 100_000;
    let f = Universal2DBoxKalmanFilter::default();
    let mut pt = FeatGen2::new(-10.0, 2.0, 0.2);

    b.iter(|| {
        let v = pt.next().unwrap().feature().as_ref().unwrap().clone();
        let n = v[0].as_array_ref();

        let bbox = Universal2DBox::new(n[0], n[1], Some(0.0), 2.0, 5.0);

        let mut state = f.initiate(bbox);
        for _i in 0..N {
            let v = pt.next().unwrap().feature().as_ref().unwrap().clone();
            let n = v[0].as_array_ref();
            let bb = Universal2DBox::new(n[0], n[1], Some(0.0), 2.0, 5.0);

            state = f.predict(state);
            state = f.update(state, bb);
        }
    });
}
