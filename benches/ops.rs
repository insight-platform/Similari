#![feature(test)]

extern crate test;

use similari::simd::F32x16;
use test::Bencher;

#[bench]
fn almost_same(b: &mut Bencher) {
    b.iter(move || {
        let n1 = F32x16::default();
        let n2 = F32x16::default();
        n1.almost_same(&n2);
    })
}

#[bench]
fn mul_assign(b: &mut Bencher) {
    b.iter(move || {
        let mut n1 = F32x16::default();
        let n2 = F32x16::default();
        n1.mul_assign(n2);
    })
}

#[bench]
fn sub_assign(b: &mut Bencher) {
    b.iter(move || {
        let mut n1 = F32x16::default();
        let n2 = F32x16::default();
        n1.sub_assign(n2);
    })
}
