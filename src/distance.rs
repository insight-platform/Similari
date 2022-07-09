use crate::track::Feature;
use std::ops::{Mul, Sub};
use ultraviolet::f32x8;

/// Euclidian distance between two vectors
pub fn euclidean(f1: &Feature, f2: &Feature) -> f32 {
    let mut acc = 0.0;
    for i in 0..f1.len().max(f2.len()) {
        let block1 = if f1.len() > i { f1[i] } else { f32x8::ZERO };
        let block2 = if f2.len() > i { f2[i] } else { f32x8::ZERO };
        let res = block1.sub(block2);
        let res = res.mul(res);
        acc += res.reduce_add();
    }
    acc.sqrt()
}

/// Cosine distance between two vectors
pub fn cosine(f1: &Feature, f2: &Feature) -> f32 {
    let mut divided = 0.0;
    for i in 0..f1.len().max(f2.len()) {
        let block1 = if f1.len() > i { f1[i] } else { f32x8::ZERO };
        let block2 = if f2.len() > i { f2[i] } else { f32x8::ZERO };
        let res = block1.mul(block2);
        divided += res.reduce_add();
    }

    let f1_divisor = f1
        .iter()
        .fold(0.0_f32, |acc, a| acc + a.mul(a).reduce_add())
        .sqrt();

    let f2_divisor = f2
        .iter()
        .fold(0.0_f32, |acc, a| acc + a.mul(a).reduce_add())
        .sqrt();

    divided / (f1_divisor * f2_divisor)
}

#[cfg(test)]
mod tests {
    use crate::distance::{cosine, euclidean};
    use crate::track::{Feature, FromVec};
    use crate::EPS;

    #[test]
    fn euclidean_distances() {
        let v1 = dbg!(Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]));
        let v2 = dbg!(Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]));
        let d = euclidean(&v1, &v1);
        assert!(d.abs() < EPS);

        let d = euclidean(&v1, &v2);
        assert!((d - 2.0f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn cosine_distances() {
        let v1 = dbg!(Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]));
        let v2 = dbg!(Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]));
        let v3 = dbg!(Feature::from_vec(1, 3, vec![-1.0f32, 0.0, 0.0]));
        let d = cosine(&v1, &v1);
        assert!((d - 1.0).abs() < EPS);
        let d = cosine(&v1, &v3);
        assert!((d + 1.0).abs() < EPS);
        let d = cosine(&v1, &v2);
        assert!(d.abs() < EPS);
    }
}
