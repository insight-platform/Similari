use crate::track::Feature;
use nalgebra::SimdComplexField;

/// Euclidian distance between two vectors
pub fn euclidean(f1: &Feature, f2: &Feature) -> f32 {
    f1.metric_distance(&f2)
}

pub fn cosine(f1: &Feature, f2: &Feature) -> f32 {
    let divided = f1.component_mul(&f2).sum();

    let f1_divisor = f1
        .fold(0.0_f32, |acc, a| acc + a.simd_modulus_squared())
        .simd_sqrt();

    let f2_divisor = f2
        .fold(0.0_f32, |acc, a| acc + a.simd_modulus_squared())
        .simd_sqrt();

    divided / (f1_divisor * f2_divisor)
}

#[cfg(test)]
mod tests {
    use crate::distance::{cosine, euclidean};
    use crate::track::Feature;
    use crate::EPS;

    #[test]
    fn euclidean_distances() {
        let v1 = Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]);
        let v2 = Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]);
        let d = euclidean(&v1, &v1);
        assert!(d.abs() < EPS);

        let d = euclidean(&v1, &v2);
        assert!((d - 2.0f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn cosine_distances() {
        let v1 = Feature::from_vec(1, 3, vec![1f32, 0.0, 0.0]);
        let v2 = Feature::from_vec(1, 3, vec![0f32, 1.0f32, 0.0]);
        let v3 = Feature::from_vec(1, 3, vec![-1.0f32, 0.0, 0.0]);
        let d = cosine(&v1, &v1);
        assert!((d - 1.0).abs() < EPS);
        let d = cosine(&v1, &v3);
        assert!((d + 1.0).abs() < EPS);
        let d = cosine(&v1, &v2);
        assert!(d.abs() < EPS);
    }
}
