use crate::track::Feature;
use std::ops::Sub;

pub fn euclidean(f1: &Feature, f2: &Feature) -> f32 {
    f1.sub(f2).map(|component| component * component).sum()
}

pub fn cosine(f1: &Feature, f2: &Feature) -> f32 {
    let divided = f1.component_mul(&f2).sum();
    let divisor = f1.component_mul(&f1).sum().sqrt() * f2.component_mul(&f2).sum().sqrt();
    divided / divisor
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
        assert!((d - 2.0f32).abs() < EPS);
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
