use crate::track::Feature;

/// Euclidian distance between two vectors
///
/// When the features distances lengths don't match, the longer feature vector is truncated to
/// shorter one when the distance is calculated
///
pub fn euclidean(f1: &Feature, f2: &Feature) -> f32 {
    let mut acc = 0.0;
    for i in 0..f1.len().min(f2.len()) {
        let mut block1 = f1[i];
        let block2 = f2[i];
        block1 -= block2;
        block1 *= block1;
        acc += block1.sum();
    }
    acc.sqrt()
}

/// Cosine distance between two vectors
///
/// When the features distances lengths don't match, the longer feature vector is truncated to
/// shorter one when the distance is calculated
///  
pub fn cosine(f1: &Feature, f2: &Feature) -> f32 {
    let mut divided = 0.0;
    let len = f1.len().min(f2.len());
    for i in 0..len {
        let mut block1 = f1[i];
        let block2 = f2[i];
        block1 *= block2;
        divided += block1.sum();
    }

    let f1_divisor = f1
        .iter()
        .take(len)
        .fold(0.0_f32, |acc, a| acc + (*a * *a).sum());

    let f2_divisor = f2
        .iter()
        .take(len)
        .fold(0.0_f32, |acc, a| acc + (*a * *a).sum());

    divided / (f1_divisor * f2_divisor).sqrt()
}

#[cfg(test)]
mod tests {
    use crate::distance::{cosine, euclidean};
    use crate::track::{Feature, FromVec};
    use crate::EPS;

    #[test]
    fn euclidean_distances() {
        let v1 = Feature::from_vec(vec![1f32, 0.0, 0.0]);
        let v2 = Feature::from_vec(vec![0f32, 1.0f32, 0.0]);
        let d = euclidean(&v1, &v1);
        assert!(d.abs() < EPS);

        let d = euclidean(&v1, &v2);
        assert!((d - 2.0f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn cosine_distances() {
        let v1 = dbg!(Feature::from_vec(vec![1f32, 0.0, 0.0]));
        let v2 = dbg!(Feature::from_vec(vec![0f32, 1.0f32, 0.0]));
        let v3 = dbg!(Feature::from_vec(vec![-1.0f32, 0.0, 0.0]));
        let d = cosine(&v1, &v1);
        assert!((d - 1.0).abs() < EPS);
        let d = cosine(&v1, &v3);
        assert!((d + 1.0).abs() < EPS);
        let d = cosine(&v1, &v2);
        assert!(d.abs() < EPS);
    }
}
