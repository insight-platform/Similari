use crate::track::{Feature, Observation, ObservationAttributes, FEATURE_LANES_SIZE};
use std::cmp::Ordering;
use ultraviolet::f32x8;

/// Utility function that can be used by [ObservationMetric](crate::track::ObservationMetric::metric) implementors to sort
/// features by attributes decreasingly.
///
pub fn feature_attributes_sort_dec<FA: ObservationAttributes + PartialOrd>(
    e1: &Observation<FA>,
    e2: &Observation<FA>,
) -> Ordering {
    e2.0.partial_cmp(&e1.0).unwrap()
}

/// Utility function that can be used by [ObservationMetric](crate::track::ObservationMetric::metric) implementors to sort
/// features by attributes increasingly.
///
pub fn feature_attributes_sort_inc<FA: ObservationAttributes + PartialOrd>(
    e1: &Observation<FA>,
    e2: &Observation<FA>,
) -> Ordering {
    e1.0.partial_cmp(&e2.0).unwrap()
}

impl FromVec<&Feature, Vec<f32>> for Vec<f32> {
    fn from_vec(vec: &Feature) -> Vec<f32> {
        let mut res = Vec::with_capacity(vec.len() * FEATURE_LANES_SIZE);
        for e in vec {
            res.extend_from_slice(e.as_array_ref());
        }
        res
    }
}

/// Feature from &Vec<f32>
///
impl FromVec<Vec<f32>, Feature> for Feature {
    fn from_vec(vec: Vec<f32>) -> Feature {
        Feature::from_vec(&vec)
    }
}

/// Feature from &Vec<f32>
///
impl FromVec<&Vec<f32>, Feature> for Feature {
    fn from_vec(vec: &Vec<f32>) -> Feature {
        let mut feature = {
            let one_more = usize::from(vec.len() % FEATURE_LANES_SIZE > 0);
            Feature::with_capacity(vec.len() / FEATURE_LANES_SIZE + one_more)
        };

        let mut acc: [f32; FEATURE_LANES_SIZE] = [0.0; FEATURE_LANES_SIZE];
        let mut part = 0;
        for (counter, i) in vec.iter().enumerate() {
            part = counter % FEATURE_LANES_SIZE;
            if part == 0 {
                acc = [0.0; FEATURE_LANES_SIZE];
            }
            acc[part] = *i;
            if part == FEATURE_LANES_SIZE - 1 {
                feature.push(f32x8::new(acc));
                part = FEATURE_LANES_SIZE;
            }
        }

        if part < FEATURE_LANES_SIZE {
            feature.push(f32x8::new(acc));
        }
        feature
    }
}

/// Utility trait to get conversion between feature vector representations
///
pub trait FromVec<V, R> {
    fn from_vec(vec: V) -> R;
}

#[cfg(test)]
mod tests {
    use crate::track::utils::FromVec;
    use crate::track::Feature;

    #[test]
    fn conv_tests() {
        let v = vec![0.0, 0.2, 0.3];
        let o = Feature::from_vec(v);
        let v2 = Vec::from_vec(&o);
        assert_eq!(v2, vec![0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
}
