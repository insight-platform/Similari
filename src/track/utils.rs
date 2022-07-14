use crate::track::{Observation, ObservationAttributes, ObservationSpec, FEATURE_LANES_SIZE};
use std::cmp::Ordering;
use ultraviolet::f32x8;

/// Utility function that can be used by [Metric](Metric::optimize) implementors to sort
/// features by attributes decreasingly.
///
pub fn feature_attributes_sort_dec<FA: ObservationAttributes>(
    e1: &ObservationSpec<FA>,
    e2: &ObservationSpec<FA>,
) -> Ordering {
    e2.0.partial_cmp(&e1.0).unwrap()
}

/// Utility function that can be used by [Metric](Metric::optimize) implementors to sort
/// features by attributes increasingly.
///
pub fn feature_attributes_sort_inc<FA: ObservationAttributes>(
    e1: &ObservationSpec<FA>,
    e2: &ObservationSpec<FA>,
) -> Ordering {
    e1.0.partial_cmp(&e2.0).unwrap()
}

impl FromVec<Vec<f32>> for Observation {
    fn from_vec(vec: Vec<f32>) -> Observation {
        let mut feature = {
            let one_more = if vec.len() % FEATURE_LANES_SIZE > 0 {
                1
            } else {
                0
            };
            Observation::with_capacity(vec.len() / FEATURE_LANES_SIZE + one_more)
        };

        let mut acc: [f32; FEATURE_LANES_SIZE] = [0.0; FEATURE_LANES_SIZE];
        let mut part = 0;
        for (counter, i) in vec.into_iter().enumerate() {
            part = counter % FEATURE_LANES_SIZE;
            if part == 0 {
                acc = [0.0; FEATURE_LANES_SIZE];
            }
            acc[part] = i;
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

pub trait FromVec<V> {
    fn from_vec(vec: V) -> Observation;
}
