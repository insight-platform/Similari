// use crate::track::{ObservationAttributes, ObservationMetricOk};
// use std::collections::HashMap;
//
// pub struct ObservationWeights<OA> {
//     candidates: HashMap<u64, Vec<ObservationMetricOk<OA>>>,
// }
//
// trait AssignmentCandidates<OA>
// where
//     OA: ObservationAttributes,
// {
//     fn at(&self, row: usize, col: usize) -> f32;
//
//     fn rows(&self) -> usize;
//
//     fn columns(&self) -> usize;
//
//     fn neg(&self) -> Self;
// }
