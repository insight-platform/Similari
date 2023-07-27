/// The struct allows defining the constraints for objects comprared across different epochs.
///
/// When the new objects batch is passed to the tracker it has a newer epoch that the tracks that are kept
/// within the trackers. It may happen that epoch difference is 1 when the track was updated in the previous
/// epoch or more than one, if the track wasn't updated lastly.
///
/// The constraint defines how far the object may be from the track for certain epoch difference to still count
/// that it can relate to it. The distance is measured in Nx(R_Obj+R_Track), where
/// * `N` is the float number that defines the expected maximal distance;
/// * `R_Obj` - radius of the circle surrounding the candidate object;
/// * `R_Track` - radius of the circle surrounding the last bounding box of the track.
///

#[derive(Default, Debug, Clone)]
pub struct SpatioTemporalConstraints {
    constraints: Vec<(usize, f32)>,
}

impl SpatioTemporalConstraints {
    /// Allows adding new constraints to the constraints engine
    ///
    /// # Parameters
    /// * `constraints` - slice of tuples (epoch_delta, max_allowed_distance)
    ///
    pub fn constraints(mut self, constraints: &[(usize, f32)]) -> Self {
        self.add_constraints(constraints.to_vec());
        self
    }
}

impl SpatioTemporalConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_constraints(&mut self, constraints: Vec<(usize, f32)>) {
        for (delta, max_distance) in constraints {
            assert!(
                max_distance > 0.0,
                "The distance is expected to be a positive float"
            );
            self.constraints.push((delta, max_distance));
        }
        self.constraints.sort_by(|(e1, _), (e2, _)| e1.cmp(e2));
        self.constraints.dedup_by(|(e1, _), (e2, _)| *e1 == *e2);
    }

    pub fn validate(&self, epoch_delta: usize, dist: f32) -> bool {
        assert!(
            dist >= 0.0,
            "The distance is expected to be a positive float"
        );
        let constraint = self.constraints.iter().find(|(d, _)| *d >= epoch_delta);

        match constraint {
            None => true,
            Some((_, max_dist)) => dist <= *max_dist,
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    use super::SpatioTemporalConstraints;

    #[pyclass]
    #[derive(Default, Debug, Clone)]
    #[pyo3(name = "SpatioTemporalConstraints")]
    pub struct PySpatioTemporalConstraints(pub(crate) SpatioTemporalConstraints);

    #[pymethods]
    impl PySpatioTemporalConstraints {
        #[new]
        pub fn new() -> Self {
            Self(SpatioTemporalConstraints::default())
        }

        /// Allows adding new constraints to the constraints engine
        ///
        /// # Parameters
        /// * `constraints` - Vec of tuples (epoch_delta, max_allowed_distance)
        ///
        #[pyo3(text_signature = "($self, l: [(epoch_delta, max_allowed_distance)]")]
        pub fn add_constraints(&mut self, constraints: Vec<(usize, f32)>) {
            self.0.add_constraints(constraints)
        }

        /// Validates the distance for specified epoch delta
        ///
        #[pyo3(text_signature = "($self, epoch_delta, dist)")]
        pub fn validate(&self, epoch_delta: usize, dist: f32) -> bool {
            self.0.validate(epoch_delta, dist)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;

    #[test]
    fn test() {
        let mut spc = SpatioTemporalConstraints::default();
        spc.add_constraints(vec![(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0)]);
        spc.add_constraints(vec![(3, 2.5), (4, 4.5), (7, 8.5)]);

        assert!(spc.validate(1, 0.4));
        assert!(!spc.validate(1, 0.6));

        assert!(spc.validate(6, 7.0));
        assert!(!spc.validate(6, 9.0));

        assert!(spc.validate(7, 8.4));
        assert!(spc.validate(7, 8.5));
        assert!(!spc.validate(7, 8.7));

        assert!(spc.validate(9, 8.7));
        assert!(spc.validate(9, 100.0));
    }
}
