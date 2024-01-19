use std::ops::SubAssign;
// Original source code idea from
// https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
//
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::{KalmanState, CHI2INV95, CHI2_UPPER_BOUND, DT};
use nalgebra::{SMatrix, SVector};

pub const DIM_2D_BOX: usize = 5;
pub const DIM_2D_BOX_X2: usize = DIM_2D_BOX * 2;

/// Kalman filter
///
#[derive(Debug)]
pub struct Universal2DBoxKalmanFilter {
    motion_matrix: SMatrix<f32, DIM_2D_BOX_X2, DIM_2D_BOX_X2>,
    update_matrix: SMatrix<f32, DIM_2D_BOX, DIM_2D_BOX_X2>,
    std_position_weight: f32,
    std_velocity_weight: f32,
}

/// Default initializer
impl Default for Universal2DBoxKalmanFilter {
    fn default() -> Self {
        Universal2DBoxKalmanFilter::new(1.0 / 20.0, 1.0 / 160.0)
    }
}

impl Universal2DBoxKalmanFilter {
    /// Constructor with custom weights (shouldn't be used without the need)
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        let mut motion_matrix: SMatrix<f32, DIM_2D_BOX_X2, DIM_2D_BOX_X2> = SMatrix::identity();

        for i in 0..DIM_2D_BOX {
            motion_matrix[(i, DIM_2D_BOX + i)] = DT as f32;
        }

        Universal2DBoxKalmanFilter {
            motion_matrix,
            update_matrix: SMatrix::identity(),
            std_position_weight: position_weight,
            std_velocity_weight: velocity_weight,
        }
    }

    fn std_position(&self, k: f32, cnst: f32, p: f32) -> [f32; DIM_2D_BOX] {
        let pos_weight = k * self.std_position_weight * p;
        [pos_weight, pos_weight, pos_weight, cnst, pos_weight]
    }

    fn std_velocity(&self, k: f32, cnst: f32, p: f32) -> [f32; DIM_2D_BOX] {
        let vel_weight = k * self.std_velocity_weight * p;
        [vel_weight, vel_weight, vel_weight, cnst, vel_weight]
    }

    /// Initialize the filter with the first observation
    ///
    pub fn initiate(&self, bbox: &Universal2DBox) -> KalmanState<DIM_2D_BOX_X2> {
        let mean: SVector<f32, DIM_2D_BOX_X2> = SVector::from_iterator([
            bbox.xc,
            bbox.yc,
            bbox.angle.unwrap_or(0.0),
            bbox.aspect,
            bbox.height,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        let mut std: SVector<f32, DIM_2D_BOX_X2> = SVector::from_iterator(
            self.std_position(2.0, 1e-2, bbox.height)
                .into_iter()
                .chain(self.std_velocity(10.0, 1e-5, bbox.height)),
        );

        std = std.component_mul(&std);

        let covariance: SMatrix<f32, DIM_2D_BOX_X2, DIM_2D_BOX_X2> = SMatrix::from_diagonal(&std);
        KalmanState { mean, covariance }
    }

    /// Predicts the state from the last state
    ///
    pub fn predict(&self, state: &KalmanState<DIM_2D_BOX_X2>) -> KalmanState<DIM_2D_BOX_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let std_pos = self.std_position(1.0, 1e-2, mean[4]);
        let std_vel = self.std_velocity(1.0, 1e-5, mean[4]);

        let mut std: SVector<f32, DIM_2D_BOX_X2> =
            SVector::from_iterator(std_pos.into_iter().chain(std_vel));

        std = std.component_mul(&std);

        let motion_cov: SMatrix<f32, DIM_2D_BOX_X2, DIM_2D_BOX_X2> = SMatrix::from_diagonal(&std);

        let mean = self.motion_matrix * mean;
        let covariance =
            self.motion_matrix * covariance * self.motion_matrix.transpose() + motion_cov;
        KalmanState { mean, covariance }
    }

    fn project(
        &self,
        mean: SVector<f32, DIM_2D_BOX_X2>,
        covariance: SMatrix<f32, DIM_2D_BOX_X2, DIM_2D_BOX_X2>,
    ) -> KalmanState<DIM_2D_BOX> {
        let mut std: SVector<f32, DIM_2D_BOX> =
            SVector::from_iterator(self.std_position(1.0, 1e-1, mean[4]));

        std = std.component_mul(&std);

        let innovation_cov: SMatrix<f32, DIM_2D_BOX, DIM_2D_BOX> = SMatrix::from_diagonal(&std);

        let mean = self.update_matrix * mean;
        let covariance =
            self.update_matrix * covariance * self.update_matrix.transpose() + innovation_cov;
        KalmanState { mean, covariance }
    }

    /// Updates the state with the current observation
    ///
    pub fn update(
        &self,
        state: &KalmanState<DIM_2D_BOX_X2>,
        measurement: &Universal2DBox,
    ) -> KalmanState<DIM_2D_BOX_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (projected_mean, projected_cov) = (projected_state.mean, projected_state.covariance);
        let b = (covariance * self.update_matrix.transpose()).transpose();
        let kalman_gain = projected_cov.solve_lower_triangular(&b).unwrap();

        let innovation = SVector::from_iterator([
            measurement.xc,
            measurement.yc,
            measurement.angle.unwrap_or(0.0),
            measurement.aspect,
            measurement.height,
        ]) - projected_mean;

        let innovation: SMatrix<f32, 1, DIM_2D_BOX> = innovation.transpose();

        let mean = mean + (innovation * kalman_gain).transpose();
        let covariance = covariance - kalman_gain.transpose() * projected_cov * kalman_gain;
        KalmanState { mean, covariance }
    }

    pub fn distance(&self, state: KalmanState<DIM_2D_BOX_X2>, measurement: &Universal2DBox) -> f32 {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (mean, covariance) = (projected_state.mean, projected_state.covariance);

        let measurements = {
            let mut r: SVector<f32, DIM_2D_BOX> = SVector::from_vec(vec![
                measurement.xc,
                measurement.yc,
                measurement.angle.unwrap_or(0.0),
                measurement.aspect,
                measurement.height,
            ]);
            r.sub_assign(&mean);
            r
        };

        let choletsky = covariance.cholesky().unwrap().l();
        let res = choletsky.solve_lower_triangular(&measurements).unwrap();
        res.component_mul(&res).sum()
    }

    pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
        if !inverted {
            if distance > CHI2INV95[4] {
                CHI2_UPPER_BOUND
            } else {
                distance
            }
        } else if distance > CHI2INV95[4] {
            0.0
        } else {
            CHI2_UPPER_BOUND - distance
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::bbox::{BoundingBox, Universal2DBox};
    use crate::utils::kalman::kalman_2d_box::Universal2DBoxKalmanFilter;
    use crate::utils::kalman::CHI2INV95;

    #[test]
    fn constructor() {
        let f = Universal2DBoxKalmanFilter::default();
        let bbox = BoundingBox::new(1.0, 2.0, 5.0, 5.0);

        let state = f.initiate(&bbox.into());
        let new_bb = BoundingBox::try_from(state);
        assert_eq!(new_bb.unwrap(), bbox);
    }

    #[test]
    fn step() {
        let f = Universal2DBoxKalmanFilter::default();
        let bbox = BoundingBox::new(-10.0, 2.0, 2.0, 5.0);

        let state = f.initiate(&bbox.into());
        let state = f.predict(&state);
        let p = Universal2DBox::try_from(state).unwrap();

        let est_p = Universal2DBox::new(-9.0, 4.5, None, 0.4, 5.0);
        assert_eq!(p, est_p);

        let bbox = Universal2DBox::new(8.75, 52.35, None, 0.150_849_15, 100.1);
        let state = f.update(&state, &bbox);
        let est_p = Universal2DBox::new(10.070248, 55.90909, None, 0.3951147, 107.173546);

        let state = f.predict(&state);
        let p = Universal2DBox::try_from(state).unwrap();
        assert_eq!(p, est_p);
    }

    #[test]
    fn gating_distance() {
        let f = Universal2DBoxKalmanFilter::default();
        let bbox = BoundingBox::new(-10.0, 2.0, 2.0, 5.0);

        let upd_bbox = BoundingBox::new(-9.5, 2.1, 2.0, 5.0);

        let new_bbox_1 = BoundingBox::new(-9.0, 2.2, 2.0, 5.0);

        let new_bbox_2 = BoundingBox::new(-5.0, 1.5, 2.2, 5.0);

        let state = f.initiate(&bbox.into());
        let state = f.predict(&state);
        let state = f.update(&state, &upd_bbox.into());
        let state = f.predict(&state);

        let dist = f.distance(state, &new_bbox_1.into());
        let dist = Universal2DBoxKalmanFilter::calculate_cost(dist, false);
        dbg!(&dist);
        assert!((0.0..CHI2INV95[4]).contains(&dist));

        let dist = f.distance(state, &new_bbox_2.into());
        let dist = Universal2DBoxKalmanFilter::calculate_cost(dist, false);
        dbg!(&dist);
        assert!(dist > CHI2INV95[4]);
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::prelude::Universal2DBox;
    use crate::utils::bbox::python::{PyBoundingBox, PyUniversal2DBox};
    use crate::utils::kalman::kalman_2d_box::{Universal2DBoxKalmanFilter, DIM_2D_BOX_X2};
    use crate::utils::kalman::KalmanState;
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "Universal2DBoxKalmanFilter")]
    pub struct PyUniversal2DBoxKalmanFilter {
        filter: Universal2DBoxKalmanFilter,
    }

    #[derive(Clone)]
    #[pyclass]
    #[pyo3(name = "Universal2DBoxKalmanFilterState")]
    pub struct PyUniversal2DBoxKalmanFilterState {
        state: KalmanState<{ DIM_2D_BOX_X2 }>,
    }

    #[pymethods]
    impl PyUniversal2DBoxKalmanFilterState {
        #[pyo3(signature = ())]
        pub fn universal_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(Universal2DBox::try_from(self.state).unwrap())
        }

        #[pyo3(signature = ())]
        pub fn bbox(&self) -> PyResult<PyBoundingBox> {
            self.universal_bbox().as_ltwh()
        }
    }

    #[pymethods]
    impl PyUniversal2DBoxKalmanFilter {
        #[new]
        #[pyo3(signature = (position_weight = 0.05, velocity_weight = 0.00625))]
        pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
            Self {
                filter: Universal2DBoxKalmanFilter::new(position_weight, velocity_weight),
            }
        }

        #[pyo3(signature = (bbox))]
        pub fn initiate(&self, bbox: PyUniversal2DBox) -> PyUniversal2DBoxKalmanFilterState {
            PyUniversal2DBoxKalmanFilterState {
                state: self.filter.initiate(&bbox.0),
            }
        }

        #[pyo3(signature = (state))]
        pub fn predict(
            &self,
            state: PyUniversal2DBoxKalmanFilterState,
        ) -> PyUniversal2DBoxKalmanFilterState {
            PyUniversal2DBoxKalmanFilterState {
                state: self.filter.predict(&state.state),
            }
        }

        #[pyo3(signature = (state, bbox))]
        pub fn update(
            &self,
            state: PyUniversal2DBoxKalmanFilterState,
            bbox: PyUniversal2DBox,
        ) -> PyUniversal2DBoxKalmanFilterState {
            PyUniversal2DBoxKalmanFilterState {
                state: self.filter.update(&state.state, &bbox.0),
            }
        }

        #[pyo3(signature = (state, bbox))]
        pub fn distance(
            &self,
            state: PyUniversal2DBoxKalmanFilterState,
            bbox: PyUniversal2DBox,
        ) -> f32 {
            self.filter.distance(state.state, &bbox.0)
        }

        #[staticmethod]
        #[pyo3(signature = (distance, inverted))]
        pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
            Universal2DBoxKalmanFilter::calculate_cost(distance, inverted)
        }
    }
}
