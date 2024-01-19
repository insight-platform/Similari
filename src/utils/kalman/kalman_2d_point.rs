use crate::utils::kalman::{KalmanState, CHI2INV95, CHI2_UPPER_BOUND, DT};
use nalgebra::{Point2, SMatrix, SVector};
use std::ops::SubAssign;

pub const DIM_2D_POINT: usize = 2;
pub const DIM_2D_POINT_X2: usize = DIM_2D_POINT * 2;

/// Kalman filter
///
#[derive(Debug)]
pub struct Point2DKalmanFilter {
    motion_matrix: SMatrix<f32, DIM_2D_POINT_X2, DIM_2D_POINT_X2>,
    update_matrix: SMatrix<f32, DIM_2D_POINT, DIM_2D_POINT_X2>,
    std_position_weight: f32,
    std_velocity_weight: f32,
}

/// Default initializer
impl Default for Point2DKalmanFilter {
    fn default() -> Self {
        Point2DKalmanFilter::new(1.0 / 20.0, 1.0 / 160.0)
    }
}

impl Point2DKalmanFilter {
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        let mut motion_matrix: SMatrix<f32, DIM_2D_POINT_X2, DIM_2D_POINT_X2> = SMatrix::identity();

        for i in 0..DIM_2D_POINT {
            motion_matrix[(i, DIM_2D_POINT + i)] = DT as f32;
        }

        Point2DKalmanFilter {
            motion_matrix,
            update_matrix: SMatrix::identity(),
            std_position_weight: position_weight,
            std_velocity_weight: velocity_weight,
        }
    }

    fn std_position(&self, k: f32) -> [f32; DIM_2D_POINT] {
        let pos_weight = k * self.std_position_weight;
        [pos_weight, pos_weight]
    }

    fn std_velocity(&self, k: f32) -> [f32; DIM_2D_POINT] {
        let vel_weight = k * self.std_velocity_weight;
        [vel_weight, vel_weight]
    }

    pub fn initiate(&self, p: &Point2<f32>) -> KalmanState<DIM_2D_POINT_X2> {
        let mean: SVector<f32, DIM_2D_POINT_X2> = SVector::from_iterator([p.x, p.y, 0.0, 0.0]);

        let mut std: SVector<f32, DIM_2D_POINT_X2> = SVector::from_iterator(
            self.std_position(2.0)
                .into_iter()
                .chain(self.std_velocity(10.0)),
        );

        std = std.component_mul(&std);

        let covariance: SMatrix<f32, DIM_2D_POINT_X2, DIM_2D_POINT_X2> =
            SMatrix::from_diagonal(&std);
        KalmanState { mean, covariance }
    }

    pub fn predict(&self, state: &KalmanState<DIM_2D_POINT_X2>) -> KalmanState<DIM_2D_POINT_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let std_pos = self.std_position(1.0);
        let std_vel = self.std_velocity(1.0);

        let mut std: SVector<f32, DIM_2D_POINT_X2> =
            SVector::from_iterator(std_pos.into_iter().chain(std_vel));

        std = std.component_mul(&std);

        let motion_cov: SMatrix<f32, DIM_2D_POINT_X2, DIM_2D_POINT_X2> =
            SMatrix::from_diagonal(&std);

        let mean = self.motion_matrix * mean;
        let covariance =
            self.motion_matrix * covariance * self.motion_matrix.transpose() + motion_cov;
        KalmanState { mean, covariance }
    }

    fn project(
        &self,
        mean: SVector<f32, DIM_2D_POINT_X2>,
        covariance: SMatrix<f32, DIM_2D_POINT_X2, DIM_2D_POINT_X2>,
    ) -> KalmanState<DIM_2D_POINT> {
        let mut std: SVector<f32, DIM_2D_POINT> = SVector::from_iterator(self.std_position(1.0));

        std = std.component_mul(&std);

        let innovation_cov: SMatrix<f32, DIM_2D_POINT, DIM_2D_POINT> = SMatrix::from_diagonal(&std);

        let mean = self.update_matrix * mean;
        let covariance =
            self.update_matrix * covariance * self.update_matrix.transpose() + innovation_cov;
        KalmanState { mean, covariance }
    }

    pub fn update(
        &self,
        state: &KalmanState<DIM_2D_POINT_X2>,
        p: &Point2<f32>,
    ) -> KalmanState<DIM_2D_POINT_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (projected_mean, projected_cov) = (projected_state.mean, projected_state.covariance);
        let b = (covariance * self.update_matrix.transpose()).transpose();
        let kalman_gain = projected_cov.solve_lower_triangular(&b).unwrap();

        let innovation = SVector::from_iterator([p.x, p.y]) - projected_mean;

        let innovation: SMatrix<f32, 1, DIM_2D_POINT> = innovation.transpose();

        let mean = mean + (innovation * kalman_gain).transpose();
        let covariance = covariance - kalman_gain.transpose() * projected_cov * kalman_gain;
        KalmanState { mean, covariance }
    }

    pub fn distance(&self, state: &KalmanState<DIM_2D_POINT_X2>, p: &Point2<f32>) -> f32 {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (mean, covariance) = (projected_state.mean, projected_state.covariance);

        let measurements = {
            let mut r: SVector<f32, DIM_2D_POINT> = SVector::from_vec(vec![p.x, p.y]);
            r.sub_assign(&mean);
            r
        };

        let choletsky = covariance.cholesky().unwrap().l();
        let res = choletsky.solve_lower_triangular(&measurements).unwrap();
        res.component_mul(&res).sum()
    }

    pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
        if !inverted {
            if distance > CHI2INV95[1] {
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

impl From<KalmanState<{ DIM_2D_POINT_X2 }>> for Point2<f32> {
    fn from(s: KalmanState<{ DIM_2D_POINT_X2 }>) -> Self {
        Point2::from([s.mean.x, s.mean.y])
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::kalman::kalman_2d_point::Point2DKalmanFilter;
    use nalgebra::Point2;

    #[test]
    fn test() {
        let p = Point2::from([1.0, 0.0]);
        let f = Point2DKalmanFilter::default();
        let state = f.initiate(&p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.1, 0.1]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.2, 0.2]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.3, 0.3]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.4, 0.4]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.5, 0.5]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.6, 0.6]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.7, 0.7]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.8, 0.67]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let p = Point2::from([1.9, 0.60]);
        let state = f.update(&state, &p);
        let state = f.predict(&state);
        dbg!(Point2::from(state));

        let dist = f.distance(&state, &Point2::from([2.0, 0.57]));
        dbg!(&dist);
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::utils::kalman::kalman_2d_point::{Point2DKalmanFilter, DIM_2D_POINT_X2};
    use crate::utils::kalman::KalmanState;
    use nalgebra::Point2;
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "Point2DKalmanFilter")]
    pub struct PyPoint2DKalmanFilter {
        filter: Point2DKalmanFilter,
    }

    #[derive(Clone)]
    #[pyclass]
    #[pyo3(name = "Point2DKalmanFilterState")]
    pub struct PyPoint2DKalmanFilterState {
        state: KalmanState<{ DIM_2D_POINT_X2 }>,
    }

    impl PyPoint2DKalmanFilterState {
        pub fn new(state: KalmanState<{ DIM_2D_POINT_X2 }>) -> Self {
            Self { state }
        }

        pub fn inner(&self) -> &KalmanState<{ DIM_2D_POINT_X2 }> {
            &self.state
        }
    }

    #[pymethods]
    impl PyPoint2DKalmanFilterState {
        #[pyo3(signature = ())]
        pub fn x(&self) -> f32 {
            self.state.mean[0]
        }

        #[pyo3(signature = ())]
        pub fn y(&self) -> f32 {
            self.state.mean[1]
        }
    }

    #[pymethods]
    impl PyPoint2DKalmanFilter {
        #[new]
        #[pyo3(signature = (position_weight = 0.05, velocity_weight = 0.00625))]
        pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
            Self {
                filter: Point2DKalmanFilter::new(position_weight, velocity_weight),
            }
        }

        #[pyo3(signature = (x, y))]
        pub fn initiate(&self, x: f32, y: f32) -> PyPoint2DKalmanFilterState {
            PyPoint2DKalmanFilterState {
                state: self.filter.initiate(&Point2::from([x, y])),
            }
        }

        #[pyo3(signature = (state))]
        pub fn predict(&self, state: PyPoint2DKalmanFilterState) -> PyPoint2DKalmanFilterState {
            PyPoint2DKalmanFilterState {
                state: self.filter.predict(&state.state),
            }
        }

        #[pyo3(signature = (state, x, y))]
        pub fn update(
            &self,
            state: PyPoint2DKalmanFilterState,
            x: f32,
            y: f32,
        ) -> PyPoint2DKalmanFilterState {
            PyPoint2DKalmanFilterState {
                state: self.filter.update(&state.state, &Point2::from([x, y])),
            }
        }

        #[pyo3(signature = (state, x, y))]
        pub fn distance(&self, state: PyPoint2DKalmanFilterState, x: f32, y: f32) -> f32 {
            self.filter.distance(&state.state, &Point2::from([x, y]))
        }

        #[staticmethod]
        #[pyo3(signature = (distance, inverted))]
        pub fn calculate_cost(distance: f32, inverted: bool) -> f32 {
            Point2DKalmanFilter::calculate_cost(distance, inverted)
        }
    }
}
