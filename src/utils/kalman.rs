use std::ops::SubAssign;
// Original source code idea from
// https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
//
use crate::utils::bbox::{BBox, GenericBBox};
use anyhow::Result;
use nalgebra::{DMatrix, DVector, Dynamic, OMatrix, SMatrix, SVector, U1};

pub fn chi2inv95() -> [f32; 9] {
    [
        3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919,
    ]
}

const DIM: usize = 5;
const DIM_X2: usize = DIM * 2;
const DT: u64 = 1;

macro_rules! pretty_print {
    ($arr:expr) => {{
        let indent = 4;
        let prefix = String::from_utf8(vec![b' '; indent]).unwrap();
        let mut result_els = vec!["".to_string()];
        for i in 0..$arr.nrows() {
            let mut row_els = vec![];
            for j in 0..$arr.ncols() {
                row_els.push(format!("{:12.3}", $arr[(i, j)]));
            }
            let row_str = row_els.into_iter().collect::<Vec<_>>().join(" ");
            let row_str = format!("{}{}", prefix, row_str);
            result_els.push(row_str);
        }
        result_els.into_iter().collect::<Vec<_>>().join("\n")
    }};
}

/// Kalman filter current state
///
#[derive(Copy, Clone, Debug)]
pub struct State<const X: usize = DIM_X2> {
    mean: SVector<f32, X>,
    covariance: SMatrix<f32, X, X>,
}

impl<const X: usize> State<X> {
    /// Fetch predicted bbox in (x,y,w,h) format from the state
    ///
    pub fn bbox(&self) -> Result<BBox> {
        self.generic_bbox().into()
    }

    /// Fetch predicted bbox in (x,y,a,h) format from the state
    ///
    pub fn generic_bbox(&self) -> GenericBBox {
        GenericBBox::new(
            self.mean[0],
            self.mean[1],
            if self.mean[2] == 0.0 {
                None
            } else {
                Some(self.mean[2])
            },
            self.mean[3],
            self.mean[4],
        )
    }

    /// dump the state
    ///
    pub fn dump(&self) {
        eprintln!("Mean={}", pretty_print!(self.mean.transpose()));
        eprintln!("Covariance={}", pretty_print!(self.covariance));
    }
}

/// Kalman filter
///
#[derive(Debug)]
pub struct KalmanFilter {
    motion_matrix: SMatrix<f32, DIM_X2, DIM_X2>,
    update_matrix: SMatrix<f32, DIM, DIM_X2>,
    std_position_weight: f32,
    std_velocity_weight: f32,
}

/// Default initializer
impl Default for KalmanFilter {
    fn default() -> Self {
        KalmanFilter::new(1.0 / 20.0, 1.0 / 160.0)
    }
}

impl KalmanFilter {
    /// Constructor with custom weights (shouldn't be used without the need)
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        let mut motion_matrix: SMatrix<f32, DIM_X2, DIM_X2> = SMatrix::identity();

        for i in 0..DIM {
            motion_matrix[(i, DIM + i)] = DT as f32;
        }

        KalmanFilter {
            motion_matrix,
            update_matrix: SMatrix::identity(),
            std_position_weight: position_weight,
            std_velocity_weight: velocity_weight,
        }
    }

    fn std_position(&self, k: f32, cnst: f32, p: f32) -> [f32; DIM] {
        let pos_weight = k * self.std_position_weight * p;
        [pos_weight, pos_weight, pos_weight, cnst, pos_weight]
    }

    fn std_velocity(&self, k: f32, cnst: f32, p: f32) -> [f32; DIM] {
        let vel_weight = k * self.std_velocity_weight * p;
        [vel_weight, vel_weight, vel_weight, cnst, vel_weight]
    }

    /// Initialize the filter with the first observation
    ///
    pub fn initiate(&self, bbox: GenericBBox) -> State<DIM_X2> {
        let mean: SVector<f32, DIM_X2> = SVector::from_iterator([
            bbox.x,
            bbox.y,
            bbox.angle.unwrap_or(0.0),
            bbox.aspect,
            bbox.height,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        let mut std: SVector<f32, DIM_X2> = SVector::from_iterator(
            self.std_position(2.0, 1e-2, bbox.height)
                .into_iter()
                .chain(self.std_velocity(10.0, 1e-5, bbox.height).into_iter()),
        );

        std = std.component_mul(&std);

        let covariance: SMatrix<f32, DIM_X2, DIM_X2> = SMatrix::from_diagonal(&std);
        State { mean, covariance }
    }

    /// Predicts the state from the last state
    ///
    pub fn predict(&self, state: State<DIM_X2>) -> State<DIM_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let std_pos = self.std_position(1.0, 1e-2, mean[4]);
        let std_vel = self.std_velocity(1.0, 1e-5, mean[4]);

        let mut std: SVector<f32, DIM_X2> =
            SVector::from_iterator(std_pos.into_iter().chain(std_vel.into_iter()));

        std = std.component_mul(&std);

        let motion_cov: SMatrix<f32, DIM_X2, DIM_X2> = SMatrix::from_diagonal(&std);

        let mean = self.motion_matrix * mean;
        let covariance =
            self.motion_matrix * covariance * self.motion_matrix.transpose() + motion_cov;
        State { mean, covariance }
    }

    fn project(
        &self,
        mean: SVector<f32, DIM_X2>,
        covariance: SMatrix<f32, DIM_X2, DIM_X2>,
    ) -> State<DIM> {
        let mut std: SVector<f32, DIM> =
            SVector::from_iterator(self.std_position(1.0, 1e-1, mean[4]));

        std = std.component_mul(&std);

        let innovation_cov: SMatrix<f32, DIM, DIM> = SMatrix::from_diagonal(&std);

        let mean = self.update_matrix * mean;
        let covariance =
            self.update_matrix * covariance * self.update_matrix.transpose() + innovation_cov;
        State { mean, covariance }
    }

    /// Updates the state with the current observation
    ///
    pub fn update(&self, state: State<DIM_X2>, measurement: GenericBBox) -> State<DIM_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (projected_mean, projected_cov) = (projected_state.mean, projected_state.covariance);
        let b = (covariance * self.update_matrix.transpose()).transpose();
        let kalman_gain = projected_cov.solve_lower_triangular(&b).unwrap();

        let innovation = SVector::from_iterator([
            measurement.x,
            measurement.y,
            measurement.angle.unwrap_or(0.0),
            measurement.aspect,
            measurement.height,
        ]) - projected_mean;

        let innovation: SMatrix<f32, 1, DIM> = innovation.transpose();

        let mean = mean + (innovation * kalman_gain).transpose();
        let covariance = covariance - kalman_gain.transpose() * projected_cov * kalman_gain;
        State { mean, covariance }
    }

    pub fn distances(
        &self,
        state: State<DIM_X2>,
        measurements: &[GenericBBox],
        only_position: bool,
    ) -> OMatrix<f32, Dynamic, U1> {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean, covariance);
        let (mean, covariance) = (projected_state.mean, projected_state.covariance);

        let (covariance, measurements) = if only_position {
            let mean = mean.resize(2, 1, 0.0);
            let covariance = covariance.resize(2, 2, 0.0);
            let measurements = DMatrix::from_columns(
                measurements
                    .iter()
                    .map(|e| {
                        let mut r = DVector::from_vec(vec![e.x, e.y]);
                        r.sub_assign(mean.clone());
                        r
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            (covariance, measurements)
        } else {
            let measurements = DMatrix::from_columns(
                measurements
                    .iter()
                    .map(|e| {
                        let mut r = DVector::from_vec(vec![
                            e.x,
                            e.y,
                            e.angle.unwrap_or(0.0),
                            e.aspect,
                            e.height,
                        ]);
                        r.sub_assign(mean);
                        r
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            (
                covariance.resize(covariance.nrows(), covariance.ncols(), 0.0),
                measurements,
            )
        };

        let choletsky = covariance.cholesky().unwrap().l();
        let res = choletsky.solve_lower_triangular(&measurements).unwrap();
        res.component_mul(&res).row_sum().transpose()
    }

    pub fn calculate_final_weights(
        mut distances: OMatrix<f32, Dynamic, U1>,
        only_position: bool,
    ) -> OMatrix<f32, Dynamic, U1> {
        let chi_index = if only_position {
            chi2inv95()[1]
        } else {
            chi2inv95()[4]
        };

        distances.iter_mut().for_each(|e| {
            if *e > chi_index {
                *e = f32::MAX;
            }
        });
        distances
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::bbox::{BBox, GenericBBox};
    use crate::utils::kalman::{chi2inv95, KalmanFilter};
    use crate::{EstimateClose, EPS};

    #[test]
    fn constructor() {
        let f = KalmanFilter::default();
        let bbox = BBox {
            x: 1.0,
            y: 2.0,
            width: 5.0,
            height: 5.0,
        };

        let state = f.initiate(bbox.clone().into());
        let new_bb = state.bbox();
        assert_eq!(new_bb.unwrap(), bbox.clone());
    }

    #[test]
    fn step() {
        let f = KalmanFilter::default();
        let bbox = BBox {
            x: -10.0,
            y: 2.0,
            width: 2.0,
            height: 5.0,
        };

        let state = f.initiate(bbox.clone().into());
        let state = f.predict(state);
        let p = state.generic_bbox();

        let est_p = GenericBBox::new(-9.0, 4.5, None, 0.4, 5.0);
        assert_eq!(p.almost_same(&est_p, EPS), true);

        let bbox = GenericBBox::new(8.75, 52.349999999999994, None, 0.15084915084915085, 100.1);
        let state = f.update(state, bbox);
        let est_p = GenericBBox::new(10.070248, 55.90909, None, 0.3951147, 107.173546);

        let state = f.predict(state);
        let p = state.generic_bbox();
        assert_eq!(p.almost_same(&est_p, EPS), true);
    }

    #[test]
    fn gating_distance() {
        let f = KalmanFilter::default();
        let bbox = BBox {
            x: -10.0,
            y: 2.0,
            width: 2.0,
            height: 5.0,
        };

        let upd_bbox = BBox {
            x: -9.5,
            y: 2.1,
            width: 2.0,
            height: 5.0,
        };

        let new_bbox_1 = BBox {
            x: -9.0,
            y: 2.2,
            width: 2.0,
            height: 5.0,
        };

        let new_bbox_2 = BBox {
            x: -5.0,
            y: 1.5,
            width: 2.2,
            height: 5.0,
        };

        let state = f.initiate(bbox.clone().into());
        let state = f.predict(state);
        let state = f.update(state, upd_bbox.into());
        let state = f.predict(state);

        let dists = f.distances(state, &vec![new_bbox_1.into(), new_bbox_2.into()], false);
        let dists = KalmanFilter::calculate_final_weights(dists, false);
        assert!(dists[0] >= 0.0 && dists[0] < chi2inv95()[1]);
        assert!(dists[1] > chi2inv95()[1]);

        let dists = f.distances(state, &vec![new_bbox_1.into(), new_bbox_2.into()], true);
        let dists = KalmanFilter::calculate_final_weights(dists, true);
        dbg!(&dists);
        assert!(dists[0] >= 0.0 && dists[0] < chi2inv95()[4]);
        assert!(dists[1] > chi2inv95()[4]);
    }
}
