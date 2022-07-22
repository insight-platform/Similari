// Original source code idea from
// https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
//
use crate::utils::bbox::{AspectBBox, BBox};
use nalgebra::{SMatrix, SVector};

pub fn chi2inv95() -> [f32; 9] {
    [
        3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919,
    ]
}

const DIM: usize = 4;
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

pub struct State<const X: usize = DIM_X2> {
    mean: SVector<f32, X>,
    covariance: SMatrix<f32, X, X>,
}

impl<const X: usize> State<X> {
    pub fn bbox(&self) -> BBox {
        AspectBBox {
            x: self.mean[0],
            y: self.mean[1],
            aspect: self.mean[2],
            height: self.mean[3],
        }
        .into()
    }

    pub fn dump(&self) {
        eprintln!("Mean={}", pretty_print!(self.mean.transpose()));
        eprintln!("Covariance={}", pretty_print!(self.covariance));
    }
}

#[derive(Debug)]
pub struct KalmanFilter {
    motion_matrix: SMatrix<f32, DIM_X2, DIM_X2>,
    update_matrix: SMatrix<f32, DIM, DIM_X2>,
    std_position_weight: f32,
    std_velocity_weight: f32,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        KalmanFilter::new(1.0 / 20.0, 1.0 / 160.0)
    }
}

impl KalmanFilter {
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

    fn std_position(&self, k: f32, cnst: f32, p: f32) -> Vec<f32> {
        vec![
            k * self.std_position_weight * p,
            k * self.std_position_weight * p,
            cnst,
            k * self.std_position_weight * p,
        ]
    }

    fn std_velocity(&self, k: f32, cnst: f32, p: f32) -> Vec<f32> {
        vec![
            k * self.std_velocity_weight * p,
            k * self.std_velocity_weight * p,
            cnst,
            k * self.std_velocity_weight * p,
        ]
    }

    pub fn initiate(&mut self, bbox: AspectBBox) -> State<DIM_X2> {
        let mean: SVector<f32, DIM_X2> = SVector::from_vec(vec![
            bbox.x,
            bbox.y,
            bbox.aspect,
            bbox.height,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        let mut std: SVector<f32, DIM_X2> = SVector::from_vec(
            self.std_position(2.0, 1e-2, bbox.height)
                .into_iter()
                .chain(self.std_velocity(10.0, 1e-5, bbox.height).into_iter())
                .collect(),
        );

        std.apply(|x| {
            let v = *x;
            *x = v * v;
        });
        let covariance: SMatrix<f32, DIM_X2, DIM_X2> = SMatrix::from_diagonal(&std);
        State { mean, covariance }
    }

    pub fn predict(&self, state: State<DIM_X2>) -> State<DIM_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let std_pos = self.std_position(1.0, 1e-2, mean[3]);
        let std_vel = self.std_velocity(1.0, 1e-5, mean[3]);

        let mut std: SVector<f32, DIM_X2> =
            SVector::from_vec(std_pos.into_iter().chain(std_vel.into_iter()).collect());

        std.apply(|x| {
            let v = *x;
            *x = v * v;
        });

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
        let mut std: SVector<f32, DIM> = SVector::from_vec(self.std_position(1.0, 1e-1, mean[3]));

        std.apply(|x| {
            let v = *x;
            *x = v * v;
        });

        let innovation_cov: SMatrix<f32, DIM, DIM> = SMatrix::from_diagonal(&std);

        let mean = self.update_matrix * mean;
        let covariance =
            self.update_matrix * covariance * self.update_matrix.transpose() + innovation_cov;
        State { mean, covariance }
    }

    pub fn update(&self, state: State<DIM_X2>, measurement: AspectBBox) -> State<DIM_X2> {
        let (mean, covariance) = (state.mean, state.covariance);
        let projected_state = self.project(mean.clone(), covariance.clone());
        let (projected_mean, projected_cov) = (projected_state.mean, projected_state.covariance);
        let choletsky_factor = projected_cov.cholesky().unwrap().l();
        let kalman_gain = choletsky_factor
            .solve_lower_triangular(&(covariance * self.update_matrix.transpose()).transpose())
            .unwrap();

        let innovation = SVector::from_vec(vec![
            measurement.x,
            measurement.y,
            measurement.aspect,
            measurement.height,
        ]) - projected_mean;

        let innovation: SMatrix<f32, 1, DIM> = innovation.transpose();

        let mean = mean + (innovation * kalman_gain).transpose();
        let covariance = covariance - kalman_gain.transpose() * projected_cov * kalman_gain;
        State { mean, covariance }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::bbox::BBox;
    use crate::utils::kalman::KalmanFilter;

    /// convert an nalgebra array to a String

    #[test]
    fn constructor() {
        let mut f = KalmanFilter::default();
        let bbox = BBox {
            x: 1.0,
            y: 2.0,
            width: 5.0,
            height: 5.0,
        };

        let state = f.initiate(bbox.clone().into());
        let new_bb = state.bbox();
        assert_eq!(new_bb, bbox.clone());
    }

    #[test]
    fn step() {
        let mut f = KalmanFilter::default();
        let bbox = BBox {
            x: -10.0,
            y: 2.0,
            width: 2.0,
            height: 5.0,
        };

        let mut state = f.initiate(bbox.clone().into());
        state.dump();

        let int_state = f.project(state.mean, state.covariance);
        int_state.dump();

        for i in 0..50 {
            let bb = BBox {
                x: 1.2 + i as f32,
                y: 2.3 + i as f32,
                width: 15.1,
                height: 100.1,
            };
            let bb_xyah = bb.clone().into();
            dbg!(&bb_xyah);

            state = f.update(state, bb_xyah);
            state.dump();

            state = f.predict(state);
            eprintln!("Orig={:?}, Predict={:?}", bb, &state.bbox());
            state.dump();
        }

        let mut range: Vec<_> = (10..50).into_iter().collect();
        range.reverse();

        for i in range {
            let bb = BBox {
                x: 1.2 + i as f32,
                y: 2.3 + i as f32,
                width: 15.1,
                height: 20.1,
            };

            state = f.update(state, bb.clone().into());

            state = f.predict(state);
            eprintln!("Orig={:?}, Predict={:?}", bb, &state.bbox());
        }
    }
}
