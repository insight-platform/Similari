use crate::prelude::{BoundingBox, Universal2DBox};
use crate::Errors;
use nalgebra::{SMatrix, SVector};

/// Kalman filter for the prediction of axis-aligned and oriented bounding boxes
///
pub mod kalman_2d_box;
/// Kalman filter for 2d point
///
pub mod kalman_2d_point;
/// Kalman filter for Vector of 2d points
///
pub mod kalman_2d_point_vec;

pub const CHI2_UPPER_BOUND: f32 = 100.0;

pub const CHI2INV95: [f32; 9] = [
    3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919,
];

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
pub struct KalmanState<const X: usize> {
    mean: SVector<f32, X>,
    covariance: SMatrix<f32, X, X>,
}

impl<const X: usize> KalmanState<X> {
    /// dump the state
    ///
    pub fn dump(&self) {
        eprintln!("Mean={}", pretty_print!(self.mean.transpose()));
        eprintln!("Covariance={}", pretty_print!(self.covariance));
    }
}

impl<const X: usize> TryFrom<KalmanState<X>> for Universal2DBox {
    type Error = Errors;

    fn try_from(value: KalmanState<X>) -> Result<Self, Self::Error> {
        if value.mean.len() < 5 {
            Err(Self::Error::OutOfRange)
        } else {
            Ok(Universal2DBox::new(
                value.mean[0],
                value.mean[1],
                if value.mean[2] == 0.0 {
                    None
                } else {
                    Some(value.mean[2])
                },
                value.mean[3],
                value.mean[4],
            ))
        }
    }
}

impl<const X: usize> TryFrom<KalmanState<X>> for BoundingBox {
    type Error = Errors;

    fn try_from(value: KalmanState<X>) -> Result<Self, Self::Error> {
        let bb = Universal2DBox::try_from(value)?;
        BoundingBox::try_from(&bb)
    }
}

pub const DT: u64 = 1;
