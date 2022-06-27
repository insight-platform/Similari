pub mod track;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum Errors {
    #[error("Attributes are incompatible to be merged.")]
    IncompatibleAttributes,
    #[error("Requested observations are missing - distance cannot be calculated.")]
    MissingObservation,
    #[error("Missing track.")]
    MissingTrack,
    #[error("Distance with self must not be used")]
    SelfDistanceCalculation,
}

pub(crate) const EPS: f32 = 0.00001;
