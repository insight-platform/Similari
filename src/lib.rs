/// Holds auxiliary functions that calculate distances between two features.
///
pub mod distance;

/// Holds basic abstractions for tracking - [Track](track::Track), auxiliary structures, traits, and functions. It defines the track's
/// look and feel, provides `Track` structure that holds track attributes and features, can accumulate track features and
/// calculate feature distances between pair of tracks.
///
pub mod track;

pub use track::store as db;

use thiserror::Error;

/// Errors
#[derive(Error, Debug, Clone)]
pub enum Errors {
    /// Compared tracks have incompatible attributes, so cannot be used in calculations.
    #[error("Attributes are incompatible between tracks and cannot be used in calculations.")]
    IncompatibleAttributes,
    /// One of tracks doesn't have features for specified class
    ///
    #[error("Requested observations for class={2}= are missing in track={0} or track={1} - distance cannot be calculated.")]
    ObservationForClassNotFound(u64, u64, u64),
    /// Requested track is not found in the store
    ///
    #[error("Missing track={0}.")]
    TrackNotFound(u64),
    /// The distance is calculated against self. Ignore it.
    ///
    #[error("Distance with self must not be used")]
    SelfDistanceCalculation,
}

#[cfg(test)]
const EPS: f32 = 0.00001;
