//!
//! # Similari
//!
//! The purpose of the crate is to provide tools to config in-memory vector (feature) similarity engines.
//! Similarity calculation is an important resource demanding task broadly used in machine learning and AI systems.
//! Read more about it at GitHub [README](https://github.com/insight-platform/Similari/blob/main/README.md)
//!

/// Holds auxiliary functions that calculate distances between two features.
///
pub mod distance;

/// Various auxiliary testing and example components
///
pub mod examples;

/// Frequently used components
///
pub mod prelude;

/// Holds basic abstractions for tracking - [Track](track::Track), auxiliary structures, traits, and functions. It defines the track's
/// look and feel, provides `Track` structure that holds track attributes and features, can accumulate track features and
/// calculate feature distances between pair of tracks.
///
pub mod track;

/// Ready-to-use trackers - SORT
///
pub mod trackers;

/// Utility objects - bounding boxes, kalman filter
///
pub mod utils;

pub use track::store;
pub use track::voting;

use thiserror::Error;

/// Errors
#[derive(Error, Debug, Clone)]
pub enum Errors {
    /// Compared tracks have incompatible attributes, so cannot be used in calculations.
    #[error("Attributes are incompatible between tracks and cannot be used in calculations.")]
    IncompatibleAttributes,
    /// One of tracks doesn't have features for specified class
    ///
    #[error("Requested observations for class={2} are missing in track={0} or track={1} - distance cannot be calculated.")]
    ObservationForClassNotFound(u64, u64, u64),
    /// Requested track is not found in the store
    ///
    #[error("Missing track={0}.")]
    TrackNotFound(u64),

    #[error("Missing requested tracks.")]
    TracksNotFound,

    /// The distance is calculated against self. Ignore it.
    ///
    #[error("Calculation with self id={0} not permitted")]
    SameTrackCalculation(u64),

    /// Track ID is duplicate
    ///
    #[error("Duplicate track id={0}")]
    DuplicateTrackId(u64),
}

pub const EPS: f32 = 0.00001;

/// Trait is used to implement fuzzy epsilon-comparison of two objects
///
pub trait EstimateClose {
    fn almost_same(&self, other: &Self, eps: f32) -> bool;
}
