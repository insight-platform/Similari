//!
//! # Similari
//!
//! The purpose of the crate is to provide tools to build in-memory vector (feature) similarity engines.
//! Similarity calculation is an important resource demanding task broadly used in machine learning and AI systems.
//!
//! Vectors (or features) in similarity engines are compared by calculation of n-dimensional distances - Euclidian, Cosine or another one.
//! The distance is used to estimate how the vectors are close between each other.
//!
//! The library helps building various kinds of similarity engines - the simplest one is that holds single vectors and supports comparing
//! a received vector versus the ones kept in the database. More sophisticated engines may operate with tracks - series of observations for the
//! same feature kinds collected during the object or phenomenon lifecycle. Such kind of systems are often used in video processing or other
//! systems where observer receives fuzzy, unstable or time-changing observation results.
//!
//! The crate provides the necessary primitives to gather tracks, build track storages, find similar tracks, and merge them. The crate doesn't provide
//! any persistence layer yet.
//!
//! ## Performance
//!
//! To provide state-of-art performance the crate stands on:
//! * [rayon](https://docs.rs/rayon/latest/rayon/) - most of track storage operations are parallelized calculations;
//! * [nalgebra](https://nalgebra.org/) - fast linear algebra library that uses simd optimization (and GPU acceleration, which is not used in Similari right now).
//!
//! The performance of `nalgebra` depends a lot of the optimization level defined for the build. When lower or default optimization levels in use
//! Rust may not use f32 vectorization, so the performance may be far from the perfect.
//!
//! When running benchmarks take care of proper optimization levels configured. Levels 2 and 3 will lead to best results.

/// Holds auxiliary functions that calculate distances between two features.
///
pub mod distance;

/// Holds basic abstractions for tracking - [Track](track::Track), auxiliary structures, traits, and functions. It defines the track's
/// look and feel, provides `Track` structure that holds track attributes and features, can accumulate track features and
/// calculate feature distances between pair of tracks.
///
pub mod track;

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
