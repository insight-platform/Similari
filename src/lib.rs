//!
//! # Similari
//!
//! The purpose of crate is to provide tools to build vector embedded im-memory similarity engines.
//! Similarity calculation is the important resource demanding task in machine learning and AI systems.
//!
//! Vectors in similarity engines are compared by calculating of n-dimensional distance - Euclidian, Cosine or another one.
//! The distance is used to estimate how the vectors are close between each other.
//!
//! The library helps building various kinds of similarity engines - the simplest one is that holds vector features and allows compare
//! new vectors against the ones kept in the database. More sophisticated engines operates over tracks - series of observations for the
//! same feature collected during the lifecycle. Such kind of systems are often used in video processing or other class of systems where
//! observer receives fuzzy or changing observation results.
//!
//! The crate provides the tools to gather tracks build track storages, find similar tracks, and merge them. The crate doesn't provide
//! any persistence layer yet.
//!
//! ## Performance
//!
//! To keep the calculations performant the crate uses:
//! * [rayon](https://docs.rs/rayon/latest/rayon/) - parallel calculations are implemented within track storage operations;
//! * [nalgebra](https://nalgebra.org/) - fast linear algebra library.
//!
//! **The performance depends a lot of the optimization level defined for build. On lower or default optimization levels Rust
//! may not use vectorized optimizations, so when running benchmarks take care of proper optimization levels configured.**

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
