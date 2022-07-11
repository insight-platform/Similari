pub mod topn;

use crate::track::FeatureDistance;

/// Trait to implement distance voting engines.
///
/// Distance voting engine is used to select winning tracks among distances
/// resulted from the distance calculation.
///
pub trait Voting<R> {
    /// Method that selects winning tracks
    ///
    ///
    /// # Arguments
    /// * `distances` - distances resulted from the distance calculation.
    ///   * `.0` is the track_id
    ///   * `.1` is the distance
    ///
    fn winners(&self, distances: &[FeatureDistance]) -> Vec<R>;
}
