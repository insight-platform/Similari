pub mod topn;

use anyhow::Result;

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
    fn winners(&self, distances: &[(u64, Result<f32>)]) -> Vec<R>;
}
