pub mod topn;

use crate::track::{ObservationAttributes, ObservationMetricResult};

/// Trait to implement distance voting engines.
///
/// Distance voting engine is used to select winning tracks among distances
/// resulted from the distance calculation.
///
pub trait Voting<R, FA>
where
    FA: ObservationAttributes,
{
    /// Method that selects winning tracks
    ///
    ///
    /// # Arguments
    /// * `distances` - distances resulted from the distance calculation.
    ///   * `.0` is the track_id
    ///   * `.1` is the distance
    ///
    fn winners(&self, distances: &[ObservationMetricResult<FA::MetricObject>]) -> Vec<R>;
}
