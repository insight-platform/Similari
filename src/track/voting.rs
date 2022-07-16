pub mod topn;

use crate::track::{ObservationAttributes, ObservationMetricResult};
use std::collections::HashMap;

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
    fn winners(
        &self,
        distances: &[ObservationMetricResult<FA::MetricObject>],
    ) -> HashMap<u64, Vec<R>>;
}
