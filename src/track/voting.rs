pub mod topn;

use crate::track::{ObservationAttributes, ObservationMetricResult};
use std::collections::HashMap;

/// Trait to implement distance voting engines.
///
/// Distance voting engine is used to select winning tracks among distances
/// resulted from the distance calculation.
///
pub trait Voting<OA>
where
    OA: ObservationAttributes,
{
    type WinnerObject;
    /// Method that selects winning tracks
    ///
    ///
    /// # Arguments
    /// * `distances` - distances resulted from the distance calculation.
    ///
    /// # Return
    /// Map of track_ids -> Vec<Result>
    ///
    fn winners(
        &self,
        distances: &[ObservationMetricResult<OA>],
    ) -> HashMap<u64, Vec<Self::WinnerObject>>;
}
