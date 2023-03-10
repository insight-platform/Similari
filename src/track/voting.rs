pub mod best;
pub mod topn;

use crate::track::{ObservationAttributes, ObservationMetricOk};
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
    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<Self::WinnerObject>>
    where
        T: IntoIterator<Item = ObservationMetricOk<OA>>;
}
