use crate::store::TrackStore;
use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{ObservationAttributes, ObservationMetric, TrackAttributes};
use std::marker::PhantomData;

/// Builder for TrackStore
///
pub struct TrackStoreBuilder<TA, M, OA, N = NoopNotifier>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    metric: Option<M>,
    default_attributes: Option<TA>,
    notifier: Option<N>,
    shards: usize,
    _phantom_oa: PhantomData<OA>,
}

/// Default builder
/// shards count is set to number cpu cores (threads)
///
impl<TA, M, OA, N> Default for TrackStoreBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    fn default() -> TrackStoreBuilder<TA, M, OA, N> {
        Self::new(num_cpus::get())
    }
}

impl<TA, M, OA, N> TrackStoreBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    /// Creates a new builder
    ///
    /// # Parameters
    /// * `shards` - number of shards to use (the parameter by default is equal to cores count), should be chosen experimentally, depending on tracker kind.
    ///
    pub fn new(shards: usize) -> Self {
        TrackStoreBuilder {
            shards,
            metric: None,
            default_attributes: None,
            notifier: None,
            _phantom_oa: PhantomData,
        }
    }

    /// Sets the metric object to use
    ///
    pub fn metric(mut self, metric: M) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Sets the notifier object to use
    ///
    pub fn notifier(mut self, notifier: N) -> Self {
        self.notifier = Some(notifier);
        self
    }

    /// Sets the default track attributes to use for new tracks
    ///
    pub fn default_attributes(mut self, attrs: TA) -> Self {
        self.default_attributes = Some(attrs);
        self
    }

    /// Builds the TrackStore
    ///
    pub fn build(self) -> TrackStore<TA, M, OA, N> {
        TrackStore::new(
            self.metric,
            self.default_attributes,
            self.notifier,
            self.shards,
        )
    }
}
