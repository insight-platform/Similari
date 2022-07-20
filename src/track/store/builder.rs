use crate::store::TrackStore;
use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{ObservationAttributes, ObservationMetric, TrackAttributes};
use std::marker::PhantomData;

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

impl<TA, M, OA, N> TrackStoreBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    pub fn new() -> Self {
        TrackStoreBuilder {
            shards: num_cpus::get(),
            metric: None,
            default_attributes: None,
            notifier: None,
            _phantom_oa: PhantomData,
        }
    }

    pub fn shards(mut self, shards: usize) -> Self {
        self.shards = shards;
        self
    }

    pub fn metric(mut self, metric: M) -> Self {
        self.metric = Some(metric);
        self
    }

    pub fn notifier(mut self, notifier: N) -> Self {
        self.notifier = Some(notifier);
        self
    }

    pub fn default_attributes(mut self, attrs: TA) -> Self {
        self.default_attributes = Some(attrs);
        self
    }

    pub fn build(self) -> TrackStore<TA, M, OA, N> {
        TrackStore::new(
            self.metric,
            self.default_attributes,
            self.notifier,
            self.shards,
        )
    }
}
