use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{Observation, ObservationAttributes, ObservationMetric, Track, TrackAttributes};
use anyhow::Result;
use rand::Rng;

type TrackBuilderObservationRepr<OA, TAU> = (u64, Option<OA>, Option<Observation>, Option<TAU>);

/// Builder is used to build an observation
///
pub struct ObservationBuilder<TAU, OA>
where
    OA: ObservationAttributes,
{
    feature_class: u64,
    observation_attributes: Option<OA>,
    observation: Option<Observation>,
    track_attributes_update: Option<TAU>,
}

impl<TAU, OA> ObservationBuilder<TAU, OA>
where
    OA: ObservationAttributes,
{
    /// Constructor method
    ///
    /// The observation is created for a certain feature class
    ///
    /// # Parameters
    /// * `feature_cals` - feature class
    ///
    pub fn new(feature_class: u64) -> Self {
        Self {
            feature_class: feature_class,
            observation_attributes: None,
            observation: None,
            track_attributes_update: None,
        }
    }

    /// Sets observation custom attributes
    ///
    pub fn observation_attributes(mut self, attrs: OA) -> Self {
        self.observation_attributes = Some(attrs);
        self
    }

    /// Sets a unified feature vector for observation
    ///
    pub fn observation(mut self, observation: Observation) -> Self {
        self.observation = Some(observation);
        self
    }

    /// Sets the track attributes update, connected to the observation
    ///
    pub fn track_attributes_update(mut self, upd: TAU) -> Self {
        self.track_attributes_update = Some(upd);
        self
    }

    /// Builds observation tuple suitable for [TrackBuilder::observation](TrackBuilder::observation) method
    ///
    pub fn build(self) -> TrackBuilderObservationRepr<OA, TAU> {
        (
            self.feature_class,
            self.observation_attributes,
            self.observation,
            self.track_attributes_update,
        )
    }
}

/// Builder object for Track
///
pub struct TrackBuilder<TA, M, OA, N = NoopNotifier>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    id: u64,
    track_attrs: Option<TA>,
    metric: Option<M>,
    notifier: Option<N>,
    observations: Vec<TrackBuilderObservationRepr<OA, TA::Update>>,
}

impl<TA, M, OA, N> Default for TrackBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        TrackBuilder::new(rng.gen::<u64>())
    }
}

impl<TA, M, OA, N> TrackBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    /// Empty track constructor
    ///
    /// # Parameters
    /// * `id` - unique track ID
    ///
    pub fn new(id: u64) -> TrackBuilder<TA, M, OA, N> {
        Self {
            id,
            track_attrs: None,
            metric: None,
            notifier: None,
            observations: Vec::new(),
        }
    }

    pub fn track_attrs(mut self, track_attrs: TA) -> Self {
        self.track_attrs = Some(track_attrs);
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

    /// Sets additional observation. The method can be called multiple times to add several observations.
    ///
    /// # Parameters
    /// * `observation` is the tuple produced by [ObservationBuilder](ObservationBuilder)
    ///
    pub fn observation(mut self, observation: TrackBuilderObservationRepr<OA, TA::Update>) -> Self {
        let (feature_class, observation_attributes, observation, track_attributes_update) =
            observation;
        self.observations.push((
            feature_class,
            observation_attributes,
            observation,
            track_attributes_update,
        ));
        self
    }

    pub fn build(self) -> Result<Track<TA, M, OA, N>> {
        let mut track = Track::new(self.id, self.metric, self.track_attrs, self.notifier);
        for (cls, oa, feat, upd) in self.observations {
            track.add_observation(cls, oa, feat, upd)?;
        }
        Ok(track)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_stuff::{UnboundAttributeUpdate, UnboundAttrs, UnboundMetric};
    use crate::track::builder::{ObservationBuilder, TrackBuilder};
    use crate::track::notify::NoopNotifier;
    use crate::track::utils::FromVec;
    use crate::track::Observation;
    use anyhow::Result;

    #[test]
    fn builder_id() -> Result<()> {
        let track = TrackBuilder::new(10)
            .notifier(NoopNotifier)
            .metric(UnboundMetric)
            .track_attrs(UnboundAttrs)
            .observation(
                ObservationBuilder::new(0)
                    .observation(Observation::from_vec(vec![0.0, 1.0]))
                    .observation_attributes(0.1)
                    .track_attributes_update(UnboundAttributeUpdate)
                    .build(),
            )
            .build()?;
        assert_eq!(track.get_track_id(), 10);
        Ok(())
    }

    #[test]
    fn builder_noid() -> Result<()> {
        let track = TrackBuilder::default()
            .notifier(NoopNotifier)
            .metric(UnboundMetric)
            .track_attrs(UnboundAttrs)
            .observation(
                ObservationBuilder::new(0)
                    .observation(Observation::from_vec(vec![0.0, 1.0]))
                    .observation_attributes(0.1)
                    .track_attributes_update(UnboundAttributeUpdate)
                    .build(),
            )
            .build()?;
        assert!(track.get_track_id() > 0);
        Ok(())
    }
}
