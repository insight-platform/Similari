use crate::track::notify::{ChangeNotifier, NoopNotifier};
use crate::track::{Observation, ObservationAttributes, ObservationMetric, Track, TrackAttributes};
use anyhow::Result;
use rand::Rng;

type TrackBuilderObservationRepr<OA, TAU> = (u64, Option<OA>, Option<Observation>, Option<TAU>);

pub struct ObservationBuilder<TAU, OA>
where
    OA: ObservationAttributes,
{
    cls: u64,
    observation_attributes: Option<OA>,
    observation: Option<Observation>,
    track_attributes_update: Option<TAU>,
}

impl<TAU, OA> ObservationBuilder<TAU, OA>
where
    OA: ObservationAttributes,
{
    pub fn new(cls: u64) -> Self {
        Self {
            cls,
            observation_attributes: None,
            observation: None,
            track_attributes_update: None,
        }
    }

    pub fn observation_attributes(mut self, attrs: OA) -> Self {
        self.observation_attributes = Some(attrs);
        self
    }

    pub fn observation(mut self, observation: Observation) -> Self {
        self.observation = Some(observation);
        self
    }

    pub fn track_attributes_update(mut self, upd: TAU) -> Self {
        self.track_attributes_update = Some(upd);
        self
    }

    pub fn build(self) -> TrackBuilderObservationRepr<OA, TAU> {
        (
            self.cls,
            self.observation_attributes,
            self.observation,
            self.track_attributes_update,
        )
    }
}

pub struct TrackBuilder<TA, M, OA, N = NoopNotifier>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    id: Option<u64>,
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
        TrackBuilder::new()
    }
}

impl<TA, M, OA, N> TrackBuilder<TA, M, OA, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
{
    pub fn new() -> TrackBuilder<TA, M, OA, N> {
        Self {
            id: None,
            track_attrs: None,
            metric: None,
            notifier: None,
            observations: Vec::new(),
        }
    }

    pub fn id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
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
        let id = if let Some(id) = self.id {
            id
        } else {
            let mut rng = rand::thread_rng();
            rng.gen::<u64>()
        };

        let mut track = Track::new(id, self.metric, self.track_attrs, self.notifier);
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
        let track = TrackBuilder::new()
            .id(10)
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
        let track = TrackBuilder::new()
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
