use crate::store::{ObservationMetricErr, Results};
use crate::track::{ObservationAttributes, ObservationMetricOk};
use crossbeam::channel::Receiver;
use std::vec::IntoIter;

///
trait TrackDistanceResponse<OA>: IntoIterator
where
    OA: ObservationAttributes,
{
    type Output;
    fn get_all(&self) -> Vec<Self::Output> {
        let mut results = Vec::new();
        for _ in 0..self.count() {
            let res = self.channel().recv().unwrap();
            Self::extend(&mut results, self.elt(res));
        }
        results
    }

    fn count(&self) -> usize;
    fn elt(&self, res: Results<OA>) -> Vec<Self::Output>;
    fn extend(output: &mut Vec<Self::Output>, elt: Vec<Self::Output>);
    fn channel(&self) -> &Receiver<Results<OA>>;
}

///
pub struct TrackDistanceOk<OA>
where
    OA: ObservationAttributes,
{
    count: usize,
    channel: Receiver<Results<OA>>,
}

pub struct TrackDistanceOkIterator<OA>
where
    OA: ObservationAttributes,
{
    iterator_count: usize,
    channel: Receiver<Results<OA>>,
    current_chunk: IntoIter<ObservationMetricOk<OA>>,
}

pub struct TrackDistanceErrIterator<OA>
where
    OA: ObservationAttributes,
{
    iterator_count: usize,
    channel: Receiver<Results<OA>>,
    current_chunk: IntoIter<ObservationMetricErr<OA>>,
}

impl<OA> TrackDistanceOk<OA>
where
    OA: ObservationAttributes,
{
    pub fn all(self) -> Vec<ObservationMetricOk<OA>> {
        self.get_all()
    }

    pub(crate) fn new(count: usize, channel: Receiver<Results<OA>>) -> Self {
        Self { count, channel }
    }
}

///
pub struct TrackDistanceErr<OA>
where
    OA: ObservationAttributes,
{
    count: usize,
    channel: Receiver<Results<OA>>,
}

impl<OA> TrackDistanceErr<OA>
where
    OA: ObservationAttributes,
{
    pub fn all(self) -> Vec<ObservationMetricErr<OA>> {
        self.get_all()
    }

    pub(crate) fn new(count: usize, channel: Receiver<Results<OA>>) -> Self {
        Self { count, channel }
    }
}

impl<OA> Iterator for TrackDistanceOkIterator<OA>
where
    OA: ObservationAttributes,
{
    type Item = ObservationMetricOk<OA>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let elt = self.current_chunk.next();
            if elt.is_some() {
                return elt;
            } else if self.iterator_count == 0 {
                return None;
            } else {
                self.iterator_count -= 1;
                let elt = self.channel.recv().unwrap();
                match elt {
                    Results::DistanceOk(elt) => {
                        self.current_chunk = elt.into_iter();
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<OA> Iterator for TrackDistanceErrIterator<OA>
where
    OA: ObservationAttributes,
{
    type Item = ObservationMetricErr<OA>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let elt = self.current_chunk.next();
            if elt.is_some() {
                return elt;
            } else if self.iterator_count == 0 {
                return None;
            } else {
                self.iterator_count -= 1;
                let elt = self.channel.recv().unwrap();
                match elt {
                    Results::DistanceErr(elt) => {
                        self.current_chunk = elt.into_iter();
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<OA> IntoIterator for TrackDistanceOk<OA>
where
    OA: ObservationAttributes,
{
    type Item = ObservationMetricOk<OA>;
    type IntoIter = TrackDistanceOkIterator<OA>;

    fn into_iter(self) -> Self::IntoIter {
        TrackDistanceOkIterator {
            iterator_count: self.count,
            channel: self.channel,
            current_chunk: Vec::default().into_iter(),
        }
    }
}

impl<OA> IntoIterator for TrackDistanceErr<OA>
where
    OA: ObservationAttributes,
{
    type Item = ObservationMetricErr<OA>;
    type IntoIter = TrackDistanceErrIterator<OA>;

    fn into_iter(self) -> Self::IntoIter {
        TrackDistanceErrIterator {
            iterator_count: self.count,
            channel: self.channel,
            current_chunk: Vec::default().into_iter(),
        }
    }
}

impl<OA> TrackDistanceResponse<OA> for TrackDistanceOk<OA>
where
    OA: ObservationAttributes,
{
    type Output = ObservationMetricOk<OA>;

    fn count(&self) -> usize {
        self.count
    }

    fn elt(&self, res: Results<OA>) -> Vec<Self::Output> {
        match res {
            Results::DistanceOk(r) => r,
            _ => {
                unreachable!();
            }
        }
    }

    fn extend(output: &mut Vec<Self::Output>, elt: Vec<Self::Output>) {
        output.extend_from_slice(&elt);
    }

    fn channel(&self) -> &Receiver<Results<OA>> {
        &self.channel
    }
}

impl<OA> TrackDistanceResponse<OA> for TrackDistanceErr<OA>
where
    OA: ObservationAttributes,
{
    type Output = ObservationMetricErr<OA>;

    fn count(&self) -> usize {
        self.count
    }

    fn elt(&self, res: Results<OA>) -> Vec<Self::Output> {
        match res {
            Results::DistanceErr(r) => r,
            _ => {
                unreachable!();
            }
        }
    }
    fn extend(output: &mut Vec<Self::Output>, elt: Vec<Self::Output>) {
        output.extend(elt);
    }

    fn channel(&self) -> &Receiver<Results<OA>> {
        &self.channel
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::examples::vec2;
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
    use crate::track::{
        MetricOutput, MetricQuery, NoopLookup, Observation, ObservationAttributes,
        ObservationMetric, ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate,
        TrackStatus,
    };
    use anyhow::Result;

    #[derive(Debug, Clone, Default)]
    struct MockAttrs;

    #[derive(Default, Clone)]
    pub struct MockAttrsUpdate;

    impl TrackAttributesUpdate<MockAttrs> for MockAttrsUpdate {
        fn apply(&self, _attrs: &mut MockAttrs) -> Result<()> {
            Ok(())
        }
    }

    impl TrackAttributes<MockAttrs, f32> for MockAttrs {
        type Update = MockAttrsUpdate;
        type Lookup = NoopLookup<MockAttrs, f32>;

        fn compatible(&self, _other: &MockAttrs) -> bool {
            true
        }

        fn merge(&mut self, _other: &MockAttrs) -> Result<()> {
            Ok(())
        }

        fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
            Ok(TrackStatus::Ready)
        }
    }

    #[derive(Default, Clone)]
    pub struct MockMetric;

    impl ObservationMetric<MockAttrs, f32> for MockMetric {
        fn metric(&self, mq: &MetricQuery<MockAttrs, f32>) -> MetricOutput<f32> {
            let (e1, e2) = (mq.candidate_observation, mq.track_observation);
            Some((
                f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
                match (e1.feature().as_ref(), e2.feature().as_ref()) {
                    (Some(x), Some(y)) => Some(euclidean(x, y)),
                    _ => None,
                },
            ))
        }

        fn optimize(
            &mut self,
            _feature_class: u64,
            _merge_history: &[u64],
            _attrs: &mut MockAttrs,
            _features: &mut Vec<Observation<f32>>,
            _prev_length: usize,
            _is_merge: bool,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn result_iterators() {
        let mut store = TrackStoreBuilder::default()
            .default_attributes(MockAttrs)
            .metric(MockMetric)
            .notifier(NoopNotifier)
            .build();
        const N: usize = 10000;
        for _ in 0..N {
            let t = store
                .new_track_random_id()
                .observation(
                    ObservationBuilder::new(0)
                        .observation(vec2(1.0, 0.0))
                        .build(),
                )
                .build()
                .unwrap();
            store.add_track(t).unwrap();
        }

        let t1: Track<MockAttrs, MockMetric, f32> = store
            .new_track_random_id()
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.0, 0.0))
                    .build(),
            )
            .build()
            .unwrap();

        let t2: Track<MockAttrs, MockMetric, f32> = store
            .new_track_random_id()
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(-1.0, 0.0))
                    .build(),
            )
            .build()
            .unwrap();

        let (dists, errs) = store.foreign_track_distances(vec![t1.clone(), t2.clone()], 0, false);
        assert!(errs.all().is_empty());
        assert_eq!(dists.all().len(), 2 * N);

        let (dists, errs) = store.foreign_track_distances(vec![t1.clone(), t2.clone()], 0, false);
        assert!(errs.into_iter().next().is_none());
        assert_eq!(dists.into_iter().count(), 2 * N);

        let (dists, errs) = store.foreign_track_distances(vec![t1, t2], 0, false);
        drop(store);
        drop(dists);
        drop(errs);
    }
}
