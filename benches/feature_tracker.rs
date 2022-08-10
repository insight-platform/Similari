#![feature(test)]

extern crate test;

use similari::distance::euclidean;
use similari::examples::FeatGen;
use similari::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
use similari::track::{
    MetricOutput, MetricQuery, NoopLookup, Observation, ObservationMetric, ObservationMetricOk,
    ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;
use std::time::Instant;
use test::Bencher;

const FEAT0: u64 = 0;

#[derive(Debug, Clone, Default)]
struct NoopAttributes;

#[derive(Clone, Debug)]
struct NoopAttributesUpdate;

impl TrackAttributesUpdate<NoopAttributes> for NoopAttributesUpdate {
    fn apply(&self, _attrs: &mut NoopAttributes) -> anyhow::Result<()> {
        Ok(())
    }
}

impl TrackAttributes<NoopAttributes, ()> for NoopAttributes {
    type Update = NoopAttributesUpdate;
    type Lookup = NoopLookup<NoopAttributes, ()>;

    fn compatible(&self, _other: &NoopAttributes) -> bool {
        true
    }

    fn merge(&mut self, _other: &NoopAttributes) -> anyhow::Result<()> {
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<()>) -> anyhow::Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone, Default)]
pub struct TrackMetric;

impl ObservationMetric<NoopAttributes, ()> for TrackMetric {
    fn metric(&self, mq: &MetricQuery<NoopAttributes, ()>) -> MetricOutput<()> {
        let (e1, e2) = (mq.candidate_observation, mq.track_observation);
        Some((
            None,
            match (e1.feature(), e2.feature()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        _attrs: &mut NoopAttributes,
        observations: &mut Vec<Observation<()>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> anyhow::Result<()> {
        observations.reverse();
        observations.truncate(3);
        observations.reverse();
        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<()>>,
    ) -> Vec<ObservationMetricOk<()>> {
        unfiltered
            .into_iter()
            .filter(|x| {
                if let Some(d) = x.feature_distance {
                    d < 100.0
                } else {
                    false
                }
            })
            .collect()
    }
}

fn benchmark(objects: usize, flen: usize, b: &mut Bencher) {
    let ncores = match objects {
        10 => 1,
        100 => 2,
        _ => num_cpus::get(),
    };

    let mut store = TrackStoreBuilder::new(ncores)
        .metric(TrackMetric::default())
        .default_attributes(NoopAttributes::default())
        .notifier(NoopNotifier)
        .build();

    let voting: TopNVoting<()> = TopNVoting::new(1, 100.0, 1);

    let pos_drift = 0.1;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(FeatGen::new(1000.0 * i as f32, flen, pos_drift));
    }

    let mut iteration = 0;
    b.iter(|| {
        let mut tracks = Vec::new();
        let tm = Instant::now();
        for i in &mut iterators {
            iteration += 1;
            let b = i.next().unwrap().feature().clone();
            let t = store
                .new_track(iteration)
                .observation(
                    ObservationBuilder::new(FEAT0)
                        .observation(b.unwrap())
                        .build(),
                )
                .build()
                .unwrap();
            tracks.push(t);
        }

        let search_tracks = tracks.clone();
        let elapsed = tm.elapsed();
        eprintln!("Construction time: {:?}", elapsed);

        let tm = Instant::now();
        let (dists, errs) = store.foreign_track_distances(search_tracks, FEAT0, false);
        let elapsed = tm.elapsed();
        eprintln!("Lookup time: {:?}", elapsed);

        let tm = Instant::now();
        let winners = voting.winners(dists);
        let elapsed = tm.elapsed();
        eprintln!("Voting time: {:?}", elapsed);
        assert!(errs.all().is_empty());

        let tm = Instant::now();
        for t in tracks {
            let winners_opt = winners.get(&t.get_track_id());
            if let Some(winners) = winners_opt {
                let _res = store
                    .merge_external_noblock(winners[0].winner_track, t, None, false)
                    .unwrap();
            } else {
                store.add_track(t).unwrap();
            }
        }
        let elapsed = tm.elapsed();
        eprintln!("Merging time: {:?}", elapsed);
        eprintln!("Store stats: {:?}", store.shard_stats());
    });
}

#[bench]
fn ft_0010_256x3(b: &mut Bencher) {
    benchmark(10, 256, b);
}

#[bench]
fn ft_0100_256x3(b: &mut Bencher) {
    benchmark(100, 256, b);
}

#[bench]
fn ft_0500_256x3(b: &mut Bencher) {
    benchmark(500, 256, b);
}
