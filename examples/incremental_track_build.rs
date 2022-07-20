use similari::distance::euclidean;
use similari::prelude::*;
use similari::test_stuff::{BBox, BoxGen2, FeatGen2};
use similari::track::{
    MetricOutput, ObservationAttributes, ObservationMetric, ObservationMetricResult,
    ObservationSpec, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;
use std::thread;
use std::time::Duration;

const FEAT0: u64 = 0;
const MAX_DIST: f32 = 0.1;

#[derive(Debug, Clone, Default)]
struct BBoxAttributes {
    bboxes: Vec<BBox>,
}

#[derive(Clone, Debug)]
struct BBoxAttributesUpdate {
    bbox: BBox,
}

impl TrackAttributesUpdate<BBoxAttributes> for BBoxAttributesUpdate {
    fn apply(&self, attrs: &mut BBoxAttributes) -> anyhow::Result<()> {
        attrs.bboxes.push(self.bbox.clone());
        Ok(())
    }
}

impl TrackAttributes<BBoxAttributes, f32> for BBoxAttributes {
    type Update = BBoxAttributesUpdate;

    fn compatible(&self, _other: &BBoxAttributes) -> bool {
        true
    }

    fn merge(&mut self, other: &BBoxAttributes) -> anyhow::Result<()> {
        self.bboxes.extend_from_slice(&other.bboxes);
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<f32>) -> anyhow::Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone, Default)]
pub struct TrackMetric;

impl ObservationMetric<BBoxAttributes, f32> for TrackMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &BBoxAttributes,
        _attrs2: &BBoxAttributes,
        e1: &ObservationSpec<f32>,
        e2: &ObservationSpec<f32>,
    ) -> MetricOutput<f32> {
        Some((
            f32::calculate_metric_object(&e1.0, &e2.0),
            match (e1.1.as_ref(), e2.1.as_ref()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _attrs: &mut BBoxAttributes,
        observations: &mut Vec<ObservationSpec<f32>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> anyhow::Result<()> {
        observations.reverse();
        observations.truncate(5);
        observations.reverse();
        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricResult<f32>>,
    ) -> Vec<ObservationMetricResult<f32>> {
        unfiltered
            .into_iter()
            .filter(|r| r.feature_distance.unwrap() < MAX_DIST)
            .collect()
    }
}

fn main() {
    let mut store = TrackStoreBuilder::<BBoxAttributes, TrackMetric, f32>::default().build();
    let voting: TopNVoting<f32> = TopNVoting::new(1, MAX_DIST, 1);
    let feature_drift = 0.01;
    let pos_drift = 5.0;
    let box_drift = 2.0;

    let mut p1 = FeatGen2::new(0.0, 0.0, feature_drift);
    let mut b1 = BoxGen2::new(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);

    let mut p2 = FeatGen2::new(1.0, 1.0, feature_drift);
    let mut b2 = BoxGen2::new(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let (obj1f, obj1b) = (p1.next().unwrap(), b1.next().unwrap());

        let obj1t = TrackBuilder::default()
            .observation(
                ObservationBuilder::new(FEAT0)
                    .observation_attributes(obj1f.0.unwrap())
                    .observation(obj1f.1.unwrap())
                    .track_attributes_update(BBoxAttributesUpdate { bbox: obj1b })
                    .build(),
            )
            .build()
            .unwrap();

        let (obj2f, obj2b) = (p2.next().unwrap(), b2.next().unwrap());

        let obj2t = TrackBuilder::default()
            .observation(
                ObservationBuilder::new(FEAT0)
                    .observation_attributes(obj2f.0.unwrap())
                    .observation(obj2f.1.unwrap())
                    .track_attributes_update(BBoxAttributesUpdate { bbox: obj2b })
                    .build(),
            )
            .build()
            .unwrap();

        thread::sleep(Duration::from_millis(2));

        for t in [obj1t, obj2t] {
            let search_track = t.clone();
            let (dists, errs) = store.foreign_track_distances(vec![search_track], FEAT0, false);
            assert!(errs.is_empty());
            let winners = voting.winners(&dists);
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                store
                    .merge_external(
                        winners.get(&t.get_track_id()).unwrap()[0].winner_track,
                        &t,
                        None,
                        false,
                    )
                    .unwrap();
            }
        }
    }

    let tracks = store.find_usable();
    for (t, _) in tracks {
        let t = store.fetch_tracks(&vec![t]);
        eprintln!("Track id: {}", t[0].get_track_id());
        eprintln!("Boxes: {:#?}", t[0].get_attributes());
    }
}
