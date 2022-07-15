#![feature(test)]

extern crate test;

use anyhow::Result;
use itertools::Itertools;
use similari::store::TrackStore;
use similari::test_stuff::{current_time_ms, BBox, BoxGen2};
use similari::track::{
    ObservationAttributes, ObservationMetric, ObservationMetricResult, ObservationSpec,
    ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVotingElt;
use similari::voting::Voting;
use std::sync::Arc;
use test::Bencher;

const FEAT0: u64 = 0;

#[derive(Debug, Clone, Default)]
struct BBoxAttributes {
    bboxes: Vec<BBox>,
}

#[derive(Clone, Debug)]
struct BBoxAttributesUpdate;

impl TrackAttributesUpdate<BBoxAttributes> for BBoxAttributesUpdate {
    fn apply(&self, _attrs: &mut BBoxAttributes) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<BBoxAttributes, BBox> for BBoxAttributes {
    fn compatible(&self, _other: &BBoxAttributes) -> bool {
        true
    }

    fn merge(&mut self, other: &BBoxAttributes) -> Result<()> {
        self.bboxes.extend_from_slice(&other.bboxes);
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<BBox>) -> Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone)]
pub struct IOUMetric {
    history: usize,
}

impl Default for IOUMetric {
    fn default() -> Self {
        Self { history: 3 }
    }
}

impl ObservationMetric<BBoxAttributes, BBox> for IOUMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &BBoxAttributes,
        _attrs2: &BBoxAttributes,
        e1: &ObservationSpec<BBox>,
        e2: &ObservationSpec<BBox>,
    ) -> (Option<f32>, Option<f32>) {
        (BBox::calculate_metric_object(&e1.0, &e2.0), None)
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        attrs: &mut BBoxAttributes,
        features: &mut Vec<ObservationSpec<BBox>>,
        prev_length: usize,
        is_merge: bool,
    ) -> Result<()> {
        if !is_merge {
            if let Some(bb) = &features[prev_length].0 {
                attrs.bboxes.push(bb.clone());
            }
        }
        // Kalman filter should be used here to generate better prediction for next
        // comparison
        features.reverse();
        features.truncate(self.history);
        features.reverse();
        Ok(())
    }
}

pub struct TopNVoting {
    topn: usize,
    min_distance: f32,
    min_votes: usize,
}

impl Voting<TopNVotingElt, f32> for TopNVoting {
    fn winners(&self, distances: &[ObservationMetricResult<f32>]) -> Vec<TopNVotingElt> {
        let mut tracks: Vec<_> = distances
            .iter()
            .filter(
                |ObservationMetricResult(_, f_attr_dist, _)| match f_attr_dist {
                    Some(e) => *e >= self.min_distance,
                    _ => false,
                },
            )
            .map(|ObservationMetricResult(track, _, _)| track)
            .collect();
        tracks.sort_unstable();
        let mut counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|(e, c)| TopNVotingElt {
                track_id: *e,
                votes: c,
            })
            .collect::<Vec<_>>();

        counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
        counts.truncate(self.topn);
        counts
    }
}

#[bench]
fn bench_iou_0010(b: &mut Bencher) {
    bench_iou(10, b);
}

#[bench]
fn bench_iou_0100(b: &mut Bencher) {
    bench_iou(100, b);
}

#[bench]
fn bench_iou_1000(b: &mut Bencher) {
    bench_iou(1000, b);
}

fn bench_iou(objects: usize, b: &mut Bencher) {
    let mut store: TrackStore<BBoxAttributes, BBoxAttributesUpdate, IOUMetric, BBox> =
        TrackStore::new(None, None, None, num_cpus::get());

    let voting = TopNVoting {
        topn: 1,
        min_distance: 0.5,
        min_votes: 1,
    };

    let pos_drift = 1.0;
    let box_drift = 1.0;
    let mut iterators = Vec::default();
    for i in 0..objects {
        iterators.push(BoxGen2::new(
            10.0 * i as f32,
            100.0 * i as f32,
            10.0,
            15.0,
            pos_drift,
            box_drift,
        ))
    }

    let mut iteration = 0;
    b.iter(|| {
        iteration += 1;
        for (c, i) in &mut iterators.iter_mut().enumerate() {
            let b = i.next();
            let mut t: Track<BBoxAttributes, IOUMetric, BBoxAttributesUpdate, BBox> = Track::new(
                u64::try_from(current_time_ms() * iteration + c as u128).unwrap(),
                None,
                None,
                None,
            );

            t.add_observation(FEAT0, b, None, Some(BBoxAttributesUpdate))
                .unwrap();

            let search_track = Arc::new(t.clone());
            let (dists, errs) = store.foreign_track_distances(search_track, FEAT0, false, None);
            assert!(errs.is_empty());
            let winners = voting.winners(&dists);
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                store
                    .merge_external(winners[0].track_id, &t, None, false)
                    .unwrap();
            }
        }
    });
}
