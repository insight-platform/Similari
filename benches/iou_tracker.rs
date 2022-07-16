#![feature(test)]

extern crate test;

use anyhow::Result;
use itertools::Itertools;
use similari::store::TrackStore;
use similari::test_stuff::{BBox, BoxGen2};
use similari::track::{
    ObservationAttributes, ObservationMetric, ObservationMetricResult, ObservationSpec,
    ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVotingElt;
use similari::voting::Voting;
use std::collections::HashMap;
use std::time::Instant;
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
        Self { history: 1 }
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
    fn winners(
        &self,
        distances: &[ObservationMetricResult<f32>],
    ) -> HashMap<u64, Vec<TopNVotingElt>> {
        let mut tracks: Vec<_> = distances
            .iter()
            .filter(
                |ObservationMetricResult {
                     from: _,
                     to: _track,
                     attribute_metric: attr_dist,
                     feature_distance: _,
                 }| match attr_dist {
                    Some(e) => *e >= self.min_distance,
                    _ => false,
                },
            )
            .map(
                |ObservationMetricResult {
                     from: src_track,
                     to: dest_track,
                     attribute_metric: _,
                     feature_distance: _,
                 }| (src_track, dest_track),
            )
            .collect();
        tracks.sort_unstable();

        let counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|((q, w), c)| TopNVotingElt {
                query_track: *q,
                winner_track: *w,
                votes: c,
            })
            .collect::<Vec<_>>();

        let mut results: HashMap<u64, Vec<TopNVotingElt>> = HashMap::new();

        for c in counts {
            let key = c.query_track;
            if let Some(val) = results.get_mut(&key) {
                val.push(c);
            } else {
                results.insert(key, vec![c]);
            }
        }

        for (_query_track, counts) in &mut results {
            counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
            counts.truncate(self.topn);
        }

        results
    }
}

#[bench]
fn bench_iou_00010(b: &mut Bencher) {
    bench_iou(10, b);
}

#[bench]
fn bench_iou_00100(b: &mut Bencher) {
    bench_iou(100, b);
}

#[bench]
fn bench_iou_01000(b: &mut Bencher) {
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
        let mut tracks = Vec::new();
        for i in &mut iterators {
            iteration += 1;
            let b = i.next();
            let mut t: Track<BBoxAttributes, IOUMetric, BBoxAttributesUpdate, BBox> =
                Track::new(iteration, None, None, None);

            t.add_observation(FEAT0, b, None, Some(BBoxAttributesUpdate))
                .unwrap();
            tracks.push(t);
        }

        let search_tracks = tracks.clone();
        let (dists, errs) = store.foreign_track_distances(search_tracks, FEAT0, false, None);
        assert!(errs.is_empty());
        let tm = Instant::now();
        let mut winners = voting.winners(&dists);
        for t in tracks {
            let winner = winners.get(&t.get_track_id());
            if let Some(winners) = winner {
                store
                    .merge_external(winners[0].winner_track, &t, None, false)
                    .unwrap();
            } else {
                store.add_track(t);
            }
        }

        let elapsed = tm.elapsed();
        eprintln!("Voting time: {:?}", elapsed.as_micros());
    });
}
