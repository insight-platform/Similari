use crate::test_stuff::BBox;
use crate::track::{
    MetricOutput, ObservationAttributes, ObservationMetric, ObservationMetricResult,
    ObservationSpec, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::voting::topn::TopNVotingElt;
use crate::voting::Voting;
use anyhow::Result;
use itertools::Itertools;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct BBoxAttributes {
    pub bboxes: Vec<BBox>,
}

#[derive(Clone, Debug)]
pub struct BBoxAttributesUpdate;

impl TrackAttributesUpdate<BBoxAttributes> for BBoxAttributesUpdate {
    fn apply(&self, _attrs: &mut BBoxAttributes) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<BBoxAttributes, BBox> for BBoxAttributes {
    type Update = BBoxAttributesUpdate;

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
    ) -> MetricOutput<f32> {
        let box_m_opt = BBox::calculate_metric_object(&e1.0, &e2.0);
        if let Some(box_m) = &box_m_opt {
            if *box_m < 0.01 {
                None
            } else {
                Some((box_m_opt, None))
            }
        } else {
            None
        }
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
        // eprintln!(
        //     "Features: {:?}, attrs: {:?}",
        //     &features[0].0,
        //     &attrs.bboxes.len()
        // );
        Ok(())
    }
}

pub struct IOUTopNVoting {
    pub topn: usize,
    pub min_distance: f32,
    pub min_votes: usize,
}

impl Voting<BBox> for IOUTopNVoting {
    type WinnerObject = TopNVotingElt;

    fn winners(
        &self,
        distances: &[ObservationMetricResult<BBox>],
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

        for counts in results.values_mut() {
            counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
            counts.truncate(self.topn);
        }

        results
    }
}
