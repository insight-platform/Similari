use crate::track::{
    MetricOutput, MetricQuery, NoopLookup, Observation, ObservationAttributes, ObservationMetric,
    ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::BoundingBox;
use anyhow::Result;

#[derive(Debug, Clone, Default)]
pub struct BBoxAttributes {
    pub bboxes: Vec<BoundingBox>,
}

#[derive(Clone, Debug)]
pub struct BBoxAttributesUpdate;

impl TrackAttributesUpdate<BBoxAttributes> for BBoxAttributesUpdate {
    fn apply(&self, _attrs: &mut BBoxAttributes) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<BBoxAttributes, BoundingBox> for BBoxAttributes {
    type Update = BBoxAttributesUpdate;
    type Lookup = NoopLookup<BBoxAttributes, BoundingBox>;

    fn compatible(&self, _other: &BBoxAttributes) -> bool {
        true
    }

    fn merge(&mut self, other: &BBoxAttributes) -> Result<()> {
        self.bboxes.extend_from_slice(&other.bboxes);
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<BoundingBox>) -> Result<TrackStatus> {
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

impl ObservationMetric<BBoxAttributes, BoundingBox> for IOUMetric {
    fn metric(&self, mq: &MetricQuery<'_, BBoxAttributes, BoundingBox>) -> MetricOutput<f32> {
        let (e1, e2) = (mq.candidate_observation, mq.track_observation);
        let box_m_opt =
            BoundingBox::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref());
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
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut BBoxAttributes,
        features: &mut Vec<Observation<BoundingBox>>,
        prev_length: usize,
        is_merge: bool,
    ) -> Result<()> {
        if !is_merge {
            if let Some(bb) = &features[prev_length].attr() {
                attrs.bboxes.push(*bb);
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
