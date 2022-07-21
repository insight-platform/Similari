use crate::track::{
    MetricOutput, NoopLookup, ObservationAttributes, ObservationMetric, ObservationSpec,
    ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::BBox;
use anyhow::Result;

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
    type Lookup = NoopLookup<BBoxAttributes, BBox>;

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
