use crate::track::{
    MetricOutput, NoopLookup, ObservationAttributes, ObservationMetric, ObservationMetricOk,
    ObservationSpec, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::BBox;
use crate::utils::kalman::{KalmanFilter, State};
use anyhow::Result;

pub const DEFAULT_IOU_THRESHOLD: f32 = 0.3;

#[derive(Debug, Clone, Default)]
pub struct SortAttributes {
    pub bboxes: Vec<BBox>,
    pub state: Option<State>,
}

#[derive(Clone, Debug)]
pub struct SortAttributesUpdate;

impl TrackAttributesUpdate<SortAttributes> for SortAttributesUpdate {
    fn apply(&self, _attrs: &mut SortAttributes) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<SortAttributes, BBox> for SortAttributes {
    type Update = SortAttributesUpdate;
    type Lookup = NoopLookup<SortAttributes, BBox>;

    fn compatible(&self, _other: &SortAttributes) -> bool {
        true
    }

    fn merge(&mut self, other: &SortAttributes) -> Result<()> {
        self.bboxes.extend_from_slice(&other.bboxes);
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<BBox>) -> Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone)]
pub struct SortMetric {
    threshold: f32,
}

impl Default for SortMetric {
    fn default() -> Self {
        Self::new(DEFAULT_IOU_THRESHOLD)
    }
}

impl SortMetric {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl ObservationMetric<SortAttributes, BBox> for SortMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &SortAttributes,
        _attrs2: &SortAttributes,
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
        attrs: &mut SortAttributes,
        features: &mut Vec<ObservationSpec<BBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.0.unwrap();
        features.clear();

        let f = KalmanFilter::default();

        let predicted_bbox = if let Some(state) = attrs.state {
            let prediction = f.predict(state);
            attrs.state = Some(f.update(prediction, observation_bbox.into()));
            prediction.bbox()
        } else {
            attrs.state = Some(f.initiate(observation_bbox.into()));
            observation_bbox
        };

        observation.0 = Some(predicted_bbox);
        features.push(observation);
        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<BBox>>,
    ) -> Vec<ObservationMetricOk<BBox>> {
        unfiltered
            .into_iter()
            .filter(|x| x.attribute_metric.unwrap_or(0.0) > self.threshold)
            .collect()
    }
}
