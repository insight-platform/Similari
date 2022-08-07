use crate::track::{
    MetricOutput, Observation, ObservationAttributes, ObservationMetric, ObservationMetricOk,
};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
use crate::utils::bbox::Universal2DBox;

#[derive(Clone)]
pub struct IOUSortMetric {
    threshold: f32,
}

impl Default for IOUSortMetric {
    fn default() -> Self {
        Self::new(DEFAULT_SORT_IOU_THRESHOLD)
    }
}

impl IOUSortMetric {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl ObservationMetric<SortAttributes, Universal2DBox> for IOUSortMetric {
    fn metric(
        &self,
        _feature_class: u64,
        _attrs1: &SortAttributes,
        _attrs2: &SortAttributes,
        e1: &Observation<Universal2DBox>,
        e2: &Observation<Universal2DBox>,
    ) -> MetricOutput<f32> {
        let box_m_opt = Universal2DBox::calculate_metric_object(&e1.attr(), &e2.attr());
        box_m_opt.filter(|e| *e >= 0.01).map(|e| (Some(e), None))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut SortAttributes,
        features: &mut Vec<Observation<Universal2DBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> anyhow::Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.attr().as_ref().unwrap();
        features.clear();

        let predicted_bbox = attrs.make_prediction(observation_bbox);
        attrs.update_history(observation_bbox, &predicted_bbox);

        *observation.attr_mut() = Some(predicted_bbox.gen_vertices());
        features.push(observation);

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<Universal2DBox>>,
    ) -> Vec<ObservationMetricOk<Universal2DBox>> {
        unfiltered
            .into_iter()
            .filter(|x| x.attribute_metric.unwrap_or(0.0) > self.threshold)
            .collect()
    }
}
