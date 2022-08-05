use crate::track::{
    MetricOutput, ObservationAttributes, ObservationMetric, ObservationMetricOk, ObservationSpec,
};
use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;

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
        e1: &ObservationSpec<Universal2DBox>,
        e2: &ObservationSpec<Universal2DBox>,
    ) -> MetricOutput<f32> {
        let box_m_opt = Universal2DBox::calculate_metric_object(&e1.0, &e2.0);
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
        features: &mut Vec<ObservationSpec<Universal2DBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> anyhow::Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.0.as_ref().unwrap();
        features.clear();

        let f = KalmanFilter::default();

        let state = if let Some(state) = attrs.state {
            f.update(state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(state);
        attrs.state = Some(prediction);
        let predicted_bbox = prediction.universal_bbox();
        attrs.length += 1;

        attrs.observed_boxes.push_back(observation_bbox.clone());
        attrs.predicted_boxes.push_back(predicted_bbox.clone());

        if attrs.max_history_len > 0 && attrs.observed_boxes.len() > attrs.max_history_len {
            attrs.observed_boxes.pop_front();
            attrs.predicted_boxes.pop_front();
        }

        observation.0 = Some(predicted_bbox.gen_vertices());
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
