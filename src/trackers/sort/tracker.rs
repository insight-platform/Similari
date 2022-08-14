use crate::track::{
    MetricOutput, MetricQuery, Observation, ObservationAttributes, ObservationMetric,
    ObservationMetricOk,
};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::PositionalMetricType;
use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;
use crate::EPS;

#[derive(Clone)]
pub struct SortMetric {
    method: PositionalMetricType,
}

impl Default for SortMetric {
    fn default() -> Self {
        Self::new(PositionalMetricType::IoU(DEFAULT_SORT_IOU_THRESHOLD))
    }
}

impl SortMetric {
    pub fn new(method: PositionalMetricType) -> Self {
        Self { method }
    }
}

impl ObservationMetric<SortAttributes, Universal2DBox> for SortMetric {
    fn metric(&self, mq: &MetricQuery<SortAttributes, Universal2DBox>) -> MetricOutput<f32> {
        let (candidate_bbox, track_bbox) = (
            mq.candidate_observation.attr().as_ref().unwrap(),
            mq.track_observation.attr().as_ref().unwrap(),
        );
        if Universal2DBox::too_far(candidate_bbox, track_bbox) {
            None
        } else {
            Some(match self.method {
                PositionalMetricType::Mahalanobis => {
                    let state = mq.track_attrs.get_state().unwrap();
                    let f = KalmanFilter::default();
                    let dist = f.distance(state, candidate_bbox);
                    (
                        Some(
                            KalmanFilter::calculate_cost(dist, true)
                                / (candidate_bbox.confidence + EPS),
                        ),
                        None,
                    )
                }
                PositionalMetricType::IoU(threshold) => {
                    let box_m_opt = Universal2DBox::calculate_metric_object(
                        &Some(&candidate_bbox),
                        &Some(&track_bbox),
                    );
                    (
                        box_m_opt
                            .map(|e| e * candidate_bbox.confidence)
                            .filter(|e| *e >= threshold),
                        None,
                    )
                }
            })
        }
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

        *observation.attr_mut() = Some(match self.method {
            PositionalMetricType::Mahalanobis => predicted_bbox,
            PositionalMetricType::IoU(_) => predicted_bbox.gen_vertices(),
        });

        features.push(observation);

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<Universal2DBox>>,
    ) -> Vec<ObservationMetricOk<Universal2DBox>> {
        unfiltered
            .into_iter()
            .filter(|res| res.attribute_metric.is_some())
            .collect()
    }
}
