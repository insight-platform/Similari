use crate::track::{
    MetricOutput, MetricQuery, Observation, ObservationAttributes, ObservationMetric,
    ObservationMetricOk,
};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::PositionalMetricType;
use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::Universal2DBoxKalmanFilter;

pub const DEFAULT_MINIMAL_SORT_CONFIDENCE: f32 = 0.05;

#[derive(Clone)]
pub struct SortMetric {
    method: PositionalMetricType,
    min_confidence: f32,
}

impl Default for SortMetric {
    fn default() -> Self {
        Self::new(
            PositionalMetricType::IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
        )
    }
}

impl SortMetric {
    pub fn new(method: PositionalMetricType, min_confidence: f32) -> Self {
        Self {
            method,
            min_confidence,
        }
    }
}

impl ObservationMetric<SortAttributes, Universal2DBox> for SortMetric {
    fn metric(&self, mq: &MetricQuery<SortAttributes, Universal2DBox>) -> MetricOutput<f32> {
        let (candidate_bbox, track_bbox) = (
            mq.candidate_observation.attr().as_ref().unwrap(),
            mq.track_observation.attr().as_ref().unwrap(),
        );
        let conf = if candidate_bbox.confidence < self.min_confidence {
            self.min_confidence
        } else {
            candidate_bbox.confidence
        };

        if Universal2DBox::too_far(candidate_bbox, track_bbox) {
            None
        } else {
            Some(match self.method {
                PositionalMetricType::Mahalanobis => {
                    let state = mq.track_attrs.get_state().unwrap();
                    let f = Universal2DBoxKalmanFilter::new(
                        mq.track_attrs.get_position_weight(),
                        mq.track_attrs.get_velocity_weight(),
                    );
                    let dist = f.distance(state, candidate_bbox);
                    (
                        Some(Universal2DBoxKalmanFilter::calculate_cost(dist, true) / conf),
                        None,
                    )
                }
                PositionalMetricType::IoU(threshold) => {
                    let box_m_opt = Universal2DBox::calculate_metric_object(
                        &Some(candidate_bbox),
                        &Some(track_bbox),
                    );
                    (
                        box_m_opt.map(|e| e * conf).filter(|e| *e >= threshold),
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

        let mut predicted_bbox = attrs.make_prediction(observation_bbox);
        attrs.update_history(observation_bbox, &predicted_bbox);

        *observation.attr_mut() = Some(match self.method {
            PositionalMetricType::Mahalanobis => predicted_bbox,
            PositionalMetricType::IoU(_) => {
                predicted_bbox.gen_vertices();
                predicted_bbox
            }
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

#[cfg(test)]
mod tests {
    use crate::prelude::{BoundingBox, PositionalMetricType};
    use crate::track::{MetricQuery, Observation, ObservationMetric};
    use crate::trackers::sort::metric::{SortMetric, DEFAULT_MINIMAL_SORT_CONFIDENCE};
    use crate::trackers::sort::{
        SortAttributes, SortAttributesOptions, DEFAULT_SORT_IOU_THRESHOLD,
    };
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
    use crate::EPS;
    use std::sync::Arc;

    #[test]
    fn confidence_preserved_during_optimization() {
        let mut attrs = SortAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )));

        let mut metric = SortMetric::new(
            PositionalMetricType::IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
        );

        let mut obs = vec![Observation::new(
            Some(BoundingBox::new_with_confidence(0.0, 0.0, 8.0, 10.0, 0.8).as_xyaah()),
            None,
        )];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, true)
            .unwrap();

        assert_eq!(
            obs[0].0.as_ref().unwrap().confidence,
            0.8,
            "Confidence must be preserved during optimization"
        );
    }

    #[test]
    fn confidence_used_in_distance_calculation() {
        let attr_opts = Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        ));

        let candidate_attrs = SortAttributes::new(attr_opts.clone());
        let track_attrs = SortAttributes::new(attr_opts.clone());

        let metric = SortMetric::new(
            PositionalMetricType::IoU(DEFAULT_SORT_IOU_THRESHOLD),
            DEFAULT_MINIMAL_SORT_CONFIDENCE,
        );

        let candidate_obs = Observation::new(
            Some(BoundingBox::new_with_confidence(0.0, 0.0, 8.0, 10.0, 0.8).as_xyaah()),
            None,
        );

        let track_obs = Observation::new(
            Some(BoundingBox::new_with_confidence(0.0, 0.0, 8.0, 10.0, 1.0).as_xyaah()),
            None,
        );

        let mq = MetricQuery {
            feature_class: 0,
            candidate_attrs: &candidate_attrs,
            candidate_observation: &candidate_obs,
            track_attrs: &track_attrs,
            track_observation: &track_obs,
        };

        let res = metric.metric(&mq);
        assert!(
            (res.unwrap().0.unwrap() - 0.8).abs() < EPS,
            "Confidence value in candidate box must be used."
        );

        let mq = MetricQuery {
            feature_class: 0,
            candidate_attrs: &track_attrs,
            candidate_observation: &track_obs,
            track_attrs: &candidate_attrs,
            track_observation: &candidate_obs,
        };

        let res = metric.metric(&mq);
        assert!(
            (res.unwrap().0.unwrap() - 1.0).abs() < EPS,
            "Confidence in track box must NOT be used."
        );
    }
}
