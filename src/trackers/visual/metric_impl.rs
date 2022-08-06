use crate::distance::{cosine, euclidean};
use crate::track::ObservationAttributes;
use crate::track::{MetricOutput, ObservationMetric, ObservationMetricOk, ObservationSpec};
use crate::trackers::visual::{
    PositionalMetricType, VisualAttributes, VisualMetric, VisualMetricType,
};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;
use anyhow::Result;
use std::default::Default;
use std::iter::{IntoIterator, Iterator};

impl ObservationMetric<VisualAttributes, Universal2DBox> for VisualMetric {
    fn metric(
        &self,
        _feature_class: u64,
        _candidate_attributes: &VisualAttributes,
        track_attributes: &VisualAttributes,
        candidate_observation: &ObservationSpec<Universal2DBox>,
        track_observation: &ObservationSpec<Universal2DBox>,
    ) -> MetricOutput<f32> {
        let candidate_observation_bbox = candidate_observation.0.as_ref().unwrap();
        let track_observation_bbox = track_observation.0.as_ref().unwrap();

        let candidate_observation_feature = candidate_observation.1.as_ref().unwrap();
        let track_observation_feature = track_observation.1.as_ref().unwrap();

        if !matches!(self.positional_kind, PositionalMetricType::Ignore)
            && Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox)
        {
            None
        } else {
            let f = KalmanFilter::default();
            let state = track_attributes.state.unwrap();
            Some((
                match self.positional_kind {
                    PositionalMetricType::Mahalanobis => {
                        let dist = f.distance(state, candidate_observation_bbox);
                        Some(KalmanFilter::calculate_cost(dist, true))
                    }
                    PositionalMetricType::IoU(threshold) => {
                        let box_m_opt = Universal2DBox::calculate_metric_object(
                            &candidate_observation.0,
                            &track_observation.0,
                        );
                        if let Some(box_m) = &box_m_opt {
                            if *box_m <= threshold {
                                None
                            } else {
                                box_m_opt
                            }
                        } else {
                            None
                        }
                    }
                    PositionalMetricType::Ignore => None,
                },
                if self.minimal_visual_track_len >= track_attributes.track_length {
                    Some(match self.visual_kind {
                        VisualMetricType::Euclidean => {
                            euclidean(candidate_observation_feature, track_observation_feature)
                        }
                        VisualMetricType::Cosine => {
                            cosine(candidate_observation_feature, track_observation_feature)
                        }
                    })
                } else {
                    None
                },
            ))
        }
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut VisualAttributes,
        features: &mut Vec<ObservationSpec<Universal2DBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation_spec = features.pop().unwrap();
        let observation_bbox = observation_spec.0.as_ref().unwrap();
        let observation_feature = observation_spec.1.clone();

        let f = KalmanFilter::default();

        let state = if let Some(state) = attrs.state {
            f.update(state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(state);
        attrs.state = Some(prediction);
        let predicted_bbox = prediction.universal_bbox();
        attrs.track_length += 1;

        attrs.observed_boxes.push_back(observation_bbox.clone());
        attrs.predicted_boxes.push_back(predicted_bbox.clone());
        attrs.observed_features.push_back(observation_feature);

        if attrs.max_history_len > 0 && attrs.observed_boxes.len() > attrs.max_history_len {
            attrs.observed_boxes.pop_front();
            attrs.predicted_boxes.pop_front();
            attrs.observed_features.pop_front();
        }

        observation_spec.0 = Some(predicted_bbox);
        features.push(observation_spec);
        if features.len() > attrs.max_observations {
            features.swap_remove(0);
        }

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<Universal2DBox>>,
    ) -> Vec<ObservationMetricOk<Universal2DBox>> {
        unfiltered
            .into_iter()
            .filter(|x| {
                x.attribute_metric.unwrap_or(0.0)
                    > match self.positional_kind {
                        PositionalMetricType::Mahalanobis => 0.0,
                        PositionalMetricType::IoU(t) => t,
                        PositionalMetricType::Ignore => f32::MAX,
                    }
            })
            .collect()
    }
}

#[cfg(test)]
mod postprocess_distances {
    use crate::track::{ObservationMetric, ObservationMetricOk};
    use crate::trackers::visual::{PositionalMetricType, VisualMetricBuilder};

    #[test]
    fn postprocess_distances_maha() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Mahalanobis)
            .build();
        let dists = metric.postprocess_distances(vec![
            ObservationMetricOk {
                from: 1,
                to: 0,
                attribute_metric: Some(0.0),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 2,
                to: 1,
                attribute_metric: Some(1.0),
                feature_distance: None,
            },
        ]);
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 2,
                to: 1,
                attribute_metric: Some(w),
                feature_distance: None,
            } if w == 1.0
        ));
    }

    #[test]
    fn postprocess_distances_iou() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .build();
        let dists = metric.postprocess_distances(vec![
            ObservationMetricOk {
                from: 1,
                to: 0,
                attribute_metric: Some(0.2),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 2,
                to: 1,
                attribute_metric: Some(1.0),
                feature_distance: None,
            },
        ]);
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 2,
                to: 1,
                attribute_metric: Some(w),
                feature_distance: None,
            } if w == 1.0
        ));
    }

    #[test]
    fn postprocess_distances_ignore() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Ignore)
            .build();
        let dists = metric.postprocess_distances(vec![ObservationMetricOk {
            from: 1,
            to: 0,
            attribute_metric: Some(f32::MAX - 1.0),
            feature_distance: None,
        }]);
        assert_eq!(dists.len(), 0);
    }
}

#[cfg(test)]
mod optimize {
    use crate::examples::vec2;
    use crate::track::{ObservationMetric, ObservationSpec};
    use crate::trackers::visual::{
        PositionalMetricType, VisualAttributes, VisualMetricBuilder, VisualMetricType,
    };
    use crate::utils::bbox::{BoundingBox, Universal2DBox};
    use std::collections::VecDeque;

    #[test]
    fn optimization_steps() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualMetricType::Euclidean)
            .build();

        let mut attrs = VisualAttributes {
            predicted_boxes: VecDeque::default(),
            observed_boxes: VecDeque::default(),
            observed_features: VecDeque::default(),
            last_updated_epoch: 0,
            track_length: 0,
            scene_id: 0,
            custom_object_id: None,
            state: None,
            max_observations: 2,
            max_history_len: 2,
            max_idle_epochs: 3,
            current_epochs: None,
        };

        let mut obs = vec![ObservationSpec(
            Some(BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah()),
            Some(vec2(0.0, 1.0)),
        )];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, false)
            .unwrap();

        assert_eq!(attrs.observed_features.len(), 1);
        assert_eq!(attrs.observed_boxes.len(), 1);
        assert_eq!(attrs.predicted_boxes.len(), 1);
        assert_eq!(attrs.track_length, 1);
        assert_eq!(obs.len(), 1);

        let mut obs = vec![
            ObservationSpec(
                Some(BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah()),
                Some(vec2(0.0, 1.0)),
            ),
            ObservationSpec(Some(BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah()), None),
        ];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, false)
            .unwrap();

        assert_eq!(attrs.observed_features.len(), 2);
        assert_eq!(attrs.observed_boxes.len(), 2);
        assert_eq!(attrs.predicted_boxes.len(), 2);
        assert_eq!(attrs.track_length, 2);
        assert_eq!(obs.len(), 2);

        let mut obs = vec![
            ObservationSpec(
                Some(BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah()),
                Some(vec2(0.0, 1.0)),
            ),
            ObservationSpec(Some(BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah()), None),
            ObservationSpec(
                Some(BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah()),
                Some(vec2(0.1, 1.1)),
            ),
        ];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, false)
            .unwrap();

        assert_eq!(attrs.observed_features.len(), 2);
        assert_eq!(attrs.observed_boxes.len(), 2);
        assert_eq!(attrs.predicted_boxes.len(), 2);
        assert_eq!(attrs.track_length, 3);
        assert_eq!(obs.len(), 2);
        assert!(matches!(
            obs[0].clone(),
            ObservationSpec(Some(Universal2DBox { .. }), Some(o)) if o[0].to_array()[..2] == [0.1 , 1.1]
        ));
    }
}

#[cfg(test)]
mod metric {
    use crate::trackers::visual::{PositionalMetricType, VisualMetricBuilder, VisualMetricType};

    #[test]
    fn metric_ignore() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Ignore)
            .visual_metric(VisualMetricType::Euclidean)
            .build();
        drop(metric);
    }

    #[test]
    fn metric_far() {}

    #[test]
    fn metric_maha() {}

    #[test]
    fn metric_iou() {}

    #[test]
    fn metric_short() {}

    #[test]
    fn metric_euclidean() {}

    #[test]
    fn metric_cosine() {}
}
