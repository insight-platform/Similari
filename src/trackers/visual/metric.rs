use crate::distance::{cosine, euclidean};
use crate::track::{MetricOutput, ObservationMetric, ObservationSpec};
use crate::track::{Observation, ObservationAttributes};
use crate::trackers::visual::{
    PositionalMetricType, VisualAttributes, VisualMetric, VisualMetricType,
    VisualObservationAttributes,
};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;
use anyhow::Result;
use itertools::Itertools;
use std::default::Default;
use std::iter::Iterator;

#[pyclass]
#[derive(Clone, Default)]
pub enum VisualMetricType {
    #[default]
    Euclidean,
    Cosine,
}

#[pymethods]
impl VisualMetricType {
    #[staticmethod]
    pub fn euclidean() -> Self {
        VisualMetricType::Euclidean
    }

    #[staticmethod]
    pub fn cosine() -> Self {
        VisualMetricType::Cosine
    }
}

#[derive(Clone, Default)]
pub enum PositionalMetricType {
    #[default]
    Mahalanobis,
    IoU(f32),
    Ignore,
}

#[derive(Clone)]
pub struct VisualMetric {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    visual_minimal_track_length: usize,
    visual_minimal_area: f32,
    visual_minimal_quality_use: f32,
    visual_minimal_quality_collect: f32,
}

impl Default for VisualMetric {
    fn default() -> Self {
        VisualMetricBuilder::default().build()
    }
}

pub struct VisualMetricBuilder {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    visual_minimal_track_length: usize,
    visual_minimal_area: f32,
    visual_minimal_quality_use: f32,
    visual_minimal_quality_collect: f32,
}

/// By default the metric object is constructed with: Euclidean visual metric, IoU(0.3) positional metric
/// and minimal visual track length = 3
///
impl Default for VisualMetricBuilder {
    fn default() -> Self {
        VisualMetricBuilder {
            visual_kind: VisualMetricType::Euclidean,
            positional_kind: PositionalMetricType::IoU(0.3),
            visual_minimal_track_length: 3,
            visual_minimal_area: 0.0,
            visual_minimal_quality_use: 0.0,
            visual_minimal_quality_collect: 0.0,
        }
    }
}

impl VisualMetricBuilder {
    pub fn visual_metric(mut self, metric: VisualMetricType) -> Self {
        self.visual_kind = metric;
        self
    }

    pub fn positional_metric(mut self, metric: PositionalMetricType) -> Self {
        if let PositionalMetricType::IoU(t) = metric {
            assert!(
                t > 0.0 && t < 1.0,
                "Threshold must lay between (0.0 and 1.0)"
            );
        }
        self.positional_kind = metric;
        self
    }

    pub fn visual_minimal_track_length(mut self, length: usize) -> Self {
        assert!(
            length > 0,
            "The minimum amount of visual features collected before visual metric is applied should be greater than 0."
        );
        self.visual_minimal_track_length = length;
        self
    }

    pub fn visual_minimal_area(mut self, area: f32) -> Self {
        assert!(
            area >= 0.0,
            "The minimum area of bbox for visual feature distance calculated and feature collected should be greater than 0."
        );
        self.visual_minimal_area = area;
        self
    }

    pub fn visual_minimal_quality_use(mut self, q: f32) -> Self {
        assert!(
            q >= 0.0,
            "The minimum quality of visual feature should be greater than or equal to 0.0."
        );
        self.visual_minimal_quality_use = q;
        self
    }

    pub fn visual_minimal_quality_collect(mut self, q: f32) -> Self {
        assert!(
            q >= 0.0,
            "The minimum quality of visual feature should be greater than or equal to 0.0."
        );
        self.visual_minimal_quality_collect = q;
        self
    }

    pub fn build(self) -> VisualMetric {
        VisualMetric {
            visual_kind: self.visual_kind,
            positional_kind: self.positional_kind,
            visual_minimal_track_length: self.visual_minimal_track_length,
            visual_minimal_area: self.visual_minimal_area,
            visual_minimal_quality_use: self.visual_minimal_quality_use,
            visual_minimal_quality_collect: self.visual_minimal_quality_collect,
        }
    }
}

impl VisualMetric {
    fn cleanup_observations(
        attrs: &mut VisualAttributes,
        observations: &mut Vec<ObservationSpec<VisualObservationAttributes>>,
    ) {
        if observations.len() > attrs.max_observations {
            observations.swap_remove(0);
        } else {
            let last = observations.len() - 1;
            observations.swap(0, last);
        }

        // remove all old bboxes
        observations.iter_mut().skip(1).for_each(|f| {
            if let Some(e) = &mut f.0 {
                e.bbox = None;
            }
        });

        // if historic element doesn't hold the feature, remove it from the observations
        let to_remove_no_feature = observations
            .iter()
            .skip(1)
            .enumerate()
            .filter(|(_, e)| e.1.is_none())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        for i in to_remove_no_feature {
            observations.remove(i);
        }
    }

    fn positional_metric(
        &self,
        candidate_observation_bbox_opt: &Option<Universal2DBox>,
        track_observation_bbox_opt: &Option<Universal2DBox>,
        track_attributes: &VisualAttributes,
    ) -> Option<f32> {
        if let (Some(candidate_observation_bbox), Some(track_observation_bbox)) =
            (candidate_observation_bbox_opt, track_observation_bbox_opt)
        {
            match self.positional_kind {
                PositionalMetricType::Mahalanobis => {
                    let f = KalmanFilter::default();
                    let state = track_attributes.state.unwrap();

                    if Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox) {
                        None
                    } else {
                        let dist = f.distance(state, candidate_observation_bbox);
                        Some(KalmanFilter::calculate_cost(dist, true))
                    }
                }
                PositionalMetricType::IoU(threshold) => {
                    if Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox) {
                        None
                    } else {
                        let box_m_opt = Universal2DBox::calculate_metric_object(
                            &candidate_observation_bbox_opt,
                            &track_observation_bbox_opt,
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
                }
                PositionalMetricType::Ignore => None,
            }
        } else {
            None
        }
    }

    fn visual_metric(
        &self,
        candidate_observation_feature: &Observation,
        track_observation_feature: &Observation,
        track_attributes: &VisualAttributes,
    ) -> Option<f32> {
        if track_attributes.visual_features_collected_count >= self.visual_minimal_track_length {
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
        }
    }
}

impl ObservationMetric<VisualAttributes, VisualObservationAttributes> for VisualMetric {
    fn metric(
        &self,
        _feature_class: u64,
        _candidate_attributes: &VisualAttributes,
        track_attributes: &VisualAttributes,
        candidate_observation: &ObservationSpec<VisualObservationAttributes>,
        track_observation: &ObservationSpec<VisualObservationAttributes>,
    ) -> MetricOutput<f32> {
        let candidate_observation_bbox_opt = &candidate_observation.0.as_ref().unwrap().bbox;
        let track_observation_bbox_opt = &track_observation.0.as_ref().unwrap().bbox;

        let candidate_observation_feature = candidate_observation.1.as_ref().unwrap();
        let track_observation_feature = track_observation.1.as_ref().unwrap();

        Some((
            self.positional_metric(
                candidate_observation_bbox_opt,
                track_observation_bbox_opt,
                track_attributes,
            ),
            self.visual_metric(
                candidate_observation_feature,
                track_observation_feature,
                track_attributes,
            ),
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut VisualAttributes,
        observations: &mut Vec<ObservationSpec<VisualObservationAttributes>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation_spec = observations.pop().unwrap();
        let observation_bbox = observation_spec.0.as_ref().unwrap().bbox.as_ref().unwrap();
        let feature_quality = observation_spec.0.as_ref().unwrap().visual_quality;
        let observation_feature = observation_spec.1.clone();

        let predicted_bbox = attrs.update_bbox_prediction(observation_bbox);

        attrs.update_history(observation_bbox, &predicted_bbox, observation_feature);

        observation_spec.0 = Some(VisualObservationAttributes::new(
            feature_quality,
            match self.positional_kind {
                PositionalMetricType::Mahalanobis | PositionalMetricType::Ignore => predicted_bbox,
                PositionalMetricType::IoU(_) => predicted_bbox.gen_vertices(),
            },
        ));

        observations.push(observation_spec);

        VisualMetric::cleanup_observations(attrs, observations);

        attrs.visual_features_collected_count =
            observations.iter().filter(|f| f.1.is_some()).count();

        Ok(())
    }
}

#[cfg(test)]
mod optimize {
    use crate::examples::vec2;
    use crate::track::{ObservationMetric, ObservationSpec};
    use crate::trackers::visual::{
        PositionalMetricType, VisualAttributes, VisualMetricBuilder, VisualMetricType,
        VisualObservationAttributes,
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
            visual_features_collected_count: 0,
            scene_id: 0,
            custom_object_id: None,
            state: None,
            max_observations: 2,
            max_history_len: 2,
            max_idle_epochs: 3,
            current_epochs: None,
        };

        let mut obs = vec![ObservationSpec(
            Some(VisualObservationAttributes::new(
                1.0,
                BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
            )),
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
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            ObservationSpec(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah(),
                )),
                None,
            ),
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
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            ObservationSpec(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah(),
                )),
                None,
            ),
            ObservationSpec(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                )),
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
            ObservationSpec(Some(VisualObservationAttributes{ visual_quality: _, bbox: Some(Universal2DBox { .. })}), Some(o)) if o[0].to_array()[..2] == [0.1 , 1.1]
        ));
    }
}

#[cfg(test)]
mod metric {
    use crate::examples::vec2;
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
    use crate::store::TrackStore;
    use crate::track::ObservationMetricOk;
    use crate::trackers::visual::{
        PositionalMetricType, VisualAttributes, VisualMetric, VisualMetricBuilder,
        VisualMetricType, VisualObservationAttributes,
    };
    use crate::utils::bbox::BoundingBox;
    use crate::EPS;
    use std::collections::VecDeque;

    fn default_attrs() -> VisualAttributes {
        VisualAttributes {
            predicted_boxes: VecDeque::default(),
            observed_boxes: VecDeque::default(),
            observed_features: VecDeque::default(),
            last_updated_epoch: 0,
            track_length: 0,
            visual_features_collected_count: 0,
            scene_id: 0,
            custom_object_id: None,
            state: None,
            max_observations: 2,
            max_history_len: 2,
            max_idle_epochs: 3,
            current_epochs: None,
        }
    }

    fn default_store(
        metric: VisualMetric,
    ) -> TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes, NoopNotifier> {
        TrackStoreBuilder::default()
            .metric(metric)
            .notifier(NoopNotifier)
            .default_attributes(default_attrs())
            .build()
    }

    #[test]
    fn metric_ignore() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Ignore)
            .visual_minimal_track_length(1)
            .build();

        let store = default_store(metric);

        let track1 = store
            .track_builder(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.1))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let track2 = store
            .track_builder(2)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let dists = track1.distances(&track2, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: None, // ignored because of ignore
                feature_distance: Some(x)
            } if x > 0.0));
    }

    #[test]
    fn metric_far() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(1)
            .build();

        let store = default_store(metric);

        let track1 = store
            .track_builder(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.1))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(100.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let track2 = store
            .track_builder(2)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let dists = track1.distances(&track2, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: None, // ignored because objects are too far
                feature_distance: Some(x)
            } if x > 0.0));
    }

    #[test]
    fn metric_iou() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualMetricType::Cosine)
            .visual_minimal_track_length(1)
            .build();
        let store = default_store(metric);

        let track1 = store
            .track_builder(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(1.0, 0.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let track2 = store
            .track_builder(2)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(1.0, 0.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let dists = track1.distances(&track2, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: Some(y)
            } if (x - 1.0).abs() < EPS && (y - 1.0).abs() < EPS));
    }

    #[test]
    fn visual_track_too_short() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualMetricType::Euclidean)
            .visual_minimal_track_length(3)
            .build();

        let store = default_store(metric);

        let track1 = store
            .track_builder(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.1))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let track2 = store
            .track_builder(2)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let dists = track1.distances(&track2, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: None     // track too short
            } if (x - 1.0).abs() < EPS));
    }

    #[test]
    fn visual_track_long_enough() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualMetricType::Euclidean)
            .visual_minimal_track_length(2)
            .build();

        let store = default_store(metric);

        let track1 = store
            .track_builder(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let track2 = store
            .track_builder(2)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                    ))
                    .build(),
            )
            .build()
            .unwrap();

        let dists = track1.distances(&track2, 0).unwrap();
        assert_eq!(dists.len(), 2);
        for i in 0..1 {
            assert!(matches!(
            dists[i],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: Some(y)     // track too short
            } if (x - 1.0).abs() < EPS && y.abs() < EPS));
        }
    }
}
