use crate::distance::{cosine, euclidean};
use crate::track::{Feature, MetricQuery, ObservationAttributes};
use crate::track::{MetricOutput, Observation, ObservationMetric};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual::track_attributes::VisualAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;
use anyhow::Result;
use pyo3::prelude::*;
use std::default::Default;
use std::iter::Iterator;
use std::sync::Arc;

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
}

pub struct VisualMetricOptions {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    visual_minimal_track_length: usize,
    // visual_minimal_area: f32,
    // visual_minimal_quality_use: f32,
    // visual_minimal_quality_collect: f32,
    visual_max_observations: usize,
}

#[derive(Clone)]
pub struct VisualMetric {
    opts: Arc<VisualMetricOptions>,
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
    visual_max_observations: usize,
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
            visual_max_observations: 5,
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

    pub fn visual_max_observations(mut self, n: usize) -> Self {
        self.visual_max_observations = n;
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
            opts: Arc::new(VisualMetricOptions {
                visual_kind: self.visual_kind,
                positional_kind: self.positional_kind,
                visual_minimal_track_length: self.visual_minimal_track_length,
                // visual_minimal_area: self.visual_minimal_area,
                // visual_minimal_quality_use: self.visual_minimal_quality_use,
                // visual_minimal_quality_collect: self.visual_minimal_quality_collect,
                visual_max_observations: self.visual_max_observations,
            }),
        }
    }
}

impl VisualMetric {
    fn optimize_observations(
        &self,
        observations: &mut Vec<Observation<VisualObservationAttributes>>,
    ) {
        if observations.len() > self.opts.visual_max_observations {
            observations.swap_remove(0);
        } else {
            let last = observations.len() - 1;
            observations.swap(0, last);
        }

        // remove all old bboxes
        observations.iter_mut().skip(1).for_each(|f| {
            if let Some(e) = &mut f.attr_mut() {
                e.drop_bbox();
            }
        });

        // if historic elements don't hold the feature, remove them from the observations
        let mut skip = true;
        observations.retain(|e| {
            if skip {
                skip = false;
                true
            } else {
                e.1.is_some()
            }
        });
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
            match self.opts.positional_kind {
                PositionalMetricType::Mahalanobis => {
                    if Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox) {
                        None
                    } else {
                        let state = track_attributes.get_state().unwrap();
                        let f = KalmanFilter::default();
                        let dist = f.distance(state, candidate_observation_bbox);
                        Some(KalmanFilter::calculate_cost(dist, true))
                    }
                }
                PositionalMetricType::IoU(threshold) => {
                    if Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox) {
                        None
                    } else {
                        let box_m_opt = Universal2DBox::calculate_metric_object(
                            &candidate_observation_bbox_opt.as_ref(),
                            &track_observation_bbox_opt.as_ref(),
                        );
                        box_m_opt.filter(|e| *e >= threshold)
                    }
                }
            }
        } else {
            None
        }
    }

    fn visual_metric(
        &self,
        candidate_observation_feature: &Feature,
        track_observation_feature: &Feature,
        track_attributes: &VisualAttributes,
    ) -> Option<f32> {
        if track_attributes.visual_features_collected_count >= self.opts.visual_minimal_track_length
        {
            Some(match self.opts.visual_kind {
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
        mq: &MetricQuery<VisualAttributes, VisualObservationAttributes>,
    ) -> MetricOutput<f32> {
        let candidate_bbox_opt = mq
            .candidate_observation
            .attr()
            .as_ref()
            .expect("Observation attributes must always present.")
            .bbox_opt();

        let track_bbox_opt = mq
            .track_observation
            .attr()
            .as_ref()
            .expect("Observation attributes must always present.")
            .bbox_opt();

        let candidate_feature_opt = mq.candidate_observation.feature().as_ref();
        let track_feature_opt = mq.track_observation.feature().as_ref();

        Some((
            self.positional_metric(candidate_bbox_opt, track_bbox_opt, mq.track_attrs),
            match (candidate_feature_opt, track_feature_opt) {
                (Some(c), Some(t)) => self.visual_metric(c, t, mq.track_attrs),
                _ => None,
            },
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut VisualAttributes,
        observations: &mut Vec<Observation<VisualObservationAttributes>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation = observations
            .pop()
            .expect("At least one observation must present in the track.");

        let observation_bbox = observation
            .attr()
            .as_ref()
            .expect("New track element must have bbox.")
            .unchecked_bbox_ref();

        let feature_quality = observation
            .attr()
            .as_ref()
            .expect("New track element must have feature quality parameter.")
            .visual_quality();

        let predicted_bbox = attrs.make_prediction(observation_bbox);
        attrs.update_history(
            observation_bbox,
            &predicted_bbox,
            observation.feature().clone(),
        );

        *observation.attr_mut() = Some(VisualObservationAttributes::new(
            feature_quality,
            match self.opts.positional_kind {
                PositionalMetricType::Mahalanobis => predicted_bbox,
                PositionalMetricType::IoU(_) => predicted_bbox.gen_vertices(),
            },
        ));

        observations.push(observation);

        self.optimize_observations(observations);

        attrs.visual_features_collected_count = observations
            .iter()
            .filter(|f| f.feature().is_some())
            .count();

        Ok(())
    }
}

#[cfg(test)]
mod optimize {
    use crate::examples::vec2;
    use crate::track::{Observation, ObservationMetric};
    use crate::trackers::sort::SortAttributesOptions;
    use crate::trackers::visual::metric::{
        PositionalMetricType, VisualMetricBuilder, VisualMetricType,
    };
    use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual::track_attributes::VisualAttributes;
    use crate::utils::bbox::{BoundingBox, Universal2DBox};
    use std::sync::Arc;

    #[test]
    fn optimization_steps() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualMetricType::Euclidean)
            .build();

        let mut attrs = VisualAttributes::new(Arc::new(SortAttributesOptions::new(None, 0, 5)));

        let mut obs = vec![Observation(
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
            Observation(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            Observation(
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
        dbg!(&obs);
        assert!(
            {
                let e = &obs[0];
                e.1.is_none()
                    && matches!(
                        e.0.as_ref().unwrap().bbox_opt(),
                        Some(Universal2DBox { .. })
                    )
            } && {
                let e = &obs[1];
                e.1.is_some() && matches!(e.0.as_ref().unwrap().bbox_opt(), None)
            }
        );

        let mut obs = vec![
            Observation(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            Observation(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah(),
                )),
                None,
            ),
            Observation(
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

        assert_eq!(attrs.observed_features.len(), 3);
        assert_eq!(attrs.observed_boxes.len(), 3);
        assert_eq!(attrs.predicted_boxes.len(), 3);
        assert_eq!(attrs.track_length, 3);
        assert_eq!(obs.len(), 2);
        assert!(matches!(
            &obs[0],
            Observation(Some(a), Some(o)) if a.bbox_opt().is_some() && o[0].to_array()[..2] == [0.1 , 1.1]
        ));
    }
}

#[cfg(test)]
mod metric_tests {
    use crate::examples::vec2;
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
    use crate::store::TrackStore;
    use crate::track::ObservationMetricOk;
    use crate::trackers::sort::SortAttributesOptions;
    use crate::trackers::visual::metric::{
        PositionalMetricType, VisualMetric, VisualMetricBuilder, VisualMetricType,
    };
    use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual::track_attributes::VisualAttributes;
    use crate::utils::bbox::BoundingBox;
    use crate::EPS;
    use std::sync::Arc;

    fn default_attrs() -> VisualAttributes {
        VisualAttributes::new(Arc::new(SortAttributesOptions::new(None, 0, 5)))
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
    fn pos_metric_far() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(1)
            .build();

        let store = default_store(metric);

        let track1 = store
            .new_track(1)
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
            .new_track(2)
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
            .new_track(1)
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
            .new_track(2)
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
            .new_track(1)
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
            .new_track(2)
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
            .new_track(1)
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
            .new_track(2)
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
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: Some(y)     // track too short
            } if (x - 1.0).abs() < EPS && y.abs() < EPS));

        assert!(matches!(
            dists[1],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: None,
                feature_distance: Some(y)     // track too short
            } if y.abs() < EPS));
    }
}