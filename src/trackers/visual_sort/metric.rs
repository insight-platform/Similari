/// Auxiliary class that helps to build a metric object
pub mod builder;

use crate::distance::{cosine, euclidean};
use crate::track::{Feature, MetricQuery, ObservationAttributes, ObservationMetricOk};
use crate::track::{MetricOutput, Observation, ObservationMetric};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::PositionalMetricType;
use crate::trackers::visual_sort::metric::builder::VisualMetricBuilder;
use crate::trackers::visual_sort::metric::VisualSortMetricType::{Cosine, Euclidean};
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual_sort::track_attributes::VisualAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::Universal2DBoxKalmanFilter;
use anyhow::Result;
use std::default::Default;
use std::iter::Iterator;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum VisualSortMetricType {
    Euclidean(f32),
    Cosine(f32),
}

impl Default for VisualSortMetricType {
    fn default() -> Self {
        Euclidean(f32::MAX)
    }
}

impl VisualSortMetricType {
    pub fn euclidean(threshold: f32) -> Self {
        assert!(threshold > 0.0, "Threshold must be a positive number");
        VisualSortMetricType::Euclidean(threshold)
    }

    pub fn cosine(threshold: f32) -> Self {
        assert!(
            (-1.0..=1.0).contains(&threshold),
            "Threshold must lay within [-1.0:1:0]"
        );
        VisualSortMetricType::Cosine(threshold)
    }

    pub fn threshold(&self) -> f32 {
        match self {
            Euclidean(t) | Cosine(t) => *t,
        }
    }

    pub fn is_ok(&self, dist: f32) -> bool {
        match self {
            Euclidean(t) => dist <= *t,
            Cosine(t) => dist >= *t,
        }
    }

    pub fn distance_to_weight(&self, dist: f32) -> f32 {
        match self {
            Euclidean(_) => dist,
            Cosine(_) => 1.0 - dist,
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::VisualSortMetricType;
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "VisualSortMetricType")]
    #[derive(Clone, Debug)]
    pub struct PyVisualSortMetricType(pub VisualSortMetricType);

    #[pymethods]
    impl PyVisualSortMetricType {
        #[staticmethod]
        pub fn euclidean(threshold: f32) -> Self {
            PyVisualSortMetricType(VisualSortMetricType::euclidean(threshold))
        }

        #[staticmethod]
        pub fn cosine(threshold: f32) -> Self {
            PyVisualSortMetricType(VisualSortMetricType::cosine(threshold))
        }

        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{self:?}")
        }

        fn __str__(&self) -> String {
            format!("{self:#?}")
        }
    }
}

#[derive(Debug)]
pub struct VisualMetricOptions {
    pub visual_max_observations: usize,
    pub visual_min_votes: usize,
    pub visual_kind: VisualSortMetricType,
    pub positional_kind: PositionalMetricType,
    pub visual_minimal_track_length: usize,
    pub visual_minimal_area: f32,
    pub visual_minimal_quality_use: f32,
    pub visual_minimal_quality_collect: f32,
    pub visual_minimal_own_area_percentage_use: f32,
    pub visual_minimal_own_area_percentage_collect: f32,
    pub positional_min_confidence: f32,
}

#[derive(Clone, Debug)]
pub struct VisualMetric {
    pub opts: Arc<VisualMetricOptions>,
}

impl Default for VisualMetric {
    fn default() -> Self {
        VisualMetricBuilder::default().build()
    }
}

impl VisualMetric {
    fn optimize_observations(
        &self,
        observations: &mut Vec<Observation<VisualObservationAttributes>>,
    ) {
        observations.retain(|e| e.feature().is_some());

        // remove all old bboxes
        observations.iter_mut().for_each(|f| {
            if let Some(e) = &mut f.attr_mut() {
                e.drop_bbox();
            }
        });

        observations.sort_by(|e1, e2| {
            e2.attr()
                .as_ref()
                .unwrap()
                .visual_quality()
                .partial_cmp(&e1.attr().as_ref().unwrap().visual_quality())
                .unwrap()
        });

        if observations.len() >= self.opts.visual_max_observations {
            observations.truncate(observations.len() - 1);
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
            if Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox) {
                None
            } else {
                let conf = if candidate_observation_bbox.confidence
                    < self.opts.positional_min_confidence
                {
                    self.opts.positional_min_confidence
                } else {
                    candidate_observation_bbox.confidence
                };

                match self.opts.positional_kind {
                    PositionalMetricType::Mahalanobis => {
                        let state = track_attributes.get_state().unwrap();
                        let f = Universal2DBoxKalmanFilter::new(
                            track_attributes.get_position_weight(),
                            track_attributes.get_velocity_weight(),
                        );
                        let dist = f.distance(state, candidate_observation_bbox);
                        Some(Universal2DBoxKalmanFilter::calculate_cost(dist, true) / conf)
                    }
                    PositionalMetricType::IoU(threshold) => {
                        let box_m_opt = Universal2DBox::calculate_metric_object(
                            &candidate_observation_bbox_opt.as_ref(),
                            &track_observation_bbox_opt.as_ref(),
                        );
                        box_m_opt.map(|e| e * conf).filter(|e| *e >= threshold)
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
            let d = match self.opts.visual_kind {
                VisualSortMetricType::Euclidean(_) => {
                    euclidean(candidate_observation_feature, track_observation_feature)
                }
                VisualSortMetricType::Cosine(_) => {
                    cosine(candidate_observation_feature, track_observation_feature)
                }
            };

            if self.opts.visual_kind.is_ok(d) {
                Some(self.opts.visual_kind.distance_to_weight(d))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn feature_can_be_used(
        &self,
        bbox_opt: &Option<&Universal2DBox>,
        feature_quality: f32,
        visual_minimal_quality: f32,
        visual_own_area_percentage: &Option<f32>,
        visual_minimal_area_percentage: f32,
    ) -> bool {
        let quality_is_ok = feature_quality >= visual_minimal_quality;

        let percentage_is_ok = visual_own_area_percentage
            .map(|p| p >= visual_minimal_area_percentage)
            .unwrap_or(true);

        let bbox_is_ok = if let Some(bbox) = bbox_opt {
            let area = bbox.area();
            area >= self.opts.visual_minimal_area
        } else {
            unreachable!("The bbox must always present for candidate track");
        };

        bbox_is_ok && quality_is_ok && percentage_is_ok
    }
}

impl ObservationMetric<VisualAttributes, VisualObservationAttributes> for VisualMetric {
    fn metric(
        &self,
        mq: &MetricQuery<VisualAttributes, VisualObservationAttributes>,
    ) -> MetricOutput<f32> {
        let candidate_obs_attrs = mq
            .candidate_observation
            .attr()
            .as_ref()
            .expect("Observation attributes must always present.");

        let track_obs_attrs = mq
            .track_observation
            .attr()
            .as_ref()
            .expect("Observation attributes must always present.");

        let candidate_bbox_opt = candidate_obs_attrs.bbox_opt();
        let candidate_feature_q = candidate_obs_attrs.visual_quality();
        let candidate_own_area_percentage_opt = candidate_obs_attrs.own_area_percentage_opt();

        let track_bbox_opt = track_obs_attrs.bbox_opt();

        let candidate_feature_opt = mq.candidate_observation.feature().as_ref();
        let track_feature_opt = mq.track_observation.feature().as_ref();

        Some((
            self.positional_metric(candidate_bbox_opt, track_bbox_opt, mq.track_attrs),
            if self.feature_can_be_used(
                &candidate_bbox_opt.as_ref(),
                candidate_feature_q,
                self.opts.visual_minimal_quality_use,
                candidate_own_area_percentage_opt,
                self.opts.visual_minimal_own_area_percentage_use,
            ) {
                match (candidate_feature_opt, track_feature_opt) {
                    (Some(c), Some(t)) => self.visual_metric(c, t, mq.track_attrs),
                    _ => None,
                }
            } else {
                None
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
        is_merge: bool,
    ) -> Result<()> {
        let mut observation = observations
            .pop()
            .expect("At least one observation must present in the track.");

        let obs_attrs = observation
            .attr()
            .as_ref()
            .expect("Observation attributes must always present.");

        let observation_bbox = obs_attrs.unchecked_bbox_ref();
        let feature_quality = obs_attrs.visual_quality();
        let own_area_percentage_opt = *obs_attrs.own_area_percentage_opt();

        let mut predicted_bbox = attrs.make_prediction(observation_bbox);
        attrs.update_history(
            observation_bbox,
            &predicted_bbox,
            observation.feature().clone(),
        );

        if is_merge
            && !self.feature_can_be_used(
                &Some(observation_bbox),
                feature_quality,
                self.opts.visual_minimal_quality_collect,
                &own_area_percentage_opt,
                self.opts.visual_minimal_own_area_percentage_collect,
            )
        {
            *observation.feature_mut() = None;
        }

        *observation.attr_mut() = Some(if let Some(percentage) = own_area_percentage_opt {
            VisualObservationAttributes::with_own_area_percentage(
                feature_quality,
                match self.opts.positional_kind {
                    PositionalMetricType::Mahalanobis => predicted_bbox,
                    PositionalMetricType::IoU(_) => {
                        predicted_bbox.gen_vertices();
                        predicted_bbox
                    }
                },
                percentage,
            )
        } else {
            VisualObservationAttributes::new(
                feature_quality,
                match self.opts.positional_kind {
                    PositionalMetricType::Mahalanobis => predicted_bbox,
                    PositionalMetricType::IoU(_) => {
                        predicted_bbox.gen_vertices();
                        predicted_bbox
                    }
                },
            )
        });

        self.optimize_observations(observations);
        observations.push(observation);
        let current_len = observations.len();
        observations.swap(0, current_len - 1);

        attrs.visual_features_collected_count = observations
            .iter()
            .filter(|f| f.feature().is_some())
            .count();

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<VisualObservationAttributes>>,
    ) -> Vec<ObservationMetricOk<VisualObservationAttributes>> {
        unfiltered
            .into_iter()
            .filter(|res| res.feature_distance.is_some() || res.attribute_metric.is_some())
            .collect()
    }
}

#[cfg(test)]
mod optimize {
    use crate::examples::vec2;
    use crate::track::{Observation, ObservationMetric};
    use crate::trackers::sort::{PositionalMetricType, SortAttributesOptions};
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
    use crate::trackers::visual_sort::metric::builder::VisualMetricBuilder;
    use crate::trackers::visual_sort::metric::VisualSortMetricType;
    use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual_sort::track_attributes::VisualAttributes;
    use crate::utils::bbox::{BoundingBox, Universal2DBox};
    use std::sync::Arc;

    #[test]
    fn optimization_regular() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .build();

        let mut attrs = VisualAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )));

        let mut obs = vec![Observation::new(
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
            Observation::new(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            Observation::new(
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
        assert_eq!(attrs.visual_features_collected_count, 1);
        assert_eq!(obs.len(), 2);
        assert!(
            {
                let e = &obs[0];
                e.feature().is_none()
                    && matches!(
                        e.attr().as_ref().unwrap().bbox_opt(),
                        Some(Universal2DBox { .. })
                    )
            } && {
                let e = &obs[1];
                e.feature().is_some() && matches!(e.attr().as_ref().unwrap().bbox_opt(), None)
            }
        );

        let mut obs = vec![
            Observation::new(
                Some(VisualObservationAttributes::new(
                    0.8,
                    BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.0)),
            ),
            Observation::new(
                Some(VisualObservationAttributes::new(
                    0.7,
                    BoundingBox::new(0.2, 0.2, 5.0, 10.0).as_xyaah(),
                )),
                None,
            ),
            Observation::new(
                Some(VisualObservationAttributes::new(
                    1.0,
                    BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                )),
                Some(vec2(0.1, 1.1)),
            ),
            Observation::new(
                Some(VisualObservationAttributes::new(
                    0.6,
                    BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                )),
                Some(vec2(0.0, 1.1)),
            ),
        ];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, false)
            .unwrap();

        assert_eq!(attrs.observed_features.len(), 3);
        assert_eq!(attrs.observed_boxes.len(), 3);
        assert_eq!(attrs.predicted_boxes.len(), 3);
        assert_eq!(attrs.track_length, 3);
        assert_eq!(attrs.visual_features_collected_count, 3);
        assert_eq!(obs.len(), 3);
        assert!(matches!(
            &obs[2],
            Observation(Some(a), Some(o)) if a.bbox_opt().is_none() && a.visual_quality() == 1.0 && o[0].to_array()[..2] == [0.1 , 1.1]
        ));
        assert!(matches!(
            &obs[1],
            Observation(Some(a), Some(o)) if a.bbox_opt().is_none() && a.visual_quality() == 0.8 && o[0].to_array()[..2] == [0.0 , 1.0]
        ));
        assert!(matches!(
            &obs[0],
            Observation(Some(a), Some(o)) if a.bbox_opt().is_some() && a.visual_quality() == 0.6 && o[0].to_array()[..2] == [0.0 , 1.1]
        ));
    }

    #[test]
    fn optimize_low_quality() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_quality_collect(0.3)
            .build();

        let mut attrs = VisualAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )));

        let mut obs = vec![Observation::new(
            Some(VisualObservationAttributes::new(
                0.25,
                BoundingBox::new(0.0, 0.0, 5.0, 10.0).as_xyaah(),
            )),
            Some(vec2(0.0, 1.0)),
        )];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, true)
            .unwrap();

        assert!(
            obs[0].feature().is_none(),
            "Feature must be removed because the quality is lower than minimal required quality for collected features"
        );
    }

    #[test]
    fn optimize_small_box() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_area(1.0)
            .build();

        let mut attrs = VisualAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )));

        let mut obs = vec![Observation::new(
            Some(VisualObservationAttributes::new(
                0.25,
                BoundingBox::new(0.0, 0.0, 0.8, 1.0).as_xyaah(),
            )),
            Some(vec2(0.0, 1.0)),
        )];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, true)
            .unwrap();

        assert!(
            obs[0].feature().is_none(),
            "Feature must be removed because the box area is lower than minimal area required for collected features"
        );
    }

    #[test]
    fn optimize_overlap() {
        let mut metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_area(1.0)
            .visual_minimal_own_area_percentage_collect(0.6)
            .build();

        let mut attrs = VisualAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )));

        let mut obs = vec![Observation::new(
            Some(VisualObservationAttributes::with_own_area_percentage(
                0.8,
                BoundingBox::new(0.0, 0.0, 8.0, 10.0).as_xyaah(),
                0.5,
            )),
            Some(vec2(0.0, 1.0)),
        )];

        metric
            .optimize(0, &[], &mut attrs, &mut obs, 0, true)
            .unwrap();

        assert!(
            obs[0].feature().is_none(),
            "Feature must be removed because the minimum own area percentage is lower than specified in metric options"
        );
    }
}

#[cfg(test)]
mod metric_tests {
    use crate::examples::vec2;
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
    use crate::store::TrackStore;
    use crate::track::ObservationMetricOk;
    use crate::trackers::sort::{PositionalMetricType, SortAttributesOptions};
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
    use crate::trackers::visual_sort::metric::builder::VisualMetricBuilder;
    use crate::trackers::visual_sort::metric::{VisualMetric, VisualSortMetricType};
    use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual_sort::track_attributes::VisualAttributes;
    use crate::utils::bbox::BoundingBox;
    use crate::EPS;
    use std::sync::Arc;

    fn default_attrs() -> VisualAttributes {
        VisualAttributes::new(Arc::new(SortAttributesOptions::new(
            None,
            0,
            5,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        )))
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
                    .observation_attributes(VisualObservationAttributes::with_own_area_percentage(
                        1.0,
                        BoundingBox::new(100.3, 0.3, 5.1, 10.0).as_xyaah(),
                        0.1,
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
                    .observation_attributes(VisualObservationAttributes::with_own_area_percentage(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                        0.2,
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
            .visual_metric(VisualSortMetricType::cosine(1.0))
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
            } if (x - 1.0).abs() < EPS && y.abs() < EPS));
    }

    #[test]
    fn metric_maha() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_metric(VisualSortMetricType::Euclidean(10.0))
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
        dbg!(&dists[0]);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: Some(y)
            } if (x - 100.0).abs() < EPS && y.abs() < EPS));
    }

    #[test]
    fn visual_track_too_short() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
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
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
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

    #[test]
    fn visual_track_small_bbox() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_track_length(1)
            .visual_minimal_area(1.0)
            .build();

        let store = default_store(metric);

        let track1 = store
            .new_track(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 0.8, 1.0).as_xyaah(),
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
        dbg!(&dists);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: None,
                feature_distance: None // feature box is too small to use feature
            }
        ));
    }

    #[test]
    fn visual_quality_low() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_quality_use(0.3)
            .visual_minimal_track_length(1)
            .build();

        let store = default_store(metric);

        let track1 = store
            .new_track(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::new(
                        0.2,
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
                feature_distance: None     // quality is low
            } if (x - 1.0).abs() < EPS));
    }

    #[test]
    fn visual_own_area_percentage_low() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::IoU(0.3))
            .visual_metric(VisualSortMetricType::Euclidean(f32::MAX))
            .visual_minimal_own_area_percentage_use(0.6)
            .visual_minimal_track_length(1)
            .build();

        let store = default_store(metric);

        let track1 = store
            .new_track(1)
            .observation(
                ObservationBuilder::new(0)
                    .observation(vec2(0.1, 1.0))
                    .observation_attributes(VisualObservationAttributes::with_own_area_percentage(
                        1.0,
                        BoundingBox::new(0.3, 0.3, 5.1, 10.0).as_xyaah(),
                        0.5,
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
        dbg!(&dists);
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 2,
                attribute_metric: Some(x),
                feature_distance: None     // own area percentage is low
            } if (x - 1.0).abs() < EPS));
    }
}
