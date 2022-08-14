use crate::trackers::sort::PositionalMetricType;
use crate::trackers::visual::metric::{
    PyVisualMetricType, VisualMetric, VisualMetricOptions, VisualMetricType,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct VisualMetricBuilder {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    visual_minimal_track_length: usize,
    visual_minimal_area: f32,
    visual_minimal_quality_use: f32,
    visual_minimal_quality_collect: f32,
    visual_max_observations: usize,
    visual_min_votes: usize,
    visual_minimal_own_area_percentage_use: f32,
    visual_minimal_own_area_percentage_collect: f32,
}

/// By default the metric object is constructed with: Euclidean visual metric, IoU(0.3) positional metric
/// and minimal visual track length = 3
///
impl Default for VisualMetricBuilder {
    fn default() -> Self {
        VisualMetricBuilder {
            visual_kind: VisualMetricType::Euclidean(f32::MAX),
            positional_kind: PositionalMetricType::IoU(0.3),
            visual_minimal_track_length: 3,
            visual_minimal_area: 0.0,
            visual_minimal_quality_use: 0.0,
            visual_minimal_quality_collect: 0.0,
            visual_max_observations: 5,
            visual_min_votes: 1,
            visual_minimal_own_area_percentage_use: 0.0,
            visual_minimal_own_area_percentage_collect: 0.0,
        }
    }
}

impl VisualMetricBuilder {
    pub(crate) fn visual_metric_py(&mut self, metric: PyVisualMetricType) {
        self.visual_kind = metric.0;
    }

    pub(crate) fn visual_min_votes_py(&mut self, n: i64) {
        self.visual_min_votes =
            usize::try_from(n).expect("The number of required votes must be a positive number.");
    }

    pub(crate) fn positional_metric_py(&mut self, metric: PositionalMetricType) {
        if let PositionalMetricType::IoU(t) = metric {
            assert!(
                t > 0.0 && t < 1.0,
                "Threshold must lay between (0.0 and 1.0)"
            );
        }
        self.positional_kind = metric;
    }

    pub(crate) fn visual_minimal_track_length_py(&mut self, length: usize) {
        assert!(
            length > 0,
            "The minimum amount of visual features collected before visual metric is applied should be greater than 0."
        );
        self.visual_minimal_track_length = length;
    }

    pub(crate) fn visual_minimal_area_py(&mut self, area: f32) {
        assert!(
            area >= 0.0,
            "The minimum area of bbox for visual feature distance calculated and feature collected should be greater than 0."
        );
        self.visual_minimal_area = area;
    }

    pub(crate) fn visual_minimal_own_area_percentage_use_py(&mut self, area: f32) {
        assert!(
            (0.0..=1.0).contains(&area),
            "Argument must be contained within (0.0..=1.0)"
        );
        self.visual_minimal_own_area_percentage_use = area;
    }

    pub(crate) fn visual_minimal_own_area_percentage_collect_py(&mut self, area: f32) {
        assert!(
            (0.0..=1.0).contains(&area),
            "Argument must be contained within (0.0..=1.0)"
        );
        self.visual_minimal_own_area_percentage_collect = area;
    }

    pub(crate) fn visual_minimal_quality_use_py(&mut self, q: f32) {
        assert!(
            q >= 0.0,
            "The minimum quality of visual feature should be greater than or equal to 0.0."
        );
        self.visual_minimal_quality_use = q;
    }

    pub(crate) fn visual_max_observations_py(&mut self, n: usize) {
        self.visual_max_observations = n;
    }

    pub(crate) fn visual_minimal_quality_collect_py(&mut self, q: f32) {
        assert!(
            q >= 0.0,
            "The minimum quality of visual feature should be greater than or equal to 0.0."
        );
        self.visual_minimal_quality_collect = q;
    }
}

impl VisualMetricBuilder {
    pub fn visual_minimal_own_area_percentage_use(mut self, area: f32) -> Self {
        self.visual_minimal_own_area_percentage_use_py(area);
        self
    }

    pub fn visual_minimal_own_area_percentage_collect(mut self, area: f32) -> Self {
        self.visual_minimal_own_area_percentage_collect_py(area);
        self
    }

    pub fn visual_min_votes(mut self, n: usize) -> Self {
        self.visual_min_votes = n;
        self
    }

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
        assert!(
            0 < self.visual_min_votes
                && 0 < self.visual_minimal_track_length
                && self.visual_minimal_track_length <= self.visual_max_observations,
            "Ratios for (visual_min_votes, visual_minimal_track_length, visual_max_observations) are broken"
        );
        VisualMetric {
            opts: Arc::new(VisualMetricOptions {
                visual_kind: self.visual_kind,
                positional_kind: self.positional_kind,
                visual_minimal_track_length: self.visual_minimal_track_length,
                visual_minimal_area: self.visual_minimal_area,
                visual_minimal_quality_use: self.visual_minimal_quality_use,
                visual_minimal_quality_collect: self.visual_minimal_quality_collect,
                visual_max_observations: self.visual_max_observations,
                visual_min_votes: self.visual_min_votes,
                visual_minimal_own_area_percentage_use: self.visual_minimal_own_area_percentage_use,
                visual_minimal_own_area_percentage_collect: self
                    .visual_minimal_own_area_percentage_collect,
            }),
        }
    }
}
