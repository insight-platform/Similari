use crate::trackers::visual::metric::{
    PositionalMetricType, VisualMetric, VisualMetricOptions, VisualMetricType,
};
use std::sync::Arc;

#[derive(Debug)]
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
    pub(crate) fn visual_metric_py(&mut self, metric: VisualMetricType) {
        self.visual_kind = metric;
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
                visual_minimal_area: self.visual_minimal_area,
                visual_minimal_quality_use: self.visual_minimal_quality_use,
                visual_minimal_quality_collect: self.visual_minimal_quality_collect,
                visual_max_observations: self.visual_max_observations,
            }),
        }
    }
}
