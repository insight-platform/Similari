use crate::trackers::sort::SortAttributesOptions;
use crate::trackers::visual::metric::builder::VisualMetricBuilder;
use crate::trackers::visual::metric::{PositionalMetricType, VisualMetric, VisualMetricType};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

#[pyclass]
pub struct VisualSortOptions {
    max_idle_epochs: usize,
    history_length: usize,
    metric_builder: VisualMetricBuilder,
}

impl VisualSortOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> (SortAttributesOptions, VisualMetric) {
        (
            SortAttributesOptions::new(
                Some(RwLock::new(HashMap::default())),
                self.max_idle_epochs,
                self.history_length,
            ),
            self.metric_builder.build(),
        )
    }

    pub fn max_idle_epochs(mut self, n: usize) -> Self {
        self.max_idle_epochs = n;
        self
    }

    pub fn history_length(mut self, n: usize) -> Self {
        self.history_length = n;
        self
    }

    pub fn visual_metric(mut self, metric: VisualMetricType) -> Self {
        self.metric_builder = self.metric_builder.visual_metric(metric);
        self
    }

    pub fn positional_metric(mut self, metric: PositionalMetricType) -> Self {
        self.metric_builder = self.metric_builder.positional_metric(metric);
        self
    }

    pub fn visual_minimal_track_length(mut self, length: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_track_length(length);
        self
    }

    pub fn visual_minimal_area(mut self, area: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_area(area);
        self
    }

    pub fn visual_minimal_quality_use(mut self, q: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_quality_use(q);
        self
    }

    pub fn visual_max_observations(mut self, n: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_max_observations(n);
        self
    }

    pub fn visual_minimal_quality_collect(mut self, q: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_quality_collect(q);
        self
    }
}

impl Default for VisualSortOptions {
    fn default() -> Self {
        Self {
            max_idle_epochs: 10,
            history_length: 300,
            metric_builder: VisualMetricBuilder::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::visual::simple_visual::options::VisualSortOptions;

    #[test]
    fn visual_sort_options_builder() {
        let (opts, metric) = VisualSortOptions::new()
            .history_length(10)
            .visual_max_observations(25)
            .build();
        assert_eq!(opts.history_length, 10);
        assert_eq!(metric.opts.visual_max_observations, 25);
    }
}
