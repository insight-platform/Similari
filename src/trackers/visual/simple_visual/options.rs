use crate::trackers::sort::SortAttributesOptions;
use crate::trackers::visual::metric::builder::VisualMetricBuilder;
use crate::trackers::visual::metric::{
    PositionalMetricType, PyPositionalMetricType, PyVisualMetricType, VisualMetric,
    VisualMetricType,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

#[pyclass]
#[derive(Debug, Clone)]
pub struct VisualSortOptions {
    max_idle_epochs: usize,
    kept_history_length: usize,
    metric_builder: VisualMetricBuilder,
}

impl VisualSortOptions {
    pub fn build(self) -> (SortAttributesOptions, VisualMetric) {
        (
            SortAttributesOptions::new(
                Some(RwLock::new(HashMap::default())),
                self.max_idle_epochs,
                self.kept_history_length,
            ),
            self.metric_builder.build(),
        )
    }

    pub fn max_idle_epochs(mut self, n: usize) -> Self {
        self.max_idle_epochs = n;
        self
    }

    pub fn kept_history_length(mut self, n: usize) -> Self {
        self.kept_history_length = n;
        self
    }

    pub fn visual_metric(mut self, metric: VisualMetricType) -> Self {
        self.metric_builder = self.metric_builder.visual_metric(metric);
        self
    }

    pub fn visual_min_votes(mut self, n: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_min_votes(n);
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
            max_idle_epochs: 2,
            kept_history_length: 10,
            metric_builder: VisualMetricBuilder::default(),
        }
    }
}

#[pymethods]
impl VisualSortOptions {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    #[pyo3(name = "max_idle_epochs", text_signature = "($self, n)")]
    fn max_idle_epochs_py(&mut self, n: i64) {
        self.max_idle_epochs = n.try_into().expect("Parameter must be a positive number");
    }

    #[pyo3(name = "kept_history_length", text_signature = "($self, n)")]
    fn kept_history_length_py(&mut self, n: i64) {
        self.kept_history_length = n.try_into().expect("Parameter must be a positive number");
    }

    #[pyo3(name = "visual_min_votes", text_signature = "($self, n)")]
    fn visual_min_votes_py(&mut self, n: i64) {
        self.metric_builder.visual_min_votes_py(n);
    }

    #[pyo3(name = "visual_metric", text_signature = "($self, metric)")]
    fn visual_metric_py(&mut self, metric: PyVisualMetricType) {
        self.metric_builder.visual_metric_py(metric);
    }

    #[pyo3(name = "positional_metric", text_signature = "($self, metric)")]
    fn positional_metric_py(&mut self, metric: PyPositionalMetricType) {
        self.metric_builder.positional_metric_py(metric.0);
    }

    #[pyo3(
        name = "visual_minimal_track_length",
        text_signature = "($self, length)"
    )]
    fn visual_minimal_track_length_py(&mut self, length: i64) {
        self.metric_builder.visual_minimal_track_length_py(
            length
                .try_into()
                .expect("Parameter must be a positive number"),
        );
    }

    #[pyo3(name = "visual_minimal_area", text_signature = "($self, area)")]
    fn visual_minimal_area_py(&mut self, area: f32) {
        self.metric_builder.visual_minimal_area_py(area);
    }

    #[pyo3(name = "visual_minimal_quality_use", text_signature = "($self, q)")]
    fn visual_minimal_quality_use_py(&mut self, q: f32) {
        self.metric_builder.visual_minimal_quality_use_py(q);
    }

    #[pyo3(name = "visual_max_observations", text_signature = "($self, n)")]
    fn visual_max_observations_py(&mut self, n: i64) {
        self.metric_builder
            .visual_max_observations_py(n.try_into().expect("Parameter must be a positive number"));
    }

    #[pyo3(name = "visual_minimal_quality_collect", text_signature = "($self, q)")]
    fn visual_minimal_quality_collect_py(&mut self, q: f32) {
        self.metric_builder.visual_minimal_quality_collect_py(q);
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::visual::metric::{
        PositionalMetricType, PyPositionalMetricType, PyVisualMetricType, VisualMetricType,
    };
    use crate::trackers::visual::simple_visual::options::VisualSortOptions;

    #[test]
    fn visual_sort_options_builder() {
        let (opts, metric) = dbg!(VisualSortOptions::new()
            .max_idle_epochs(3)
            .kept_history_length(10)
            .visual_metric(VisualMetricType::Euclidean(100.0))
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(3)
            .visual_minimal_area(5.0)
            .visual_minimal_quality_use(0.45)
            .visual_minimal_quality_collect(0.5)
            .visual_max_observations(25)
            .visual_min_votes(5)
            .build());

        let mut opts_builder = VisualSortOptions::new();
        opts_builder.max_idle_epochs_py(3);
        opts_builder.kept_history_length_py(10);
        opts_builder.visual_metric_py(PyVisualMetricType::euclidean(100.0));
        opts_builder.positional_metric_py(PyPositionalMetricType::maha());
        opts_builder.visual_minimal_track_length_py(3);
        opts_builder.visual_minimal_area_py(5.0);
        opts_builder.visual_minimal_quality_use_py(0.45);
        opts_builder.visual_minimal_quality_collect_py(0.5);
        opts_builder.visual_max_observations_py(25);
        opts_builder.visual_min_votes_py(5);
        let (opts_py, metric_py) = dbg!(opts_builder.build());

        assert_eq!(format!("{:?}", opts), format!("{:?}", opts_py));
        assert_eq!(format!("{:?}", metric), format!("{:?}", metric_py));
    }
}
