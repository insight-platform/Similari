use crate::trackers::sort::{PositionalMetricType, SortAttributesOptions};
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::trackers::visual_sort::metric::builder::VisualMetricBuilder;
use crate::trackers::visual_sort::metric::{VisualMetric, VisualSortMetricType};
use std::collections::HashMap;
use std::sync::RwLock;

/// Class that is used to configure the Visual Tracker
#[derive(Debug, Clone)]
pub struct VisualSortOptions {
    max_idle_epochs: usize,
    kept_history_length: usize,
    spatio_temporal_constraints: SpatioTemporalConstraints,
    metric_builder: VisualMetricBuilder,
    kalman_position_weight: f32,
    kalman_velocity_weight: f32,
}

impl VisualSortOptions {
    pub(crate) fn build(self) -> (SortAttributesOptions, VisualMetric) {
        (
            SortAttributesOptions::new(
                Some(RwLock::new(HashMap::default())),
                self.max_idle_epochs,
                self.kept_history_length,
                self.spatio_temporal_constraints,
                self.kalman_position_weight,
                self.kalman_velocity_weight,
            ),
            self.metric_builder.build(),
        )
    }

    /// The number of epochs the track remains active.
    ///
    /// Lets the Frame Rate per second is `30`, setting `max_idle_epochs` to `30` means that the
    /// track in store will be active even if only one new observation was merged with it during the
    /// second. If during `30` invocations of `predict` for the scene, where the track is defined,
    /// no observations are merged with it, the track will be marked as wasted, and no further
    /// observations will be merged with it.
    ///
    pub fn max_idle_epochs(mut self, n: usize) -> Self {
        self.max_idle_epochs = n;
        self
    }

    /// The number of last observations, predictions, and features kept within the track attributes.
    ///
    /// The track's attributes may accumulate last observations, predictions, and features for the
    /// caller's purpose. To protect the system from overflow `kept_history_length` parameter is used.
    /// It forces the track to keep only the last `N` values instead of unlimited history. The parameter
    /// is important when one uses the tracker in offline mode -  when the wasted tracks are used to get
    /// the history. If the tracker is an online tracker, setting `1` is a reasonable value to keep memory
    /// utilization low.
    ///
    pub fn kept_history_length(mut self, n: usize) -> Self {
        assert!(n > 0, "History length must be a positive number");
        self.kept_history_length = n;
        self
    }

    /// The method is used to calculate the distance for visual_sort feature vectors.
    ///
    /// Currently, cosine and euclidean metrics are supported. The one you choose
    /// is defined by the ReID model used.
    ///    
    pub fn visual_metric(mut self, metric: VisualSortMetricType) -> Self {
        self.metric_builder = self.metric_builder.visual_metric(metric);
        self
    }

    /// The minimal number of votes that is required to allow a track candidate to surpass the enabling
    /// threshold of the visual_sort voting. The maximum allowed number of visual_sort features kept for the track
    /// is defined by `visual_max_observations`.
    ///
    /// _Don't confuse `visual_max_observations` with `kept_history_length` - they have no relation.
    /// The later is only used for caller purposes, not for track prediction._
    ///
    /// When the track candidate consisting of the single observation is compared versus tracks kept in
    /// the store the system calculates up to `N` distances (`1 X N`), where `N` at most is equal to
    /// `visual_max_observations`, but can be less if the track is short or previous observations were
    /// ignored due to quality or other constraints.
    ///
    /// Only when `N >= visual_min_votes`, the track candidate is used in leader selection.
    ///
    pub fn visual_min_votes(mut self, n: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_min_votes(n);
        self
    }

    /// The maximum number of visual_sort observations kept in the track for visual_sort estimations. The features
    /// are collected in the track from the candidates, and when the `visual_max_observations` is
    /// reached, the features with lower quality are wiped from the track.
    ///
    pub fn visual_max_observations(mut self, n: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_max_observations(n);
        self
    }

    /// Minimal allowed confidence for bounding boxes. If the confidence is less than specified it is
    /// corrected to be the minimal
    ///
    pub fn positional_min_confidence(mut self, conf: f32) -> Self {
        self.metric_builder = self.metric_builder.positional_min_confidence(conf);
        self
    }

    /// The constraints define how far the candidate is allowed to be from a track’s last box to
    /// participate in the selection for the track. If the track candidate is too far from the
    /// track kept in the store, it is skipped from the comparison.
    ///
    pub fn spatio_temporal_constraints(mut self, constraints: SpatioTemporalConstraints) -> Self {
        self.spatio_temporal_constraints = constraints;
        self
    }

    /// The parameter defines which positional metric is used to calculate distances between the track
    /// candidate and tracks kept in the store. There are two metrics are supported - the Mahalanobis metric
    /// and the IoU metric.
    ///
    pub fn positional_metric(mut self, metric: PositionalMetricType) -> Self {
        self.metric_builder = self.metric_builder.positional_metric(metric);
        self
    }

    /// The minimally required number of visual_sort features in the track that enables their usage in
    /// candidates estimation. If the track is short and there are fewer features collected than
    /// `visual_minimal_track_length` then candidates are estimated against it only by positional
    /// distance. Keep in mind that this parameter must be less than or equal
    /// to `visual_max_observations` to have sense.
    ///
    pub fn visual_minimal_track_length(mut self, length: usize) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_track_length(length);
        self
    }

    /// The minimal required area of track candidate's bounding box to use the visual_sort feature in estimation.
    /// This parameter protects from the low-quality features received from the smallish boxes.
    ///
    pub fn visual_minimal_area(mut self, area: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_area(area);
        self
    }

    /// The visual_sort quality threshold of a feature that activates the visual_sort estimation of a candidate
    /// versus the tracks kept in the store.
    ///
    pub fn visual_minimal_quality_use(mut self, q: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_quality_use(q);
        self
    }

    /// The visual_sort quality threshold of a feature that activates the adding of the visual_sort feature
    /// to the track's visual_sort features.
    ///
    pub fn visual_minimal_quality_collect(mut self, q: f32) -> Self {
        self.metric_builder = self.metric_builder.visual_minimal_quality_collect(q);
        self
    }

    /// The threshold is calculated as `solely_owned_area / all_area` of the bounding box that
    /// prevents low-quality visual_sort features received in a messy environment from being used in
    /// visual_sort predictions.
    ///
    pub fn visual_minimal_own_area_percentage_use(mut self, area: f32) -> Self {
        self.metric_builder = self
            .metric_builder
            .visual_minimal_own_area_percentage_use(area);
        self
    }

    /// The threshold is calculated as `solely_owned_area / all_area` of the bounding box that prevents
    /// low-quality visual_sort features received in a messy environment from being collected to a track
    /// for making visual_sort predictions.
    ///
    pub fn visual_minimal_own_area_percentage_collect(mut self, area: f32) -> Self {
        self.metric_builder = self
            .metric_builder
            .visual_minimal_own_area_percentage_collect(area);
        self
    }

    pub fn kalman_position_weight(mut self, weight: f32) -> Self {
        self.kalman_position_weight = weight;
        self
    }

    pub fn kalman_velocity_weight(mut self, weight: f32) -> Self {
        self.kalman_velocity_weight = weight;
        self
    }
}

impl Default for VisualSortOptions {
    fn default() -> Self {
        Self {
            max_idle_epochs: 2,
            kept_history_length: 10,
            metric_builder: VisualMetricBuilder::default(),
            spatio_temporal_constraints: SpatioTemporalConstraints::default(),
            kalman_position_weight: 1.0 / 20.0,
            kalman_velocity_weight: 1.0 / 160.0,
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::trackers::sort::python::PyPositionalMetricType;
    use crate::trackers::spatio_temporal_constraints::python::PySpatioTemporalConstraints;
    use crate::trackers::visual_sort::metric::python::PyVisualSortMetricType;

    use super::VisualSortOptions;
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "VisualSortOptions")]
    pub struct PyVisualSortOptions(pub(crate) VisualSortOptions);

    #[pymethods]
    impl PyVisualSortOptions {
        #[new]
        pub(crate) fn new() -> Self {
            Self(VisualSortOptions::default())
        }

        #[pyo3(text_signature = "($self, n)")]
        pub(crate) fn max_idle_epochs(&mut self, n: i64) {
            self.0.max_idle_epochs = n.try_into().expect("Parameter must be a positive number");
        }

        #[pyo3(text_signature = "($self, n)")]
        pub(crate) fn kept_history_length(&mut self, n: i64) {
            self.0.kept_history_length = n.try_into().expect("Parameter must be a positive number");
        }

        #[pyo3(text_signature = "($self, n)")]
        pub(crate) fn visual_min_votes(&mut self, n: i64) {
            self.0.metric_builder.set_visual_min_votes(n as _);
        }

        #[pyo3(text_signature = "($self, metric)")]
        pub(crate) fn visual_metric(&mut self, metric: PyVisualSortMetricType) {
            self.0.metric_builder.set_visual_kind(metric.0);
        }

        #[pyo3(text_signature = "($self, constraints)")]
        pub(crate) fn spatio_temporal_constraints(
            &mut self,
            constraints: PySpatioTemporalConstraints,
        ) {
            self.0.spatio_temporal_constraints = constraints.0;
        }

        #[pyo3(text_signature = "($self, metric)")]
        pub(crate) fn positional_metric(&mut self, metric: PyPositionalMetricType) {
            self.0.metric_builder.set_positional_kind(metric.0);
        }

        #[pyo3(text_signature = "($self, length)")]
        pub(crate) fn visual_minimal_track_length(&mut self, length: i64) {
            self.0.metric_builder.set_visual_minimal_track_length(
                length
                    .try_into()
                    .expect("Parameter must be a positive number"),
            );
        }

        #[pyo3(text_signature = "($self, area)")]
        pub(crate) fn visual_minimal_area(&mut self, area: f32) {
            self.0.metric_builder.set_visual_minimal_area(area);
        }

        #[pyo3(text_signature = "($self, q)")]
        pub(crate) fn visual_minimal_quality_use(&mut self, q: f32) {
            self.0.metric_builder.set_visual_minimal_quality_use(q);
        }

        #[pyo3(text_signature = "($self, conf)")]
        pub(crate) fn positional_min_confidence(&mut self, conf: f32) {
            self.0.metric_builder.set_positional_min_confidence(conf);
        }

        #[pyo3(text_signature = "($self, n)")]
        pub(crate) fn visual_max_observations(&mut self, n: i64) {
            self.0.metric_builder.set_visual_max_observations(
                n.try_into().expect("Parameter must be a positive number"),
            );
        }

        #[pyo3(text_signature = "($self, q)")]
        pub(crate) fn visual_minimal_quality_collect(&mut self, q: f32) {
            self.0.metric_builder.set_visual_minimal_quality_collect(q);
        }

        #[pyo3(text_signature = "($self, area)")]
        pub(crate) fn visual_minimal_own_area_percentage_use(&mut self, area: f32) {
            self.0
                .metric_builder
                .set_visual_minimal_own_area_percentage_use(area);
        }

        #[pyo3(text_signature = "($self, area)")]
        pub(crate) fn visual_minimal_own_area_percentage_collect(&mut self, area: f32) {
            self.0
                .metric_builder
                .set_visual_minimal_own_area_percentage_collect(area);
        }

        #[pyo3(text_signature = "($self, weight)")]
        pub(crate) fn kalman_position_weight(&mut self, weight: f32) {
            self.0.kalman_position_weight = weight;
        }

        #[pyo3(text_signature = "($self, weight)")]
        pub(crate) fn kalman_velocity_weight(&mut self, weight: f32) {
            self.0.kalman_velocity_weight = weight;
        }

        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn __str__(&self) -> String {
            format!("{:#?}", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::sort::python::PyPositionalMetricType;
    use crate::trackers::sort::PositionalMetricType;
    use crate::trackers::spatio_temporal_constraints::python::PySpatioTemporalConstraints;
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
    use crate::trackers::visual_sort::metric::python::PyVisualSortMetricType;
    use crate::trackers::visual_sort::metric::VisualSortMetricType;
    use crate::trackers::visual_sort::options::python::PyVisualSortOptions;
    use crate::trackers::visual_sort::options::VisualSortOptions;

    #[test]
    fn visual_sort_options_builder() {
        let (opts, metric) = dbg!(VisualSortOptions::default()
            .max_idle_epochs(3)
            .kept_history_length(10)
            .visual_metric(VisualSortMetricType::Euclidean(100.0))
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(3)
            .visual_minimal_area(5.0)
            .visual_minimal_quality_use(0.45)
            .visual_minimal_quality_collect(0.5)
            .visual_max_observations(25)
            .visual_min_votes(5)
            .positional_min_confidence(0.13)
            .visual_minimal_own_area_percentage_use(0.1)
            .visual_minimal_own_area_percentage_collect(0.2)
            .spatio_temporal_constraints(
                SpatioTemporalConstraints::default().constraints(&[(5, 7.0)])
            )
            .build());

        let mut opts_builder = PyVisualSortOptions::new();
        opts_builder.max_idle_epochs(3);
        opts_builder.kept_history_length(10);
        opts_builder.visual_metric(PyVisualSortMetricType::euclidean(100.0));
        opts_builder.positional_metric(PyPositionalMetricType::maha());
        opts_builder.visual_minimal_track_length(3);
        opts_builder.visual_minimal_area(5.0);
        opts_builder.visual_minimal_quality_use(0.45);
        opts_builder.visual_minimal_quality_collect(0.5);
        opts_builder.visual_max_observations(25);
        opts_builder.positional_min_confidence(0.13);
        opts_builder.visual_minimal_own_area_percentage_use(0.1);
        opts_builder.visual_minimal_own_area_percentage_collect(0.2);
        opts_builder.visual_min_votes(5);
        let mut constraints = PySpatioTemporalConstraints::new();
        constraints.add_constraints(vec![(5, 7.0)]);
        opts_builder.spatio_temporal_constraints(constraints);
        let (opts_py, metric_py) = dbg!(opts_builder.0.build());

        assert_eq!(format!("{:?}", opts), format!("{:?}", opts_py));
        assert_eq!(format!("{:?}", metric), format!("{:?}", metric_py));
    }
}
