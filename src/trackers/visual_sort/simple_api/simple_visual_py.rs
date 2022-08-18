use crate::prelude::SortTrack;
use crate::trackers::tracker_api::TrackerAPI;
use crate::trackers::visual_sort::simple_api::options::VisualSortOptions;
use crate::trackers::visual_sort::simple_api::VisualSort;
use crate::trackers::visual_sort::{PyWastedVisualSortTrack, VisualObservation};
use crate::utils::bbox::Universal2DBox;
use pyo3::prelude::*;

#[pyclass(
    text_signature = "(feature_opt, feature_quality_opt, bounding_box, custom_object_id_opt)"
)]
#[derive(Debug, Clone)]
#[pyo3(name = "VisualSortObservation")]
pub struct PyVisualSortObservation {
    pub feature: Option<Vec<f32>>,
    pub feature_quality: Option<f32>,
    pub bounding_box: Universal2DBox,
    pub custom_object_id: Option<i64>,
}

#[pymethods]
impl PyVisualSortObservation {
    #[new]
    pub fn new(
        feature: Option<Vec<f32>>,
        feature_quality: Option<f32>,
        bounding_box: Universal2DBox,
        custom_object_id: Option<i64>,
    ) -> Self {
        Self {
            feature,
            feature_quality,
            bounding_box,
            custom_object_id,
        }
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

#[pyclass(
    text_signature = "(feature_opt, feature_quality_opt, bounding_box, custom_object_id_opt)"
)]
#[derive(Debug)]
#[pyo3(name = "VisualSortObservationSet")]
pub struct PyVisualSortObservationSet {
    inner: Vec<PyVisualSortObservation>,
}

#[pymethods]
impl PyVisualSortObservationSet {
    #[new]
    fn new() -> Self {
        Self {
            inner: Vec::default(),
        }
    }

    #[pyo3(text_signature = "($self, observation)")]
    fn add(&mut self, observation: PyVisualSortObservation) {
        self.inner.push(observation);
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

#[pymethods]
impl VisualSort {
    #[new]
    pub fn new_py(shards: i64, opts: &VisualSortOptions) -> Self {
        assert!(shards > 0);
        Self::new(shards.try_into().unwrap(), opts)
    }

    #[pyo3(name = "skip_epochs", text_signature = "($self, n)")]
    pub fn skip_epochs_py(&mut self, n: i64) {
        assert!(n > 0);
        self.skip_epochs(n.try_into().unwrap())
    }

    #[pyo3(
        name = "skip_epochs_for_scene",
        text_signature = "($self, scene_id, n)"
    )]
    pub fn skip_epochs_for_scene_py(&mut self, scene_id: i64, n: i64) {
        assert!(n > 0 && scene_id >= 0);
        self.skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap())
    }

    /// Get the amount of stored tracks per shard
    ///
    #[pyo3(name = "shard_stats", text_signature = "($self)")]
    pub fn shard_stats_py(&self) -> Vec<i64> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        py.allow_threads(|| {
            self.active_shard_stats()
                .into_iter()
                .map(|e| i64::try_from(e).unwrap())
                .collect()
        })
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    #[pyo3(name = "current_epoch", text_signature = "($self)")]
    pub fn current_epoch_py(&self) -> i64 {
        self.current_epoch_with_scene(0).try_into().unwrap()
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    #[pyo3(
        name = "current_epoch_with_scene",
        text_signature = "($self, scene_id)"
    )]
    pub fn current_epoch_with_scene_py(&self, scene_id: i64) -> isize {
        assert!(scene_id >= 0);
        self.current_epoch_with_scene(scene_id.try_into().unwrap())
            .try_into()
            .unwrap()
    }

    /// Receive tracking information for observed bboxes of `scene_id` == 0
    ///
    /// # Parameters
    /// * `bboxes` - bounding boxes received from a detector
    ///
    #[pyo3(name = "predict", text_signature = "($self, observation_set)")]
    pub fn predict_py(&mut self, observation_set: &PyVisualSortObservationSet) -> Vec<SortTrack> {
        self.predict_with_scene_py(0, observation_set)
    }

    /// Receive tracking information for observed bboxes of `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id provided by a user (class, camera id, etc...)
    /// * `bboxes` - bounding boxes received from a detector
    ///
    #[pyo3(
        name = "predict_with_scene",
        text_signature = "($self, scene_id, observations)"
    )]
    pub fn predict_with_scene_py(
        &mut self,
        scene_id: i64,
        observation_set: &PyVisualSortObservationSet,
    ) -> Vec<SortTrack> {
        assert!(scene_id >= 0);
        let gil = Python::acquire_gil();
        let py = gil.python();
        let observations = observation_set
            .inner
            .iter()
            .map(|e| {
                VisualObservation::new(
                    e.feature.as_ref(),
                    e.feature_quality,
                    e.bounding_box.clone(),
                    e.custom_object_id,
                )
            })
            .collect::<Vec<_>>();
        py.allow_threads(|| self.predict_with_scene(scene_id.try_into().unwrap(), &observations))
    }

    /// Remove all the tracks with expired life
    ///
    #[pyo3(name = "wasted", text_signature = "($self)")]
    pub fn wasted_py(&mut self) -> Vec<PyWastedVisualSortTrack> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        py.allow_threads(|| {
            self.wasted()
                .into_iter()
                .map(PyWastedVisualSortTrack::from)
                .collect()
        })
    }
}
