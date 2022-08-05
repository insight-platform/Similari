use crate::prelude::SortTrack;
use crate::trackers::sort::simple_maha::MahaSort;
use crate::trackers::sort::PyWastedSortTrack;
use crate::utils::bbox::Universal2DBox;
use pyo3::prelude::*;

#[pymethods]
impl MahaSort {
    #[new]
    #[args(shards = "4", bbox_history = "1", max_idle_epochs = "5")]
    pub fn new_py(shards: i64, bbox_history: i64, max_idle_epochs: i64) -> Self {
        assert!(shards > 0 && bbox_history > 0 && max_idle_epochs > 0);
        Self::new(
            shards.try_into().unwrap(),
            bbox_history.try_into().unwrap(),
            max_idle_epochs.try_into().unwrap(),
        )
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
        self.store
            .shard_stats()
            .into_iter()
            .map(|e| i64::try_from(e).unwrap())
            .collect()
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
    #[pyo3(name = "predict", text_signature = "($self, bboxes)")]
    pub fn predict_py(&mut self, bboxes: Vec<Universal2DBox>) -> Vec<SortTrack> {
        self.predict(&bboxes)
    }

    /// Receive tracking information for observed bboxes of `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id provided by a user (class, camera id, etc...)
    /// * `bboxes` - bounding boxes received from a detector
    ///
    #[pyo3(
        name = "predict_with_scene",
        text_signature = "($self, scene_id, bboxes)"
    )]
    pub fn predict_with_scene_py(
        &mut self,
        scene_id: i64,
        bboxes: Vec<Universal2DBox>,
    ) -> Vec<SortTrack> {
        assert!(scene_id >= 0);
        self.predict_with_scene(scene_id.try_into().unwrap(), &bboxes)
    }

    /// Remove all the tracks with expired life
    ///
    #[pyo3(name = "wasted", text_signature = "($self)")]
    pub fn wasted_py(&mut self) -> Vec<PyWastedSortTrack> {
        self.wasted()
            .into_iter()
            .map(PyWastedSortTrack::from)
            .collect()
    }
}
