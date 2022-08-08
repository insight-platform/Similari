use crate::track::{
    NoopLookup, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanState;
use anyhow::Result;
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

/// SORT implementation with IoU distance
pub mod iou;
/// SORT implementation with Mahalanobis distance
pub mod maha;

/// SORT implementation with a very tiny interface (IoU)
pub mod simple_iou;
/// SORT implementation with a very tiny interface (Mahalanobis)
pub mod simple_maha;

/// Voting engine with Hungarian algorithm
pub mod voting;

/// Default IoU threshold that is defined by SORT author in the original repo
pub const DEFAULT_SORT_IOU_THRESHOLD: f32 = 0.3;

#[derive(Debug, Default)]
pub struct SortAttributesOptions {
    /// The map that stores current epochs for the scene_id
    epoch_db: Option<RwLock<HashMap<u64, usize>>>,
    /// The maximum number of epochs without update while the track is alive
    max_idle_epochs: usize,
    /// The maximum length of collected objects for the track
    pub history_length: usize,
}

impl EpochDb for SortAttributesOptions {
    fn epoch_db(&self) -> &Option<RwLock<HashMap<u64, usize>>> {
        &self.epoch_db
    }

    fn max_idle_epochs(&self) -> usize {
        self.max_idle_epochs
    }
}

impl SortAttributesOptions {
    pub fn new(
        epoch_db: Option<RwLock<HashMap<u64, usize>>>,
        max_idle_epochs: usize,
        history_length: usize,
    ) -> Self {
        Self {
            epoch_db,
            max_idle_epochs,
            history_length,
        }
    }
}

/// Attributes associated with SORT track
///
#[derive(Debug, Clone)]
pub struct SortAttributes {
    /// The lastly predicted boxes
    pub predicted_boxes: VecDeque<Universal2DBox>,
    /// The lastly observed boxes
    pub observed_boxes: VecDeque<Universal2DBox>,
    /// The epoch when the track was lastly updated
    pub last_updated_epoch: usize,
    /// The length of the track
    pub track_length: usize,
    /// Customer-specific scene identifier that splits the objects by classes, realms, etc.
    pub scene_id: u64,

    /// Kalman filter predicted state
    state: Option<KalmanState>,
    opts: Arc<SortAttributesOptions>,
}

impl TrackAttributesKalmanPrediction for SortAttributes {
    fn get_state(&self) -> Option<KalmanState> {
        self.state
    }

    fn set_state(&mut self, state: KalmanState) {
        self.state = Some(state);
    }
}

impl Default for SortAttributes {
    fn default() -> Self {
        Self {
            predicted_boxes: VecDeque::default(),
            observed_boxes: VecDeque::default(),
            last_updated_epoch: 0,
            track_length: 0,
            scene_id: 0,
            state: None,
            opts: Arc::new(SortAttributesOptions::default()),
        }
    }
}

impl SortAttributes {
    /// Creates new attributes with limited history
    ///
    /// # Parameters
    /// * `opts` - options
    ///
    pub fn new(opts: Arc<SortAttributesOptions>) -> Self {
        Self {
            opts,
            ..Default::default()
        }
    }

    fn update_history(
        &mut self,
        observation_bbox: &Universal2DBox,
        predicted_bbox: &Universal2DBox,
    ) {
        self.track_length += 1;

        self.observed_boxes.push_back(observation_bbox.clone());
        self.predicted_boxes.push_back(predicted_bbox.clone());

        if self.opts.history_length > 0 && self.observed_boxes.len() > self.opts.history_length {
            self.observed_boxes.pop_front();
            self.predicted_boxes.pop_front();
        }
    }
}

/// Update object for SortAttributes
///
#[derive(Clone, Debug, Default)]
pub struct SortAttributesUpdate {
    epoch: usize,
    scene_id: u64,
}

impl SortAttributesUpdate {
    /// update epoch with scene_id == 0
    ///
    /// # Parameters
    /// * `epoch` - epoch update
    ///
    pub fn new(epoch: usize) -> Self {
        Self { epoch, scene_id: 0 }
    }
    /// update epoch for a specific scene_id
    ///
    /// # Parameters
    /// * `epoch` - epoch
    /// * `scene_id` - scene_id
    pub fn new_with_scene(epoch: usize, scene_id: u64) -> Self {
        Self { epoch, scene_id }
    }
}

impl TrackAttributesUpdate<SortAttributes> for SortAttributesUpdate {
    fn apply(&self, attrs: &mut SortAttributes) -> Result<()> {
        attrs.last_updated_epoch = self.epoch;
        attrs.scene_id = self.scene_id;
        Ok(())
    }
}

impl TrackAttributes<SortAttributes, Universal2DBox> for SortAttributes {
    type Update = SortAttributesUpdate;
    type Lookup = NoopLookup<SortAttributes, Universal2DBox>;

    fn compatible(&self, other: &SortAttributes) -> bool {
        self.scene_id == other.scene_id
    }

    fn merge(&mut self, other: &SortAttributes) -> Result<()> {
        self.last_updated_epoch = other.last_updated_epoch;
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<Universal2DBox>) -> Result<TrackStatus> {
        self.opts.baked(self.scene_id, self.last_updated_epoch)
    }
}

#[cfg(test)]
mod track_tests {
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackBuilder};
    use crate::trackers::sort::iou::IOUSortMetric;
    use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
    use crate::utils::bbox::BoundingBox;
    use crate::utils::kalman::KalmanFilter;
    use crate::{EstimateClose, EPS};

    #[test]
    fn construct() {
        let observation_bb_0 = BoundingBox::new(1.0, 1.0, 10.0, 15.0);
        let observation_bb_1 = BoundingBox::new(1.1, 1.3, 10.0, 15.0);

        let f = KalmanFilter::default();
        let init_state = f.initiate(observation_bb_0.into());

        let mut t1 = TrackBuilder::new(1)
            .attributes(SortAttributes::default())
            .metric(IOUSortMetric::new(DEFAULT_SORT_IOU_THRESHOLD))
            .notifier(NoopNotifier)
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(observation_bb_0.into())
                    .build(),
            )
            .build()
            .unwrap();

        assert!(t1.get_attributes().state.is_some());
        assert_eq!(t1.get_attributes().predicted_boxes.len(), 1);
        assert_eq!(t1.get_attributes().observed_boxes.len(), 1);
        assert_eq!(t1.get_merge_history().len(), 1);
        assert!(t1.get_attributes().predicted_boxes[0].almost_same(&observation_bb_0.into(), EPS));

        let predicted_state = f.predict(init_state);
        assert!(predicted_state
            .bbox()
            .unwrap()
            .almost_same(&observation_bb_0, EPS));

        let t2 = TrackBuilder::new(2)
            .attributes(SortAttributes::default())
            .metric(IOUSortMetric::new(DEFAULT_SORT_IOU_THRESHOLD))
            .notifier(NoopNotifier)
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(observation_bb_1.into())
                    .build(),
            )
            .build()
            .unwrap();

        t1.merge(&t2, &[0], false).unwrap();

        assert!(t1.get_attributes().state.is_some());
        assert_eq!(t1.get_attributes().predicted_boxes.len(), 2);
        assert_eq!(t1.get_attributes().observed_boxes.len(), 2);

        let predicted_state = f.predict(f.update(predicted_state, observation_bb_1.into()));
        assert!(t1.get_attributes().predicted_boxes[1]
            .almost_same(&predicted_state.universal_bbox(), EPS));
    }
}

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
#[pyclass]
pub struct SortTrack {
    /// id of the track
    ///
    #[pyo3(get)]
    pub id: u64,
    /// when the track was lastly updated
    ///
    #[pyo3(get)]
    pub epoch: usize,
    /// the bbox predicted by KF
    ///
    #[pyo3(get)]
    pub predicted_bbox: Universal2DBox,
    /// the bbox passed by detector
    ///
    #[pyo3(get)]
    pub observed_bbox: Universal2DBox,
    /// user-defined scene id that splits tracking space on isolated realms
    ///
    #[pyo3(get)]
    pub scene_id: u64,
    /// current track length
    ///
    #[pyo3(get)]
    pub length: usize,
}

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
#[pyclass]
#[pyo3(name = "WastedSortTrack")]
pub struct PyWastedSortTrack {
    /// id of the track
    ///
    #[pyo3(get)]
    pub id: u64,
    /// when the track was lastly updated
    ///
    #[pyo3(get)]
    pub epoch: usize,
    /// the bbox predicted by KF
    ///
    #[pyo3(get)]
    pub predicted_bbox: Universal2DBox,
    /// the bbox passed by detector
    ///
    #[pyo3(get)]
    pub observed_bbox: Universal2DBox,
    /// user-defined scene id that splits tracking space on isolated realms
    ///
    #[pyo3(get)]
    pub scene_id: u64,
    /// current track length
    ///
    #[pyo3(get)]
    pub length: usize,
    /// history of predicted boxes
    ///
    #[pyo3(get)]
    pub predicted_boxes: Vec<Universal2DBox>,
    /// history of observed boxes
    ///
    #[pyo3(get)]
    pub observed_boxes: Vec<Universal2DBox>,
}

#[pymethods]
impl SortTrack {
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
impl PyWastedSortTrack {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}
