use crate::track::{
    LookupRequest, ObservationsDb, Track, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::DIM_2D_BOX_X2;
use crate::utils::kalman::KalmanState;
use anyhow::Result;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use self::metric::SortMetric;

/// SORT metric implementation with IoU and Mahalanobis distances
pub mod metric;

/// SORT implementation with a very tiny interface
pub mod simple_api;

/// Voting engine with Hungarian algorithm
///
pub mod voting;

/// SORT tracker with Batch API
pub mod batch_api;

/// Default IoU threshold that is defined by SORT author in the original repo
pub const DEFAULT_SORT_IOU_THRESHOLD: f32 = 0.3;

#[derive(Debug)]
pub struct SortAttributesOptions {
    /// The map that stores current epochs for the scene_id
    epoch_db: Option<RwLock<HashMap<u64, usize>>>,
    /// The maximum number of epochs without update while the track is alive
    max_idle_epochs: usize,
    /// The maximum length of collected objects for the track
    pub history_length: usize,
    pub spatio_temporal_constraints: SpatioTemporalConstraints,
    pub position_weight: f32,
    pub velocity_weight: f32,
}

impl Default for SortAttributesOptions {
    fn default() -> Self {
        Self {
            epoch_db: None,
            max_idle_epochs: 0,
            history_length: 0,
            spatio_temporal_constraints: SpatioTemporalConstraints::default(),
            position_weight: 1.0 / 20.0,
            velocity_weight: 1.0 / 160.0,
        }
    }
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
        spatio_temporal_constraints: SpatioTemporalConstraints,
        position_weight: f32,
        velocity_weight: f32,
    ) -> Self {
        Self {
            epoch_db,
            max_idle_epochs,
            history_length,
            spatio_temporal_constraints,
            position_weight,
            velocity_weight,
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
    /// Custom object id
    pub custom_object_id: Option<i64>,

    /// Kalman filter predicted state
    state: Option<KalmanState<{ DIM_2D_BOX_X2 }>>,
    opts: Arc<SortAttributesOptions>,
}

impl TrackAttributesKalmanPrediction for SortAttributes {
    fn get_state(&self) -> Option<KalmanState<{ DIM_2D_BOX_X2 }>> {
        self.state
    }

    fn set_state(&mut self, state: KalmanState<{ DIM_2D_BOX_X2 }>) {
        self.state = Some(state);
    }

    fn get_position_weight(&self) -> f32 {
        self.opts.position_weight
    }

    fn get_velocity_weight(&self) -> f32 {
        self.opts.velocity_weight
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
            custom_object_id: None,
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
    custom_object_id: Option<i64>,
}

/// Lookup object for SortAttributes
///
#[derive(Clone, Debug)]
pub enum SortLookup {
    IdleLookup(u64),
}

impl LookupRequest<SortAttributes, Universal2DBox> for SortLookup {
    fn lookup(
        &self,
        attributes: &SortAttributes,
        _observations: &ObservationsDb<Universal2DBox>,
        _merge_history: &[u64],
    ) -> bool {
        match self {
            SortLookup::IdleLookup(scene_id) => {
                *scene_id == attributes.scene_id
                    && attributes.last_updated_epoch
                        != attributes
                            .opts
                            .current_epoch_with_scene(attributes.scene_id)
                            .unwrap()
            }
        }
    }
}

impl SortAttributesUpdate {
    /// update epoch with scene_id == 0
    ///
    /// # Parameters
    /// * `epoch` - epoch update
    ///
    pub fn new(epoch: usize, custom_object_id: Option<i64>) -> Self {
        Self {
            epoch,
            scene_id: 0,
            custom_object_id,
        }
    }
    /// update epoch for a specific scene_id
    ///
    /// # Parameters
    /// * `epoch` - epoch
    /// * `scene_id` - scene_id
    pub fn new_with_scene(epoch: usize, scene_id: u64, custom_object_id: Option<i64>) -> Self {
        Self {
            epoch,
            scene_id,
            custom_object_id,
        }
    }
}

impl TrackAttributesUpdate<SortAttributes> for SortAttributesUpdate {
    fn apply(&self, attrs: &mut SortAttributes) -> Result<()> {
        attrs.last_updated_epoch = self.epoch;
        attrs.scene_id = self.scene_id;
        attrs.custom_object_id = self.custom_object_id;
        Ok(())
    }
}

impl TrackAttributes<SortAttributes, Universal2DBox> for SortAttributes {
    type Update = SortAttributesUpdate;
    type Lookup = SortLookup;

    fn compatible(&self, other: &SortAttributes) -> bool {
        if self.scene_id == other.scene_id {
            let o1 = self.predicted_boxes.back().unwrap();
            let o2 = other.predicted_boxes.back().unwrap();

            let epoch_delta = (self.last_updated_epoch as i128 - other.last_updated_epoch as i128)
                .abs()
                .try_into()
                .unwrap();

            let center_dist = Universal2DBox::dist_in_2r(o1, o2);

            self.opts.max_idle_epochs() >= epoch_delta
                && self
                    .opts
                    .spatio_temporal_constraints
                    .validate(epoch_delta, center_dist)
        } else {
            false
        }
    }

    fn merge(&mut self, other: &SortAttributes) -> Result<()> {
        self.last_updated_epoch = other.last_updated_epoch;
        self.custom_object_id = other.custom_object_id;
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<Universal2DBox>) -> Result<TrackStatus> {
        self.opts.baked(self.scene_id, self.last_updated_epoch)
    }
}

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
pub struct SortTrack {
    /// id of the track
    ///
    pub id: u64,
    /// when the track was lastly updated
    ///
    pub epoch: usize,
    /// the bbox predicted by KF
    ///
    pub predicted_bbox: Universal2DBox,
    /// the bbox passed by detector
    ///
    pub observed_bbox: Universal2DBox,
    /// user-defined scene id that splits tracking space on isolated realms
    ///
    pub scene_id: u64,
    /// current track length
    ///
    pub length: usize,
    /// what kind of voting was led to the current merge
    ///
    pub voting_type: VotingType,
    /// custom object id passed by the user to find the track easily
    ///
    pub custom_object_id: Option<i64>,
}

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
pub struct WastedSortTrack {
    /// id of the track
    ///
    pub id: u64,
    /// when the track was lastly updated
    ///
    pub epoch: usize,
    /// the bbox predicted by KF
    ///
    pub predicted_bbox: Universal2DBox,
    /// the bbox passed by detector
    ///
    pub observed_bbox: Universal2DBox,
    /// user-defined scene id that splits tracking space on isolated realms
    ///
    pub scene_id: u64,
    /// current track length
    ///
    pub length: usize,
    /// history of predicted boxes
    ///
    pub predicted_boxes: Vec<Universal2DBox>,
    /// history of observed boxes
    ///
    pub observed_boxes: Vec<Universal2DBox>,
}

impl From<Track<SortAttributes, SortMetric, Universal2DBox>> for WastedSortTrack {
    fn from(track: Track<SortAttributes, SortMetric, Universal2DBox>) -> Self {
        let attrs = track.get_attributes();
        WastedSortTrack {
            id: track.get_track_id(),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            length: attrs.track_length,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum VotingType {
    #[default]
    Visual,
    Positional,
}

#[derive(Clone, Default, Copy, Debug)]
pub enum PositionalMetricType {
    #[default]
    Mahalanobis,
    IoU(f32),
}

pub struct AutoWaste {
    pub periodicity: usize,
    pub counter: usize,
}

pub(crate) const DEFAULT_AUTO_WASTE_PERIODICITY: usize = 100;
pub(crate) const MAHALANOBIS_NEW_TRACK_THRESHOLD: f32 = 1.0;

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    use crate::utils::bbox::python::PyUniversal2DBox;

    use super::{PositionalMetricType, SortTrack, VotingType, WastedSortTrack};

    #[pyclass]
    #[pyo3(name = "PositionalMetricType")]
    #[derive(Clone, Debug)]
    pub struct PyPositionalMetricType(pub PositionalMetricType);

    #[pymethods]
    impl PyPositionalMetricType {
        #[staticmethod]
        pub fn maha() -> Self {
            PyPositionalMetricType(PositionalMetricType::Mahalanobis)
        }

        #[staticmethod]
        pub fn iou(threshold: f32) -> Self {
            assert!(
                threshold > 0.0 && threshold < 1.0,
                "Threshold must lay between (0.0 and 1.0)"
            );

            PyPositionalMetricType(PositionalMetricType::IoU(threshold))
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

    #[pyclass]
    #[pyo3(name = "SortTrack")]
    #[derive(Debug, Clone)]
    #[repr(transparent)]
    pub struct PySortTrack(pub(crate) SortTrack);

    #[pymethods]
    impl PySortTrack {
        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{self:?}")
        }

        fn __str__(&self) -> String {
            format!("{self:#?}")
        }

        #[getter]
        fn get_id(&self) -> u64 {
            self.0.id
        }

        #[getter]
        fn get_epoch(&self) -> usize {
            self.0.epoch
        }

        #[getter]
        fn get_predicted_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.predicted_bbox.clone())
        }

        #[getter]
        fn get_observed_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.observed_bbox.clone())
        }

        #[getter]
        fn get_scene_id(&self) -> u64 {
            self.0.scene_id
        }

        #[getter]
        fn get_length(&self) -> usize {
            self.0.length
        }

        #[getter]
        fn get_voting_type(&self) -> PyVotingType {
            PyVotingType(self.0.voting_type)
        }

        #[getter]
        fn get_custom_object_id(&self) -> Option<i64> {
            self.0.custom_object_id
        }
    }

    #[pyclass]
    #[pyo3(name = "WastedSortTrack")]
    #[derive(Debug, Clone)]
    #[repr(transparent)]
    pub struct PyWastedSortTrack(pub(crate) WastedSortTrack);

    #[pymethods]
    impl PyWastedSortTrack {
        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn __str__(&self) -> String {
            format!("{:#?}", self.0)
        }

        #[getter]
        fn id(&self) -> u64 {
            self.0.id
        }

        #[getter]
        fn epoch(&self) -> usize {
            self.0.epoch
        }

        #[getter]
        fn predicted_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.predicted_bbox.clone())
        }

        #[getter]
        fn observed_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.observed_bbox.clone())
        }

        #[getter]
        fn scene_id(&self) -> u64 {
            self.0.scene_id
        }

        #[getter]
        fn length(&self) -> usize {
            self.0.length
        }

        #[getter]
        fn predicted_boxes(&self) -> Vec<PyUniversal2DBox> {
            unsafe { std::mem::transmute(self.0.predicted_boxes.clone()) }
        }

        #[getter]
        fn observed_boxes(&self) -> Vec<PyUniversal2DBox> {
            unsafe { std::mem::transmute(self.0.observed_boxes.clone()) }
        }
    }

    #[pyclass]
    #[pyo3(name = "VotingType")]
    #[derive(Default, Debug, Clone, Copy)]
    pub struct PyVotingType(pub(crate) VotingType);

    #[pymethods]
    impl PyVotingType {
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

#[cfg(test)]
mod track_tests {
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackBuilder};
    use crate::trackers::sort::metric::{SortMetric, DEFAULT_MINIMAL_SORT_CONFIDENCE};
    use crate::trackers::sort::PositionalMetricType::IoU;
    use crate::trackers::sort::{SortAttributes, DEFAULT_SORT_IOU_THRESHOLD};
    use crate::utils::bbox::BoundingBox;
    use crate::utils::kalman::kalman_2d_box::Universal2DBoxKalmanFilter;

    #[test]
    fn construct() {
        let observation_bb_0 = BoundingBox::new(1.0, 1.0, 10.0, 15.0);
        let observation_bb_1 = BoundingBox::new(1.1, 1.3, 10.0, 15.0);

        let f = Universal2DBoxKalmanFilter::default();
        let init_state = f.initiate(&observation_bb_0.into());

        let mut t1 = TrackBuilder::new(1)
            .attributes(SortAttributes::default())
            .metric(SortMetric::new(
                IoU(DEFAULT_SORT_IOU_THRESHOLD),
                DEFAULT_MINIMAL_SORT_CONFIDENCE,
            ))
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
        assert_eq!(
            t1.get_attributes().predicted_boxes[0],
            observation_bb_0.into()
        );

        let predicted_state = f.predict(&init_state);
        assert_eq!(
            BoundingBox::try_from(predicted_state).unwrap(),
            observation_bb_0
        );

        let t2 = TrackBuilder::new(2)
            .attributes(SortAttributes::default())
            .metric(SortMetric::new(
                IoU(DEFAULT_SORT_IOU_THRESHOLD),
                DEFAULT_MINIMAL_SORT_CONFIDENCE,
            ))
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
    }
}
