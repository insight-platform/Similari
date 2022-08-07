use crate::track::{
    NoopLookup, Observation, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::{KalmanFilter, KalmanState};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

#[derive(Debug, Default)]
pub struct VisualAttributesOpts {
    // how long history (bboxes, observations) can be kept in attributes
    history_length: usize,
    // when the track becomes wasted
    max_idle_epochs: usize,
    epoch_db: Option<RwLock<HashMap<u64, usize>>>,
}

impl EpochDb for VisualAttributesOpts {
    fn epoch_db(&self) -> &Option<RwLock<HashMap<u64, usize>>> {
        &self.epoch_db
    }

    fn max_idle_epochs(&self) -> usize {
        self.max_idle_epochs
    }
}

impl VisualAttributesOpts {
    pub fn new(
        epoch_db: Option<RwLock<HashMap<u64, usize>>>,
        max_idle_epochs: usize,
        history_length: usize,
    ) -> Self {
        Self {
            history_length,
            max_idle_epochs,
            epoch_db,
        }
    }
}

/// Universal visual attributes for visual trackers
///
#[derive(Debug, Clone)]
pub struct VisualAttributes {
    /// Boxes predicted by Kalman filter
    pub predicted_boxes: VecDeque<Universal2DBox>,
    /// Boxes observed by detector
    pub observed_boxes: VecDeque<Universal2DBox>,
    /// Features observed by feature extractor model
    pub observed_features: VecDeque<Option<Observation>>,
    /// The last epoch when attributes were updated
    pub last_updated_epoch: usize,
    /// The length of the track
    pub track_length: usize,
    /// Visual track elements amount collected
    pub visual_features_collected_count: usize,
    /// Custom scene id provided by the user
    pub scene_id: u64,
    /// Custom object id provided for the bbox and observation
    pub custom_object_id: Option<u64>,

    state: Option<KalmanState>,
    opts: Arc<VisualAttributesOpts>,
}

impl Default for VisualAttributes {
    fn default() -> Self {
        Self {
            predicted_boxes: VecDeque::default(),
            observed_boxes: VecDeque::default(),
            observed_features: VecDeque::default(),
            last_updated_epoch: 0,
            track_length: 0,
            visual_features_collected_count: 0,
            scene_id: 0,
            custom_object_id: None,
            state: None,
            opts: Arc::new(VisualAttributesOpts::default()),
        }
    }
}

impl VisualAttributes {
    /// Creates new attributes with limited history
    ///
    /// # Parameters
    /// * `history_len` - how long history to hold. 0 means all history.
    /// * `max_idle_epochs` - how long to wait before exclude the track from store.
    /// * `current_epoch` - current epoch counter.
    ///
    pub fn new(opts: Arc<VisualAttributesOpts>) -> Self {
        Self {
            opts,
            ..Default::default()
        }
    }

    fn update_history(
        &mut self,
        observation_bbox: &Universal2DBox,
        predicted_bbox: &Universal2DBox,
        observation_feature: Option<Observation>,
    ) {
        self.track_length += 1;

        self.observed_boxes.push_back(observation_bbox.clone());
        self.predicted_boxes.push_back(predicted_bbox.clone());
        self.observed_features.push_back(observation_feature);

        if self.opts.history_length > 0 && self.observed_boxes.len() > self.opts.history_length {
            self.observed_boxes.pop_front();
            self.predicted_boxes.pop_front();
            self.observed_features.pop_front();
        }
    }

    fn update_bbox_prediction(&mut self, observation_bbox: &Universal2DBox) -> Universal2DBox {
        let f = KalmanFilter::default();

        let state = if let Some(state) = self.state {
            f.update(state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(state);
        self.state = Some(prediction);
        prediction.universal_bbox()
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisualAttributesUpdate {
    epoch: usize,
    scene_id: u64,
    custom_object_id: Option<u64>,
}

impl VisualAttributesUpdate {
    pub fn new(epoch: usize, custom_object_id: Option<u64>) -> Self {
        Self {
            epoch,
            scene_id: 0,
            custom_object_id,
        }
    }
    pub fn new_with_scene(epoch: usize, scene_id: u64, custom_object_id: Option<u64>) -> Self {
        Self {
            epoch,
            scene_id,
            custom_object_id,
        }
    }
}

impl TrackAttributesUpdate<VisualAttributes> for VisualAttributesUpdate {
    fn apply(&self, attrs: &mut VisualAttributes) -> Result<()> {
        attrs.last_updated_epoch = self.epoch;
        attrs.scene_id = self.scene_id;
        attrs.custom_object_id = self.custom_object_id;
        Ok(())
    }
}

impl TrackAttributes<VisualAttributes, VisualObservationAttributes> for VisualAttributes {
    type Update = VisualAttributesUpdate;
    type Lookup = NoopLookup<VisualAttributes, VisualObservationAttributes>;

    fn compatible(&self, other: &VisualAttributes) -> bool {
        self.scene_id == other.scene_id
    }

    fn merge(&mut self, other: &VisualAttributes) -> Result<()> {
        self.last_updated_epoch = other.last_updated_epoch;
        self.custom_object_id = other.custom_object_id;
        Ok(())
    }

    fn baked(
        &self,
        _observations: &ObservationsDb<VisualObservationAttributes>,
    ) -> Result<TrackStatus> {
        self.opts.baked(self.scene_id, self.last_updated_epoch)
    }
}
