use crate::track::{
    Feature, NoopLookup, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::SortAttributesOptions;
use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanState;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;

/// Universal visual attributes for visual trackers
///
#[derive(Debug, Clone)]
pub struct VisualAttributes {
    /// Boxes predicted by Kalman filter
    pub predicted_boxes: VecDeque<Universal2DBox>,
    /// Boxes observed by detector
    pub observed_boxes: VecDeque<Universal2DBox>,
    /// Features observed by feature extractor model
    pub observed_features: VecDeque<Option<Feature>>,
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
    opts: Arc<SortAttributesOptions>,
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
            opts: Arc::new(SortAttributesOptions::default()),
        }
    }
}

impl VisualAttributes {
    /// Creates new attributes with limited history
    ///
    /// # Parameters
    /// * `opts` - attribute options.
    ///
    pub fn new(opts: Arc<SortAttributesOptions>) -> Self {
        Self {
            opts,
            ..Default::default()
        }
    }

    pub fn update_history(
        &mut self,
        observation_bbox: &Universal2DBox,
        predicted_bbox: &Universal2DBox,
        observation_feature: Option<Feature>,
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
}

impl TrackAttributesKalmanPrediction for VisualAttributes {
    fn get_state(&self) -> Option<KalmanState> {
        self.state
    }

    fn set_state(&mut self, state: KalmanState) {
        self.state = Some(state);
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
        Self::new_with_scene(epoch, 0, custom_object_id)
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

#[cfg(test)]
mod tests {
    use crate::trackers::sort::SortAttributesOptions;
    use crate::trackers::visual::track_attributes::VisualAttributes;
    use crate::utils::bbox::BoundingBox;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    #[test]
    fn attribute_operations() {
        let opts = SortAttributesOptions::new(Some(RwLock::new(HashMap::default())), 5, 1);
        let mut attributes = VisualAttributes::new(Arc::new(opts));
        attributes.update_history(
            &BoundingBox::new(0.0, 3.0, 5.0, 7.0).as_xyaah(),
            &BoundingBox::new(0.1, 3.1, 5.1, 7.1).as_xyaah(),
            None,
        );

        assert_eq!(attributes.observed_boxes.len(), 1);
        assert_eq!(attributes.predicted_boxes.len(), 1);
        assert_eq!(attributes.observed_features.len(), 1);
        assert_eq!(attributes.track_length, 1);

        attributes.update_history(
            &BoundingBox::new(0.0, 3.0, 5.0, 7.0).as_xyaah(),
            &BoundingBox::new(0.1, 3.1, 5.1, 7.1).as_xyaah(),
            None,
        );

        assert_eq!(attributes.observed_boxes.len(), 1);
        assert_eq!(attributes.predicted_boxes.len(), 1);
        assert_eq!(attributes.observed_features.len(), 1);
        assert_eq!(attributes.track_length, 2);
    }
}
