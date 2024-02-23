use crate::track::{
    Feature, LookupRequest, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::{SortAttributesOptions, VotingType};
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::DIM_2D_BOX_X2;
use crate::utils::kalman::KalmanState;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;

/// Universal visual_sort attributes for visual_sort trackers
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
    pub custom_object_id: Option<i64>,
    /// Last voting type
    pub voting_type: Option<VotingType>,

    state: Option<KalmanState<{ DIM_2D_BOX_X2 }>>,
    opts: Arc<SortAttributesOptions>,
}

impl Default for VisualAttributes {
    fn default() -> Self {
        Self {
            voting_type: None,
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

#[derive(Clone, Debug)]
pub enum VisualSortLookup {
    IdleLookup(u64),
}

impl LookupRequest<VisualAttributes, VisualObservationAttributes> for VisualSortLookup {
    fn lookup(
        &self,
        attributes: &VisualAttributes,
        _observations: &ObservationsDb<VisualObservationAttributes>,
        _merge_history: &[u64],
    ) -> bool {
        match self {
            VisualSortLookup::IdleLookup(scene_id) => {
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

#[derive(Clone, Debug)]
pub enum VisualAttributesUpdate {
    Init {
        epoch: usize,
        scene_id: u64,
        custom_object_id: Option<i64>,
    },
    VotingType(VotingType),
}

impl VisualAttributesUpdate {
    pub fn new_init(epoch: usize, custom_object_id: Option<i64>) -> Self {
        Self::new_init_with_scene(epoch, 0, custom_object_id)
    }

    pub fn new_init_with_scene(epoch: usize, scene_id: u64, custom_object_id: Option<i64>) -> Self {
        Self::Init {
            epoch,
            scene_id,
            custom_object_id,
        }
    }

    pub fn new_voting_type(vt: VotingType) -> Self {
        Self::VotingType(vt)
    }
}

impl TrackAttributesUpdate<VisualAttributes> for VisualAttributesUpdate {
    fn apply(&self, attrs: &mut VisualAttributes) -> Result<()> {
        match self {
            Self::Init {
                epoch,
                scene_id,
                custom_object_id,
            } => {
                attrs.last_updated_epoch = *epoch;
                attrs.scene_id = *scene_id;
                attrs.custom_object_id = *custom_object_id;
            }
            VisualAttributesUpdate::VotingType(vt) => {
                attrs.voting_type = Some(*vt);
            }
        }
        Ok(())
    }
}

impl TrackAttributes<VisualAttributes, VisualObservationAttributes> for VisualAttributes {
    type Update = VisualAttributesUpdate;
    type Lookup = VisualSortLookup;

    fn compatible(&self, other: &VisualAttributes) -> bool {
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

    fn merge(&mut self, other: &VisualAttributes) -> Result<()> {
        self.last_updated_epoch = other.last_updated_epoch;
        self.custom_object_id = other.custom_object_id;
        self.voting_type = other.voting_type;
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
    use crate::trackers::spatio_temporal_constraints::SpatioTemporalConstraints;
    use crate::trackers::visual_sort::track_attributes::VisualAttributes;
    use crate::utils::bbox::BoundingBox;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    #[test]
    fn attribute_operations() {
        let opts = SortAttributesOptions::new(
            Some(RwLock::new(HashMap::default())),
            5,
            1,
            SpatioTemporalConstraints::default(),
            1.0 / 20.0,
            1.0 / 160.0,
        );
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
