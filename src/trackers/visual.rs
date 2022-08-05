use crate::distance::{cosine, euclidean};
use crate::track::{
    MetricOutput, NoopLookup, Observation, ObservationAttributes, ObservationMetric,
    ObservationMetricOk, ObservationSpec, ObservationsDb, TrackAttributes, TrackAttributesUpdate,
    TrackStatus,
};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::{KalmanFilter, State};
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

pub mod deep_sort;

#[derive(Debug, Clone, Default)]
pub struct VisualAttributes {
    pub predicted_boxes: VecDeque<Universal2DBox>,
    pub observed_boxes: VecDeque<Universal2DBox>,
    pub observed_features: VecDeque<Observation>,
    pub state: Option<State>,
    pub epoch: usize,
    pub length: usize,
    pub scene_id: u64,
    pub custom_object_id: u64,

    max_observations: usize,
    max_history_len: usize,
    max_idle_epochs: usize,
    current_epochs: Option<Arc<RwLock<HashMap<u64, usize>>>>,
}

impl VisualAttributes {
    /// Creates new attributes with limited history
    ///
    /// # Parameters
    /// * `history_len` - how long history to hold. 0 means all history.
    /// * `max_idle_epochs` - how long to wait before exclude the track from store.
    /// * `current_epoch` - current epoch counter.
    ///
    pub fn new_with_epochs(
        history_len: usize,
        max_idle_epochs: usize,
        current_epoch: Arc<RwLock<HashMap<u64, usize>>>,
    ) -> Self {
        Self {
            max_history_len: history_len,
            max_idle_epochs,
            current_epochs: Some(current_epoch),
            ..Default::default()
        }
    }

    /// Creates new attributes with limited history
    ///
    /// # Parameters
    /// * `history_len` - how long history to hold. 0 means all history.
    ///
    pub fn new(history_len: usize) -> Self {
        Self {
            max_history_len: history_len,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VisualAttributesUpdate {
    epoch: usize,
    scene_id: u64,
}

impl VisualAttributesUpdate {
    pub fn new(epoch: usize) -> Self {
        Self { epoch, scene_id: 0 }
    }
    pub fn new_with_scene(epoch: usize, scene_id: u64) -> Self {
        Self { epoch, scene_id }
    }
}

impl TrackAttributesUpdate<VisualAttributes> for VisualAttributesUpdate {
    fn apply(&self, attrs: &mut VisualAttributes) -> Result<()> {
        attrs.epoch = self.epoch;
        attrs.scene_id = self.scene_id;
        Ok(())
    }
}

impl TrackAttributes<VisualAttributes, Universal2DBox> for VisualAttributes {
    type Update = VisualAttributesUpdate;
    type Lookup = NoopLookup<VisualAttributes, Universal2DBox>;

    fn compatible(&self, other: &VisualAttributes) -> bool {
        self.scene_id == other.scene_id
    }

    fn merge(&mut self, other: &VisualAttributes) -> Result<()> {
        self.epoch = other.epoch;
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<Universal2DBox>) -> Result<TrackStatus> {
        let scene_id = self.scene_id;
        if let Some(current_epoch) = &self.current_epochs {
            let current_epoch = current_epoch.read().unwrap();
            if self.epoch + self.max_idle_epochs < *current_epoch.get(&scene_id).unwrap_or(&0) {
                Ok(TrackStatus::Wasted)
            } else {
                Ok(TrackStatus::Pending)
            }
        } else {
            // If epoch expiration is not set the tracks are always ready.
            // If set, then only when certain amount of epochs pass they are Wasted.
            //
            Ok(TrackStatus::Ready)
        }
    }
}

#[derive(Clone, Default)]
pub enum VisualMetricKind {
    #[default]
    Euclidean,
    Cosine,
}

#[derive(Clone, Default)]
pub enum PositionalMetricKind {
    #[default]
    Mahalanobis,
    IoU,
    Ignore,
}

#[derive(Clone, Default)]
pub struct VisualMetric {
    visual_kind: VisualMetricKind,
    positional_kind: PositionalMetricKind,
    bbox_distance_threshold: f32,
    min_required_track_visual_length: usize,
}

impl ObservationMetric<VisualAttributes, Universal2DBox> for VisualMetric {
    fn metric(
        &self,
        _feature_class: u64,
        _candidate_attributes: &VisualAttributes,
        track_attributes: &VisualAttributes,
        candidate_observation: &ObservationSpec<Universal2DBox>,
        track_observation: &ObservationSpec<Universal2DBox>,
    ) -> MetricOutput<f32> {
        let candidate_observation_bbox = candidate_observation.0.as_ref().unwrap();
        let track_observation_bbox = track_observation.0.as_ref().unwrap();

        let candidate_observation_feature = candidate_observation.1.as_ref().unwrap();
        let track_observation_feature = track_observation.1.as_ref().unwrap();

        if !matches!(self.positional_kind, PositionalMetricKind::Ignore)
            && Universal2DBox::too_far(candidate_observation_bbox, track_observation_bbox)
        {
            None
        } else {
            let f = KalmanFilter::default();
            let state = track_attributes.state.unwrap();
            Some((
                match self.positional_kind {
                    PositionalMetricKind::Mahalanobis => {
                        let dist = f.distance(state, candidate_observation_bbox);
                        Some(KalmanFilter::calculate_cost(dist, true))
                    }
                    PositionalMetricKind::IoU => {
                        let box_m_opt = Universal2DBox::calculate_metric_object(
                            &candidate_observation.0,
                            &track_observation.0,
                        );
                        if let Some(box_m) = &box_m_opt {
                            if *box_m < 0.01 {
                                None
                            } else {
                                box_m_opt
                            }
                        } else {
                            None
                        }
                    }
                    PositionalMetricKind::Ignore => None,
                },
                if self.min_required_track_visual_length >= track_attributes.length {
                    Some(match self.visual_kind {
                        VisualMetricKind::Euclidean => {
                            euclidean(candidate_observation_feature, track_observation_feature)
                        }
                        VisualMetricKind::Cosine => {
                            cosine(candidate_observation_feature, track_observation_feature)
                        }
                    })
                } else {
                    None
                },
            ))
        }
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        attrs: &mut VisualAttributes,
        features: &mut Vec<ObservationSpec<Universal2DBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.0.as_ref().unwrap();

        let f = KalmanFilter::default();

        let state = if let Some(state) = attrs.state {
            f.update(state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(state);
        attrs.state = Some(prediction);
        let predicted_bbox = prediction.universal_bbox();
        attrs.length += 1;

        attrs.observed_boxes.push_back(observation_bbox.clone());
        attrs.predicted_boxes.push_back(predicted_bbox.clone());

        if attrs.max_history_len > 0 && attrs.observed_boxes.len() > attrs.max_history_len {
            attrs.observed_boxes.pop_front();
            attrs.predicted_boxes.pop_front();
        }

        observation.0 = Some(predicted_bbox);
        features.push(observation);
        if features.len() > attrs.max_observations {
            features.swap_remove(0);
        }

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<Universal2DBox>>,
    ) -> Vec<ObservationMetricOk<Universal2DBox>> {
        unfiltered
            .into_iter()
            .filter(|x| x.attribute_metric.unwrap_or(0.0) > self.bbox_distance_threshold)
            .collect()
    }
}
