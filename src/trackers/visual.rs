use crate::track::{
    NoopLookup, Observation, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::State;
use anyhow::Result;
use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

mod metric_impl;

/// Implementation of Python-only structs and their implementations
///
pub mod visual_py;

/// Universal visual attributes for visual trackers
///
#[derive(Debug, Clone, Default)]
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
    /// Custom scene id provided by the user
    pub scene_id: u64,
    /// Custom object id provided for the bbox and observation
    pub custom_object_id: Option<u64>,

    state: Option<State>,
    // how many observations store for voting
    max_observations: usize,
    // how long history (bboxes, observations) can be kept in attributes
    max_history_len: usize,
    // when the track becomes wasted
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

impl TrackAttributes<VisualAttributes, Universal2DBox> for VisualAttributes {
    type Update = VisualAttributesUpdate;
    type Lookup = NoopLookup<VisualAttributes, Universal2DBox>;

    fn compatible(&self, other: &VisualAttributes) -> bool {
        self.scene_id == other.scene_id
    }

    fn merge(&mut self, other: &VisualAttributes) -> Result<()> {
        self.last_updated_epoch = other.last_updated_epoch;
        self.custom_object_id = other.custom_object_id;
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<Universal2DBox>) -> Result<TrackStatus> {
        let scene_id = self.scene_id;
        if let Some(current_epoch) = &self.current_epochs {
            let current_epoch = current_epoch.read().unwrap();
            if self.last_updated_epoch + self.max_idle_epochs
                < *current_epoch.get(&scene_id).unwrap_or(&0)
            {
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

#[pyclass]
#[derive(Clone, Default)]
pub enum VisualMetricType {
    #[default]
    Euclidean,
    Cosine,
}

#[pymethods]
impl VisualMetricType {
    #[staticmethod]
    pub fn euclidean() -> Self {
        VisualMetricType::Euclidean
    }

    #[staticmethod]
    pub fn cosine() -> Self {
        VisualMetricType::Cosine
    }
}

#[derive(Clone, Default)]
pub enum PositionalMetricType {
    #[default]
    Mahalanobis,
    IoU(f32),
    Ignore,
}

#[derive(Clone)]
pub struct VisualMetric {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    minimal_visual_track_len: usize,
}

impl Default for VisualMetric {
    fn default() -> Self {
        VisualMetricBuilder::default().build()
    }
}

pub struct VisualMetricBuilder {
    visual_kind: VisualMetricType,
    positional_kind: PositionalMetricType,
    minimal_visual_track_len: usize,
}

/// By default the metric object is constructed with: Euclidean visual metric, IoU(0.3) positional metric
/// and minimal visual track length = 3
///
impl Default for VisualMetricBuilder {
    fn default() -> Self {
        VisualMetricBuilder {
            visual_kind: VisualMetricType::Euclidean,
            positional_kind: PositionalMetricType::IoU(0.3),
            minimal_visual_track_len: 3,
        }
    }
}

impl VisualMetricBuilder {
    pub fn visual_metric(mut self, metric: VisualMetricType) -> Self {
        self.visual_kind = metric;
        self
    }

    pub fn positional_metric(mut self, metric: PositionalMetricType) -> Self {
        if let PositionalMetricType::IoU(t) = metric {
            assert!(
                t > 0.0 && t < 1.0,
                "Threshold must lay between (0.0 and 1.0)"
            );
        }
        self.positional_kind = metric;
        self
    }

    pub fn minimal_visual_track_len(mut self, length: usize) -> Self {
        assert!(
            length > 0,
            "The minimum amount of visual features collected before visual metric is applied."
        );
        self.minimal_visual_track_len = length;
        self
    }

    pub fn build(self) -> VisualMetric {
        VisualMetric {
            visual_kind: self.visual_kind,
            positional_kind: self.positional_kind,
            minimal_visual_track_len: self.minimal_visual_track_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::visual::*;

    #[test]
    fn build_default_metric() {
        let metric = VisualMetricBuilder::default().build();
        assert!(matches!(
            metric.positional_kind,
            PositionalMetricType::IoU(t) if t == 0.3
        ));
        assert!(matches!(metric.visual_kind, VisualMetricType::Euclidean));
        assert_eq!(metric.minimal_visual_track_len, 3);
    }

    #[test]
    fn build_customized_metric() {
        let metric = VisualMetricBuilder::default()
            .visual_metric(VisualMetricType::Cosine)
            .positional_metric(PositionalMetricType::Mahalanobis)
            .minimal_visual_track_len(5)
            .build();
        drop(metric);
    }

    #[test]
    fn postprocess_distances_maha() {
        let metric = VisualMetricBuilder::default()
            .positional_metric(PositionalMetricType::Mahalanobis)
            .build();
        drop(metric);
    }
}
