use crate::distance::euclidean;
use crate::track::utils::FromVec;
use crate::track::{
    Metric, Observation, ObservationAttributes, ObservationSpec, ObservationsDb, TrackAttributes,
    TrackAttributesUpdate, TrackStatus,
};
use anyhow::Result;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::Rng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Only single feature vector can be set")]
    SetError,
    #[error("Incompatible attributes")]
    Incompatible,
}

#[derive(Debug, Clone, Default)]
pub struct SimpleAttrs {
    set: bool,
}

#[derive(Default, Clone)]
pub struct SimpleAttributeUpdate;

impl TrackAttributesUpdate<SimpleAttrs> for SimpleAttributeUpdate {
    fn apply(&self, attrs: &mut SimpleAttrs) -> Result<()> {
        if attrs.set {
            return Err(AppError::SetError.into());
        }
        attrs.set = true;
        Ok(())
    }
}

impl TrackAttributes<SimpleAttrs, f32> for SimpleAttrs {
    fn compatible(&self, other: &SimpleAttrs) -> bool {
        self.set && other.set
    }

    fn merge(&mut self, other: &SimpleAttrs) -> Result<()> {
        if self.compatible(other) {
            Ok(())
        } else {
            Err(AppError::Incompatible.into())
        }
    }

    fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
        if self.set {
            Ok(TrackStatus::Ready)
        } else {
            Ok(TrackStatus::Pending)
        }
    }
}

#[derive(Default, Clone)]
pub struct SimpleMetric;

impl Metric<f32> for SimpleMetric {
    fn distance(
        _feature_class: u64,
        e1: &ObservationSpec<f32>,
        e2: &ObservationSpec<f32>,
    ) -> Option<f32> {
        Some(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _features: &mut Vec<ObservationSpec<f32>>,
        _prev_length: usize,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct UnboundAttrs;

#[derive(Default, Clone)]
pub struct UnboundAttributeUpdate;

impl TrackAttributesUpdate<UnboundAttrs> for UnboundAttributeUpdate {
    fn apply(&self, _attrs: &mut UnboundAttrs) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<UnboundAttrs, f32> for UnboundAttrs {
    fn compatible(&self, _other: &UnboundAttrs) -> bool {
        true
    }

    fn merge(&mut self, _other: &UnboundAttrs) -> Result<()> {
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Default, Clone)]
pub struct UnboundMetric;

impl Metric<f32> for UnboundMetric {
    fn distance(
        _feature_class: u64,
        e1: &ObservationSpec<f32>,
        e2: &ObservationSpec<f32>,
    ) -> Option<f32> {
        Some(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _features: &mut Vec<ObservationSpec<f32>>,
        _prev_length: usize,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn vec2(x: f32, y: f32) -> Observation {
    Observation::from_vec(vec![x, y])
}

impl ObservationAttributes for f32 {}
impl ObservationAttributes for () {}

pub struct FeatGen2 {
    x: f32,
    y: f32,
    gen: ThreadRng,
    dist: Uniform<f32>,
}

impl FeatGen2 {
    pub fn new(x: f32, y: f32, drift: f32) -> Self {
        Self {
            x,
            y,
            gen: rand::thread_rng(),
            dist: Uniform::new(-drift, drift),
        }
    }
}

impl Iterator for FeatGen2 {
    type Item = ObservationSpec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist);
        self.y += self.gen.sample(&self.dist);
        Some(ObservationSpec(
            self.gen.sample(&self.dist) + 0.7,
            vec2(self.x, self.y),
        ))
    }
}
