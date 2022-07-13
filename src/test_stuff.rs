use crate::distance::euclidean;
use crate::track::{
    Feature, FeatureAttribute, FeatureObservationsGroups, FeatureSpec, FromVec, Metric,
    TrackAttributes, TrackAttributesUpdate, TrackBakingStatus,
};
use anyhow::Result;
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

    fn baked(&self, _observations: &FeatureObservationsGroups<f32>) -> Result<TrackBakingStatus> {
        if self.set {
            Ok(TrackBakingStatus::Ready)
        } else {
            Ok(TrackBakingStatus::Pending)
        }
    }
}

#[derive(Default, Clone)]
pub struct SimpleMetric;

impl Metric<f32> for SimpleMetric {
    fn distance(_feature_class: u64, e1: &FeatureSpec<f32>, e2: &FeatureSpec<f32>) -> Option<f32> {
        Some(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _features: &mut Vec<FeatureSpec<f32>>,
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

    fn baked(&self, _observations: &FeatureObservationsGroups<f32>) -> Result<TrackBakingStatus> {
        Ok(TrackBakingStatus::Ready)
    }
}

#[derive(Default, Clone)]
pub struct UnboundMetric;

impl Metric<f32> for UnboundMetric {
    fn distance(_feature_class: u64, e1: &FeatureSpec<f32>, e2: &FeatureSpec<f32>) -> Option<f32> {
        Some(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _features: &mut Vec<FeatureSpec<f32>>,
        _prev_length: usize,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn vec2(x: f32, y: f32) -> Feature {
    Feature::from_vec(vec![x, y])
}

impl FeatureAttribute for f32 {}
impl FeatureAttribute for () {}
