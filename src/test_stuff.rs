use crate::distance::euclidean;
use crate::track::{
    feat_confidence_cmp, AttributeMatch, AttributeUpdate, FeatureObservationsGroups, FeatureSpec,
    Metric, TrackBakingStatus,
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

#[derive(Debug, Clone)]
pub struct SimpleAttrs {
    set: bool,
}

impl Default for SimpleAttrs {
    fn default() -> Self {
        Self { set: false }
    }
}

#[derive(Default)]
pub struct SimpleAttributeUpdate;

impl AttributeUpdate<SimpleAttrs> for SimpleAttributeUpdate {
    fn apply(&self, attrs: &mut SimpleAttrs) -> Result<()> {
        if attrs.set {
            return Err(AppError::SetError.into());
        }
        attrs.set = true;
        Ok(())
    }
}

impl AttributeMatch<SimpleAttrs> for SimpleAttrs {
    fn compatible(&self, other: &SimpleAttrs) -> bool {
        self.set && other.set
    }

    fn merge(&mut self, other: &SimpleAttrs) -> Result<()> {
        if self.compatible(&other) {
            Ok(())
        } else {
            Err(AppError::Incompatible.into())
        }
    }

    fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
        if self.set {
            Ok(TrackBakingStatus::Ready)
        } else {
            Ok(TrackBakingStatus::Pending)
        }
    }
}

#[derive(Default, Clone)]
pub struct SimpleMetric;

impl Metric for SimpleMetric {
    fn distance(_feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32> {
        Ok(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        features: &mut Vec<FeatureSpec>,
        _prev_length: usize,
    ) -> Result<()> {
        features.sort_by(feat_confidence_cmp);
        features.truncate(1);
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct UnboundAttrs;

#[derive(Default)]
pub struct UnboundAttributeUpdate;

impl AttributeUpdate<UnboundAttrs> for UnboundAttributeUpdate {
    fn apply(&self, _attrs: &mut UnboundAttrs) -> Result<()> {
        Ok(())
    }
}

impl AttributeMatch<UnboundAttrs> for UnboundAttrs {
    fn compatible(&self, _other: &UnboundAttrs) -> bool {
        true
    }

    fn merge(&mut self, _other: &UnboundAttrs) -> Result<()> {
        Ok(())
    }

    fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
        Ok(TrackBakingStatus::Ready)
    }
}

#[derive(Default, Clone)]
pub struct UnboundMetric;

impl Metric for UnboundMetric {
    fn distance(_feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Result<f32> {
        Ok(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        features: &mut Vec<FeatureSpec>,
        _prev_length: usize,
    ) -> Result<()> {
        features.sort_by(feat_confidence_cmp);
        features.truncate(1);
        Ok(())
    }
}
