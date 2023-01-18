pub mod iou;

use crate::distance::euclidean;
use crate::track::utils::FromVec;
use crate::track::{
    Feature, MetricOutput, MetricQuery, NoopLookup, Observation, ObservationAttributes,
    ObservationMetric, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::BoundingBox;
use anyhow::Result;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    type Update = SimpleAttributeUpdate;
    type Lookup = NoopLookup<SimpleAttrs, f32>;

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

impl ObservationMetric<SimpleAttrs, f32> for SimpleMetric {
    fn metric(&self, mq: &MetricQuery<'_, SimpleAttrs, f32>) -> MetricOutput<f32> {
        let (e1, e2) = (mq.candidate_observation, mq.track_observation);
        Some((
            f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
            match (e1.feature().as_ref(), e2.feature().as_ref()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        _attrs: &mut SimpleAttrs,
        _features: &mut Vec<Observation<f32>>,
        _prev_length: usize,
        _is_merge: bool,
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
    type Update = UnboundAttributeUpdate;
    type Lookup = NoopLookup<UnboundAttrs, f32>;

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

impl ObservationMetric<UnboundAttrs, f32> for UnboundMetric {
    fn metric(&self, mq: &MetricQuery<'_, UnboundAttrs, f32>) -> MetricOutput<f32> {
        let (e1, e2) = (mq.candidate_observation, mq.track_observation);
        Some((
            f32::calculate_metric_object(&e1.attr().as_ref(), &e2.attr().as_ref()),
            match (e1.feature().as_ref(), e2.feature().as_ref()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        ))
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        _attrs: &mut UnboundAttrs,
        _features: &mut Vec<Observation<f32>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn vec2(x: f32, y: f32) -> Feature {
    Feature::from_vec(vec![x, y])
}

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
    type Item = Observation<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(self.dist);
        self.y += self.gen.sample(self.dist);
        Some(Observation::new(
            Some(self.gen.sample(self.dist) + 0.7),
            Some(vec2(self.x, self.y)),
        ))
    }
}

pub struct BoxGen2 {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    gen: ThreadRng,
    dist_pos: Uniform<f32>,
    dist_box: Uniform<f32>,
}

impl BoxGen2 {
    pub fn new(x: f32, y: f32, width: f32, height: f32, pos_drift: f32, box_drift: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            gen: rand::thread_rng(),
            dist_pos: Uniform::new(-pos_drift, pos_drift),
            dist_box: Uniform::new(-box_drift, box_drift),
        }
    }
    pub fn new_monotonous(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        pos_drift: f32,
        box_drift: f32,
    ) -> Self {
        Self {
            x,
            y,
            width,
            height,
            gen: rand::thread_rng(),
            dist_pos: Uniform::new(0.0, pos_drift),
            dist_box: Uniform::new(-box_drift, box_drift),
        }
    }
}

impl Iterator for BoxGen2 {
    type Item = BoundingBox;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(self.dist_pos);
        self.y += self.gen.sample(self.dist_pos);

        self.width += self.gen.sample(self.dist_box);
        self.height += self.gen.sample(self.dist_box);

        if self.width < 1.0 {
            self.width = 1.0;
        }
        if self.height < 1.0 {
            self.height = 1.0;
        }

        Some(BoundingBox::new(self.x, self.y, self.width, self.height))
    }
}

#[inline]
pub fn current_time_span() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap()
}

#[inline]
pub fn current_time_ms() -> u128 {
    current_time_span().as_millis()
}

#[inline]
pub fn current_time_sec() -> u64 {
    current_time_span().as_secs()
}

pub struct FeatGen {
    x: f32,
    len: usize,
    gen: ThreadRng,
    dist: Uniform<f32>,
}

impl FeatGen {
    pub fn new(x: f32, len: usize, drift: f32) -> Self {
        Self {
            x,
            len,
            gen: rand::thread_rng(),
            dist: Uniform::new(-drift, drift),
        }
    }
}

impl Iterator for FeatGen {
    type Item = Observation<()>;

    fn next(&mut self) -> Option<Self::Item> {
        let v = (0..self.len)
            .map(|_| self.x + self.gen.sample(self.dist))
            .collect::<Vec<_>>();
        Some(Observation::<()>::new(None, Some(Feature::from_vec(v))))
    }
}
