use crate::distance::euclidean;
use crate::track::utils::FromVec;
use crate::track::{
    Metric, Observation, ObservationAttributes, ObservationSpec, ObservationsDb, TrackAttributes,
    TrackAttributesUpdate, TrackStatus,
};
use crate::EPS;
use anyhow::Result;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::cmp::Ordering;
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
}

impl Iterator for BoxGen2 {
    type Item = Bbox;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist_pos);
        self.y += self.gen.sample(&self.dist_pos);
        self.width += self.gen.sample(&self.dist_box);
        self.height += self.gen.sample(&self.dist_box);

        Some(Bbox {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        })
    }
}

pub fn current_time_span() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap()
}

pub fn current_time_ms() -> u128 {
    current_time_span().as_millis()
}

pub fn current_time_sec() -> u64 {
    current_time_span().as_secs()
}

#[derive(Clone, Default, Debug)]
pub struct Bbox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl ObservationAttributes for Bbox {}

impl PartialOrd for Bbox {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        unreachable!()
    }
}

impl PartialEq<Self> for Bbox {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPS
            && (self.y - other.y).abs() < EPS
            && (self.width - other.width).abs() < EPS
            && (self.height - other.height).abs() < EPS
    }
}
