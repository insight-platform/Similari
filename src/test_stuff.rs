use crate::distance::euclidean;
use crate::track::utils::FromVec;
use crate::track::{
    Observation, ObservationAttributes, ObservationMetric, ObservationSpec, ObservationsDb,
    TrackAttributes, TrackAttributesUpdate, TrackStatus,
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

impl ObservationMetric<SimpleAttrs, f32> for SimpleMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &SimpleAttrs,
        _attrs2: &SimpleAttrs,
        e1: &ObservationSpec<f32>,
        e2: &ObservationSpec<f32>,
    ) -> (Option<f32>, Option<f32>) {
        (
            f32::calculate_metric_object(&e1.0, &e2.0),
            match (e1.1.as_ref(), e2.1.as_ref()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        )
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _attrs: &mut SimpleAttrs,
        _features: &mut Vec<ObservationSpec<f32>>,
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
    fn metric(
        _feature_class: u64,
        _attrs1: &UnboundAttrs,
        _attrs2: &UnboundAttrs,
        e1: &ObservationSpec<f32>,
        e2: &ObservationSpec<f32>,
    ) -> (Option<f32>, Option<f32>) {
        (
            f32::calculate_metric_object(&e1.0, &e2.0),
            match (e1.1.as_ref(), e2.1.as_ref()) {
                (Some(x), Some(y)) => Some(euclidean(x, y)),
                _ => None,
            },
        )
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _attrs: &mut UnboundAttrs,
        _features: &mut Vec<ObservationSpec<f32>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        Ok(())
    }
}

pub fn vec2(x: f32, y: f32) -> Observation {
    Observation::from_vec(vec![x, y])
}

impl ObservationAttributes for f32 {
    type MetricObject = f32;

    fn calculate_metric_object(
        left: &Option<Self>,
        right: &Option<Self>,
    ) -> Option<Self::MetricObject> {
        if let (Some(left), Some(right)) = (left, right) {
            Some((left - right).abs())
        } else {
            None
        }
    }
}
impl ObservationAttributes for () {
    type MetricObject = ();

    fn calculate_metric_object(
        _left: &Option<Self>,
        _right: &Option<Self>,
    ) -> Option<Self::MetricObject> {
        None
    }
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
    type Item = ObservationSpec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist);
        self.y += self.gen.sample(&self.dist);
        Some(ObservationSpec(
            Some(self.gen.sample(&self.dist) + 0.7),
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
}

impl Iterator for BoxGen2 {
    type Item = BBox;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist_pos);
        self.y += self.gen.sample(&self.dist_pos);

        self.width += self.gen.sample(&self.dist_box);
        self.height += self.gen.sample(&self.dist_box);

        if self.width < 1.0 {
            self.width = 2.0;
        }
        if self.height < 1.0 {
            self.height = 2.0;
        }

        Some(BBox {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        })
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

#[derive(Clone, Default, Debug)]
pub struct BBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl ObservationAttributes for BBox {
    type MetricObject = f32;

    fn calculate_metric_object(
        _left: &Option<Self>,
        _right: &Option<Self>,
    ) -> Option<Self::MetricObject> {
        match (_left, _right) {
            (Some(l), Some(r)) => {
                assert!(l.width > 0.0);
                assert!(l.height > 0.0);
                assert!(r.width > 0.0);
                assert!(r.height > 0.0);

                let (ax0, ay0, ax1, ay1) = (l.x, l.y, l.x + l.width, l.y + l.height);
                let (bx0, by0, bx1, by1) = (r.x, r.y, r.x + r.width, r.y + r.height);

                let (x1, y1) = (ax0.max(bx0), ay0.max(by0));
                let (x2, y2) = (ax1.min(bx1), ay1.min(by1));

                let int_width = x2 - x1;
                let int_height = y2 - y1;

                let intersection = if int_width > 0.0 && int_height > 0.0 {
                    int_width * int_height
                } else {
                    0.0
                };

                let union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection;
                Some(intersection / union)
            }
            _ => None,
        }
    }
}

impl PartialOrd for BBox {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        unreachable!()
    }
}

impl PartialEq<Self> for BBox {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPS
            && (self.y - other.y).abs() < EPS
            && (self.width - other.width).abs() < EPS
            && (self.height - other.height).abs() < EPS
    }
}

#[cfg(test)]
mod tests {
    use crate::test_stuff::BBox;
    use crate::track::ObservationAttributes;

    #[test]
    fn test_iou() {
        let bb1 = BBox {
            x: -1.0,
            y: -1.0,
            width: 2.0,
            height: 2.0,
        };

        let bb2 = BBox {
            x: -0.9,
            y: -0.9,
            width: 2.0,
            height: 2.0,
        };
        let bb3 = BBox {
            x: 1.0,
            y: 1.0,
            width: 3.0,
            height: 3.0,
        };

        assert!(
            BBox::calculate_metric_object(&Some(bb1.clone()), &Some(bb1.clone())).unwrap() > 0.999
        );
        assert!(
            BBox::calculate_metric_object(&Some(bb2.clone()), &Some(bb2.clone())).unwrap() > 0.999
        );
        assert!(
            BBox::calculate_metric_object(&Some(bb1.clone()), &Some(bb2.clone())).unwrap() > 0.8
        );
        assert!(
            BBox::calculate_metric_object(&Some(bb1.clone()), &Some(bb3.clone())).unwrap() < 0.001
        );
        assert!(
            BBox::calculate_metric_object(&Some(bb2.clone()), &Some(bb3.clone())).unwrap() < 0.001
        );
    }
}
