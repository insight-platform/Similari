use crate::track::{ObservationAttributes, ObservationMetricOk};
use crate::voting::topn::TopNVotingElt;
use crate::voting::Voting;
use crate::Errors::GenericBBoxConversionError;
use crate::{EstimateClose, EPS};
use anyhow::Result;
use geo::{Area, BooleanOps, Coordinate, LineString, Polygon};
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Bounding box in the format (x,y, width, height)
///
#[derive(Clone, Default, Debug, Copy)]
pub struct BBox {
    /// top-left point X
    pub x: f32,
    /// top-left point Y
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BBox {
    /// Constructor
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl EstimateClose for BBox {
    /// Allows comparing bboxes
    ///
    fn almost_same(&self, other: &Self, eps: f32) -> bool {
        (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.width - other.width) < eps
            && (self.height - other.height) < eps
    }
}

/// Bounding box in the format (x, y, angle, aspect, height)
#[derive(Default, Debug)]
pub struct GenericBBox {
    /// Top-left point X
    pub x: f32,
    /// Top-left point Y
    pub y: f32,
    /// Angle
    pub angle: Option<f32>,
    /// Width/Height ratio
    pub aspect: f32,
    /// Height
    pub height: f32,

    vertex_cache: Option<Polygon<f64>>,
}

impl Clone for GenericBBox {
    fn clone(&self) -> Self {
        GenericBBox::new(self.x, self.y, self.angle, self.aspect, self.height).gen_vertices()
    }
}

impl GenericBBox {
    pub fn get_radius(&self) -> f32 {
        let hw = self.aspect * self.height / 2.0_f32;
        let hh = self.height / 2.0_f32;
        (hw * hw + hh * hh).sqrt()
    }

    pub fn gen_vertices(mut self) -> Self {
        if self.angle.is_some() {
            self.vertex_cache = Some(Polygon::from(&self));
        }
        self
    }
    /// Constructor
    pub fn new(x: f32, y: f32, angle: Option<f32>, aspect: f32, height: f32) -> Self {
        Self {
            x,
            y,
            angle,
            aspect,
            height,
            vertex_cache: None,
        }
        .gen_vertices()
    }

    pub fn rotate(self, angle: f32) -> Self {
        Self {
            x: self.x,
            y: self.y,
            angle: Some(angle),
            aspect: self.aspect,
            height: self.height,
            vertex_cache: None,
        }
        .gen_vertices()
    }
}

impl EstimateClose for GenericBBox {
    /// Allows comparing bboxes
    ///
    fn almost_same(&self, other: &Self, eps: f32) -> bool {
        (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.angle.unwrap_or(0.0) - other.angle.unwrap_or(0.0)) < eps
            && (self.aspect - other.aspect) < eps
            && (self.height - other.height) < eps
    }
}

impl From<BBox> for GenericBBox {
    fn from(f: BBox) -> Self {
        GenericBBox {
            x: f.x + f.width / 2.0,
            y: f.y + f.height / 2.0,
            angle: None,
            aspect: f.width / f.height,
            height: f.height,
            vertex_cache: None,
        }
        .gen_vertices()
    }
}

impl From<GenericBBox> for Result<BBox> {
    /// This is a lossy translation. It is valid only when the angle is 0
    fn from(f: GenericBBox) -> Self {
        if f.angle.is_some() {
            Err(GenericBBoxConversionError.into())
        } else {
            let width = f.height * f.aspect;
            Ok(BBox {
                x: f.x - width / 2.0,
                y: f.y - f.height / 2.0,
                width,
                height: f.height,
            })
        }
    }
}

impl From<&GenericBBox> for Polygon<f64> {
    fn from(b: &GenericBBox) -> Self {
        let angle = b.angle.unwrap_or(0.0) as f64;
        let height = b.height as f64;
        let aspect = b.aspect as f64;

        let c = angle.cos();
        let s = angle.sin();

        let half_width = height * aspect / 2.0;
        let half_height = height / 2.0;

        let r1x = -half_width * c - half_height * s;
        let r1y = -half_width * s + half_height * c;

        let r2x = half_width * c - half_height * s;
        let r2y = half_width * s + half_height * c;

        let x = b.x as f64;
        let y = b.y as f64;

        Polygon::new(
            LineString(vec![
                Coordinate {
                    x: x + r1x,
                    y: y + r1y,
                },
                Coordinate {
                    x: x + r2x,
                    y: y + r2y,
                },
                Coordinate {
                    x: x - r1x,
                    y: y - r1y,
                },
                Coordinate {
                    x: x - r2x,
                    y: y - r2y,
                },
            ]),
            vec![],
        )
    }
}

#[cfg(test)]
mod polygons {
    use crate::track::ObservationAttributes;
    use crate::utils::bbox::GenericBBox;
    use crate::utils::clipping::sutherland_hodgman_clip;
    use crate::EPS;
    use geo::{Area, BooleanOps, Polygon};
    use std::f32::consts::PI;

    #[test]
    fn transform() {
        let bbox1 = GenericBBox::new(0.0, 0.0, Some(2.0), 0.5, 2.0);
        let polygon1 = Polygon::from(&bbox1);
        let bbox2 = GenericBBox::new(0.0, 0.0, Some(2.0 + PI / 2.0), 0.5, 2.0);
        let polygon2 = Polygon::from(&bbox2);
        let clip = sutherland_hodgman_clip(&polygon1, &polygon2);
        let int_area = clip.unsigned_area();
        let int = polygon1.intersection(&polygon2).unsigned_area();
        assert!((int - int_area).abs() < EPS as f64);

        let union = polygon1.union(&polygon2).unsigned_area();
        assert!((union - 3.0).abs() < EPS as f64);

        let res = GenericBBox::calculate_metric_object(&Some(bbox1.clone()), &Some(bbox2)).unwrap()
            as f64;
        assert!((res - int / union).abs() < EPS as f64);

        let bbox3 = GenericBBox::new(10.0, 0.0, Some(2.0 + PI / 2.0), 0.5, 2.0);
        let polygon3 = Polygon::from(&bbox3);

        let int = polygon1.intersection(&polygon3).unsigned_area();
        assert!((int - 0.0).abs() < EPS as f64);

        let union = polygon1.union(&polygon3).unsigned_area();
        assert!((union - 4.0).abs() < EPS as f64);

        let res = GenericBBox::calculate_metric_object(&Some(bbox1), &Some(bbox3)).unwrap() as f64;
        assert!((res - int / union).abs() < EPS as f64);
    }

    #[test]
    fn corner_case_f32() {
        let x = GenericBBox::new(8044.315, 8011.0454, Some(2.67877485), 1.00801, 49.8073);
        let polygon_x = Polygon::from(&x);

        let y = GenericBBox::new(8044.455, 8011.338, Some(2.67877485), 1.0083783, 49.79979);
        let polygon_y = Polygon::from(&y);

        dbg!(&polygon_x, &polygon_y);
    }
}

impl From<&GenericBBox> for BBox {
    /// This is a lossy translation. It is valid only when the angle is 0
    fn from(f: &GenericBBox) -> Self {
        let width = f.height * f.aspect;
        BBox {
            x: f.x - width / 2.0,
            y: f.y - f.height / 2.0,
            width,
            height: f.height,
        }
    }
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
        self.almost_same(other, EPS)
    }
}

pub fn normalize_angle(a: f32) -> f32 {
    let pix2 = 2.0 * PI;
    let n = (a / pix2).floor();
    let a = a - n * pix2;
    if a < 0.0 {
        a + pix2
    } else {
        a
    }
}

#[cfg(test)]
mod tests_normalize_angle {
    use crate::utils::bbox::normalize_angle;
    use crate::EPS;

    #[test]
    fn normalize() {
        assert!((normalize_angle(0.3) - 0.3).abs() < EPS);
        assert!((normalize_angle(-0.3) - 5.983184).abs() < EPS);
        assert!((normalize_angle(-0.3) - 5.983184).abs() < EPS);
        assert!((normalize_angle(6.583184) - 0.3).abs() < EPS);
    }
}

impl ObservationAttributes for GenericBBox {
    type MetricObject = f32;

    fn calculate_metric_object(
        left: &Option<Self>,
        rigth: &Option<Self>,
    ) -> Option<Self::MetricObject> {
        match (left, rigth) {
            (Some(l), Some(r)) => {
                if (normalize_angle(l.angle.unwrap_or(0.0))
                    - normalize_angle(r.angle.unwrap_or(0.0)))
                .abs()
                    < EPS
                {
                    BBox::calculate_metric_object(&Some(l.into()), &Some(r.into()))
                } else {
                    assert!(l.aspect > 0.0);
                    assert!(l.height > 0.0);
                    assert!(r.aspect > 0.0);
                    assert!(r.height > 0.0);

                    let max_distance = l.get_radius() + r.get_radius();
                    let x = l.x - r.x;
                    let y = l.y - r.y;
                    if x * x + y * y > max_distance * max_distance {
                        None
                    } else {
                        let mut l = l.clone();
                        let mut r = r.clone();

                        if l.vertex_cache.is_none() {
                            let angle = l.angle.unwrap_or(0.0);
                            l = l.rotate(angle).gen_vertices();
                        }

                        if r.vertex_cache.is_none() {
                            let angle = r.angle.unwrap_or(0.0);
                            r = r.rotate(angle).gen_vertices();
                        }

                        let p1 = l.vertex_cache.as_ref().unwrap();
                        let p2 = r.vertex_cache.as_ref().unwrap();

                        let intersection = p1.intersection(&p2).unsigned_area();
                        let union = p1.union(&p2).unsigned_area();
                        let res = intersection / union;
                        Some(res as f32)
                    }
                }
            }
            _ => None,
        }
    }
}

impl PartialOrd for GenericBBox {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        unreachable!()
    }
}

impl PartialEq<Self> for GenericBBox {
    fn eq(&self, other: &Self) -> bool {
        self.almost_same(other, EPS)
    }
}

#[cfg(test)]
mod tests {
    use crate::track::ObservationAttributes;
    use crate::utils::bbox::BBox;

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

/// TopN Voting engine for IoU metric
///
pub struct IOUTopNVoting {
    pub topn: usize,
    pub min_distance: f32,
    pub min_votes: usize,
}

impl Voting<BBox> for IOUTopNVoting {
    type WinnerObject = TopNVotingElt;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<TopNVotingElt>>
    where
        T: IntoIterator<Item = ObservationMetricOk<BBox>>,
    {
        let counts: Vec<_> = distances
            .into_iter()
            .filter(
                |ObservationMetricOk {
                     from: _,
                     to: _track,
                     attribute_metric: attr_dist,
                     feature_distance: _,
                 }| match attr_dist {
                    Some(e) => *e >= self.min_distance,
                    _ => false,
                },
            )
            .map(
                |ObservationMetricOk {
                     from: src_track,
                     to: dest_track,
                     attribute_metric: _,
                     feature_distance: _,
                 }| (src_track, dest_track),
            )
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|((q, w), c)| TopNVotingElt {
                query_track: q,
                winner_track: w,
                votes: c,
            })
            .collect::<Vec<_>>();

        let mut results: HashMap<u64, Vec<TopNVotingElt>> = HashMap::new();

        for c in counts {
            let key = c.query_track;
            if let Some(val) = results.get_mut(&key) {
                val.push(c);
            } else {
                results.insert(key, vec![c]);
            }
        }

        for counts in results.values_mut() {
            counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
            counts.truncate(self.topn);
        }

        results
    }
}
