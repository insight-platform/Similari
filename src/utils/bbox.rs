use crate::track::{ObservationAttributes, ObservationMetricOk};
use crate::utils::clipping::sutherland_hodgman_clip;
use crate::voting::topn::TopNVotingElt;
use crate::voting::Voting;
use crate::Errors::GenericBBoxConversionError;
use crate::EPS;
use anyhow::Result;
use geo::{Area, Coordinate, LineString, Polygon};
use itertools::Itertools;
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Bounding box in the format (left, top, width, height)
///
#[derive(Clone, Default, Debug, Copy)]
#[pyclass]
pub struct BoundingBox {
    #[pyo3(get, set)]
    left: f32,
    #[pyo3(get, set)]
    top: f32,
    #[pyo3(get, set)]
    width: f32,
    #[pyo3(get, set)]
    height: f32,
    #[pyo3(get, set)]
    confidence: f32,
}

#[pymethods]
impl BoundingBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn as_xyaah(&self) -> Universal2DBox {
        Universal2DBox::from(self)
    }

    /// Constructor. Confidence is set to 1.0
    ///
    ///
    #[new]
    pub fn new(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            left,
            top,
            width,
            height,
            confidence: 1.0,
        }
    }

    /// Creates the bbox with custom confidence
    ///
    #[staticmethod]
    pub fn new_with_confidence(
        left: f32,
        top: f32,
        width: f32,
        height: f32,
        confidence: f32,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must lay between 0.0 and 1.0"
        );
        Self {
            left,
            top,
            width,
            height,
            confidence,
        }
    }
}

// impl EstimateClose for BoundingBox {
//     /// Allows comparing bboxes
//     ///
//     fn almost_same(&self, other: &Self, eps: f32) -> bool {
//         (self.left - other.left).abs() < eps
//             && (self.top - other.top).abs() < eps
//             && (self.width - other.width) < eps
//             && (self.height - other.height) < eps
//             && (self.confidence - other.confidence) < eps
//     }
// }

/// Bounding box in the format (x, y, angle, aspect, height)
#[derive(Default, Debug)]
#[pyclass]
pub struct Universal2DBox {
    #[pyo3(get, set)]
    pub xc: f32,
    #[pyo3(get, set)]
    pub yc: f32,
    #[pyo3(get, set)]
    pub angle: Option<f32>,
    #[pyo3(get, set)]
    pub aspect: f32,
    #[pyo3(get, set)]
    pub height: f32,
    #[pyo3(get, set)]
    pub confidence: f32,
    _vertex_cache: Option<Polygon<f64>>,
}

impl Clone for Universal2DBox {
    fn clone(&self) -> Self {
        Universal2DBox::new_with_confidence(
            self.xc,
            self.yc,
            self.angle,
            self.aspect,
            self.height,
            self.confidence,
        )
    }
}

#[pymethods]
impl Universal2DBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn get_radius(&self) -> f32 {
        let hw = self.aspect * self.height / 2.0_f32;
        let hh = self.height / 2.0_f32;
        (hw * hw + hh * hh).sqrt()
    }

    #[pyo3(name = "as_ltwh")]
    pub fn as_ltwh_py(&self) -> PyResult<BoundingBox> {
        let r: Result<BoundingBox> = Result::from(self);
        if let Ok(res) = r {
            Ok(res)
        } else {
            Err(PyAttributeError::new_err(format!("{:?}", r)))
        }
    }

    #[pyo3(name = "gen_vertices")]
    pub fn gen_vertices_py(&mut self) {
        let copy = self.clone();
        if self.angle.is_some() {
            self._vertex_cache = Some(Polygon::from(&copy));
        }
    }

    /// Sets the angle
    ///
    #[pyo3(name = "rotate")]
    pub fn rotate_py(&mut self, angle: f32) {
        self.angle = Some(angle)
    }

    /// Sets the angle
    ///
    #[pyo3(name = "set_confidence")]
    pub fn set_confidence_py(&mut self, confidence: f32) {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must lay between 0.0 and 1.0"
        );
        self.confidence = confidence;
    }

    /// Constructor. Creates new generic bbox and doesn't generate vertex cache
    ///
    #[new]
    pub fn new(xc: f32, yc: f32, angle: Option<f32>, aspect: f32, height: f32) -> Self {
        Self {
            xc,
            yc,
            angle,
            aspect,
            height,
            confidence: 1.0,
            _vertex_cache: None,
        }
    }

    /// Constructor. Creates new generic bbox and doesn't generate vertex cache
    ///
    #[staticmethod]
    pub fn new_with_confidence(
        xc: f32,
        yc: f32,
        angle: Option<f32>,
        aspect: f32,
        height: f32,
        confidence: f32,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must lay between 0.0 and 1.0"
        );

        Self {
            xc,
            yc,
            angle,
            aspect,
            height,
            confidence,
            _vertex_cache: None,
        }
    }

    /// Constructor. Creates new generic bbox and doesn't generate vertex cache
    ///
    #[staticmethod]
    pub fn ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self::from(BoundingBox::new_with_confidence(
            left, top, width, height, 1.0,
        ))
    }

    /// Constructor. Creates new generic bbox and doesn't generate vertex cache
    ///
    #[staticmethod]
    pub fn ltwh_with_confidence(
        left: f32,
        top: f32,
        width: f32,
        height: f32,
        confidence: f32,
    ) -> Self {
        Self::from(BoundingBox::new_with_confidence(
            left, top, width, height, confidence,
        ))
    }

    pub fn area(&self) -> f32 {
        let w = self.height * self.aspect;
        w * self.height
    }
}

impl Universal2DBox {
    pub fn get_vertices(&self) -> &Option<Polygon<f64>> {
        &self._vertex_cache
    }

    pub fn gen_vertices(mut self) -> Self {
        if self.angle.is_some() {
            self._vertex_cache = Some(Polygon::from(&self));
        }
        self
    }

    /// Sets the angle
    ///
    pub fn rotate(self, angle: f32) -> Self {
        Self {
            xc: self.xc,
            yc: self.yc,
            angle: Some(angle),
            aspect: self.aspect,
            height: self.height,
            confidence: self.confidence,
            _vertex_cache: None,
        }
    }
}

// impl EstimateClose for Universal2DBox {
//     /// Allows comparing bboxes
//     ///
//     fn almost_same(&self, other: &Self, eps: f32) -> bool {
//         (self.xc - other.xc).abs() < eps
//             && (self.yc - other.yc).abs() < eps
//             && (self.angle.unwrap_or(0.0) - other.angle.unwrap_or(0.0)) < eps
//             && (self.aspect - other.aspect) < eps
//             && (self.height - other.height) < eps
//     }
// }

impl From<BoundingBox> for Universal2DBox {
    fn from(f: BoundingBox) -> Self {
        Self::from(&f)
    }
}

impl From<&BoundingBox> for Universal2DBox {
    fn from(f: &BoundingBox) -> Self {
        Universal2DBox {
            xc: f.left + f.width / 2.0,
            yc: f.top + f.height / 2.0,
            angle: None,
            aspect: f.width / f.height,
            height: f.height,
            confidence: f.confidence,
            _vertex_cache: None,
        }
    }
}

impl From<Universal2DBox> for Result<BoundingBox> {
    /// This is a lossy translation. It is valid only when the angle is 0
    fn from(f: Universal2DBox) -> Self {
        let r: Result<BoundingBox> = Self::from(&f);
        r
    }
}

impl From<&Universal2DBox> for Result<BoundingBox> {
    /// This is a lossy translation. It is valid only when the angle is 0
    fn from(f: &Universal2DBox) -> Self {
        if f.angle.is_some() {
            Err(GenericBBoxConversionError.into())
        } else {
            let width = f.height * f.aspect;
            Ok(BoundingBox {
                left: f.xc - width / 2.0,
                top: f.yc - f.height / 2.0,
                width,
                height: f.height,
                confidence: f.confidence,
            })
        }
    }
}

impl From<&Universal2DBox> for Polygon<f64> {
    fn from(b: &Universal2DBox) -> Self {
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

        let x = b.xc as f64;
        let y = b.yc as f64;

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
    use crate::utils::bbox::Universal2DBox;
    use crate::utils::clipping::sutherland_hodgman_clip;
    use crate::EPS;
    use geo::{Area, BooleanOps, Polygon};
    use std::f32::consts::PI;

    #[test]
    fn transform() {
        let bbox1 = Universal2DBox::new(0.0, 0.0, Some(2.0), 0.5, 2.0);
        let polygon1 = Polygon::from(&bbox1);
        let bbox2 = Universal2DBox::new(0.0, 0.0, Some(2.0 + PI / 2.0), 0.5, 2.0);
        let polygon2 = Polygon::from(&bbox2);
        let clip = sutherland_hodgman_clip(&polygon1, &polygon2);
        let int_area = clip.unsigned_area();
        let int = polygon1.intersection(&polygon2).unsigned_area();
        assert!((int - int_area).abs() < EPS as f64);

        let union = polygon1.union(&polygon2).unsigned_area();
        assert!((union - 3.0).abs() < EPS as f64);

        let res =
            Universal2DBox::calculate_metric_object(&Some(&bbox1), &Some(&bbox2)).unwrap() as f64;
        assert!((res - int / union).abs() < EPS as f64);

        let bbox3 = Universal2DBox::new(10.0, 0.0, Some(2.0 + PI / 2.0), 0.5, 2.0);
        let polygon3 = Polygon::from(&bbox3);

        let int = polygon1.intersection(&polygon3).unsigned_area();
        assert!((int - 0.0).abs() < EPS as f64);

        let union = polygon1.union(&polygon3).unsigned_area();
        assert!((union - 4.0).abs() < EPS as f64);

        assert!(Universal2DBox::calculate_metric_object(&Some(&bbox1), &Some(&bbox3)).is_none());
    }

    #[test]
    fn corner_case_f32() {
        let x = Universal2DBox::new(8044.315, 8011.0454, Some(2.678_774_8), 1.00801, 49.8073);
        let polygon_x = Polygon::from(&x);

        let y = Universal2DBox::new(8044.455, 8011.338, Some(2.678_774_8), 1.0083783, 49.79979);
        let polygon_y = Polygon::from(&y);

        dbg!(&polygon_x, &polygon_y);
    }
}

impl From<&Universal2DBox> for BoundingBox {
    /// This is a lossy translation. It is valid only when the angle is 0
    fn from(f: &Universal2DBox) -> Self {
        let width = f.height * f.aspect;
        BoundingBox {
            left: f.xc - width / 2.0,
            top: f.yc - f.height / 2.0,
            width,
            height: f.height,
            confidence: f.confidence,
        }
    }
}

impl BoundingBox {
    pub fn intersection(l: &BoundingBox, r: &BoundingBox) -> f64 {
        assert!(l.width > 0.0);
        assert!(l.height > 0.0);
        assert!(r.width > 0.0);
        assert!(r.height > 0.0);

        let (ax0, ay0, ax1, ay1) = (l.left, l.top, l.left + l.width, l.top + l.height);
        let (bx0, by0, bx1, by1) = (r.left, r.top, r.left + r.width, r.top + r.height);

        let (x1, y1) = (ax0.max(bx0), ay0.max(by0));
        let (x2, y2) = (ax1.min(bx1), ay1.min(by1));

        let int_width = x2 - x1;
        let int_height = y2 - y1;

        if int_width > 0.0 && int_height > 0.0 {
            (int_width * int_height) as f64
        } else {
            0.0_f64
        }
    }
}

impl ObservationAttributes for BoundingBox {
    type MetricObject = f32;

    fn calculate_metric_object(
        left: &Option<&Self>,
        right: &Option<&Self>,
    ) -> Option<Self::MetricObject> {
        match (left, right) {
            (Some(l), Some(r)) => {
                let intersection = BoundingBox::intersection(l, r);
                let union = (l.height * l.width + r.height * r.width) as f64 - intersection;
                let res = intersection / union;
                Some(res as f32)
            }
            _ => None,
        }
    }
}

impl PartialEq<Self> for BoundingBox {
    fn eq(&self, other: &Self) -> bool {
        (self.left - other.left).abs() < EPS
            && (self.top - other.top).abs() < EPS
            && (self.width - other.width) < EPS
            && (self.height - other.height) < EPS
            && (self.confidence - other.confidence) < EPS
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

impl Universal2DBox {
    pub fn too_far(l: &Universal2DBox, r: &Universal2DBox) -> bool {
        assert!(l.aspect > 0.0);
        assert!(l.height > 0.0);
        assert!(r.aspect > 0.0);
        assert!(r.height > 0.0);

        let max_distance = l.get_radius() + r.get_radius();
        let x = l.xc - r.xc;
        let y = l.yc - r.yc;
        x * x + y * y > max_distance * max_distance
    }

    pub fn dist_in_2r(l: &Universal2DBox, r: &Universal2DBox) -> f32 {
        assert!(l.aspect > 0.0);
        assert!(l.height > 0.0);
        assert!(r.aspect > 0.0);
        assert!(r.height > 0.0);

        let radial_distance = l.get_radius() + r.get_radius();
        let x = l.xc - r.xc;
        let y = l.yc - r.yc;
        (x * x + y * y).sqrt() / (radial_distance * radial_distance + EPS).sqrt()
    }

    pub fn intersection(l: &Universal2DBox, r: &Universal2DBox) -> f64 {
        if (normalize_angle(l.angle.unwrap_or(0.0)) - normalize_angle(r.angle.unwrap_or(0.0))).abs()
            < EPS
        {
            BoundingBox::intersection(&l.into(), &r.into())
        } else if Universal2DBox::too_far(l, r) {
            0.0
        } else {
            let mut l = l.clone();
            let mut r = r.clone();

            if l.get_vertices().is_none() {
                let angle = l.angle.unwrap_or(0.0);
                l = l.rotate(angle).gen_vertices();
            }

            if r.get_vertices().is_none() {
                let angle = r.angle.unwrap_or(0.0);
                r = r.rotate(angle).gen_vertices();
            }

            let p1 = l.get_vertices().as_ref().unwrap();
            let p2 = r.get_vertices().as_ref().unwrap();

            sutherland_hodgman_clip(p1, p2).unsigned_area()
        }
    }
}

impl ObservationAttributes for Universal2DBox {
    type MetricObject = f32;

    fn calculate_metric_object(
        left: &Option<&Self>,
        right: &Option<&Self>,
    ) -> Option<Self::MetricObject> {
        match (left, right) {
            (Some(l), Some(r)) => {
                let intersection = Universal2DBox::intersection(l, r);
                if intersection == 0.0 {
                    None
                } else {
                    let union = (l.height * l.height * l.aspect + r.height * r.height * r.aspect)
                        as f64
                        - intersection;
                    let res = intersection / union;
                    Some(res as f32)
                }
            }
            _ => None,
        }
    }
}

impl PartialEq<Self> for Universal2DBox {
    fn eq(&self, other: &Self) -> bool {
        (self.xc - other.xc).abs() < EPS
            && (self.yc - other.yc).abs() < EPS
            && (self.angle.unwrap_or(0.0) - other.angle.unwrap_or(0.0)) < EPS
            && (self.aspect - other.aspect) < EPS
            && (self.height - other.height) < EPS
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::Universal2DBox;
    use crate::track::ObservationAttributes;
    use crate::utils::bbox::BoundingBox;
    use crate::EPS;

    #[test]
    fn test_radius() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        let r = bb1.get_radius();
        assert!((r - 5.0).abs() < EPS);
    }

    #[test]
    fn test_not_too_far() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        let bb2 = BoundingBox::new(6.0, 0.0, 6.0, 8.0).as_xyaah();
        assert!(!Universal2DBox::too_far(&bb1, &bb2));
    }

    #[test]
    fn test_same() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        assert!(!Universal2DBox::too_far(&bb1, &bb1));
    }

    #[test]
    fn test_too_far() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        let bb2 = BoundingBox::new(10.1, 0.0, 6.0, 8.0).as_xyaah();
        assert!(Universal2DBox::too_far(&bb1, &bb2));
    }

    #[test]
    fn dist_same() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        assert!(Universal2DBox::dist_in_2r(&bb1, &bb1) < EPS);
    }

    #[test]
    fn dist_less_1() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        let bb2 = BoundingBox::new(6.0, 0.0, 6.0, 8.0).as_xyaah();
        let d = Universal2DBox::dist_in_2r(&bb1, &bb2);
        assert!((d - 0.6).abs() < EPS);
    }

    #[test]
    fn dist_is_1() {
        let bb1 = BoundingBox::new(0.0, 0.0, 6.0, 8.0).as_xyaah();
        let bb2 = BoundingBox::new(10.0, 0.0, 6.0, 8.0).as_xyaah();
        let d = Universal2DBox::dist_in_2r(&bb1, &bb2);
        assert!((d - 1.0).abs() < EPS);
    }

    #[test]
    fn test_iou() {
        let bb1 = BoundingBox::new(-1.0, -1.0, 2.0, 2.0);

        let bb2 = BoundingBox::new(-0.9, -0.9, 2.0, 2.0);
        let bb3 = BoundingBox::new(1.0, 1.0, 3.0, 3.0);

        assert!(BoundingBox::calculate_metric_object(&Some(&bb1), &Some(&bb1)).unwrap() > 0.999);
        assert!(BoundingBox::calculate_metric_object(&Some(&bb2), &Some(&bb2)).unwrap() > 0.999);
        assert!(BoundingBox::calculate_metric_object(&Some(&bb1), &Some(&bb2)).unwrap() > 0.8);
        assert!(BoundingBox::calculate_metric_object(&Some(&bb1), &Some(&bb3)).unwrap() < 0.001);
        assert!(BoundingBox::calculate_metric_object(&Some(&bb2), &Some(&bb3)).unwrap() < 0.001);
    }
}
/// Voting engine for IoU metric
///
pub struct IOUTopNVoting {
    pub topn: usize,
    pub min_distance: f32,
    pub min_votes: usize,
}

impl Voting<BoundingBox> for IOUTopNVoting {
    type WinnerObject = TopNVotingElt;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<TopNVotingElt>>
    where
        T: IntoIterator<Item = ObservationMetricOk<BoundingBox>>,
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
