use crate::track::{ObservationAttributes, ObservationMetricOk};
use crate::voting::topn::TopNVotingElt;
use crate::voting::Voting;
use crate::{EstimateClose, EPS};
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Default, Debug, Copy)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BBox {
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
    fn estimate(&self, other: &Self, eps: f32) -> bool {
        (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.width - other.width) < eps
            && (self.height - other.height) < eps
    }
}

#[derive(Clone, Default, Debug, Copy)]
pub struct AspectBBox {
    pub x: f32,
    pub y: f32,
    pub aspect: f32,
    pub height: f32,
}

impl AspectBBox {
    pub fn new(x: f32, y: f32, aspect: f32, height: f32) -> Self {
        Self {
            x,
            y,
            aspect,
            height,
        }
    }
}

impl EstimateClose for AspectBBox {
    fn estimate(&self, other: &Self, eps: f32) -> bool {
        (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.aspect - other.aspect) < eps
            && (self.height - other.height) < eps
    }
}

impl From<BBox> for AspectBBox {
    fn from(f: BBox) -> Self {
        AspectBBox {
            x: f.x + f.width / 2.0,
            y: f.y + f.height / 2.0,
            aspect: f.width / f.height,
            height: f.height,
        }
    }
}

impl From<AspectBBox> for BBox {
    fn from(f: AspectBBox) -> Self {
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
        (self.x - other.x).abs() < EPS
            && (self.y - other.y).abs() < EPS
            && (self.width - other.width).abs() < EPS
            && (self.height - other.height).abs() < EPS
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
