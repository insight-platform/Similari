use crate::track::ObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::{EstimateClose, EPS};

#[derive(Clone, Debug, Default)]
pub struct VisualObservationAttributes {
    bbox: Option<Universal2DBox>,
    visual_quality: f32,
}

impl VisualObservationAttributes {
    pub fn new(q: f32, b: Universal2DBox) -> Self {
        Self {
            visual_quality: q,
            bbox: Some(b),
        }
    }

    pub fn bbox_ref(&self) -> &Universal2DBox {
        self.bbox.as_ref().unwrap()
    }
}

impl ObservationAttributes for VisualObservationAttributes {
    type MetricObject = f32;

    fn calculate_metric_object(l: &Option<Self>, r: &Option<Self>) -> Option<Self::MetricObject> {
        if let (Some(l), Some(r)) = (l, r) {
            if let (Some(l), Some(r)) = (&l.bbox, &r.bbox) {
                let intersection = Universal2DBox::intersection(l, r);
                if intersection == 0.0 {
                    None
                } else {
                    let union = (l.height() * l.height() * l.aspect()
                        + r.height() * r.height() * r.aspect())
                        as f64
                        - intersection;
                    let res = intersection / union;
                    Some(res as f32)
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl PartialEq<Self> for VisualObservationAttributes {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(my_bbox), Some(other_bbox)) = (&self.bbox, &other.bbox) {
            my_bbox.almost_same(other_bbox, EPS)
                && (self.visual_quality - other.visual_quality).abs() < EPS
        } else {
            false
        }
    }
}
