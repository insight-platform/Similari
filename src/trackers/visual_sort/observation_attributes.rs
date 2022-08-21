use crate::track::{Observation, ObservationAttributes};
use crate::utils::bbox::Universal2DBox;
use crate::EPS;
use std::fmt::Formatter;

#[derive(Clone, Debug, Default)]
pub struct VisualObservationAttributes {
    bbox: Option<Universal2DBox>,
    visual_quality: f32,
    own_area_percentage: Option<f32>,
}

impl VisualObservationAttributes {
    pub fn new(q: f32, b: Universal2DBox) -> Self {
        Self {
            visual_quality: q,
            bbox: Some(b),
            own_area_percentage: None,
        }
    }

    pub fn with_own_area_percentage(q: f32, b: Universal2DBox, own_area_percentage: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&own_area_percentage),
            "Own area percentage must be contained in (0.0..=1.0)"
        );

        Self {
            visual_quality: q,
            bbox: Some(b),
            own_area_percentage: Some(own_area_percentage),
        }
    }

    pub fn unchecked_bbox_ref(&self) -> &Universal2DBox {
        self.bbox.as_ref().unwrap()
    }

    pub fn bbox_opt(&self) -> &Option<Universal2DBox> {
        &self.bbox
    }

    pub fn own_area_percentage_opt(&self) -> &Option<f32> {
        &self.own_area_percentage
    }

    pub fn drop_bbox(&mut self) {
        self.bbox = None;
    }

    pub fn visual_quality(&self) -> f32 {
        self.visual_quality
    }
}

impl std::fmt::Debug for Observation<VisualObservationAttributes> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?})", self.attr(), self.feature())
    }
}

impl ObservationAttributes for VisualObservationAttributes {
    type MetricObject = f32;

    fn calculate_metric_object(l: &Option<&Self>, r: &Option<&Self>) -> Option<Self::MetricObject> {
        if let (Some(l), Some(r)) = (l, r) {
            if let (Some(l), Some(r)) = (&l.bbox, &r.bbox) {
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
            my_bbox.eq(other_bbox) && (self.visual_quality - other.visual_quality).abs() < EPS
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::track::ObservationAttributes;
    use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
    use crate::utils::bbox::BoundingBox;
    use crate::EPS;

    #[test]
    fn operations() {
        let mut attrs1 =
            VisualObservationAttributes::new(0.7, BoundingBox::new(0.0, 0.0, 3.0, 5.0).as_xyaah());
        let attrs2 =
            VisualObservationAttributes::new(0.7, BoundingBox::new(0.0, 0.0, 3.0, 5.0).as_xyaah());

        let dist =
            VisualObservationAttributes::calculate_metric_object(&Some(&attrs1), &Some(&attrs2))
                .unwrap();
        assert!((dist - 1.0).abs() < EPS);

        assert_eq!(&attrs1, &attrs2);

        attrs1.bbox = None;
        let dist =
            VisualObservationAttributes::calculate_metric_object(&Some(&attrs1), &Some(&attrs2));
        assert_eq!(dist, None);

        let dist = VisualObservationAttributes::calculate_metric_object(&None, &Some(&attrs2));
        assert_eq!(dist, None);

        let dist = VisualObservationAttributes::calculate_metric_object(&Some(&attrs1), &None);
        assert_eq!(dist, None);
    }
}
