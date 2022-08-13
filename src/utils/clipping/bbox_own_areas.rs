use crate::prelude::Universal2DBox;
use crate::EPS;
use geo::{Area, BooleanOps, MultiPolygon, Polygon};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

pub fn compute_own_polygons(boxes: &[Universal2DBox]) -> Vec<MultiPolygon> {
    let mut distances = HashSet::new();
    for (i, b1) in boxes.iter().enumerate() {
        for (j, b2) in boxes[i + 1..].iter().enumerate() {
            let j = j + i + 1;
            if !Universal2DBox::too_far(b1, b2) {
                distances.insert((i, j));
            }
        }
    }

    let distances = Arc::new(distances);
    boxes
        .par_iter()
        .enumerate()
        .map(|(i, own)| {
            let mut own_poly = MultiPolygon::from(Polygon::from(own));
            for (j, other) in boxes.iter().enumerate() {
                if distances.contains(&(i, j)) || distances.contains(&(j, i)) {
                    let clipping = MultiPolygon::from(Polygon::from(other));
                    own_poly = own_poly.difference(&clipping);
                }
            }
            own_poly
        })
        .collect()
}

pub fn compute_relative_own_areas(
    boxes: &[Universal2DBox],
    own_polygons: &[MultiPolygon],
) -> Vec<f32> {
    boxes
        .iter()
        .zip(own_polygons.iter())
        .map(|(b, poly)| (poly.unsigned_area() / (b.area() + EPS) as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::prelude::BoundingBox;
    use crate::utils::clipping::bbox_own_areas::{
        compute_own_polygons, compute_relative_own_areas,
    };
    use crate::EPS;
    use geo::Area;

    #[test]
    fn test() {
        let bb1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bb2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let bb3 = BoundingBox::new(10.0, 10.0, 10.0, 10.0);
        let boxes = &[bb1.into(), bb2.into(), bb3.into()];

        let own_polygons = compute_own_polygons(boxes);

        let bb1_own = own_polygons[0].unsigned_area();
        let bb2_own = own_polygons[1].unsigned_area();
        let bb3_own = own_polygons[2].unsigned_area();

        assert!((bb1_own - 75.0).abs() < EPS as f64);
        assert!((bb2_own - 50.0).abs() < EPS as f64);
        assert!((bb3_own - 75.0).abs() < EPS as f64);

        let own_areas = compute_relative_own_areas(boxes, own_polygons.as_slice());

        assert!((own_areas[0] - 0.75).abs() < EPS);
        assert!((own_areas[1] - 0.50).abs() < EPS);
        assert!((own_areas[2] - 0.75).abs() < EPS);
    }
}
