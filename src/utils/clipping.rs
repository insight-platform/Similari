/// Python interface for `sutherland_hodgman_clip`
///
#[cfg(feature = "python")]
pub mod clipping_py;

/// The function to calculate polygons solely owned by a bounding box
///
pub mod bbox_own_areas;

use geo::{Coord, CoordsIter, LineString, Polygon};

fn is_inside(q: &Coord<f64>, p1: &Coord<f64>, p2: &Coord<f64>) -> bool {
    let r = (p2.x - p1.x) * (q.y - p1.y) - (p2.y - p1.y) * (q.x - p1.x);
    r <= 0.0
}

fn compute_intersection(
    cp1: &Coord<f64>,
    cp2: &Coord<f64>,
    s: &Coord<f64>,
    e: &Coord<f64>,
) -> Coord<f64> {
    let dc = Coord {
        x: cp1.x - cp2.x,
        y: cp1.y - cp2.y,
    };
    let dp = Coord {
        x: s.x - e.x,
        y: s.y - e.y,
    };
    let n1 = cp1.x * cp2.y - cp1.y * cp2.x;
    let n2 = s.x * e.y - s.y * e.x;
    let n3 = 1.0 / (dc.x * dp.y - dc.y * dp.x);
    Coord {
        x: (n1 * dp.x - n2 * dc.x) * n3,
        y: (n1 * dp.y - n2 * dc.y) * n3,
    }
}

pub fn sutherland_hodgman_clip(
    subject_polygon: &Polygon<f64>,
    clipping_polygon: &Polygon<f64>,
) -> Polygon<f64> {
    let mut final_polygon = subject_polygon.coords_iter().collect::<Vec<_>>();
    final_polygon.pop();

    let mut clipping_polygon = clipping_polygon.coords_iter().collect::<Vec<_>>();
    clipping_polygon.pop();

    for i in 0..clipping_polygon.len() {
        let next_polygon = final_polygon;
        final_polygon = Vec::default();

        let i_i = if i == 0 {
            clipping_polygon.len() - 1
        } else {
            i - 1
        };

        let c_edge_start = clipping_polygon[i_i];

        let c_edge_end = clipping_polygon[i];
        for j in 0..next_polygon.len() {
            let j_i = if j == 0 {
                next_polygon.len() - 1
            } else {
                j - 1
            };

            let s_edge_start = next_polygon[j_i];
            let s_edge_end = next_polygon[j];
            if is_inside(&s_edge_end, &c_edge_start, &c_edge_end) {
                if !is_inside(&s_edge_start, &c_edge_start, &c_edge_end) {
                    let int = compute_intersection(
                        &s_edge_start,
                        &s_edge_end,
                        &c_edge_start,
                        &c_edge_end,
                    );
                    final_polygon.push(int);
                }
                final_polygon.push(s_edge_end);
            } else if is_inside(&s_edge_start, &c_edge_start, &c_edge_end) {
                let int =
                    compute_intersection(&s_edge_start, &s_edge_end, &c_edge_start, &c_edge_end);
                final_polygon.push(int);
            }
        }
    }
    Polygon::new(LineString::new(final_polygon), vec![])
}

#[cfg(test)]
mod tests {
    use crate::utils::clipping::sutherland_hodgman_clip;
    use geo::{polygon, Polygon};

    #[test]
    fn clip() {
        let subject_polygon: Polygon<f64> = polygon![
            (x: 8055.658, y: 7977.5537),
            (x: 8010.734, y: 7999.9697),
            (x: 8032.9717, y: 8044.537),
            (x: 8077.896, y: 8022.121),
        ];

        let clip_polygon: Polygon<f64> = polygon![
            (x: 8055.805, y: 7977.847),
            (x: 8010.871, y: 8000.2676),
            (x: 8033.105, y: 8044.8286),
            (x: 8078.039, y: 8022.408),
        ];

        let _result = sutherland_hodgman_clip(&subject_polygon, &clip_polygon);
    }
}
