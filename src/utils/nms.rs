use crate::track::ObservationAttributes;
use crate::utils::bbox::GenericBBox;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct Candidate {
    bbox: Option<GenericBBox>,
    rank: f32,
    index: usize,
}

impl Candidate {
    pub fn new(bbox: &GenericBBox, rank: &Option<f32>, index: usize) -> Self {
        Self {
            bbox: Some(bbox.clone()),
            rank: rank.unwrap_or(bbox.height),
            index,
        }
    }
}

/// NMS algorithm implementation
///
/// # Parameters
/// * `boxes` - boxes with optional scores to filter out with NMS
/// * `max_overlap` - when to exclude box from set
///
pub fn nms(detections: &[(GenericBBox, Option<f32>)], max_overlap: f32) -> Vec<GenericBBox> {
    let nms_boxes = detections
        .iter()
        .enumerate()
        .map(|(index, (b, score))| Candidate::new(b, score, index))
        .sorted_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap())
        .collect::<Vec<_>>();

    let results = nms_boxes
        .par_iter()
        .enumerate()
        .flat_map(|(index, cb)| {
            let mut excluded = Vec::new();

            for ob in &nms_boxes[index + 1..] {
                if excluded.contains(&ob.index) {
                    continue;
                }

                let metric = GenericBBox::calculate_metric_object(&cb.bbox, &ob.bbox);

                if let Some(m) = metric {
                    if m > max_overlap {
                        excluded.push(ob.index);
                    }
                }
            }
            excluded
        })
        .collect::<HashSet<_>>();

    nms_boxes
        .into_iter()
        .filter(|e| !results.contains(&e.index))
        .flat_map(|e| e.bbox)
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::utils::bbox::GenericBBox;
    use crate::utils::nms::nms;

    #[test]
    fn nms_test() {
        let bboxes = [
            (GenericBBox::new(0.0, 0.0, None, 1.0, 5.0), None),
            (GenericBBox::new(0.0, 0.0, None, 1.05, 5.1), None),
            (GenericBBox::new(0.0, 0.0, None, 1.0, 4.9), None),
            (GenericBBox::new(3.0, 4.0, None, 1.0, 4.5), None),
        ];
        // let scores = [0.75_f32, 0.86_f32];
        let res = nms(&bboxes, 0.8);
        dbg!(res);
    }
}
