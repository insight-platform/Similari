use crate::utils::bbox::GenericBBox;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct Candidate<'a> {
    bbox: &'a GenericBBox,
    rank: f32,
    index: usize,
}

impl<'a> Candidate<'a> {
    pub fn new(bbox: &'a GenericBBox, rank: &Option<f32>, index: usize) -> Self {
        Self {
            bbox,
            rank: rank.unwrap_or(bbox.height),
            index,
        }
    }
}

/// NMS algorithm implementation
///
/// # Parameters
/// * `detections` - boxes with optional scores to filter out with NMS; if `detection.1` is `None`, that the score is set as `detection.0.height`;
/// * `nms_threshold` - when to exclude the box from set by NMS;
/// * `score_threshold` - when to exclude the from set by initial score. if `score_threshold` is None, then `f32::MAX` is used.
///
pub fn nms(
    detections: &[(GenericBBox, Option<f32>)],
    nms_threshold: f32,
    score_threshold: Option<f32>,
) -> Vec<&GenericBBox> {
    let score_threshold = score_threshold.unwrap_or(f32::MIN);
    let nms_boxes = detections
        .iter()
        .filter(|(e, score)| {
            score.unwrap_or(f32::MAX) > score_threshold && e.height > 0.0 && e.aspect > 0.0
        })
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

                let metric = GenericBBox::intersection(cb.bbox, ob.bbox) as f32 / ob.bbox.area();
                if metric > nms_threshold {
                    excluded.push(ob.index);
                }
            }
            excluded
        })
        .collect::<HashSet<_>>();

    nms_boxes
        .into_iter()
        .filter(|e| !results.contains(&e.index))
        .map(|e| e.bbox)
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
        let res = nms(&bboxes, 0.8, None);
        dbg!(res);
    }
}
