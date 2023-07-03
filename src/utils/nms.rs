#[cfg(feature = "python")]
pub mod nms_py;

use crate::utils::bbox::Universal2DBox;
use itertools::Itertools;
use std::collections::HashSet;

#[derive(Clone, Debug)]
struct Candidate<'a> {
    bbox: &'a Universal2DBox,
    rank: f32,
    index: usize,
}

impl<'a> Candidate<'a> {
    pub fn new(bbox: &'a Universal2DBox, rank: &Option<f32>, index: usize) -> Self {
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
    detections: &[(Universal2DBox, Option<f32>)],
    nms_threshold: f32,
    score_threshold: Option<f32>,
) -> Vec<&Universal2DBox> {
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

    let mut excluded = HashSet::new();

    for (index, cb) in nms_boxes.iter().enumerate() {
        if excluded.contains(&cb.index) {
            continue;
        }

        for ob in &nms_boxes[index + 1..] {
            if excluded.contains(&ob.index) {
                continue;
            }

            let metric = Universal2DBox::intersection(cb.bbox, ob.bbox) as f32 / ob.bbox.area();
            if metric > nms_threshold {
                excluded.insert(ob.index);
            }
        }
    }

    nms_boxes
        .into_iter()
        .filter(|e| !excluded.contains(&e.index))
        .map(|e| e.bbox)
        .collect()
}

// /// NMS algorithm implementation
// ///
// /// # Parameters
// /// * `detections` - boxes with optional scores to filter out with NMS; if `detection.1` is `None`, that the score is set as `detection.0.height`;
// /// * `nms_threshold` - when to exclude the box from set by NMS;
// /// * `score_threshold` - when to exclude the from set by initial score. if `score_threshold` is None, then `f32::MAX` is used.
// ///
// pub fn parallel_nms(
//     detections: &[(Universal2DBox, Option<f32>)],
//     nms_threshold: f32,
//     score_threshold: Option<f32>,
// ) -> Vec<&Universal2DBox> {
//     let score_threshold = score_threshold.unwrap_or(f32::MIN);
//     let nms_boxes = detections
//         .iter()
//         .filter(|(e, score)| {
//             score.unwrap_or(f32::MAX) > score_threshold && e.height() > 0.0 && e.aspect() > 0.0
//         })
//         .enumerate()
//         .map(|(index, (b, score))| Candidate::new(b, score, index))
//         .sorted_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap())
//         .collect::<Vec<_>>();
//
//     let weight_matrix = nms_boxes
//         .par_iter()
//         .enumerate()
//         .flat_map(|(index, cb)| {
//             nms_boxes[index + 1..]
//                 .iter()
//                 .enumerate()
//                 .map(|(inner_index, ob)| {
//                     (
//                         (index, inner_index),
//                         Universal2DBox::intersection(cb.bbox, ob.bbox) as f32 / ob.bbox.area(),
//                     )
//                 })
//                 .collect::<Vec<_>>()
//         })
//         .collect::<HashMap<_, _>>();
//
//     let mut excluded = HashSet::new();
//
//     for (index, cb) in nms_boxes.iter().enumerate() {
//         if excluded.contains(&cb.index) {
//             continue;
//         }
//
//         for (internal_index, ob) in nms_boxes[index + 1..].iter().enumerate() {
//             if excluded.contains(&ob.index) {
//                 continue;
//             }
//
//             let metric = weight_matrix.get(&(index, internal_index)).unwrap();
//             if *metric > nms_threshold {
//                 excluded.insert(ob.index);
//             }
//         }
//     }
//
//     nms_boxes
//         .into_iter()
//         .filter(|e| !excluded.contains(&e.index))
//         .map(|e| e.bbox)
//         .collect()
// }
//
// #[cfg(test)]
// mod tests {
//     use crate::utils::bbox::Universal2DBox;
//     use crate::utils::nms::{nms, parallel_nms};
//
//     #[test]
//     fn nms_test() {
//         let bboxes = [
//             (Universal2DBox::new(0.0, 0.0, None, 1.0, 5.0), None),
//             (Universal2DBox::new(0.0, 0.0, None, 1.05, 5.1), None),
//             (Universal2DBox::new(0.0, 0.0, None, 1.0, 4.9), None),
//             (Universal2DBox::new(3.0, 4.0, None, 1.0, 4.5), None),
//         ];
//         let res_serial = nms(&bboxes, 0.8, None);
//         let res_parallel = parallel_nms(&bboxes, 0.8, None);
//         assert_eq!(res_serial, res_parallel);
//     }
// }
