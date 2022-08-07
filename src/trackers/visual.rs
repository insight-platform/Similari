/// Track metric implementation
pub mod metric;

// Implementation of Python-only structs and their implementations
//
//pub mod visual_py;
// Cascade voting engine for visual tracker. Combines TopN voting first for features and
// Hungarian voting for the rest of unmatched (objects, tracks)
//
//pub mod voting;

/// Track attributes for visual tracker
pub mod track_attributes;

/// Observation attributes for visual tracker
pub mod observation_attributes;

// #[cfg(test)]
// mod tests {
//     use crate::trackers::visual::*;
//
//     #[test]
//     fn build_default_metric() {
//         let metric = VisualMetricBuilder::default().build();
//         assert!(matches!(
//             metric.positional_kind,
//             PositionalMetricType::IoU(t) if t == 0.3
//         ));
//         assert!(matches!(metric.visual_kind, VisualMetricType::Euclidean));
//         assert_eq!(metric.visual_minimal_track_length, 3);
//     }
//
//     #[test]
//     fn build_customized_metric() {
//         let metric = VisualMetricBuilder::default()
//             .visual_metric(VisualMetricType::Cosine)
//             .positional_metric(PositionalMetricType::Mahalanobis)
//             .visual_minimal_track_length(5)
//             .build();
//         drop(metric);
//     }
//
//     #[test]
//     fn postprocess_distances_maha() {
//         let metric = VisualMetricBuilder::default()
//             .positional_metric(PositionalMetricType::Mahalanobis)
//             .build();
//         drop(metric);
//     }
// }
