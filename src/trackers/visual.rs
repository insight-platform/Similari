/// Track metric implementation
pub mod metric;

/// Implementation of Python-only structs and their implementations
///
pub mod visual_py;

/// Cascade voting engine for visual tracker. Combines TopN voting first for features and
/// Hungarian voting for the rest of unmatched (objects, tracks)
pub mod voting;

/// Track attributes for visual tracker
pub mod track_attributes;

/// Observation attributes for visual tracker
pub mod observation_attributes;
