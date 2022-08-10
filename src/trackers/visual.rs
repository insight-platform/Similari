use crate::utils::bbox::Universal2DBox;

/// Track metric implementation
pub mod metric;

/// Cascade voting engine for visual tracker. Combines TopN voting first for features and
/// Hungarian voting for the rest of unmatched (objects, tracks)
pub mod voting;

/// Track attributes for visual tracker
pub mod track_attributes;

/// Observation attributes for visual tracker
pub mod observation_attributes;

/// Implementation of Visual tracker with simple API
pub mod simple_visual;

#[derive(Debug, Clone)]
pub struct VisualObservation<'a> {
    feature: Option<&'a Vec<f32>>,
    feature_quality: Option<f32>,
    bounding_box: Universal2DBox,
    custom_object_id: Option<i64>,
}

impl<'a> VisualObservation<'a> {
    pub fn new(
        feature: Option<&'a Vec<f32>>,
        feature_quality: Option<f32>,
        bounding_box: Universal2DBox,
        custom_object_id: Option<i64>,
    ) -> Self {
        Self {
            feature,
            feature_quality,
            bounding_box,
            custom_object_id,
        }
    }

    // #[classattr]
    // const __hash__: Option<Py<PyAny>> = None;
    //
    // fn __repr__(&self) -> String {
    //     format!("{:?}", self)
    // }
    //
    // fn __str__(&self) -> String {
    //     format!("{:#?}", self)
    // }
}
