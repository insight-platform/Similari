use std::borrow::Cow;

use crate::{
    track::{utils::FromVec, Track},
    utils::bbox::Universal2DBox,
};

use self::{
    metric::VisualMetric, observation_attributes::VisualObservationAttributes,
    track_attributes::VisualAttributes,
};

/// Track metric implementation
pub mod metric;

/// Cascade voting engine for visual_sort tracker. Combines TopN voting first for features and
/// Hungarian voting for the rest of unmatched (objects, tracks)
pub mod voting;

/// Track attributes for visual_sort tracker
pub mod track_attributes;

/// Observation attributes for visual_sort tracker
pub mod observation_attributes;

/// Implementation of Visual tracker with simple API
pub mod simple_api;

/// Batched API that accepts the batch with multiple scenes at once
pub mod batch_api;
/// Options object to configure the tracker
pub mod options;

#[derive(Debug, Clone)]
pub struct VisualSortObservation<'a> {
    feature: Option<Cow<'a, [f32]>>,
    feature_quality: Option<f32>,
    bounding_box: Universal2DBox,
    custom_object_id: Option<i64>,
}

impl<'a> VisualSortObservation<'a> {
    pub fn new(
        feature: Option<&'a [f32]>,
        feature_quality: Option<f32>,
        bounding_box: Universal2DBox,
        custom_object_id: Option<i64>,
    ) -> Self {
        Self {
            feature: feature.map(Cow::Borrowed),
            feature_quality,
            bounding_box,
            custom_object_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisualSortObservationSet<'a> {
    pub inner: Vec<VisualSortObservation<'a>>,
}

impl<'a> VisualSortObservationSet<'a> {
    pub fn new() -> Self {
        Self {
            inner: Vec::default(),
        }
    }

    pub fn add(&mut self, observation: VisualSortObservation<'a>) {
        self.inner.push(observation);
    }
}

impl<'a> Default for VisualSortObservationSet<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Online track structure that contains tracking information for the last tracker epoch
///
#[derive(Debug, Clone)]
pub struct WastedVisualSortTrack {
    /// id of the track
    ///
    pub id: u64,

    /// when the track was lastly updated
    ///
    pub epoch: usize,

    /// the bbox predicted by KF
    ///
    pub predicted_bbox: Universal2DBox,

    /// the bbox passed by detector
    ///
    pub observed_bbox: Universal2DBox,

    /// user-defined scene id that splits tracking space on isolated realms
    ///
    pub scene_id: u64,

    /// current track length
    ///
    pub length: usize,

    /// history of predicted boxes
    ///
    pub predicted_boxes: Vec<Universal2DBox>,

    /// history of observed boxes
    ///
    pub observed_boxes: Vec<Universal2DBox>,

    /// history of features
    ///
    pub observed_features: Vec<Option<Vec<f32>>>,
}

impl From<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>>
    for WastedVisualSortTrack
{
    fn from(track: Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
        let attrs = track.get_attributes();
        WastedVisualSortTrack {
            id: track.get_track_id(),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            length: attrs.track_length,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
            observed_features: attrs
                .observed_features
                .clone()
                .iter()
                .map(|f_opt| f_opt.as_ref().map(Vec::from_vec))
                .collect(),
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use super::{VisualSortObservation, VisualSortObservationSet, WastedVisualSortTrack};
    use crate::utils::bbox::python::PyUniversal2DBox;
    use pyo3::prelude::*;
    use std::borrow::Cow;

    #[pyclass]
    #[pyo3(name = "WastedVisualSortTrack")]
    pub struct PyWastedVisualSortTrack(pub(crate) WastedVisualSortTrack);

    #[pymethods]
    impl PyWastedVisualSortTrack {
        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn __str__(&self) -> String {
            format!("{:#?}", self.0)
        }

        #[getter]
        fn id(&self) -> u64 {
            self.0.id
        }

        #[getter]
        fn epoch(&self) -> usize {
            self.0.epoch
        }

        #[getter]
        fn predicted_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.predicted_bbox.clone())
        }

        #[getter]
        fn observed_bbox(&self) -> PyUniversal2DBox {
            PyUniversal2DBox(self.0.observed_bbox.clone())
        }

        #[getter]
        fn scene_id(&self) -> u64 {
            self.0.scene_id
        }

        #[getter]
        fn length(&self) -> usize {
            self.0.length
        }

        #[getter]
        fn predicted_boxes(&self) -> Vec<PyUniversal2DBox> {
            unsafe { std::mem::transmute(self.0.predicted_boxes.clone()) }
        }

        #[getter]
        fn observed_boxes(&self) -> Vec<PyUniversal2DBox> {
            unsafe { std::mem::transmute(self.0.observed_boxes.clone()) }
        }

        #[getter]
        fn observed_features(&self) -> Vec<Option<Vec<f32>>> {
            self.0.observed_features.clone()
        }
    }

    #[pyclass]
    #[derive(Debug, Clone)]
    #[pyo3(name = "VisualSortObservation")]
    pub struct PyVisualSortObservation(pub(crate) VisualSortObservation<'static>);

    #[pymethods]
    impl PyVisualSortObservation {
        #[new]
        #[pyo3(signature = (feature, feature_quality, bounding_box, custom_object_id))]
        pub fn new(
            feature: Option<Vec<f32>>,
            feature_quality: Option<f32>,
            bounding_box: PyUniversal2DBox,
            custom_object_id: Option<i64>,
        ) -> Self {
            Self(VisualSortObservation {
                feature: feature.map(Cow::Owned),
                feature_quality,
                bounding_box: bounding_box.0,
                custom_object_id,
            })
        }

        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{self:?}")
        }

        fn __str__(&self) -> String {
            format!("{self:#?}")
        }
    }

    #[pyclass]
    #[derive(Debug)]
    #[pyo3(name = "VisualSortObservationSet")]
    pub struct PyVisualSortObservationSet(pub(crate) VisualSortObservationSet<'static>);

    #[pymethods]
    impl PyVisualSortObservationSet {
        #[new]
        fn new() -> Self {
            Self(VisualSortObservationSet::new())
        }

        #[pyo3(text_signature = "($self, observation)")]
        fn add(&mut self, observation: PyVisualSortObservation) {
            self.0.add(observation.0);
        }

        #[classattr]
        const __hash__: Option<Py<PyAny>> = None;

        fn __repr__(&self) -> String {
            format!("{self:?}")
        }

        fn __str__(&self) -> String {
            format!("{self:#?}")
        }
    }
}
