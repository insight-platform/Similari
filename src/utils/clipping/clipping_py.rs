use crate::utils::bbox::Universal2DBox;
use crate::utils::clipping::sutherland_hodgman_clip;
use geo::{Area, CoordsIter, Polygon};
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "Polygon")]
pub struct PyPolygon {
    polygon: Polygon<f64>,
}

#[pymethods]
impl PyPolygon {
    #[pyo3(text_signature = "($self)")]
    pub fn get_vertices(&self) -> Vec<(f64, f64)> {
        self.polygon.coords_iter().map(|c| (c.x, c.y)).collect()
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        format!("{:#?}", self)
    }
}

#[pyfunction]
#[pyo3(
    name = "sutherland_hodgman_clip",
    text_signature = "(subject, clipping)"
)]
pub fn sutherland_hodgman_clip_py(subject: Universal2DBox, clipping: Universal2DBox) -> PyPolygon {
    let mut subject = subject;
    let mut clipping = clipping;

    if subject.angle().is_none() {
        subject.rotate_py(0.0);
    }

    if clipping.angle().is_none() {
        clipping.rotate_py(0.0);
    }

    if subject.get_vertices().is_none() {
        subject.gen_vertices_py();
    }

    if clipping.get_vertices().is_none() {
        clipping.gen_vertices_py();
    }

    let clip = sutherland_hodgman_clip(
        subject.get_vertices().as_ref().unwrap(),
        clipping.get_vertices().as_ref().unwrap(),
    );
    PyPolygon { polygon: clip }
}

#[pyfunction]
#[pyo3(name = "intersection_area", text_signature = "(subject, clipping)")]
pub fn intersection_area_py(subject: Universal2DBox, clipping: Universal2DBox) -> f64 {
    let poly = sutherland_hodgman_clip_py(subject, clipping);
    poly.polygon.unsigned_area()
}
