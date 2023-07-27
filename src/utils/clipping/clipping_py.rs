use crate::utils::bbox::python::PyUniversal2DBox;
use geo::{Area, CoordsIter, Polygon};
use pyo3::prelude::*;

#[derive(Debug)]
#[pyclass]
#[pyo3(name = "Polygon")]
pub struct PyPolygon(pub(crate) Polygon<f64>);

#[pymethods]
impl PyPolygon {
    #[pyo3(text_signature = "($self)")]
    pub fn get_points(&self) -> Vec<(f64, f64)> {
        self.0.coords_iter().map(|c| (c.x, c.y)).collect()
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

#[pyfunction]
#[pyo3(
    name = "sutherland_hodgman_clip",
    text_signature = "(subject, clipping)"
)]
pub fn sutherland_hodgman_clip_py(
    subject: PyUniversal2DBox,
    clipping: PyUniversal2DBox,
) -> PyPolygon {
    PyPolygon(subject.0.sutherland_hodgman_clip(clipping.0))
}

#[pyfunction]
#[pyo3(name = "intersection_area", text_signature = "(subject, clipping)")]
pub fn intersection_area_py(subject: PyUniversal2DBox, clipping: PyUniversal2DBox) -> f64 {
    let poly = sutherland_hodgman_clip_py(subject, clipping);
    poly.0.unsigned_area()
}
