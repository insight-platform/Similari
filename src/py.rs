use crate::prelude::{IoUSort, MahaSort, SortTrack};
use crate::utils::bbox::{BoundingBox, Universal2DBox};
use crate::utils::nms::py::{nms_py, parallel_nms_py};
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "similari")]
fn similari(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<Universal2DBox>()?;
    m.add_class::<SortTrack>()?;
    m.add_class::<IoUSort>()?;
    m.add_class::<MahaSort>()?;
    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_nms_py, m)?)?;
    Ok(())
}
