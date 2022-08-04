use crate::prelude::{Maha_SORT, SortTrack, IOU_SORT};
use crate::utils::bbox::{BoundingBox, Universal2DBox};
use crate::utils::nms::py::{nms_py, parallel_nms_py};
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "similari")]
fn similari(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<Universal2DBox>()?;
    m.add_class::<SortTrack>()?;
    m.add_class::<IOU_SORT>()?;
    m.add_class::<Maha_SORT>()?;
    m.add_function(wrap_pyfunction!(nms_py, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_nms_py, m)?)?;
    Ok(())
}
