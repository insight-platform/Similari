use crate::prelude::{Maha_SORT, SortTrack, IOU_SORT};
use crate::utils::bbox::{BoundingBox, Universal2DBox};
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "similari")]
fn similari(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BoundingBox>()?;
    m.add_class::<Universal2DBox>()?;
    m.add_class::<SortTrack>()?;
    m.add_class::<IOU_SORT>()?;
    m.add_class::<Maha_SORT>()?;
    Ok(())
}
