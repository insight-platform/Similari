use crate::utils::bbox::Universal2DBox;
use crate::utils::nms::nms;
use pyo3::prelude::*;

/// # NMS Python interface
///
/// The python function name is `nms`. When the function is called, the GIL is released until the end of its execution.
///
/// The signature is:
/// ```python
/// def nms(detections: List[(Universal2DBox, Optional(float))], nms_threshold: float, score_threshold: Optional(float) -> List[Universal2DBox]
/// ```
/// # Parameters
/// * `detections` receives the list of tuples `(Universal2DBox, Optional(float))` where the first argument is bbox `Universal2DBox`, the second argument
///   is confidence or another ranking parameter. It is Optional value - can be `None` or `float`. If the confidence is None, the confidence is calculated
///   as height of the box.
/// * `nms_threshold` - the threshold that is used to remove excessive boxes out of the list.
/// * `score_threshold` - the threshold that filters boxes by confidence value. If it's not used, then None can be set.
///

/**
# Example

```python
from similari import nms, BoundingBox

 if __name__ == '__main__':

     print("With score")
     bbox1 = (BoundingBox(10.0, 11.0, 3.0, 3.8).as_xyaah(), 1.0)
     bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), 0.9)
     res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
     print(res[0].as_xywh())

     print("No score")
     bbox1 = (BoundingBox(10.0, 11.0, 3.0, 4.0).as_xyaah(), None)
     bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), None)
     res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
     print(res[0].as_xywh())
 ```
*/
#[pyfunction]
#[pyo3(
    name = "nms",
    text_signature = "(detections, nms_threshold, score_threshold)"
)]
pub fn nms_py(
    detections: Vec<(Universal2DBox, Option<f32>)>,
    nms_threshold: f32,
    score_threshold: Option<f32>,
) -> Vec<Universal2DBox> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    py.allow_threads(|| {
        nms(&detections, nms_threshold, score_threshold)
            .into_iter()
            .cloned()
            .collect()
    })
}

//
// #[pyfunction]
// #[pyo3(
//     name = "parallel_nms",
//     text_signature = "(detections, nms_threshold, score_threshold)"
// )]
// pub fn parallel_nms_py(
//     detections: Vec<(Universal2DBox, Option<f32>)>,
//     nms_threshold: f32,
//     score_threshold: Option<f32>,
// ) -> Vec<Universal2DBox> {
//     parallel_nms(&detections, nms_threshold, score_threshold)
//         .into_iter()
//         .cloned()
//         .collect()
// }
