/// Various bounding boxes implementations for axis-aligned and oriented (rotated)
///
pub mod bbox;

/// Kalman filter for the prediction of axis-aligned and oriented bounding boxes
///
pub mod kalman_bbox;

/// Bounding box intersection calculation for oriented bounding boxes
///
pub mod clipping;

/// Auxiliary traits implementations for primitive types
///
pub mod primitive;

/// Non maximum suppression implementation for axis-aligned and oriented bounding boxes
///
pub mod nms;

/// Kalman filter for Vector of 2d points
///
pub mod kalman_2d_vec;
