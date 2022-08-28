/// Various bounding boxes implementations for axis-aligned and oriented (rotated)
///
pub mod bbox;

/// Bounding box intersection calculation for oriented bounding boxes
///
pub mod clipping;

/// Auxiliary traits implementations for primitive types
///
pub mod primitive;

/// Non maximum suppression implementation for axis-aligned and oriented bounding boxes
///
pub mod nms;

/// Kalman filter related stuff
pub mod kalman;

/// 2D Points stuff
pub mod point;
