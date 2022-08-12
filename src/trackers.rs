/// SORT tracker implementations (middleware and simple sort implementation - IoU and Mahalanobis)
///
pub mod sort;

/// Trait that implements epoch db management
mod epoch_db;

/// Visual tracker implementations
pub mod visual;

/// Trait that implements kalman prediction for attributes
pub mod kalman_prediction;

/// The object that implements the constraints for space when objects from various epochs are compared.
/// It helps to decrease the brute-force space
///
pub mod spatio_temporal_constraints;
