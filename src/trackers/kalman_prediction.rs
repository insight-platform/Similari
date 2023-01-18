use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::{Universal2DBoxKalmanFilter, DIM_2D_BOX_X2};
use crate::utils::kalman::KalmanState;

pub trait TrackAttributesKalmanPrediction {
    fn get_state(&self) -> Option<KalmanState<{ DIM_2D_BOX_X2 }>>;
    fn set_state(&mut self, state: KalmanState<{ DIM_2D_BOX_X2 }>);

    fn make_prediction(&mut self, observation_bbox: &Universal2DBox) -> Universal2DBox {
        let f = Universal2DBoxKalmanFilter::default();

        let state = if let Some(state) = self.get_state() {
            f.update(&state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(&state);
        self.set_state(prediction);
        let mut res = Universal2DBox::try_from(prediction).unwrap();
        res.confidence = observation_bbox.confidence;

        res
    }
}
