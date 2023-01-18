use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::kalman_2d_box::{Universal2DBoxKalmanFilter, DIM_2D_BOX_X2};
use crate::utils::kalman::KalmanState;

pub trait TrackAttributesKalmanPrediction {
    fn get_state(&self) -> Option<KalmanState<{ DIM_2D_BOX_X2 }>>;
    fn set_state(&mut self, state: KalmanState<{ DIM_2D_BOX_X2 }>);

    fn make_prediction(&mut self, observation_bbox: &Universal2DBox) -> Universal2DBox {
        let f = Universal2DBoxKalmanFilter::default();

        let current_state = if let Some(state) = self.get_state() {
            state
        } else {
            f.initiate(observation_bbox)
        };

        let prediction = f.predict(&current_state);

        let new_state = f.update(&prediction, observation_bbox);
        self.set_state(new_state);

        let mut res = Universal2DBox::try_from(new_state).unwrap();
        res.confidence = observation_bbox.confidence;

        res
    }
}
