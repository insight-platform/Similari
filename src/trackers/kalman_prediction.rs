use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::{KalmanFilter, KalmanState};

pub trait TrackAttributesKalmanPrediction {
    fn get_state(&self) -> Option<KalmanState>;
    fn set_state(&mut self, state: KalmanState);

    fn make_prediction(&mut self, observation_bbox: &Universal2DBox) -> Universal2DBox {
        let f = KalmanFilter::default();

        let state = if let Some(state) = self.get_state() {
            f.update(state, observation_bbox.clone())
        } else {
            f.initiate(observation_bbox.clone())
        };

        let prediction = f.predict(state);
        self.set_state(prediction);
        let mut res = prediction.universal_bbox();
        res.confidence = observation_bbox.confidence;

        res
    }
}
