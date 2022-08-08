use crate::track::ObservationAttributes;

impl ObservationAttributes for f32 {
    type MetricObject = f32;

    fn calculate_metric_object(
        left: &Option<&Self>,
        right: &Option<&Self>,
    ) -> Option<Self::MetricObject> {
        if let (Some(left), Some(right)) = (left, right) {
            Some((*left - *right).abs())
        } else {
            None
        }
    }
}

impl ObservationAttributes for () {
    type MetricObject = ();

    fn calculate_metric_object(
        _left: &Option<&Self>,
        _right: &Option<&Self>,
    ) -> Option<Self::MetricObject> {
        None
    }
}
