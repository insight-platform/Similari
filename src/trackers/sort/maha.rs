use crate::track::{MetricOutput, ObservationMetric, ObservationMetricOk, ObservationSpec};
use crate::trackers::kalman_prediction::TrackAttributesKalmanPrediction;
use crate::trackers::sort::SortAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::utils::kalman::KalmanFilter;
use anyhow::Result;

#[derive(Clone)]
pub struct MahaSortMetric;

impl ObservationMetric<SortAttributes, Universal2DBox> for MahaSortMetric {
    fn metric(
        &self,
        _feature_class: u64,
        _candidate_attributes: &SortAttributes,
        track_attributes: &SortAttributes,
        candidate_observation: &ObservationSpec<Universal2DBox>,
        track_observation: &ObservationSpec<Universal2DBox>,
    ) -> MetricOutput<f32> {
        let candidate_observation = candidate_observation.0.as_ref().unwrap();
        let track_observation = track_observation.0.as_ref().unwrap();

        if Universal2DBox::too_far(candidate_observation, track_observation) {
            None
        } else {
            let f = KalmanFilter::default();
            let state = track_attributes.state.unwrap();
            let dist = f.distance(state, candidate_observation);
            let dist = KalmanFilter::calculate_cost(dist, true);
            Some((Some(dist), None))
        }
    }

    fn optimize(
        &mut self,
        _feature_class: u64,
        _merge_history: &[u64],
        attrs: &mut SortAttributes,
        features: &mut Vec<ObservationSpec<Universal2DBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.0.as_ref().unwrap();
        features.clear();

        let predicted_bbox = attrs.make_prediction(&observation_bbox);
        attrs.update_history(observation_bbox, &predicted_bbox);

        observation.0 = Some(predicted_bbox);
        features.push(observation);

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<Universal2DBox>>,
    ) -> Vec<ObservationMetricOk<Universal2DBox>> {
        unfiltered
            .into_iter()
            .filter(|x| x.attribute_metric.unwrap_or(0.0) > 0.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackBuilder};
    use crate::track::ObservationMetricOk;
    use crate::trackers::sort::maha::MahaSortMetric;
    use crate::trackers::sort::{SortAttributes, SortAttributesOptions};
    use crate::utils::bbox::Universal2DBox;
    use std::sync::Arc;

    #[test]
    fn maha_track() {
        let mut track = TrackBuilder::new(0)
            .metric(MahaSortMetric)
            .attributes(SortAttributes::new(Arc::new(SortAttributesOptions::new(
                None, 0, 5,
            ))))
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(
                        Universal2DBox::new(0.0, 0.0, None, 0.5, 10.0).gen_vertices(),
                    )
                    .build(),
            )
            .notifier(NoopNotifier)
            .build()
            .unwrap();
        assert!(track.get_attributes().state.is_some());

        let new_seg = TrackBuilder::new(1)
            .metric(MahaSortMetric)
            .attributes(SortAttributes::new(Arc::new(SortAttributesOptions::new(
                None, 0, 5,
            ))))
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(
                        Universal2DBox::new(0.5, 0.5, None, 0.52, 10.1).gen_vertices(),
                    )
                    .build(),
            )
            .notifier(NoopNotifier)
            .build()
            .unwrap();
        let dists = new_seg.distances(&track, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 0,
                attribute_metric: Some(x),
                feature_distance: None,
            } if x > 99.0
        ));

        track.merge(&new_seg, &[0], true).unwrap();

        let new_seg = TrackBuilder::new(1)
            .metric(MahaSortMetric)
            .attributes(SortAttributes::new(Arc::new(SortAttributesOptions::new(
                None, 0, 5,
            ))))
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(
                        Universal2DBox::new(10.0, 10.0, None, 0.52, 15.1).gen_vertices(),
                    )
                    .build(),
            )
            .notifier(NoopNotifier)
            .build()
            .unwrap();

        let dists = new_seg.distances(&track, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 0,
                attribute_metric: Some(x),
                feature_distance: None,
            } if x == 0.0
        ));

        let new_seg = TrackBuilder::new(1)
            .metric(MahaSortMetric)
            .attributes(SortAttributes::new(Arc::new(SortAttributesOptions::new(
                None, 0, 5,
            ))))
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(
                        Universal2DBox::new(1.0, 0.9, None, 0.51, 10.0).gen_vertices(),
                    )
                    .build(),
            )
            .notifier(NoopNotifier)
            .build()
            .unwrap();

        let dists = new_seg.distances(&track, 0).unwrap();
        assert_eq!(dists.len(), 1);
        assert!(matches!(
            dists[0],
            ObservationMetricOk {
                from: 1,
                to: 0,
                attribute_metric: Some(x),
                feature_distance: None,
            } if x > 99.0
        ));
    }
}
