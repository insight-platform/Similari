use crate::track::{
    MetricOutput, NoopLookup, ObservationAttributes, ObservationMetric, ObservationMetricOk,
    ObservationSpec, ObservationsDb, TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use crate::utils::bbox::BBox;
use crate::utils::kalman::{KalmanFilter, State};
use crate::voting::Voting;
use anyhow::Result;
use pathfinding::prelude::{kuhn_munkres, Matrix};
use std::collections::HashMap;

pub const DEFAULT_SORT_IOU_THRESHOLD: f32 = 0.3;
pub const F32_U64_MULT: f32 = 1_000_000.0;

#[derive(Debug, Clone, Default)]
pub struct SortAttributes {
    pub bboxes: Vec<BBox>,
    pub state: Option<State>,
}

#[derive(Clone, Debug)]
pub struct SortAttributesUpdate;

impl TrackAttributesUpdate<SortAttributes> for SortAttributesUpdate {
    fn apply(&self, _attrs: &mut SortAttributes) -> Result<()> {
        Ok(())
    }
}

impl TrackAttributes<SortAttributes, BBox> for SortAttributes {
    type Update = SortAttributesUpdate;
    type Lookup = NoopLookup<SortAttributes, BBox>;

    fn compatible(&self, _other: &SortAttributes) -> bool {
        true
    }

    fn merge(&mut self, _other: &SortAttributes) -> Result<()> {
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<BBox>) -> Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone)]
pub struct SortMetric {
    threshold: f32,
}

impl Default for SortMetric {
    fn default() -> Self {
        Self::new(DEFAULT_SORT_IOU_THRESHOLD)
    }
}

impl SortMetric {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl ObservationMetric<SortAttributes, BBox> for SortMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &SortAttributes,
        _attrs2: &SortAttributes,
        e1: &ObservationSpec<BBox>,
        e2: &ObservationSpec<BBox>,
    ) -> MetricOutput<f32> {
        let box_m_opt = BBox::calculate_metric_object(&e1.0, &e2.0);
        if let Some(box_m) = &box_m_opt {
            if *box_m < 0.01 {
                None
            } else {
                Some((box_m_opt, None))
            }
        } else {
            None
        }
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        attrs: &mut SortAttributes,
        features: &mut Vec<ObservationSpec<BBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> Result<()> {
        let mut observation = features.pop().unwrap();
        let observation_bbox = observation.0.unwrap();
        features.clear();

        let f = KalmanFilter::default();

        let state = if let Some(state) = attrs.state {
            f.update(state, observation_bbox.into())
        } else {
            f.initiate(observation_bbox.into())
        };

        let prediction = f.predict(state);
        attrs.state = Some(prediction);
        let predicted_bbox = prediction.bbox();

        attrs.bboxes.push(predicted_bbox);
        observation.0 = Some(predicted_bbox);
        features.push(observation);

        Ok(())
    }

    fn postprocess_distances(
        &self,
        unfiltered: Vec<ObservationMetricOk<BBox>>,
    ) -> Vec<ObservationMetricOk<BBox>> {
        unfiltered
            .into_iter()
            .filter(|x| x.attribute_metric.unwrap_or(0.0) > self.threshold)
            .collect()
    }
}

#[cfg(test)]
mod track_tests {
    use crate::prelude::{NoopNotifier, ObservationBuilder, TrackBuilder};
    use crate::trackers::sort::{SortAttributes, SortMetric, DEFAULT_SORT_IOU_THRESHOLD};
    use crate::utils::bbox::BBox;
    use crate::utils::kalman::KalmanFilter;
    use crate::{EstimateClose, EPS};

    #[test]
    fn construct() {
        let observation_bb_0 = BBox::new(1.0, 1.0, 10.0, 15.0);
        let observation_bb_1 = BBox::new(1.1, 1.3, 10.0, 15.0);

        let f = KalmanFilter::default();
        let init_state = f.initiate(observation_bb_0.into());

        let mut t1 = TrackBuilder::new(1)
            .track_attrs(SortAttributes::default())
            .metric(SortMetric::new(DEFAULT_SORT_IOU_THRESHOLD))
            .notifier(NoopNotifier)
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(observation_bb_0)
                    .build(),
            )
            .build()
            .unwrap();

        assert!(t1.get_attributes().state.is_some());
        assert_eq!(t1.get_attributes().bboxes.len(), 1);
        assert_eq!(t1.get_merge_history().len(), 1);
        assert!(t1.get_attributes().bboxes[0].estimate(&observation_bb_0, EPS));

        let predicted_state = f.predict(init_state);
        assert!(predicted_state.bbox().estimate(&observation_bb_0, EPS));

        let t2 = TrackBuilder::new(2)
            .track_attrs(SortAttributes::default())
            .metric(SortMetric::new(DEFAULT_SORT_IOU_THRESHOLD))
            .notifier(NoopNotifier)
            .observation(
                ObservationBuilder::new(0)
                    .observation_attributes(observation_bb_1)
                    .build(),
            )
            .build()
            .unwrap();

        t1.merge(&t2, &[0], false).unwrap();

        assert!(t1.get_attributes().state.is_some());
        assert_eq!(t1.get_attributes().bboxes.len(), 2);

        let predicted_state = f.predict(f.update(predicted_state, observation_bb_1.into()));
        assert!(t1.get_attributes().bboxes[1].estimate(&predicted_state.bbox(), EPS));
    }
}

pub struct SortVoting {
    threshold: i64,
    candidate_num: usize,
    track_num: usize,
}

impl SortVoting {
    pub fn new(threshold: f32, candidate_num: usize, track_num: usize) -> Self {
        Self {
            threshold: (threshold * F32_U64_MULT) as i64,
            candidate_num,
            track_num,
        }
    }
}

impl Voting<BBox> for SortVoting {
    type WinnerObject = u64;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<Self::WinnerObject>>
    where
        T: IntoIterator<Item = ObservationMetricOk<BBox>>,
    {
        let mut candidates_index: usize = 0;

        let mut tracks_index: Vec<u64> = Vec::default();
        tracks_index.resize(self.candidate_num, 0);
        let mut tracks_r_index: HashMap<u64, usize> = HashMap::default();

        let mut cost_matrix = Matrix::new(
            self.candidate_num,
            self.candidate_num + self.track_num,
            0i64,
        );

        for ObservationMetricOk {
            from,
            to,
            attribute_metric,
            feature_distance: _,
        } in distances
        {
            assert!(from > 0 && to > 0);

            if self.track_num == 0 {
                return HashMap::default();
            }

            let weight = (attribute_metric.unwrap() * F32_U64_MULT) as i64;

            let row = tracks_r_index
                .get(&from)
                .map(|x| *x as usize)
                .unwrap_or_else(|| {
                    let index = candidates_index;
                    candidates_index += 1;

                    tracks_index[index] = from;
                    tracks_r_index.insert(from, index);
                    index
                });

            let col = tracks_r_index
                .get(&to)
                .map(|x| *x as usize)
                .unwrap_or_else(|| {
                    let index = tracks_index.len();
                    tracks_index.push(to);
                    tracks_r_index.insert(to, index);
                    index
                });

            let v = cost_matrix.get_mut((row, col)).unwrap();
            *v = weight;
        }

        for i in 0..self.candidate_num {
            let v = cost_matrix.get_mut((i, i)).unwrap();
            *v = self.threshold;
        }

        let (_, solution) = kuhn_munkres(&cost_matrix);

        solution
            .into_iter()
            .enumerate()
            .flat_map(|(i, e)| {
                let (from, to) = (tracks_index[i], tracks_index[e]);
                if from > 0 && to > 0 {
                    Some((from, vec![to]))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod voting_tests {
    use crate::track::ObservationMetricOk;
    use crate::trackers::sort::SortVoting;
    use crate::voting::Voting;
    use std::collections::HashMap;

    #[test]
    fn test_voting() {
        let v = SortVoting::new(0.3, 3, 3);
        let winners = v.winners([
            ObservationMetricOk {
                from: 10,
                to: 20,
                attribute_metric: Some(0.6),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 10,
                to: 25,
                attribute_metric: Some(0.4),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 10,
                to: 30,
                attribute_metric: Some(0.4),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 11,
                to: 20,
                attribute_metric: Some(0.5),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 11,
                to: 25,
                attribute_metric: Some(0.69),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 11,
                to: 30,
                attribute_metric: Some(0.4),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 12,
                to: 20,
                attribute_metric: Some(0.2),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 12,
                to: 25,
                attribute_metric: Some(0.27),
                feature_distance: None,
            },
            ObservationMetricOk {
                from: 12,
                to: 30,
                attribute_metric: Some(0.28),
                feature_distance: None,
            },
        ]);

        assert_eq!(
            winners,
            HashMap::from([(10, vec![20]), (11, vec![25]), (12, vec![12])])
        );
    }
}
