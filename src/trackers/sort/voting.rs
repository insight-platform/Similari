use crate::track::ObservationMetricOk;
use crate::utils::bbox::Universal2DBox;
use crate::voting::Voting;
use core::option::Option::{None, Some};
use pathfinding::kuhn_munkres::kuhn_munkres;
use pathfinding::matrix::Matrix;
use std::collections::HashMap;

const F32_U64_MULT: f32 = 1_000_000.0;

pub struct SortVoting {
    threshold: i64,
    candidate_num: usize,
    track_num: usize,
}

impl SortVoting {
    pub fn new(threshold: f32, candidates_num: usize, tracks_num: usize) -> Self {
        Self {
            threshold: (threshold * F32_U64_MULT) as i64,
            candidate_num: candidates_num,
            track_num: tracks_num,
        }
    }
}

impl Voting<Universal2DBox> for SortVoting {
    type WinnerObject = u64;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<Self::WinnerObject>>
    where
        T: IntoIterator<Item = ObservationMetricOk<Universal2DBox>>,
    {
        let mut candidates_index: usize = 0;

        if self.track_num == 0 {
            return HashMap::default();
        }

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

            let weight = (attribute_metric.unwrap_or(0.0) * F32_U64_MULT) as i64;

            let row = tracks_r_index.get(&from).copied().unwrap_or_else(|| {
                let index = candidates_index;
                candidates_index += 1;

                tracks_index[index] = from;
                tracks_r_index.insert(from, index);
                index
            });

            let col = tracks_r_index.get(&to).copied().unwrap_or_else(|| {
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
    use crate::trackers::sort::voting::SortVoting;
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
