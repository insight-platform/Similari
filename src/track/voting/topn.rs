use crate::track::{ObservationAttributes, ObservationMetricOk};
use crate::voting::Voting;
use itertools::Itertools;
use std::collections::HashMap;
use std::marker::PhantomData;

/// TopN winners voting engine that selects Top N vectors with most close distances.
///
/// It calculates winners as:
/// 1. removes all distances that are greater than threshold
/// 2. sorts remaining tracks according to their IDs
/// 3. counts tracks by their ID's
/// 4. sorts groups by frequency decreasingly
/// 5. returns TopN
///
pub struct TopNVoting<OA>
where
    OA: ObservationAttributes,
{
    topn: usize,
    max_distance: f32,
    min_votes: usize,
    _phony: PhantomData<OA>,
}

impl<OA> TopNVoting<OA>
where
    OA: ObservationAttributes,
{
    /// Constructs new engine
    ///
    /// # Arguments
    /// * `topn` - top winners
    /// * `max_distance` - max distance permitted to participate
    /// * `min_votes` - minimal amount of votes required the track to participate
    ///
    pub fn new(topn: usize, max_distance: f32, min_votes: usize) -> Self {
        Self {
            topn,
            max_distance,
            min_votes,
            _phony: PhantomData,
        }
    }
}

/// Return type fot TopN voting engine
///
#[derive(Default, Debug, PartialEq)]
pub struct TopNVotingElt {
    pub query_track: u64,
    /// winning track
    pub winner_track: u64,
    /// number of votes it gathered
    pub weight: f64,
}

impl TopNVotingElt {
    pub fn new(query_track: u64, winner_track: u64, weight: f64) -> Self {
        Self {
            query_track,
            winner_track,
            weight,
        }
    }
}

impl<OA> Voting<OA> for TopNVoting<OA>
where
    OA: ObservationAttributes,
{
    type WinnerObject = TopNVotingElt;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<TopNVotingElt>>
    where
        T: IntoIterator<Item = ObservationMetricOk<OA>>,
    {
        let mut max_dist = -1.0_f32;
        let counts: Vec<_> = distances
            .into_iter()
            .filter(
                |ObservationMetricOk {
                     from: _,
                     to: _track,
                     attribute_metric: _f_attr_dist,
                     feature_distance: feat_dist,
                 }| match feat_dist {
                    Some(e) => {
                        if max_dist < *e {
                            max_dist = *e;
                        }
                        *e <= self.max_distance
                    }
                    _ => false,
                },
            )
            .map(
                |ObservationMetricOk {
                     from: src_track,
                     to: dest_track,
                     attribute_metric: _,
                     feature_distance: dist,
                 }| { ((src_track, dest_track), dist.unwrap()) },
            )
            .into_group_map()
            .into_iter()
            .filter(|(_, count)| count.len() >= self.min_votes)
            .map(|((q, w), c)| {
                let weight = c.into_iter().map(|d| (max_dist - d) as f64).sum();

                TopNVotingElt {
                    query_track: q,
                    winner_track: w,
                    weight,
                }
            })
            .collect::<Vec<_>>();

        let mut results: HashMap<u64, Vec<TopNVotingElt>> = HashMap::new();

        for c in counts {
            let key = c.query_track;
            if let Some(val) = results.get_mut(&key) {
                val.push(c);
            } else {
                results.insert(key, vec![c]);
            }
        }

        for counts in results.values_mut() {
            counts.sort_by(|l, r| r.weight.partial_cmp(&l.weight).unwrap());
            counts.truncate(self.topn);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use crate::track::voting::topn::{TopNVoting, TopNVotingElt, Voting};
    use crate::track::ObservationMetricOk;
    use std::collections::HashMap;

    #[test]
    fn default_voting() {
        let v: TopNVoting<()> = TopNVoting::new(5, 0.32, 1);

        let candidates = v.winners([ObservationMetricOk::new(0, 1, None, Some(0.2))]);

        assert_eq!(
            candidates,
            HashMap::from([(0, vec![TopNVotingElt::new(0, 1, 0.0)])])
        );

        let candidates = v.winners([
            ObservationMetricOk::new(0, 1, None, Some(0.2)),
            ObservationMetricOk::new(0, 1, None, Some(0.3)),
        ]);

        assert_eq!(
            candidates,
            HashMap::from([(0, vec![TopNVotingElt::new(0, 1, 0.10000000894069672)])])
        );

        let candidates = v.winners([
            ObservationMetricOk::new(0, 1, None, Some(0.2)),
            ObservationMetricOk::new(0, 1, None, Some(0.4)),
        ]);

        assert_eq!(
            candidates,
            HashMap::from([(0, vec![TopNVotingElt::new(0, 1, 0.20000000298023224)])])
        );

        let mut candidates = v.winners([
            ObservationMetricOk::new(0, 1, None, Some(0.2)),
            ObservationMetricOk::new(0, 2, None, Some(0.2)),
        ]);

        candidates
            .get_mut(&0)
            .unwrap()
            .sort_by(|l, r| l.winner_track.partial_cmp(&r.winner_track).unwrap());

        assert_eq!(
            candidates,
            HashMap::from([(
                0,
                vec![TopNVotingElt::new(0, 1, 0.0), TopNVotingElt::new(0, 2, 0.0)]
            )])
        );

        let mut candidates = v.winners([
            ObservationMetricOk::new(0, 1, None, Some(0.2)),
            ObservationMetricOk::new(0, 1, None, Some(0.22)),
            ObservationMetricOk::new(0, 2, None, Some(0.21)),
            ObservationMetricOk::new(0, 2, None, Some(0.2)),
            ObservationMetricOk::new(0, 3, None, Some(0.22)),
            ObservationMetricOk::new(0, 3, None, Some(0.2)),
            ObservationMetricOk::new(0, 4, None, Some(0.23)),
            ObservationMetricOk::new(0, 4, None, Some(0.3)),
            ObservationMetricOk::new(0, 5, None, Some(0.24)),
            ObservationMetricOk::new(0, 5, None, Some(0.3)),
            ObservationMetricOk::new(0, 6, None, Some(0.25)),
            ObservationMetricOk::new(0, 6, None, Some(0.5)),
        ]);

        candidates
            .get_mut(&0)
            .unwrap()
            .sort_by(|l, r| l.winner_track.partial_cmp(&r.winner_track).unwrap());

        assert_eq!(
            candidates,
            HashMap::from([(
                0,
                vec![
                    TopNVotingElt::new(0, 1, 0.5800000131130219),
                    TopNVotingElt::new(0, 2, 0.5900000333786011),
                    TopNVotingElt::new(0, 3, 0.5800000131130219),
                    TopNVotingElt::new(0, 4, 0.4699999690055847),
                    TopNVotingElt::new(0, 5, 0.4599999785423279)
                ]
            )])
        );
    }

    #[test]
    fn two_query_vecs() {
        let v: TopNVoting<f32> = TopNVoting::new(5, 0.32, 1);

        let mut candidates = v.winners([
            ObservationMetricOk::new(0, 1, None, Some(0.2)),
            ObservationMetricOk::new(0, 1, None, Some(0.22)),
            ObservationMetricOk::new(0, 2, None, Some(0.21)),
            ObservationMetricOk::new(0, 2, None, Some(0.2)),
            ObservationMetricOk::new(0, 3, None, Some(0.22)),
            ObservationMetricOk::new(0, 3, None, Some(0.2)),
            ObservationMetricOk::new(7, 4, None, Some(0.23)),
            ObservationMetricOk::new(7, 4, None, Some(0.3)),
            ObservationMetricOk::new(7, 5, None, Some(0.24)),
            ObservationMetricOk::new(7, 5, None, Some(0.3)),
            ObservationMetricOk::new(7, 6, None, Some(0.25)),
            ObservationMetricOk::new(7, 6, None, Some(0.5)),
        ]);

        candidates
            .get_mut(&0)
            .unwrap()
            .sort_by(|l, r| l.winner_track.partial_cmp(&r.winner_track).unwrap());

        candidates
            .get_mut(&7)
            .unwrap()
            .sort_by(|l, r| l.winner_track.partial_cmp(&r.winner_track).unwrap());

        assert_eq!(
            candidates,
            HashMap::from([
                (
                    0,
                    vec![
                        TopNVotingElt::new(0, 1, 0.5800000131130219),
                        TopNVotingElt::new(0, 2, 0.5900000333786011),
                        TopNVotingElt::new(0, 3, 0.5800000131130219),
                    ]
                ),
                (
                    7,
                    vec![
                        TopNVotingElt::new(7, 4, 0.4699999690055847),
                        TopNVotingElt::new(7, 5, 0.4599999785423279),
                        TopNVotingElt::new(7, 6, 0.250)
                    ]
                )
            ])
        );
    }
}
