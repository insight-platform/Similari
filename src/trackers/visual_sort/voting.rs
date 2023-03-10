use crate::track::ObservationMetricOk;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::sort::VotingType;
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::voting::best::BestFitVoting;
use crate::voting::Voting;
use itertools::Itertools;
use log::debug;
use std::collections::{HashMap, HashSet};

pub struct VisualVoting {
    positional_threshold: f32,
    max_allowed_feature_distance: f32,
    min_winner_feature_votes: usize,
}

impl VisualVoting {
    pub fn new(
        positional_threshold: f32,
        max_allowed_feature_distance: f32,
        min_winner_feature_votes: usize,
    ) -> Self {
        Self {
            positional_threshold,
            max_allowed_feature_distance,
            min_winner_feature_votes,
        }
    }
}

impl From<ObservationMetricOk<VisualObservationAttributes>>
    for ObservationMetricOk<Universal2DBox>
{
    fn from(e: ObservationMetricOk<VisualObservationAttributes>) -> Self {
        ObservationMetricOk {
            from: e.from,
            to: e.to,
            attribute_metric: e.attribute_metric,
            feature_distance: e.feature_distance,
        }
    }
}

impl Voting<VisualObservationAttributes> for VisualVoting {
    type WinnerObject = (u64, VotingType);

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<Self::WinnerObject>>
    where
        T: IntoIterator<Item = ObservationMetricOk<VisualObservationAttributes>>,
    {
        let topn_feature_voting: BestFitVoting<VisualObservationAttributes> = BestFitVoting::new(
            self.max_allowed_feature_distance,
            self.min_winner_feature_votes,
        );

        let (distances, distances_clone) = distances.into_iter().tee();

        let feature_winners = topn_feature_voting.winners(distances);
        debug!("TopN winners: {:#?}", &feature_winners);

        let mut excluded_tracks = HashSet::new();
        let mut feature_winners = feature_winners
            .into_iter()
            .map(|(from, w)| {
                let winner_track = w[0].winner_track;
                excluded_tracks.insert(winner_track);
                (from, vec![(winner_track, VotingType::Visual)])
            })
            .collect::<HashMap<_, _>>();

        let mut remaining_candidates = HashSet::new();
        let mut remaining_tracks = HashSet::new();
        let remaining_distances = distances_clone
            .into_iter()
            .filter(|e: &ObservationMetricOk<VisualObservationAttributes>| {
                (!(feature_winners.contains_key(&e.from) || excluded_tracks.contains(&e.to)))
                    && e.attribute_metric.is_some()
            })
            .map(|e| {
                remaining_candidates.insert(e.from);
                remaining_tracks.insert(e.to);
                e.into()
            })
            .collect::<Vec<_>>();

        let positional_voting = SortVoting::new(
            self.positional_threshold,
            remaining_candidates.len(),
            remaining_tracks.len(),
        );

        let positional_winners = positional_voting
            .winners(remaining_distances)
            .into_iter()
            .map(|(from, winner)| (from, vec![(winner[0], VotingType::Positional)]));

        feature_winners.extend(positional_winners);
        feature_winners
    }
}

#[cfg(test)]
mod voting_tests {
    use crate::track::ObservationMetricOk;
    use crate::trackers::visual_sort::voting::{VisualVoting, VotingType};
    use crate::voting::Voting;

    #[test]
    fn test_visual_match() {
        let v = VisualVoting::new(0.3, 0.7, 1);
        let w = v.winners(vec![ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7))]);
        assert!(matches!(w.get(&1).unwrap()[0].1, VotingType::Visual));
    }

    #[test]
    fn test_positional_match() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let w = v.winners(vec![ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7))]);
        assert!(matches!(w.get(&1).unwrap()[0].1, VotingType::Positional));
    }

    #[test]
    fn test_visual_competitive_match() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let w = v.winners(vec![
            ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 2, None, Some(0.68)),
            ObservationMetricOk::new(1, 2, None, Some(0.65)),
            ObservationMetricOk::new(1, 3, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 3, None, Some(0.64)),
        ]);
        let res = w.get(&1).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Visual));
        assert!(matches!(res[0].0, 2));
    }

    #[test]
    fn test_visual_competitive_match_2() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let w = v.winners(vec![
            ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 2, None, Some(0.68)),
            ObservationMetricOk::new(1, 2, None, Some(0.65)),
            ObservationMetricOk::new(4, 3, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(4, 3, None, Some(0.64)),
        ]);
        let res = w.get(&1).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Visual));
        assert!(matches!(res[0].0, 2));

        let res = w.get(&4).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Visual));
        assert!(matches!(res[0].0, 3));
    }

    #[test]
    fn test_visual_positional_competitive_match_2() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let w = v.winners(vec![
            ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 2, None, Some(0.68)),
            ObservationMetricOk::new(1, 2, None, Some(0.65)),
            ObservationMetricOk::new(1, 3, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 3, None, Some(0.64)),
            ObservationMetricOk::new(11, 2, Some(0.8), Some(0.7)),
            ObservationMetricOk::new(11, 3, Some(0.6), Some(0.64)),
        ]);
        let res = w.get(&1).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Visual));
        assert!(matches!(res[0].0, 2));

        let res = w.get(&11).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Positional));
        assert!(matches!(res[0].0, 3));
    }

    #[test]
    fn test_visual_positional_competitive_match_no_pos_metric() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let w = v.winners(vec![
            ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 2, None, Some(0.68)),
            ObservationMetricOk::new(1, 2, None, Some(0.65)),
            ObservationMetricOk::new(1, 3, Some(0.7), Some(0.7)),
            ObservationMetricOk::new(1, 3, None, Some(0.64)),
            ObservationMetricOk::new(11, 2, Some(0.8), Some(0.7)), // will be excluded by visual_sort voting (1>2)
            ObservationMetricOk::new(11, 3, None, Some(0.64)), // no pos metric, as visual_sort votes
                                                               // less 2 will go to pos voting, but no pos metric.
        ]);
        let res = w.get(&1).unwrap();
        assert_eq!(res.len(), 1);
        assert!(matches!(res[0].1, VotingType::Visual));
        assert!(matches!(res[0].0, 2));

        assert!(w.get(&11).is_none());
    }
}
