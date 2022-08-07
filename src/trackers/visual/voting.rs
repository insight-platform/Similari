use crate::track::ObservationMetricOk;
use crate::trackers::sort::voting::SortVoting;
use crate::trackers::visual::VisualObservationAttributes;
use crate::utils::bbox::Universal2DBox;
use crate::voting::topn::TopNVoting;
use crate::voting::Voting;
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

#[derive(Default, Debug, Clone)]
pub enum VotingType {
    #[default]
    Visual,
    Positional,
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
        let topn_feature_voting: TopNVoting<VisualObservationAttributes> = TopNVoting::new(
            1,
            self.max_allowed_feature_distance,
            self.min_winner_feature_votes,
        );

        let distances = distances.into_iter().collect::<Vec<_>>();

        let feature_winners = topn_feature_voting.winners(distances.clone());

        let mut excluded_tracks = HashSet::new();
        let feature_winners = feature_winners
            .into_iter()
            .map(|(from, w)| {
                let winner_track = w[0].winner_track;
                excluded_tracks.insert(winner_track);
                (from, vec![(winner_track, VotingType::Visual)])
            })
            .collect::<HashMap<_, _>>();

        let mut remaining_candidates = HashSet::new();
        let mut remaining_tracks = HashSet::new();
        let remaining_distances = distances
            .into_iter()
            .filter(|e: &ObservationMetricOk<VisualObservationAttributes>| {
                !(feature_winners.contains_key(&e.from) || excluded_tracks.contains(&e.to))
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

        let mut winners = positional_voting
            .winners(remaining_distances)
            .into_iter()
            .map(|(from, winner)| (from, vec![(winner[0], VotingType::Positional)]))
            .collect::<HashMap<_, _>>();

        winners.extend(feature_winners);

        winners
    }
}

#[cfg(test)]
mod voting {
    use crate::track::ObservationMetricOk;
    use crate::trackers::visual::voting::VisualVoting;
    use crate::voting::Voting;

    #[test]
    fn test_visual() {
        let v = VisualVoting::new(0.3, 0.7, 2);
        let _winners = v.winners(vec![ObservationMetricOk::new(1, 2, Some(0.7), Some(0.7))]);
    }
}
