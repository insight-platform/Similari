use crate::track::ObservationMetricOk;
use crate::trackers::sort::voting::SortVoting;
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

impl Voting<Universal2DBox> for VisualVoting {
    type WinnerObject = u64;

    fn winners<T>(&self, distances: T) -> HashMap<u64, Vec<Self::WinnerObject>>
    where
        T: IntoIterator<Item = ObservationMetricOk<Universal2DBox>>,
    {
        let topn_feature_voting: TopNVoting<Universal2DBox> = TopNVoting::new(
            1,
            self.max_allowed_feature_distance,
            self.min_winner_feature_votes,
        );

        let distances = distances
            .into_iter()
            .collect::<Vec<ObservationMetricOk<Universal2DBox>>>();

        let feature_winners = topn_feature_voting.winners(distances.clone());

        let mut excluded_tracks = HashSet::new();
        let feature_winners = feature_winners
            .into_iter()
            .map(|(from, w)| {
                let winner_track = w[0].winner_track;
                excluded_tracks.insert(winner_track);
                (from, vec![winner_track])
            })
            .collect::<HashMap<_, _>>();

        let mut remaining_candidates = HashSet::new();
        let mut remaining_tracks = HashSet::new();
        let remaining_distances = distances
            .into_iter()
            .filter(|e: &ObservationMetricOk<Universal2DBox>| {
                !(feature_winners.contains_key(&e.from) || excluded_tracks.contains(&e.to))
            })
            .map(|e| {
                remaining_candidates.insert(e.from);
                remaining_tracks.insert(e.to);
                e
            })
            .collect::<Vec<_>>();

        let positional_voting = SortVoting::new(
            self.positional_threshold,
            remaining_candidates.len(),
            remaining_tracks.len(),
        );

        let mut positional_winners = positional_voting.winners(remaining_distances);
        positional_winners.extend(feature_winners);

        positional_winners
    }
}

#[cfg(test)]
mod voting {
    #[test]
    fn test() {}
}
