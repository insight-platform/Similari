use anyhow::Result;
use itertools::Itertools;

pub trait Voting<R> {
    fn find_merge_candidates(&self, distances: Vec<(u64, Result<f32>)>) -> Vec<R>;
}

pub struct TopNVoting {
    topn: usize,
    max_distance: f32,
    min_votes: usize,
}

impl TopNVoting {
    pub fn new(topn: usize, max_distance: f32, min_votes: usize) -> Self {
        Self {
            topn,
            max_distance,
            min_votes,
        }
    }
}

#[derive(Default, Debug)]
pub struct TopNVotingElt {
    pub track_id: u64,
    pub votes: usize,
}

impl Voting<TopNVotingElt> for TopNVoting {
    fn find_merge_candidates(&self, distances: Vec<(u64, Result<f32>)>) -> Vec<TopNVotingElt> {
        let mut tracks: Vec<_> = distances
            .into_iter()
            .filter(|(_, e)| match e {
                Ok(e) => *e <= self.max_distance,
                _ => false,
            })
            .map(|(track, _)| track)
            .collect();
        tracks.sort();
        let mut counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|(e, c)| TopNVotingElt {
                track_id: e,
                votes: c,
            })
            .collect::<Vec<_>>();
        counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
        counts.truncate(self.topn);
        counts
    }
}

#[cfg(test)]
mod tests {
    use crate::track::voting::{TopNVoting, Voting};

    #[test]
    fn default_voting() {
        let v = TopNVoting {
            topn: 5,
            max_distance: 0.32,
            min_votes: 1,
        };

        let candidates = v.find_merge_candidates(vec![(1, Ok(0.2))]);
        assert_eq!(candidates, vec![(1, 1)]);

        let candidates = v.find_merge_candidates(vec![(1, Ok(0.2)), (1, Ok(0.3))]);
        assert_eq!(candidates, vec![(1, 2)]);

        let candidates = v.find_merge_candidates(vec![(1, Ok(0.2)), (1, Ok(0.4))]);
        assert_eq!(candidates, vec![(1, 1)]);

        let mut candidates = v.find_merge_candidates(vec![(1, Ok(0.2)), (2, Ok(0.2))]);
        candidates.sort();
        assert_eq!(candidates, vec![(1, 1), (2, 1)]);

        let mut candidates = v.find_merge_candidates(vec![
            (1, Ok(0.2)),
            (1, Ok(0.22)),
            (2, Ok(0.21)),
            (2, Ok(0.2)),
            (3, Ok(0.22)),
            (3, Ok(0.2)),
            (4, Ok(0.23)),
            (4, Ok(0.3)),
            (5, Ok(0.24)),
            (5, Ok(0.3)),
            (6, Ok(0.25)),
            (6, Ok(0.5)),
        ]);
        candidates.sort_by(|(l, _), (r, _)| l.partial_cmp(r).unwrap());
        assert_eq!(candidates, vec![(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]);
    }
}
