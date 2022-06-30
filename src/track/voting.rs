use anyhow::Result;
use itertools::Itertools;

/// Trait to implement distance voting engines.
///
/// Distance voting engine is used to select winning tracks among distances
/// resulted from the distance calculation.
///
pub trait Voting<R> {
    /// Method that selects winning tracks
    ///
    ///
    /// # Arguments
    /// * `distances` - distances resulted from the distance calculation.
    ///   * `.0` is the track_id
    ///   * `.1` is the distance
    ///
    fn winners(&self, distances: &Vec<(u64, Result<f32>)>) -> Vec<R>;
}

/// TopN winners voting engine that selects Top N vectors with most close distances.
///
/// It calculates winners as:
/// 1. removes all distances that are greater than threshold
/// 2. sorts remaining tracks according to their IDs
/// 3. counts tracks by their ID's
/// 4. sorts groups by frequency decreasingly
/// 5. returns TopN
///
pub struct TopNVoting {
    topn: usize,
    max_distance: f32,
    min_votes: usize,
}

impl TopNVoting {
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
        }
    }
}

/// Return type fot TopN voting engine
///
#[derive(Default, Debug, PartialEq)]
pub struct TopNVotingElt {
    /// winning track
    pub track_id: u64,
    /// number of votes it gathered
    pub votes: usize,
}

impl TopNVotingElt {
    pub fn new(track_id: u64, votes: usize) -> Self {
        Self { track_id, votes }
    }
}

impl Voting<TopNVotingElt> for TopNVoting {
    fn winners(&self, distances: &Vec<(u64, Result<f32>)>) -> Vec<TopNVotingElt> {
        let mut tracks: Vec<_> = distances
            .into_iter()
            .filter(|(_, e)| match e {
                Ok(e) => *e <= self.max_distance,
                _ => false,
            })
            .map(|(track, _)| track)
            .collect();
        tracks.sort_unstable();
        let mut counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|(e, c)| TopNVotingElt {
                track_id: *e,
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
    use crate::track::voting::{TopNVoting, TopNVotingElt, Voting};

    #[test]
    fn default_voting() {
        let v = TopNVoting {
            topn: 5,
            max_distance: 0.32,
            min_votes: 1,
        };

        let candidates = v.winners(&vec![(1, Ok(0.2))]);
        assert_eq!(candidates, vec![TopNVotingElt::new(1, 1)]);

        let candidates = v.winners(&vec![(1, Ok(0.2)), (1, Ok(0.3))]);
        assert_eq!(candidates, vec![TopNVotingElt::new(1, 2)]);

        let candidates = v.winners(&vec![(1, Ok(0.2)), (1, Ok(0.4))]);
        assert_eq!(candidates, vec![TopNVotingElt::new(1, 1)]);

        let mut candidates = v.winners(&vec![(1, Ok(0.2)), (2, Ok(0.2))]);
        candidates.sort_by(|l, r| l.track_id.partial_cmp(&r.track_id).unwrap());
        assert_eq!(
            candidates,
            vec![TopNVotingElt::new(1, 1), TopNVotingElt::new(2, 1)]
        );

        let mut candidates = v.winners(&vec![
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
        candidates.sort_by(|l, r| l.track_id.partial_cmp(&r.track_id).unwrap());
        assert_eq!(
            candidates,
            vec![
                TopNVotingElt::new(1, 2),
                TopNVotingElt::new(2, 2),
                TopNVotingElt::new(3, 2),
                TopNVotingElt::new(4, 2),
                TopNVotingElt::new(5, 2)
            ]
        );
    }
}
