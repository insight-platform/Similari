use itertools::Itertools;

pub trait Voting {
    fn find_merge_candidates(&self, distances: Vec<(u64, f32)>) -> Vec<u64>;
}

pub struct DefaultVoting {
    count: usize,
    max_distance: f32,
}

impl Voting for DefaultVoting {
    fn find_merge_candidates(&self, distances: Vec<(u64, f32)>) -> Vec<u64> {
        let mut tracks: Vec<_> = distances
            .into_iter()
            .filter(|(_, e)| *e <= self.max_distance)
            .map(|(track, _)| track)
            .collect();
        tracks.sort();
        let mut counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .collect::<Vec<(_, _)>>();
        counts.sort_by(|(_, c1), (_, c2)| c2.partial_cmp(c1).unwrap());
        counts.truncate(self.count);
        counts.into_iter().map(|(t, _)| t).collect()
    }
}
