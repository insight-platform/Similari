use crate::track::{AttributeMatch, AttributeUpdate, Feature, Metric, Track};
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
{
    tracks: HashMap<u64, Track<A, M, U>>,
}

impl<A, U, M> Default for TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, U, M> TrackStore<A, U, M>
where
    A: Default + AttributeMatch<A> + Send + Sync,
    U: AttributeUpdate<A> + Send + Sync,
    M: Metric + Default + Send + Sync,
{
    pub fn new() -> Self {
        Self {
            tracks: HashMap::default(),
        }
    }

    pub fn find_baked(&self) -> Vec<u64> {
        self.tracks
            .iter()
            .flat_map(|(track_id, track)| {
                if track.get_attributes().baked(&track.observations) {
                    Some(*track_id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn fetch_backed(&mut self, backed: &Vec<u64>) -> Vec<Track<A, M, U>> {
        let mut res = Vec::default();
        for track_id in backed {
            if let Some(t) = self.tracks.remove(track_id) {
                res.push(t);
            }
        }
        res
    }

    pub fn foreign_track_distances(
        &self,
        track: &Track<A, M, U>,
        feature_id: u64,
    ) -> Option<Vec<(u64, f32)>> {
        Some(
            self.tracks
                .par_iter()
                .flat_map(|(_, other)| track.distances(other, feature_id))
                .flatten()
                .collect(),
        )
    }

    pub fn owned_track_distances(&self, track_id: u64, feature_id: u64) -> Option<Vec<(u64, f32)>> {
        let track = self.tracks.get(&track_id);
        if track.is_none() {
            return None;
        }
        let track = track.unwrap();
        Some(
            self.tracks
                .par_iter()
                .flat_map(|(other_track_id, other)| {
                    if *other_track_id == track_id {
                        None
                    } else {
                        track.distances(other, feature_id)
                    }
                })
                .flatten()
                .collect(),
        )
    }

    pub fn add(
        &mut self,
        track_id: u64,
        feature_id: u64,
        reid_q: f32,
        reid_v: Feature,
        attribute_update: U,
    ) {
        match self.tracks.get_mut(&track_id) {
            None => {
                let mut t = Track {
                    attributes: A::default(),
                    track_id,
                    observations: HashMap::from([(feature_id, vec![(reid_q, reid_v)])]),
                    metric: M::default(),
                    phantom_attribute_update: PhantomData,
                    merge_history: vec![track_id],
                };
                t.update_attributes(attribute_update);
                self.tracks.insert(track_id, t);
            }
            Some(track) => {
                track.add(feature_id, reid_q, reid_v, attribute_update);
            }
        }
    }
}
