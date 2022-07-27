use crate::prelude::{ObservationBuilder, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::{Track, TrackStatus};
use crate::trackers::sort::{SortAttributes, SortAttributesUpdate, SortMetric, SortVoting};
use crate::utils::bbox::BBox;
use crate::voting::Voting;
use rand::Rng;
use std::sync::{Arc, RwLock};

pub struct SimpleSort {
    store: TrackStore<SortAttributes, SortMetric, BBox>,
    epoch: Arc<RwLock<usize>>,
    threshold: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct SortTrack {
    pub id: u64,
    pub epoch: usize,
    pub predicted_bbox: BBox,
    pub observed_bbox: BBox,
    pub length: usize,
}

impl From<Track<SortAttributes, SortMetric, BBox>> for SortTrack {
    fn from(track: Track<SortAttributes, SortMetric, BBox>) -> Self {
        let attrs = track.get_attributes();
        SortTrack {
            id: track.get_track_id(),
            epoch: attrs.epoch,
            observed_bbox: attrs.last_observation,
            predicted_bbox: attrs.last_prediction,
            length: attrs.length,
        }
    }
}

impl SimpleSort {
    pub fn new(shards: usize, bbox_history: usize, max_idle_epochs: usize, threshold: f32) -> Self {
        assert!(bbox_history > 0);
        let epoch = Arc::new(RwLock::new(0_usize));
        let store = TrackStoreBuilder::new(shards)
            .default_attributes(SortAttributes::new_with_epochs(
                bbox_history,
                max_idle_epochs,
                epoch.clone(),
            ))
            .metric(SortMetric::new(threshold))
            .build();

        Self {
            epoch,
            store,
            threshold,
        }
    }

    pub fn skip_epochs(&mut self, n: usize) {
        let mut epoch = self.epoch.write().unwrap();
        *epoch += n;
    }

    pub fn shard_stats(&self) -> Vec<usize> {
        self.store.shard_stats()
    }

    pub fn epoch(&mut self, bboxes: &[BBox]) -> Vec<SortTrack> {
        let mut rng = rand::thread_rng();
        let epoch = {
            let mut epoch = self.epoch.write().unwrap();
            *epoch += 1;
            *epoch
        };

        let tracks = bboxes
            .iter()
            .map(|bb| {
                self.store
                    .track_builder(rng.gen())
                    .observation(
                        ObservationBuilder::new(0)
                            .observation_attributes(*bb)
                            .track_attributes_update(SortAttributesUpdate::new(epoch))
                            .build(),
                    )
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let num_tracks = tracks.len();
        let (dists, errs) = self.store.foreign_track_distances(tracks.clone(), 0, false);
        assert!(errs.all().is_empty());
        let voting = SortVoting::new(
            self.threshold,
            num_tracks,
            self.store.shard_stats().iter().sum(),
        );
        let winners = voting.winners(dists);
        let mut res = Vec::default();
        for t in tracks {
            let source = t.get_track_id();
            let track_id: u64 = if let Some(dest) = winners.get(&source) {
                let dest = dest[0];
                if dest == source {
                    self.store.add_track(t).unwrap();
                    source
                } else {
                    self.store
                        .merge_external(dest, &t, Some(&[0]), false)
                        .unwrap();
                    dest
                }
            } else {
                self.store.add_track(t).unwrap();
                source
            };

            let store = self.store.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap().clone();

            res.push(track.into())
        }

        res
    }

    pub fn wasted(&mut self) -> Vec<Track<SortAttributes, SortMetric, BBox>> {
        let res = self.store.find_usable();
        let wasted = res
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.store.fetch_tracks(&wasted)
    }
}

#[cfg(test)]
mod tests {
    use crate::trackers::sort::simple::SimpleSort;
    use crate::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
    use crate::utils::bbox::BBox;
    use crate::{EstimateClose, EPS};

    #[test]
    fn sort() {
        let mut t = SimpleSort::new(1, 10, 2, DEFAULT_SORT_IOU_THRESHOLD);
        let bb = BBox::new(0.0, 0.0, 10.0, 20.0);
        let v = t.epoch(&vec![bb]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0];
        let track_id = v.id;
        assert_eq!(v.length, 1);
        assert!(v.observed_bbox.almost_same(&bb, EPS));
        assert_eq!(v.epoch, 1);

        let bb = BBox::new(0.1, 0.1, 10.1, 20.0);
        let v = t.epoch(&vec![bb]);
        let wasted = t.wasted();
        assert!(wasted.is_empty());
        assert_eq!(v.len(), 1);
        let v = v[0];
        assert_eq!(v.id, track_id);
        assert_eq!(v.length, 2);
        assert!(v.observed_bbox.almost_same(&bb, EPS));
        assert_eq!(v.epoch, 2);

        let bb = BBox::new(10.1, 10.1, 10.1, 20.0);
        let v = t.epoch(&[bb]);
        assert_eq!(v.len(), 1);
        let v = v[0];
        assert_ne!(v.id, track_id);
        let wasted = t.wasted();
        assert!(wasted.is_empty());

        let bb = t.epoch(&[]);
        assert!(bb.is_empty());
        let wasted = t.wasted();
        assert!(wasted.is_empty());

        let bb = t.epoch(&[]);
        assert!(bb.is_empty());
        let wasted = t.wasted();
        assert_eq!(wasted.len(), 1);
        assert_eq!(wasted[0].get_track_id(), track_id);
    }
}
