use crate::prelude::{NoopNotifier, ObservationBuilder, SortTrack, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::utils::FromVec;
use crate::track::{Feature, Track, TrackStatus};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::VotingType::Positional;
use crate::trackers::sort::{PyWastedSortTrack, SortAttributesOptions};
use crate::trackers::visual::metric::{PositionalMetricType, VisualMetric, VisualMetricOptions};
use crate::trackers::visual::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual::simple_visual::options::VisualSortOptions;
use crate::trackers::visual::track_attributes::{VisualAttributes, VisualAttributesUpdate};
use crate::trackers::visual::voting::VisualVoting;
use crate::trackers::visual::VisualObservation;
use crate::voting::Voting;
use pyo3::prelude::*;
use rand::Rng;
use std::sync::Arc;

/// Options object to configure the tracker
pub mod options;

/// Python implementation for Visual tracker with Simple API
pub mod simple_visual_py;

// /// Easy to use Visual SORT tracker implementation
// ///
#[pyclass(text_signature = "(shards, opts)")]
pub struct VisualSort {
    store: TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes>,
    metric_opts: Arc<VisualMetricOptions>,
    track_opts: Arc<SortAttributesOptions>,
}

impl VisualSort {
    /// Creates new tracker
    ///
    /// # Parameters
    /// * `shards` - amount of cpu threads to process the data, keep 1 for up to 100 simultaneously tracked objects, try it before setting high - higher numbers may lead to unexpected latencies.
    /// * `opts` - tracker options
    ///
    pub fn new(shards: usize, opts: &VisualSortOptions) -> Self {
        let (track_opts, metric) = opts.clone().build();
        let track_opts = Arc::new(track_opts);
        let metric_opts = metric.opts.clone();
        let store = TrackStoreBuilder::new(shards)
            .default_attributes(VisualAttributes::new(track_opts.clone()))
            .metric(metric)
            .notifier(NoopNotifier)
            .build();

        Self {
            store,
            track_opts,
            metric_opts,
        }
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id` == 0
    ///
    pub fn skip_epochs(&mut self, n: usize) {
        self.skip_epochs_for_scene(0, n)
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id`
    /// * `scene_id` - scene to skip epochs
    ///
    pub fn skip_epochs_for_scene(&mut self, scene_id: u64, n: usize) {
        self.track_opts.skip_epochs_for_scene(scene_id, n)
    }

    /// Get the amount of stored tracks per shard
    ///
    pub fn shard_stats(&self) -> Vec<usize> {
        self.store.shard_stats()
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    pub fn current_epoch(&self) -> usize {
        self.current_epoch_with_scene(0)
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    pub fn current_epoch_with_scene(&self, scene_id: u64) -> usize {
        self.track_opts.current_epoch_with_scene(scene_id).unwrap()
    }

    /// Receive tracking information for observed bboxes of `scene_id` == 0
    ///
    /// # Parameters
    /// * `observations` - object observations with (feature, feature_quality and bounding box).
    ///
    pub fn predict(&mut self, observations: &[VisualObservation]) -> Vec<SortTrack> {
        self.predict_with_scene(0, observations)
    }

    /// Receive tracking information for observed bboxes of `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - custom identifier for the group of observed objects;
    /// * `observations` - object observations with (feature, feature_quality and bounding box).
    ///
    pub fn predict_with_scene(
        &mut self,
        scene_id: u64,
        observations: &[VisualObservation],
    ) -> Vec<SortTrack> {
        let mut rng = rand::thread_rng();
        let epoch = self.track_opts.next_epoch(scene_id).unwrap();

        let mut tracks = observations
            .iter()
            .map(|o| {
                self.store
                    .new_track(rng.gen())
                    .observation({
                        let mut obs = ObservationBuilder::new(0).observation_attributes(
                            VisualObservationAttributes::new(
                                o.feature_quality.unwrap_or(1.0),
                                o.bounding_box.clone(),
                            ),
                        );

                        if let Some(feature) = &o.feature {
                            obs = obs.observation(Feature::from_vec(feature));
                        }

                        obs.track_attributes_update(VisualAttributesUpdate::new_init_with_scene(
                            epoch,
                            scene_id,
                            o.custom_object_id,
                        ))
                        .build()
                    })
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let (dists, errs) = self.store.foreign_track_distances(tracks.clone(), 0, false);
        assert!(errs.all().is_empty());
        let voting = VisualVoting::new(
            match self.metric_opts.positional_kind {
                PositionalMetricType::Mahalanobis => 1.0,
                PositionalMetricType::IoU(t) => t,
            },
            self.metric_opts.visual_max_distance,
            self.metric_opts.visual_min_votes,
        );
        let winners = voting.winners(dists);
        let mut res = Vec::default();
        for t in &mut tracks {
            let source = t.get_track_id();
            let track_id: u64 = if let Some(dest) = winners.get(&source) {
                let (dest, vt) = dest[0];
                if dest == source {
                    self.store.add_track(t.clone()).unwrap();
                    source
                } else {
                    t.add_observation(
                        0,
                        None,
                        None,
                        Some(VisualAttributesUpdate::new_voting_type(vt)),
                    )
                    .unwrap();
                    self.store
                        .merge_external(dest, t, Some(&[0]), false)
                        .unwrap();
                    dest
                }
            } else {
                self.store.add_track(t.clone()).unwrap();
                source
            };

            let store = self.store.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap().clone();

            res.push(track.into())
        }

        res
    }

    /// Receive all the tracks with expired life
    ///
    pub fn wasted(
        &mut self,
    ) -> Vec<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> {
        let res = self.store.find_usable();
        let wasted = res
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.store.fetch_tracks(&wasted)
    }
}

impl From<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> for SortTrack {
    fn from(track: Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
        let attrs = track.get_attributes();
        SortTrack {
            id: track.get_track_id(),
            voting_type: attrs.voting_type.unwrap_or(Positional),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            length: attrs.track_length,
        }
    }
}

impl From<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>>
    for PyWastedSortTrack
{
    fn from(track: Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
        let attrs = track.get_attributes();
        PyWastedSortTrack {
            id: track.get_track_id(),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            length: attrs.track_length,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
    //     use crate::trackers::sort::simple_iou::IoUSort;
    //     use crate::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
    //     use crate::utils::bbox::BoundingBox;
    //     use crate::{EstimateClose, EPS};
    //
    //     #[test]
    //     fn sort() {
    //         let mut t = IoUSort::new(1, 10, 2, DEFAULT_SORT_IOU_THRESHOLD);
    //         assert_eq!(t.current_epoch(), 0);
    //         let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
    //         let v = t.predict(&[bb.into()]);
    //         let wasted = t.wasted();
    //         assert!(wasted.is_empty());
    //         assert_eq!(v.len(), 1);
    //         let v = v[0].clone();
    //         let track_id = v.id;
    //         assert_eq!(v.length, 1);
    //         assert!(v.observed_bbox.almost_same(&bb.into(), EPS));
    //         assert_eq!(v.epoch, 1);
    //         assert_eq!(t.current_epoch(), 1);
    //
    //         let bb = BoundingBox::new(0.1, 0.1, 10.1, 20.0);
    //         let v = t.predict(&[bb.into()]);
    //         let wasted = t.wasted();
    //         assert!(wasted.is_empty());
    //         assert_eq!(v.len(), 1);
    //         let v = v[0].clone();
    //         assert_eq!(v.id, track_id);
    //         assert_eq!(v.length, 2);
    //         assert!(v.observed_bbox.almost_same(&bb.into(), EPS));
    //         assert_eq!(v.epoch, 2);
    //         assert_eq!(t.current_epoch(), 2);
    //
    //         let bb = BoundingBox::new(10.1, 10.1, 10.1, 20.0);
    //         let v = t.predict(&[bb.into()]);
    //         assert_eq!(v.len(), 1);
    //         let v = v[0].clone();
    //         assert_ne!(v.id, track_id);
    //         let wasted = t.wasted();
    //         assert!(wasted.is_empty());
    //         assert_eq!(t.current_epoch(), 3);
    //
    //         let bb = t.predict(&[]);
    //         assert!(bb.is_empty());
    //         let wasted = t.wasted();
    //         assert!(wasted.is_empty());
    //         assert_eq!(t.current_epoch(), 4);
    //         assert_eq!(t.current_epoch(), 4);
    //
    //         let bb = t.predict(&[]);
    //         assert!(bb.is_empty());
    //         let wasted = t.wasted();
    //         assert_eq!(wasted.len(), 1);
    //         assert_eq!(wasted[0].get_track_id(), track_id);
    //         assert_eq!(t.current_epoch(), 5);
    //     }
    //
    //     #[test]
    //     fn sort_with_scenes() {
    //         let mut t = IoUSort::new(1, 10, 2, DEFAULT_SORT_IOU_THRESHOLD);
    //         let bb = BoundingBox::new(0.0, 0.0, 10.0, 20.0);
    //         assert_eq!(t.current_epoch_with_scene(1), 0);
    //         assert_eq!(t.current_epoch_with_scene(2), 0);
    //
    //         let _v = t.predict_with_scene(1, &[bb.into()]);
    //         let _v = t.predict_with_scene(1, &[bb.into()]);
    //
    //         assert_eq!(t.current_epoch_with_scene(1), 2);
    //         assert_eq!(t.current_epoch_with_scene(2), 0);
    //
    //         let _v = t.predict_with_scene(2, &[bb.into()]);
    //
    //         assert_eq!(t.current_epoch_with_scene(1), 2);
    //         assert_eq!(t.current_epoch_with_scene(2), 1);
    //     }
}
