use crate::prelude::{NoopNotifier, ObservationBuilder, SortTrack, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::utils::FromVec;
use crate::track::{Feature, Track, TrackStatus};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::VotingType::Positional;
use crate::trackers::sort::{
    AutoWaste, PositionalMetricType, SortAttributesOptions, DEFAULT_AUTO_WASTE_PERIODICITY,
    MAHALANOBIS_NEW_TRACK_THRESHOLD,
};
use crate::trackers::visual_sort::metric::{VisualMetric, VisualMetricOptions};
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual_sort::simple_api::options::VisualSortOptions;
use crate::trackers::visual_sort::track_attributes::{VisualAttributes, VisualAttributesUpdate};
use crate::trackers::visual_sort::voting::VisualVoting;
use crate::trackers::visual_sort::{PyWastedVisualSortTrack, VisualObservation};
use crate::utils::clipping::bbox_own_areas::{
    exclusively_owned_areas, exclusively_owned_areas_normalized_shares,
};
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
    wasted_store: TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes>,
    metric_opts: Arc<VisualMetricOptions>,
    track_opts: Arc<SortAttributesOptions>,
    auto_waste: AutoWaste,
    track_id: u64,
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
            .metric(metric.clone())
            .notifier(NoopNotifier)
            .build();

        let wasted_store = TrackStoreBuilder::new(shards)
            .default_attributes(VisualAttributes::new(track_opts.clone()))
            .metric(metric)
            .notifier(NoopNotifier)
            .build();

        Self {
            store,
            wasted_store,
            track_opts,
            track_id: 0,
            metric_opts,
            auto_waste: AutoWaste {
                periodicity: DEFAULT_AUTO_WASTE_PERIODICITY,
                counter: DEFAULT_AUTO_WASTE_PERIODICITY,
            },
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

    /// change auto waste job periodicity
    ///
    pub fn set_auto_waste(&mut self, periodicity: usize) {
        self.auto_waste.periodicity = periodicity;
        self.auto_waste.counter = 0;
    }

    fn gen_track_id(&mut self) -> u64 {
        self.track_id += 1;
        self.track_id
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
        if self.auto_waste.counter == 0 {
            self.auto_waste();
            self.auto_waste.counter = self.auto_waste.periodicity;
        } else {
            self.auto_waste.counter -= 1;
        }

        let mut percentages = Vec::default();
        let use_own_area_percentage = self.metric_opts.visual_minimal_own_area_percentage_collect
            + self.metric_opts.visual_minimal_own_area_percentage_use
            > 0.0;

        if use_own_area_percentage {
            percentages.reserve(observations.len());
            let boxes = observations
                .iter()
                .map(|e| &e.bounding_box)
                .collect::<Vec<_>>();

            percentages = exclusively_owned_areas_normalized_shares(
                boxes.as_ref(),
                exclusively_owned_areas(boxes.as_ref()).as_ref(),
            );
        }

        let mut rng = rand::thread_rng();
        let epoch = self.track_opts.next_epoch(scene_id).unwrap();

        let mut tracks = observations
            .iter()
            .enumerate()
            .map(|(i, o)| {
                self.store
                    .new_track(rng.gen())
                    .observation({
                        let mut obs = ObservationBuilder::new(0).observation_attributes(
                            if use_own_area_percentage {
                                VisualObservationAttributes::with_own_area_percentage(
                                    o.feature_quality.unwrap_or(1.0),
                                    o.bounding_box.clone(),
                                    percentages[i],
                                )
                            } else {
                                VisualObservationAttributes::new(
                                    o.feature_quality.unwrap_or(1.0),
                                    o.bounding_box.clone(),
                                )
                            },
                        );

                        if let Some(feature) = o.feature {
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
                PositionalMetricType::Mahalanobis => MAHALANOBIS_NEW_TRACK_THRESHOLD,
                PositionalMetricType::IoU(t) => t,
            },
            f32::MAX,
            self.metric_opts.visual_min_votes,
        );
        let winners = voting.winners(dists);
        let mut res = Vec::default();
        for t in &mut tracks {
            let source = t.get_track_id();
            let track_id: u64 = if let Some(dest) = winners.get(&source) {
                let (dest, vt) = dest[0];
                if dest == source {
                    let mut t = t.clone();
                    let track_id = self.gen_track_id();
                    t.set_track_id(track_id);
                    self.store.add_track(t).unwrap();
                    track_id
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
                let mut t = t.clone();
                let track_id = self.gen_track_id();
                t.set_track_id(track_id);
                self.store.add_track(t).unwrap();
                track_id
            };

            let store = self.store.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap().clone();

            res.push(track.into())
        }

        res
    }

    /// Receive all the tracks with expired life from the main store
    ///
    fn get_main_store_wasted(
        &mut self,
    ) -> Vec<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> {
        let tracks = self.store.find_usable();
        let wasted = tracks
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.store.fetch_tracks(&wasted)
    }

    pub fn auto_waste(&mut self) {
        let tracks = self.get_main_store_wasted();
        for t in tracks {
            self.wasted_store
                .add_track(t)
                .expect("Cannot be a error, copying track to wasted store");
        }
    }

    pub fn wasted(
        &mut self,
    ) -> Vec<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> {
        self.auto_waste();
        let tracks = self.wasted_store.find_usable();
        let wasted = tracks
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.wasted_store.fetch_tracks(&wasted)
    }
}

impl From<Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> for SortTrack {
    fn from(track: Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
        let attrs = track.get_attributes();
        SortTrack {
            id: track.get_track_id(),
            custom_object_id: attrs.custom_object_id,
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
    for PyWastedVisualSortTrack
{
    fn from(track: Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
        let attrs = track.get_attributes();
        PyWastedVisualSortTrack {
            id: track.get_track_id(),
            epoch: attrs.last_updated_epoch,
            scene_id: attrs.scene_id,
            length: attrs.track_length,
            observed_bbox: attrs.observed_boxes.back().unwrap().clone(),
            predicted_bbox: attrs.predicted_boxes.back().unwrap().clone(),
            predicted_boxes: attrs.predicted_boxes.clone().into_iter().collect(),
            observed_boxes: attrs.observed_boxes.clone().into_iter().collect(),
            observed_features: attrs
                .observed_features
                .clone()
                .iter()
                .map(|f_opt| f_opt.as_ref().map(Vec::from_vec))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::track::Observation;
    use crate::trackers::sort::{PositionalMetricType, VotingType};
    use crate::trackers::visual_sort::metric::VisualSortMetricType;
    use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual_sort::simple_api::options::VisualSortOptions;
    use crate::trackers::visual_sort::simple_api::VisualSort;
    use crate::trackers::visual_sort::{PyWastedVisualSortTrack, VisualObservation};
    use crate::utils::bbox::BoundingBox;

    #[test]
    fn visual_sort() {
        let opts = VisualSortOptions::default()
            .max_idle_epochs(3)
            .kept_history_length(3)
            .visual_metric(VisualSortMetricType::Euclidean(1.0))
            .positional_metric(PositionalMetricType::Mahalanobis)
            .visual_minimal_track_length(2)
            .visual_minimal_area(5.0)
            .visual_minimal_quality_use(0.45)
            .visual_minimal_quality_collect(0.7)
            .visual_max_observations(3)
            .visual_min_votes(2);

        let mut tracker = VisualSort::new(1, &opts);

        // new track to be initialized
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![1.0, 1.0]),
                Some(0.9),
                BoundingBox::new(1.0, 1.0, 3.0, 5.0).as_xyaah(),
                Some(13),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.custom_object_id, Some(13));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 1));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 1);
        assert_eq!(attrs.track_length, 1);
        assert_eq!(attrs.observed_boxes.len(), 1);
        assert_eq!(attrs.predicted_boxes.len(), 1);
        assert_eq!(attrs.observed_features.len(), 1);
        let first_track_id = t.id;

        {
            // another scene - new track
            let tracks = tracker.predict_with_scene(
                1,
                &[VisualObservation::new(
                    Some(&vec![1.0, 1.0]),
                    Some(0.9),
                    BoundingBox::new(1.0, 1.0, 3.0, 5.0).as_xyaah(),
                    Some(133),
                )],
            );
            let t = &tracks[0];
            assert_eq!(t.custom_object_id, Some(133));
            assert_eq!(t.scene_id, 1);
            assert!(matches!(t.voting_type, VotingType::Positional));
            assert!(matches!(t.epoch, 1));
            let attrs = {
                let store = tracker.store.get_store(t.id as usize);
                let track = store.get(&t.id).unwrap();
                track.get_attributes().clone()
            };
            assert_eq!(attrs.visual_features_collected_count, 1);
            assert_eq!(attrs.track_length, 1);
            assert_eq!(attrs.observed_boxes.len(), 1);
            assert_eq!(attrs.predicted_boxes.len(), 1);
            assert_eq!(attrs.observed_features.len(), 1);
        }

        // add the segment to the track (merge by bbox pos)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.95, 0.95]),
                Some(0.93),
                BoundingBox::new(1.1, 1.1, 3.05, 5.01).as_xyaah(),
                Some(15),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert_eq!(t.custom_object_id, Some(15));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 2));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 2);
        assert_eq!(attrs.observed_boxes.len(), 2);
        assert_eq!(attrs.predicted_boxes.len(), 2);
        assert_eq!(attrs.observed_features.len(), 2);

        // add the segment to the track (no visual_sort feature)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                None,
                Some(0.93),
                BoundingBox::new(1.11, 1.15, 3.15, 5.05).as_xyaah(),
                Some(25),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert_eq!(t.custom_object_id, Some(25));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 3));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 3);
        assert_eq!(attrs.observed_boxes.len(), 3);
        assert_eq!(attrs.predicted_boxes.len(), 3);
        assert_eq!(attrs.observed_features.len(), 3);
        assert!(attrs.observed_features.back().unwrap().is_none());

        // add the segment to the track (no visual_sort feature)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                None,
                Some(0.93),
                BoundingBox::new(1.15, 1.25, 3.10, 5.05).as_xyaah(),
                Some(2),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 4));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 4);
        assert_eq!(attrs.observed_boxes.len(), 3);
        assert_eq!(attrs.predicted_boxes.len(), 3);
        assert_eq!(attrs.observed_features.len(), 3);
        assert!(attrs.observed_features.back().unwrap().is_none());

        // add the segment to the track (with visual_sort feature but low quality - no use, no collect)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.97, 0.97]),
                Some(0.44),
                BoundingBox::new(1.15, 1.25, 3.10, 5.05).as_xyaah(),
                Some(2),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert!(matches!(t.voting_type, VotingType::Positional));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 5);
        assert!(attrs.observed_features.back().unwrap().is_some());

        // add the segment to the track (with visual_sort feature but low quality - use, but no collect)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.97, 0.97]),
                Some(0.6),
                BoundingBox::new(1.15, 1.25, 3.10, 5.05).as_xyaah(),
                Some(2),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert!(matches!(t.voting_type, VotingType::Visual));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 6);
        assert!(attrs.observed_features.back().unwrap().is_some());

        // add the segment to the track (with visual_sort feature of normal quality - use, collect)
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.97, 0.97]),
                Some(0.8),
                BoundingBox::new(1.15, 1.25, 3.10, 5.05).as_xyaah(),
                Some(2),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.id, first_track_id);
        assert!(matches!(t.voting_type, VotingType::Visual));
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            let observations = track.get_observations(0).unwrap();

            fn bbox_is(b: &Observation<VisualObservationAttributes>) -> bool {
                b.attr().as_ref().unwrap().bbox_opt().is_some()
            }

            assert!(bbox_is(&observations[0]) && observations[0].feature().is_some());
            assert!(!bbox_is(&observations[1]) && observations[1].feature().is_some());
            assert!(!bbox_is(&observations[2]) && observations[2].feature().is_some());

            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 3);
        assert_eq!(attrs.track_length, 7);
        assert!(attrs.observed_features.back().unwrap().is_some());

        // new track to be initialized
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.1, 0.1]),
                Some(0.9),
                BoundingBox::new(10.0, 10.0, 3.0, 5.0).as_xyaah(),
                Some(33),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.custom_object_id, Some(33));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 8));
        assert_ne!(t.id, first_track_id);
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 1);
        assert_eq!(attrs.track_length, 1);
        assert_eq!(attrs.observed_boxes.len(), 1);
        assert_eq!(attrs.predicted_boxes.len(), 1);
        assert_eq!(attrs.observed_features.len(), 1);
        let other_track_id = t.id;

        // add segment to be initialized
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.12, 0.15]),
                Some(0.88),
                BoundingBox::new(10.1, 10.1, 3.0, 5.0).as_xyaah(),
                Some(35),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.custom_object_id, Some(35));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Positional));
        assert!(matches!(t.epoch, 9));
        assert_eq!(t.id, other_track_id);
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 2);
        assert_eq!(attrs.track_length, 2);
        assert_eq!(attrs.observed_boxes.len(), 2);
        assert_eq!(attrs.predicted_boxes.len(), 2);
        assert_eq!(attrs.observed_features.len(), 2);

        // add segment to be initialized
        //
        let tracks = tracker.predict_with_scene(
            10,
            &[VisualObservation::new(
                Some(&vec![0.12, 0.14]),
                Some(0.87),
                BoundingBox::new(10.1, 10.1, 3.0, 5.0).as_xyaah(),
                Some(31),
            )],
        );
        let t = &tracks[0];
        assert_eq!(t.custom_object_id, Some(31));
        assert_eq!(t.scene_id, 10);
        assert!(matches!(t.voting_type, VotingType::Visual));
        assert!(matches!(t.epoch, 10));
        assert_eq!(t.id, other_track_id);
        let attrs = {
            let store = tracker.store.get_store(t.id as usize);
            let track = store.get(&t.id).unwrap();
            track.get_attributes().clone()
        };
        assert_eq!(attrs.visual_features_collected_count, 3);
        assert_eq!(attrs.track_length, 3);
        assert_eq!(attrs.observed_boxes.len(), 3);
        assert_eq!(attrs.predicted_boxes.len(), 3);
        assert_eq!(attrs.observed_features.len(), 3);

        tracker.skip_epochs_for_scene(10, 5);
        let tracks = tracker
            .wasted()
            .into_iter()
            .map(PyWastedVisualSortTrack::from)
            .collect::<Vec<_>>();
        dbg!(&tracks);
    }
}
