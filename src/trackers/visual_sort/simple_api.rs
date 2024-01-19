use crate::prelude::{NoopNotifier, ObservationBuilder, SortTrack, TrackStoreBuilder};
use crate::store::TrackStore;
use crate::track::utils::FromVec;
use crate::track::{Feature, Track};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::VotingType::Positional;
use crate::trackers::sort::{
    AutoWaste, PositionalMetricType, SortAttributesOptions, DEFAULT_AUTO_WASTE_PERIODICITY,
    MAHALANOBIS_NEW_TRACK_THRESHOLD,
};
use crate::trackers::tracker_api::TrackerAPI;
use crate::trackers::visual_sort::metric::{VisualMetric, VisualMetricOptions};
use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
use crate::trackers::visual_sort::options::VisualSortOptions;
use crate::trackers::visual_sort::track_attributes::{
    VisualAttributes, VisualAttributesUpdate, VisualSortLookup,
};
use crate::trackers::visual_sort::voting::VisualVoting;
use crate::trackers::visual_sort::VisualSortObservation;
use crate::utils::clipping::bbox_own_areas::{
    exclusively_owned_areas, exclusively_owned_areas_normalized_shares,
};
use crate::voting::Voting;
use rand::Rng;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// /// Easy to use Visual SORT tracker implementation
// ///
pub struct VisualSort {
    store: RwLock<TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes>>,
    wasted_store: RwLock<TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes>>,
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
        let store = RwLock::new(
            TrackStoreBuilder::new(shards)
                .default_attributes(VisualAttributes::new(track_opts.clone()))
                .metric(metric.clone())
                .notifier(NoopNotifier)
                .build(),
        );

        let wasted_store = RwLock::new(
            TrackStoreBuilder::new(shards)
                .default_attributes(VisualAttributes::new(track_opts.clone()))
                .metric(metric)
                .notifier(NoopNotifier)
                .build(),
        );

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

    /// Receive tracking information for observed bboxes of `scene_id == 0`
    ///
    /// # Parameters
    /// * `scene_id` - custom identifier for the group of observed objects;
    /// * `observations` - object observations with (feature, feature_quality and bounding box).
    ///
    pub fn predict(&mut self, observations: &[VisualSortObservation]) -> Vec<SortTrack> {
        self.predict_with_scene(0, observations)
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
        observations: &[VisualSortObservation],
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
                    .read()
                    .unwrap()
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

                        if let Some(feature) = &o.feature {
                            obs = obs.observation(Feature::from_vec(feature.to_vec()));
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

        let (dists, errs) =
            self.store
                .write()
                .unwrap()
                .foreign_track_distances(tracks.clone(), 0, false);

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
                    self.store.write().unwrap().add_track(t).unwrap();
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
                        .write()
                        .unwrap()
                        .merge_external(dest, t, Some(&[0]), false)
                        .unwrap();
                    dest
                }
            } else {
                let mut t = t.clone();
                let track_id = self.gen_track_id();
                t.set_track_id(track_id);
                self.store.write().unwrap().add_track(t).unwrap();
                track_id
            };

            let lock = self.store.read().unwrap();
            let store = lock.get_store(track_id as usize);
            let track = store.get(&track_id).unwrap();

            res.push(SortTrack::from(track))
        }

        res
    }

    pub fn idle_tracks(&mut self) -> Vec<SortTrack> {
        self.idle_tracks_with_scene(0)
    }

    pub fn idle_tracks_with_scene(&mut self, scene_id: u64) -> Vec<SortTrack> {
        let store = self.store.read().unwrap();
        store
            .lookup(VisualSortLookup::IdleLookup(scene_id))
            .iter()
            .map(|(track_id, _status)| {
                let shard = store.get_store(*track_id as usize);
                let track = shard.get(track_id).unwrap();
                SortTrack::from(track)
            })
            .collect()
    }
}

impl
    TrackerAPI<
        VisualAttributes,
        VisualMetric,
        VisualObservationAttributes,
        SortAttributesOptions,
        NoopNotifier,
    > for VisualSort
{
    fn get_auto_waste_obj_mut(&mut self) -> &mut AutoWaste {
        &mut self.auto_waste
    }

    fn get_opts(&self) -> &SortAttributesOptions {
        &self.track_opts
    }

    fn get_main_store_mut(
        &mut self,
    ) -> RwLockWriteGuard<
        TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes, NoopNotifier>,
    > {
        self.store.write().unwrap()
    }

    fn get_wasted_store_mut(
        &mut self,
    ) -> RwLockWriteGuard<
        TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes, NoopNotifier>,
    > {
        self.wasted_store.write().unwrap()
    }

    fn get_main_store(
        &self,
    ) -> RwLockReadGuard<
        TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes, NoopNotifier>,
    > {
        self.store.read().unwrap()
    }

    fn get_wasted_store(
        &self,
    ) -> RwLockReadGuard<
        TrackStore<VisualAttributes, VisualMetric, VisualObservationAttributes, NoopNotifier>,
    > {
        self.wasted_store.read().unwrap()
    }
}

impl From<&Track<VisualAttributes, VisualMetric, VisualObservationAttributes>> for SortTrack {
    fn from(track: &Track<VisualAttributes, VisualMetric, VisualObservationAttributes>) -> Self {
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

#[cfg(test)]
mod tests {
    use crate::track::Observation;
    use crate::trackers::sort::{PositionalMetricType, VotingType};
    use crate::trackers::tracker_api::TrackerAPI;
    use crate::trackers::visual_sort::metric::VisualSortMetricType;
    use crate::trackers::visual_sort::observation_attributes::VisualObservationAttributes;
    use crate::trackers::visual_sort::options::VisualSortOptions;
    use crate::trackers::visual_sort::simple_api::VisualSort;
    use crate::trackers::visual_sort::{VisualSortObservation, WastedVisualSortTrack};
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
                &[VisualSortObservation::new(
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
                let lock = tracker.store.read().unwrap();
                let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            &[VisualSortObservation::new(
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
            let lock = tracker.store.read().unwrap();
            let store = lock.get_store(t.id as usize);
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
            .map(WastedVisualSortTrack::from)
            .collect::<Vec<_>>();
        dbg!(&tracks);
    }
}

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    use crate::{
        prelude::VisualSortObservation,
        trackers::{
            sort::python::PySortTrack,
            tracker_api::TrackerAPI,
            visual_sort::{
                options::python::PyVisualSortOptions,
                python::{PyVisualSortObservationSet, PyWastedVisualSortTrack},
                WastedVisualSortTrack,
            },
        },
    };

    use super::VisualSort;

    #[pyclass]
    #[pyo3(name = "VisualSort")]
    pub struct PyVisualSort(pub(crate) VisualSort);

    #[pymethods]
    impl PyVisualSort {
        #[new]
        pub fn new(shards: i64, opts: &PyVisualSortOptions) -> Self {
            assert!(shards > 0);
            Self(VisualSort::new(shards.try_into().unwrap(), &opts.0))
        }

        #[pyo3(signature = (n))]
        pub fn skip_epochs(&mut self, n: i64) {
            assert!(n > 0);
            self.0.skip_epochs(n.try_into().unwrap())
        }

        #[pyo3(signature = (scene_id, n))]
        pub fn skip_epochs_for_scene(&mut self, scene_id: i64, n: i64) {
            assert!(n > 0 && scene_id >= 0);
            self.0
                .skip_epochs_for_scene(scene_id.try_into().unwrap(), n.try_into().unwrap())
        }

        /// Get the amount of stored tracks per shard
        ///
        #[pyo3(signature = ())]
        pub fn shard_stats(&self) -> Vec<i64> {
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    self.0
                        .active_shard_stats()
                        .into_iter()
                        .map(|e| i64::try_from(e).unwrap())
                        .collect()
                })
            })
        }

        /// Get the current epoch for `scene_id` == 0
        ///
        #[pyo3(signature = ())]
        pub fn current_epoch(&self) -> i64 {
            self.0.current_epoch_with_scene(0).try_into().unwrap()
        }

        /// Get the current epoch for `scene_id`
        ///
        /// # Parameters
        /// * `scene_id` - scene id
        ///
        #[pyo3(signature = (scene_id))]
        pub fn current_epoch_with_scene(&self, scene_id: i64) -> isize {
            assert!(scene_id >= 0);
            self.0
                .current_epoch_with_scene(scene_id.try_into().unwrap())
                .try_into()
                .unwrap()
        }

        /// Receive tracking information for observed bboxes of `scene_id` == 0
        ///
        /// # Parameters
        /// * `bboxes` - bounding boxes received from a detector
        ///
        #[pyo3(signature = (observation_set))]
        pub fn predict(
            &mut self,
            observation_set: &PyVisualSortObservationSet,
        ) -> Vec<PySortTrack> {
            unsafe { std::mem::transmute(self.0.predict_with_scene(0, &observation_set.0.inner)) }
        }

        /// Receive tracking information for observed bboxes of `scene_id`
        ///
        /// # Parameters
        /// * `scene_id` - scene id provided by a user (class, camera id, etc...)
        /// * `observation_set` - observation set
        ///
        #[pyo3(signature = (scene_id, observation_set))]
        pub fn predict_with_scene(
            &mut self,
            scene_id: i64,
            observation_set: &PyVisualSortObservationSet,
        ) -> Vec<PySortTrack> {
            assert!(scene_id >= 0);
            let observations = observation_set
                .0
                .inner
                .iter()
                .map(|e| {
                    VisualSortObservation::new(
                        e.feature.as_deref(),
                        e.feature_quality,
                        e.bounding_box.clone(),
                        e.custom_object_id,
                    )
                })
                .collect::<Vec<_>>();

            Python::with_gil(|py| {
                py.allow_threads(|| unsafe {
                    std::mem::transmute(
                        self.0
                            .predict_with_scene(scene_id.try_into().unwrap(), &observations),
                    )
                })
            })
        }

        /// Remove all the tracks with expired life
        ///
        #[pyo3(signature = ())]
        pub fn wasted(&mut self) -> Vec<PyWastedVisualSortTrack> {
            Python::with_gil(|py| {
                py.allow_threads(|| {
                    self.0
                        .wasted()
                        .into_iter()
                        .map(WastedVisualSortTrack::from)
                        .map(PyWastedVisualSortTrack)
                        .collect()
                })
            })
        }

        /// Clear all tracks with expired life
        ///
        #[pyo3(signature = ())]
        pub fn clear_wasted(&mut self) {
            Python::with_gil(|py| py.allow_threads(|| self.0.clear_wasted()));
        }

        /// Get idle tracks with not expired life
        ///
        #[pyo3(signature = ())]
        pub fn idle_tracks(&mut self) -> Vec<PySortTrack> {
            unsafe { std::mem::transmute(self.0.idle_tracks_with_scene(0)) }
        }

        /// Get idle tracks with not expired life
        ///
        #[pyo3(signature = (scene_id))]
        pub fn idle_tracks_with_scene_py(&mut self, scene_id: i64) -> Vec<PySortTrack> {
            Python::with_gil(|py| {
                py.allow_threads(|| unsafe {
                    std::mem::transmute(self.0.idle_tracks_with_scene(scene_id.try_into().unwrap()))
                })
            })
        }
    }
}
