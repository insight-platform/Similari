use crate::store::TrackStore;
use crate::track::notify::ChangeNotifier;
use crate::track::TrackStatus;
use crate::track::{ObservationAttributes, ObservationMetric, Track, TrackAttributes};
use crate::trackers::epoch_db::EpochDb;
use crate::trackers::sort::AutoWaste;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

pub trait TrackerAPI<TA, M, OA, E, N>
where
    TA: TrackAttributes<TA, OA>,
    M: ObservationMetric<TA, OA>,
    OA: ObservationAttributes,
    N: ChangeNotifier,
    E: EpochDb,
{
    fn get_auto_waste_obj_mut(&mut self) -> &mut AutoWaste;
    fn get_opts(&self) -> &E;
    fn get_main_store_mut(&mut self) -> RwLockWriteGuard<TrackStore<TA, M, OA, N>>;
    fn get_wasted_store_mut(&mut self) -> RwLockWriteGuard<TrackStore<TA, M, OA, N>>;

    fn get_main_store(&self) -> RwLockReadGuard<TrackStore<TA, M, OA, N>>;
    fn get_wasted_store(&self) -> RwLockReadGuard<TrackStore<TA, M, OA, N>>;

    /// change auto waste job periodicity
    ///
    fn set_auto_waste(&mut self, periodicity: usize) {
        let obj = self.get_auto_waste_obj_mut();
        obj.periodicity = periodicity;
        obj.counter = 0;
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id` == 0
    ///
    fn skip_epochs(&mut self, n: usize) {
        self.skip_epochs_for_scene(0, n)
    }

    /// Skip number of epochs to force tracks to turn to terminal state
    ///
    /// # Parameters
    /// * `n` - number of epochs to skip for `scene_id`
    /// * `scene_id` - scene to skip epochs
    ///
    fn skip_epochs_for_scene(&mut self, scene_id: u64, n: usize) {
        self.get_opts().skip_epochs_for_scene(scene_id, n);
        self.auto_waste();
    }

    /// Get the current epoch for `scene_id` == 0
    ///
    fn current_epoch(&self) -> usize {
        self.current_epoch_with_scene(0)
    }

    /// Get the current epoch for `scene_id`
    ///
    /// # Parameters
    /// * `scene_id` - scene id
    ///
    fn current_epoch_with_scene(&self, scene_id: u64) -> usize {
        self.get_opts().current_epoch_with_scene(scene_id).unwrap()
    }

    /// Receive all the tracks with expired life from the main store
    ///
    fn get_main_store_wasted(&mut self) -> Vec<Track<TA, M, OA, N>> {
        let tracks = self.get_main_store_mut().find_usable();
        let wasted = tracks
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.get_main_store_mut().fetch_tracks(&wasted)
    }

    fn auto_waste(&mut self) {
        let tracks = self.get_main_store_wasted();
        for t in tracks {
            self.get_wasted_store_mut()
                .add_track(t)
                .expect("Cannot be a error, copying track to wasted store");
        }
    }

    fn wasted(&mut self) -> Vec<Track<TA, M, OA, N>> {
        self.auto_waste();
        let tracks = self.get_wasted_store_mut().find_usable();
        let wasted = tracks
            .into_iter()
            .filter(|(_, status)| matches!(status, Ok(TrackStatus::Wasted)))
            .map(|(track, _)| track)
            .collect::<Vec<_>>();

        self.get_wasted_store_mut().fetch_tracks(&wasted)
    }

    /// Get the amount of tracks kept in main store per shard
    ///
    fn active_shard_stats(&self) -> Vec<usize> {
        self.get_main_store().shard_stats()
    }

    /// Get the amount of tracks kept in wasted store per shard
    ///
    fn wasted_shard_stats(&self) -> Vec<usize> {
        self.get_main_store().shard_stats()
    }

    /// Clears wasted tracks
    fn clear_wasted(&self) {
        self.get_wasted_store().clear();
    }
}
