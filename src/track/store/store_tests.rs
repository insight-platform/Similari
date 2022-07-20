#[cfg(test)]
mod tests {
    use crate::distance::euclidean;
    use crate::prelude::{TrackBuilder, TrackStoreBuilder};
    use crate::test_stuff::{current_time_ms, vec2};
    use crate::track::store::TrackStore;
    use crate::track::utils::feature_attributes_sort_dec;
    use crate::track::{
        LookupRequest, MetricOutput, NoopLookup, NoopNotifier, ObservationAttributes,
        ObservationMetric, ObservationSpec, ObservationsDb, Track, TrackAttributes,
        TrackAttributesUpdate, TrackStatus,
    };
    use crate::EPS;
    use anyhow::Result;
    use std::thread;
    use std::time::Duration;

    #[derive(Default, Debug, Clone)]
    pub struct TimeAttrs {
        start_time: u128,
        end_time: u128,
        baked_period: u128,
    }

    #[derive(Default, Clone)]
    pub struct TimeAttrUpdates {
        time: u128,
    }

    impl TrackAttributesUpdate<TimeAttrs> for TimeAttrUpdates {
        fn apply(&self, attrs: &mut TimeAttrs) -> Result<()> {
            attrs.end_time = self.time;
            if attrs.start_time == 0 {
                attrs.start_time = self.time;
            }
            Ok(())
        }
    }

    impl TrackAttributes<TimeAttrs, f32> for TimeAttrs {
        type Update = TimeAttrUpdates;
        type Lookup = NoopLookup<TimeAttrs, f32>;

        fn compatible(&self, other: &TimeAttrs) -> bool {
            self.end_time <= other.start_time
        }

        fn merge(&mut self, other: &TimeAttrs) -> Result<()> {
            self.start_time = self.start_time.min(other.start_time);
            self.end_time = self.end_time.max(other.end_time);
            Ok(())
        }

        fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
            if current_time_ms() >= self.baked_period + self.end_time {
                Ok(TrackStatus::Ready)
            } else {
                Ok(TrackStatus::Pending)
            }
        }
    }

    #[derive(Default, Clone)]
    struct TimeMetric {
        max_length: usize,
    }

    impl ObservationMetric<TimeAttrs, f32> for TimeMetric {
        fn metric(
            _feature_class: u64,
            _attrs1: &TimeAttrs,
            _attrs2: &TimeAttrs,
            e1: &ObservationSpec<f32>,
            e2: &ObservationSpec<f32>,
        ) -> MetricOutput<f32> {
            Some((
                f32::calculate_metric_object(&e1.0, &e2.0),
                match (e1.1.as_ref(), e2.1.as_ref()) {
                    (Some(x), Some(y)) => Some(euclidean(x, y)),
                    _ => None,
                },
            ))
        }

        fn optimize(
            &mut self,
            _feature_class: &u64,
            _merge_history: &[u64],
            _attrs: &mut TimeAttrs,
            features: &mut Vec<ObservationSpec<f32>>,
            _prev_length: usize,
            _is_merge: bool,
        ) -> Result<()> {
            features.sort_by(feature_attributes_sort_dec);
            features.truncate(self.max_length);
            Ok(())
        }
    }

    #[test]
    fn new_default_store() -> Result<()> {
        let default_store: TrackStore<TimeAttrs, TimeMetric, f32> = TrackStore::default();
        drop(default_store);
        Ok(())
    }

    #[test]
    fn new_store_10_shards() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            10,
        );
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            Some(TimeAttrUpdates {
                time: current_time_ms(),
            }),
        )?;

        Ok(())
    }

    fn time_attrs_current_ts() -> Option<TimeAttrUpdates> {
        Some(TimeAttrUpdates {
            time: current_time_ms(),
        })
    }

    #[test]
    fn sharding_n_fetch() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            2,
        );

        let stats = store.shard_stats();
        assert_eq!(stats, vec![0, 0]);

        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let stats = store.shard_stats();
        assert_eq!(stats, vec![1, 0]);

        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let stats = store.shard_stats();
        assert_eq!(stats, vec![1, 1]);

        let tracks = store.fetch_tracks(&vec![0, 1]);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].track_id, 0);
        assert_eq!(tracks[1].track_id, 1);

        Ok(())
    }

    #[test]
    fn general_ops() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            1,
        );
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;
        let baked = store.find_usable();
        assert!(baked.is_empty());
        thread::sleep(Duration::from_millis(30));
        let baked = store.find_usable();
        assert_eq!(baked.len(), 1);
        assert_eq!(baked[0].0, 0);

        let vectors = store.fetch_tracks(&baked.into_iter().map(|(t, _)| t).collect::<Vec<_>>());
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].track_id, 0);
        assert_eq!(vectors[0].observations.len(), 1);

        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;
        let (dists, errs) = store.owned_track_distances(&[0], 0, false);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            Some(0.7),
            Some(vec2(1.0, 0.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[0], 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        let (dists, errs) = store.owned_track_distances(&[1], 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 0);

        let mut v = store.fetch_tracks(&vec![0]);

        let v = v.pop().unwrap();
        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 1);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!(errs.is_empty());

        // make it incompatible across the attributes
        thread::sleep(Duration::from_millis(10));
        let mut v = v.clone();
        v.attributes.end_time = current_time_ms();

        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 0);
        assert_eq!(errs.len(), 0);

        thread::sleep(Duration::from_millis(10));
        store.add(
            1,
            0,
            Some(0.7),
            Some(vec2(1.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let mut v = v.clone();
        v.attributes.end_time = store.get_store(1).get(&1).unwrap().attributes.start_time - 1;
        let (dists, errs) = store.foreign_track_distances(vec![v.clone()], 0, false);
        assert_eq!(dists.len(), 2);
        assert_eq!(dists[0].to, 1);
        assert!(dists[0].feature_distance.is_some());
        assert!((dists[0].feature_distance.as_ref().unwrap() - 2.0_f32.sqrt()).abs() < EPS);
        assert!((dists[1].feature_distance.as_ref().unwrap() - 1.0).abs() < EPS);
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn baked_similarity() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            2,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
        );

        //thread::sleep(Duration::from_millis(10));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            Some(TimeAttrUpdates {
                time: current_time_ms(),
            }),
        )?;

        let (dists, errs) = store.foreign_track_distances(vec![ext_track.clone()], 0, true);
        assert!(dists.is_empty());
        assert!(errs.is_empty());
        thread::sleep(Duration::from_millis(10));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[1], 0, true);
        assert!(dists.is_empty());
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn all_similarity() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            2,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.foreign_track_distances(vec![ext_track.clone()], 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        thread::sleep(Duration::from_millis(1));
        store.add(
            3,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let (dists, errs) = store.owned_track_distances(&[1], 0, false);
        assert_eq!(dists.len(), 1);
        assert!(errs.is_empty());

        Ok(())
    }

    #[test]
    fn add_track_ok() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        store.add_track(ext_track)?;
        Ok(())
    }

    #[test]
    fn add_track_dup_id() -> Result<()> {
        let mut ext_track = Track::new(
            0, // duplicate track id
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        assert!(store.add_track(ext_track).is_err());

        Ok(())
    }

    #[test]
    fn merge_ext_tracks() -> Result<()> {
        let mut ext_track = Track::new(
            2,
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
        );

        thread::sleep(Duration::from_millis(1));
        ext_track.add_observation(
            0,
            Some(0.8),
            Some(vec2(0.66, 0.33)),
            time_attrs_current_ts(),
        )?;

        ext_track.add_observation(
            1,
            Some(0.8),
            Some(vec2(0.65, 0.33)),
            time_attrs_current_ts(),
        )?;

        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            None,
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let res = store.merge_external(0, &ext_track, Some(&[0]), true);
        assert!(res.is_ok());
        let classes = store.get_store(0).get(&0).unwrap().get_feature_classes();
        assert_eq!(classes, vec![0]);

        let res = store.merge_external(0, &ext_track, None, true);
        assert!(res.is_ok());
        let mut classes = store.get_store(0).get(&0).unwrap().get_feature_classes();
        classes.sort();
        assert_eq!(classes, vec![0, 1]);

        Ok(())
    }

    #[test]
    fn merge_own_tracks() -> Result<()> {
        let mut store = TrackStore::new(
            Some(TimeMetric { max_length: 20 }),
            Some(TimeAttrs {
                baked_period: 10,
                ..Default::default()
            }),
            Some(NoopNotifier::default()),
            1,
        );
        thread::sleep(Duration::from_millis(1));
        store.add(
            0,
            0,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        thread::sleep(Duration::from_millis(1));
        store.add(
            1,
            1,
            Some(0.9),
            Some(vec2(0.0, 1.0)),
            time_attrs_current_ts(),
        )?;

        let res = store.merge_owned(0, 1, None, false, true);
        if let Ok(None) = res {
            ();
        } else {
            unreachable!();
        }

        let res = store.merge_owned(0, 1, None, true, true);
        if let Ok(Some(t)) = res {
            assert_eq!(t.track_id, 1);
        } else {
            unreachable!();
        }

        Ok(())
    }

    #[test]
    fn lookup() {
        #[derive(Default, Clone)]
        struct Lookup;
        impl LookupRequest<LookupAttrs, f32> for Lookup {
            fn lookup(
                &self,
                _attributes: &LookupAttrs,
                _observations: &ObservationsDb<f32>,
                _merge_history: &[u64],
            ) -> bool {
                true
            }
        }

        #[derive(Debug, Clone, Default)]
        struct LookupAttrs;

        #[derive(Default, Clone)]
        pub struct LookupAttributeUpdate;

        impl TrackAttributesUpdate<LookupAttrs> for LookupAttributeUpdate {
            fn apply(&self, _attrs: &mut LookupAttrs) -> Result<()> {
                Ok(())
            }
        }

        impl TrackAttributes<LookupAttrs, f32> for LookupAttrs {
            type Update = LookupAttributeUpdate;
            type Lookup = Lookup;

            fn compatible(&self, _other: &LookupAttrs) -> bool {
                true
            }

            fn merge(&mut self, _other: &LookupAttrs) -> Result<()> {
                Ok(())
            }

            fn baked(&self, _observations: &ObservationsDb<f32>) -> Result<TrackStatus> {
                Ok(TrackStatus::Ready)
            }
        }

        #[derive(Default, Clone)]
        pub struct LookupMetric;

        impl ObservationMetric<LookupAttrs, f32> for LookupMetric {
            fn metric(
                _feature_class: u64,
                _attrs1: &LookupAttrs,
                _attrs2: &LookupAttrs,
                e1: &ObservationSpec<f32>,
                e2: &ObservationSpec<f32>,
            ) -> MetricOutput<f32> {
                Some((
                    f32::calculate_metric_object(&e1.0, &e2.0),
                    match (e1.1.as_ref(), e2.1.as_ref()) {
                        (Some(x), Some(y)) => Some(euclidean(x, y)),
                        _ => None,
                    },
                ))
            }

            fn optimize(
                &mut self,
                _feature_class: &u64,
                _merge_history: &[u64],
                _attrs: &mut LookupAttrs,
                _features: &mut Vec<ObservationSpec<f32>>,
                _prev_length: usize,
                _is_merge: bool,
            ) -> Result<()> {
                Ok(())
            }
        }

        let mut store = TrackStoreBuilder::default().build();
        const N: usize = 10;
        for _ in 0..N {
            let t: Track<LookupAttrs, LookupMetric, f32> = TrackBuilder::default().build().unwrap();
            store.add_track(t).unwrap();
        }
        let res = store.lookup(Lookup);
        assert_eq!(res.len(), N);
    }
}
