use crate::Gender::{Female, Male};
use anyhow::Result;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rand::distributions::Uniform;
use rand::rngs::ThreadRng;
use rand::Rng;
use similari::current_time_ms;
use similari::distance::euclidean;
use similari::store::TrackStore;
use similari::test_stuff::vec2;
use similari::track::notify::NoopNotifier;
use similari::track::{
    AttributeMatch, AttributeUpdate, FeatureObservationsGroups, FeatureSpec, Metric,
    TrackBakingStatus,
};
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;
use std::cmp::{max, min};

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use thiserror::Error;

const FEATURE0: u64 = 0;

#[derive(Debug, Error)]
enum AppErrors {
    #[error("Cam id passed ({0}) != id set ({1})")]
    WrongCamID(u64, u64),
    #[error("Time passed {0} < time set {1}")]
    WrongTime(u128, u128),
    #[error("Incompatible attributes")]
    IncompatibleAttributes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum Gender {
    Female,
    Male,
    #[default]
    Unknown,
}

// person attributes
#[derive(Debug, Clone, Default)]
struct CamTrackingAttributes {
    start_time: u128, // when the track observation first appeared
    end_time: u128,   // when the track observation last appeared
    baked_period_ms: u128,
    camera_id: OnceCell<u64>, // identifier of camera that detected the object
    age: Vec<u8>,             // age detected during the observations
    gender: Vec<Gender>,      // gender detected during the observations
    screen_pos: Vec<(u16, u16)>, // person screen position
}

impl CamTrackingAttributes {
    // calculate age as average over observations
    pub fn get_age(&self) -> Option<u8> {
        if self.age.len() == 0 {
            return None;
        }
        u8::try_from(self.age.iter().map(|e| *e as u32).sum::<u32>() / self.age.len() as u32).ok()
    }

    // calculate gender as most frequent gender
    pub fn get_gender(&self) -> Gender {
        if self.gender.is_empty() {
            return Gender::Unknown;
        }

        let groups = self.gender.clone();
        let mut groups = groups.into_iter().counts().into_iter().collect::<Vec<_>>();
        groups.sort_by(|(_, l), (_, r)| r.partial_cmp(l).unwrap());
        groups[0].0.clone()
    }
}

#[test]
fn test_attributes_age_gender() {
    use Gender::*;
    let attrs = CamTrackingAttributes {
        screen_pos: vec![(0, 0), (10, 15), (20, 25), (30, 35)],
        start_time: 0,
        end_time: 0,
        baked_period_ms: 10,
        camera_id: Default::default(),
        age: vec![17, 24, 36],
        gender: vec![Male, Female, Female, Unknown],
    };
    assert_eq!(attrs.get_age(), Some(25));
    assert_eq!(attrs.get_gender(), Female);
}

// update
#[derive(Clone, Debug)]
struct CamTrackingAttributesUpdate {
    time: u128,
    gender: Option<Gender>,
    age: Option<u8>,
    camera_id: u64,
    screen_pos: (u16, u16),
}

impl AttributeUpdate<CamTrackingAttributes> for CamTrackingAttributesUpdate {
    fn apply(&self, attrs: &mut CamTrackingAttributes) -> anyhow::Result<()> {
        // initially, track start time is set to end time
        if attrs.start_time == 0 {
            attrs.start_time = self.time;
        }

        // if future track observation is submited with older timestamp
        // then it's incorrect situation, timestamp should increase.
        if attrs.end_time > self.time {
            return Err(AppErrors::WrongTime(self.time, attrs.end_time).into());
        }

        attrs.end_time = self.time;

        // update may be without the gender, if observer cannot determine the
        // gender within the observation
        if let Some(gender) = &self.gender {
            attrs.gender.push(gender.clone());
        }

        // same for age
        if let Some(age) = &self.age {
            attrs.age.push(*age);
        }

        // track with <id> always goes from the same camera. If camera id changed
        // it's a wrong case.
        if let Err(_r) = attrs.camera_id.set(self.camera_id) {
            if self.camera_id != *attrs.camera_id.get().unwrap() {
                return Err(
                    AppErrors::WrongCamID(self.camera_id, *attrs.camera_id.get().unwrap()).into(),
                );
            }
        }

        attrs.screen_pos.push(self.screen_pos);

        Ok(())
    }
}

#[test]
fn cam_tracking_attributes_update_test() {
    use Gender::*;
    let mut attrs = CamTrackingAttributes {
        start_time: 0,
        end_time: 0,
        baked_period_ms: 10,
        camera_id: Default::default(),
        age: Vec::default(),
        gender: Vec::default(),
        screen_pos: Vec::default(),
    };
    let update = CamTrackingAttributesUpdate {
        time: 10,
        gender: Some(Female),
        age: Some(30),
        camera_id: 10,
        screen_pos: (10, 10),
    };
    assert!(update.apply(&mut attrs).is_ok());

    // incorrect cam id
    let update = CamTrackingAttributesUpdate {
        time: 20,
        gender: Some(Female),
        age: Some(10),
        camera_id: 20,
        screen_pos: (10, 15),
    };
    assert!(update.apply(&mut attrs).is_err());

    // incorrect time
    let update = CamTrackingAttributesUpdate {
        time: 5,
        gender: Some(Female),
        age: Some(10),
        camera_id: 20,
        screen_pos: (20, 25),
    };
    assert!(update.apply(&mut attrs).is_err());
}

struct FeatGen2 {
    x: f32,
    y: f32,
    gen: ThreadRng,
    dist: Uniform<f32>,
}

impl FeatGen2 {
    pub fn new(x: f32, y: f32, drift: f32) -> Self {
        Self {
            x,
            y,
            gen: rand::thread_rng(),
            dist: Uniform::new(-drift, drift),
        }
    }
}

impl Iterator for FeatGen2 {
    type Item = FeatureSpec;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist);
        self.y += self.gen.sample(&self.dist);
        Some((self.gen.sample(&self.dist) + 0.7, vec2(self.x, self.y)))
    }
}

#[test]
fn feat_gen() {
    use std::ops::Sub;
    use ultraviolet::f32x8;

    let drift = 0.01;
    let mut gen = FeatGen2::new(0.0, 0.0, drift);
    let v1 = gen.next().unwrap().1[0];
    let v2 = gen.next().unwrap().1[0];
    assert!(v1.sub(v2).abs().reduce_add() <= 2.0 * f32x8::splat(drift).reduce_add());
}

impl AttributeMatch<CamTrackingAttributes> for CamTrackingAttributes {
    fn compatible(&self, other: &CamTrackingAttributes) -> bool {
        (self.start_time >= other.end_time || self.end_time <= other.start_time)
            && self.camera_id.get().unwrap() == other.camera_id.get().unwrap()
    }

    fn merge(&mut self, other: &CamTrackingAttributes) -> Result<()> {
        if self.compatible(other) {
            self.start_time = min(self.start_time, other.start_time);
            self.end_time = max(self.end_time, other.end_time);
            self.screen_pos.extend_from_slice(&other.screen_pos);
            self.age.extend_from_slice(&other.age);
            self.gender.extend_from_slice(&other.gender);
            Ok(())
        } else {
            Err(AppErrors::IncompatibleAttributes.into())
        }
    }

    fn baked(&self, _observations: &FeatureObservationsGroups) -> Result<TrackBakingStatus> {
        let now = current_time_ms();
        if now > self.end_time + self.baked_period_ms {
            Ok(TrackBakingStatus::Ready)
        } else {
            Ok(TrackBakingStatus::Pending)
        }
    }
}

#[derive(Clone)]
pub struct CamTrackingAttributesMetric {
    merge_extension: f32,
    initial_capacity: u64,
    max_capacity: u64,
}

impl Default for CamTrackingAttributesMetric {
    fn default() -> Self {
        Self {
            merge_extension: 1.5,
            initial_capacity: 4,
            max_capacity: 12,
        }
    }
}

impl Metric for CamTrackingAttributesMetric {
    fn distance(_feature_class: u64, e1: &FeatureSpec, e2: &FeatureSpec) -> Option<f32> {
        Some(euclidean(&e1.1, &e2.1))
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        merge_history: &[u64],
        features: &mut Vec<FeatureSpec>,
        _prev_length: usize,
    ) -> Result<()> {
        let merges = merge_history.len();
        let mut current_capacity =
            (self.initial_capacity as f32 * self.merge_extension.powf(merges as f32)) as u64;
        if current_capacity > self.max_capacity {
            current_capacity = self.max_capacity
        }
        features.sort_by(|(l, _), (r, _)| r.partial_cmp(l).unwrap());
        features.truncate(current_capacity as usize);
        Ok(())
    }
}

struct Observation {
    pub track_id: u64,
    pub age: Option<u8>,
    pub gender: Option<Gender>,
    pub camera_id: u64,
    pub screen_pos: (u16, u16),
    pub class: u64,
    pub feature: FeatureSpec,
}

impl Observation {
    pub fn new(
        track_id: u64,
        age: Option<u8>,
        gender: Option<Gender>,
        camera_id: u64,
        screen_pos: (u16, u16),
        feature: FeatureSpec,
    ) -> Self {
        Self {
            track_id,
            age,
            gender,
            camera_id,
            screen_pos,
            feature,
            class: FEATURE0,
        }
    }
}

fn main() {
    let drift = 0.01;
    let mut p1 = FeatGen2::new(1.0, 0.0, drift);
    let mut p2 = FeatGen2::new(1.0, 1.0, drift);

    let m = Some(Male);
    let f = Some(Female);
    let observations = vec![
        // track 1 (person 1)
        Observation::new(1, Some(13), f, 1, (30, 30), p1.next().unwrap()),
        Observation::new(1, Some(17), m, 1, (35, 30), p1.next().unwrap()),
        Observation::new(1, Some(23), m, 1, (35, 35), p1.next().unwrap()),
        Observation::new(1, None, None, 1, (40, 35), p1.next().unwrap()),
        Observation::new(1, Some(18), m, 1, (40, 40), p1.next().unwrap()),
        // track 2 (person 2)
        Observation::new(2, Some(46), f, 1, (100, 100), p2.next().unwrap()),
        Observation::new(2, Some(30), f, 1, (135, 130), p2.next().unwrap()),
        Observation::new(2, Some(40), f, 1, (135, 135), p2.next().unwrap()),
        Observation::new(2, None, None, 1, (140, 135), p2.next().unwrap()),
        Observation::new(2, Some(54), m, 1, (140, 140), p2.next().unwrap()),
        // track 3 (person 1)
        Observation::new(3, Some(18), f, 1, (50, 40), p1.next().unwrap()),
        Observation::new(3, Some(17), m, 1, (55, 50), p1.next().unwrap()),
        Observation::new(3, Some(20), m, 1, (65, 55), p1.next().unwrap()),
        Observation::new(3, Some(17), None, 1, (70, 50), p1.next().unwrap()),
        Observation::new(3, None, m, 1, (75, 55), p1.next().unwrap()),
        // track 4 (person 2)
        Observation::new(4, Some(48), f, 1, (150, 140), p2.next().unwrap()),
        Observation::new(4, Some(47), f, 1, (155, 150), p2.next().unwrap()),
        Observation::new(4, Some(30), m, 1, (165, 155), p2.next().unwrap()),
        Observation::new(4, Some(57), None, 1, (170, 150), p2.next().unwrap()),
        Observation::new(4, None, f, 1, (175, 155), p2.next().unwrap()),
        // track 5 (person 1)
        Observation::new(5, None, None, 1, (80, 55), p1.next().unwrap()),
        Observation::new(5, None, None, 1, (85, 60), p1.next().unwrap()),
        Observation::new(5, None, None, 1, (90, 65), p1.next().unwrap()),
        Observation::new(5, None, None, 1, (90, 50), p1.next().unwrap()),
        Observation::new(5, None, m, 1, (90, 50), p1.next().unwrap()),
    ];

    // collect tracks here until they are initially ready
    let mut temp_store = TrackStore::new(
        Some(CamTrackingAttributesMetric::default()),
        Some(CamTrackingAttributes {
            baked_period_ms: 20,
            ..Default::default()
        }),
        Some(NoopNotifier),
        1,
    );

    // merge tracks here until they are initially complete
    let merge_store_baked_period_ms = 60;
    let mut merge_store: TrackStore<
        CamTrackingAttributes,
        CamTrackingAttributesUpdate,
        CamTrackingAttributesMetric,
    > = TrackStore::new(
        Some(CamTrackingAttributesMetric::default()),
        Some(CamTrackingAttributes {
            baked_period_ms: merge_store_baked_period_ms,
            ..Default::default()
        }),
        Some(NoopNotifier),
        1,
    );
    let voting_machine = TopNVoting::new(1, 0.1, 3);

    let mut idx = 0;
    loop {
        if let Some(Observation {
            track_id,
            age,
            gender,
            camera_id,
            screen_pos,
            class,
            feature,
        }) = observations.get(idx)
        {
            let update = CamTrackingAttributesUpdate {
                time: current_time_ms(),
                gender: *gender,
                age: *age,
                camera_id: *camera_id,
                screen_pos: *screen_pos,
            };
            temp_store
                .add(*track_id, *class, feature.0, feature.1.clone(), update)
                .unwrap();
        }
        idx += 1;

        thread::sleep(Duration::from_millis(1));
        let baked = temp_store.find_baked();
        for (id, s) in baked {
            let mut track = temp_store.fetch_tracks(&vec![id]).pop().unwrap();
            if let Ok(TrackBakingStatus::Ready) = s {
                let search_track = Arc::new(track.clone());
                track.get_attributes_mut().baked_period_ms = 0;

                let (dists, _errs) =
                    merge_store.foreign_track_distances(search_track, FEATURE0, false, None);
                let mut winners = voting_machine.winners(&dists);
                if winners.is_empty() {
                    let _track_id = merge_store.add_track(track).unwrap();
                } else {
                    let winner = winners.pop().unwrap();
                    merge_store
                        .merge_external(winner.track_id, &track, Some(&[FEATURE0]))
                        .unwrap();
                }
            }
        }

        if idx > 100 {
            break;
        }
    }

    let baked = merge_store.find_baked();
    for (id, s) in baked {
        if let Ok(TrackBakingStatus::Ready) = s {
            let track = merge_store.fetch_tracks(&vec![id]).pop().unwrap();
            eprintln!(
                "Composite Track is ready: {}, age: {:?}, gender: {:?}\nCoordinates: {:?}",
                track.get_track_id(),
                track.get_attributes().get_age(),
                track.get_attributes().get_gender(),
                track.get_attributes().screen_pos
            );
            eprintln!("Merge history: {:?}", track.get_merge_history());
        }
    }
}
