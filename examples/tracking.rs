use itertools::Itertools;
use once_cell::sync::OnceCell;
use rand::distributions::Uniform;
use rand::rngs::ThreadRng;
use rand::Rng;
use similari::test_stuff::vec2;
use similari::track::{AttributeUpdate, Feature};
use thiserror::Error;

const FEATURE0: u64 = 0;

#[derive(Debug, Error)]
enum AppErrors {
    #[error("Cam id passed ({0}) != id set ({1})")]
    WrongCamID(u64, u64),
    #[error("Time passed {0} < time set {1}")]
    WrongTime(u64, u64),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
enum Gender {
    Female,
    Male,
    #[default]
    Unknown,
}

// person attributes
#[derive(Debug, Clone, Default)]
struct CamTrackingAttributes {
    start_time: u64,             // when the track observation first appeared
    end_time: u64,               // when the track observation last appeared
    camera_id: OnceCell<u64>,    // identifier of camera that detected the object
    age: Vec<u8>,                // age detected during the observations
    gender: Vec<Gender>,         // gender detected during the observations
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
        camera_id: Default::default(),
        age: vec![17, 24, 36],
        gender: vec![Male, Female, Female, Unknown],
    };
    assert_eq!(attrs.get_age(), Some(25));
    assert_eq!(attrs.get_gender(), Female);
}

// update
#[derive(Clone)]
struct CamTrackingAttributesUpdate {
    time: u64,
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
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            gen: rand::thread_rng(),
            dist: Uniform::new(-0.01, 0.01),
        }
    }
}

impl Iterator for FeatGen2 {
    type Item = Feature;

    fn next(&mut self) -> Option<Self::Item> {
        self.x += self.gen.sample(&self.dist);
        self.y += self.gen.sample(&self.dist);
        Some(vec2(self.x, self.y))
    }
}

fn main() {}
