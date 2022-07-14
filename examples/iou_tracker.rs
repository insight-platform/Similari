use anyhow::Result;
use itertools::Itertools;
use similari::store::TrackStore;
use similari::test_stuff::{current_time_ms, Bbox, BoxGen2};
use similari::track::{
    ObservationMetric, ObservationMetricResult, ObservationSpec, ObservationsDb, Track,
    TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVotingElt;
use similari::voting::Voting;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const FEAT0: u64 = 0;

#[derive(Debug, Clone, Default)]
struct BBoxAttributes {
    bboxes: Vec<Bbox>,
}

#[derive(Clone, Debug)]
struct BBoxAttributesUpdate {
    bbox: Bbox,
}

impl TrackAttributesUpdate<BBoxAttributes> for BBoxAttributesUpdate {
    fn apply(&self, attrs: &mut BBoxAttributes) -> Result<()> {
        attrs.bboxes.push(self.bbox.clone());
        Ok(())
    }
}

impl TrackAttributes<BBoxAttributes, Bbox> for BBoxAttributes {
    fn compatible(&self, _other: &BBoxAttributes) -> bool {
        true
    }

    fn merge(&mut self, other: &BBoxAttributes) -> Result<()> {
        self.bboxes.extend_from_slice(&other.bboxes);
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<Bbox>) -> Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone)]
pub struct IOUMetric {
    history: usize,
}

impl Default for IOUMetric {
    fn default() -> Self {
        Self { history: 3 }
    }
}

impl ObservationMetric<BBoxAttributes, Bbox> for IOUMetric {
    fn distance(
        _feature_class: u64,
        _e1: &ObservationSpec<Bbox>,
        _e2: &ObservationSpec<Bbox>,
    ) -> Option<f32> {
        None
    }

    fn optimize(
        &mut self,
        _feature_class: &u64,
        _merge_history: &[u64],
        _attrs: &mut BBoxAttributes,
        features: &mut Vec<ObservationSpec<Bbox>>,
        _prev_length: usize,
    ) -> Result<()> {
        // Kalman filter should be there to generate better predictions
        features.reverse();
        features.truncate(self.history);
        features.reverse();
        Ok(())
    }
}

pub struct TopNVoting {
    topn: usize,
    min_distance: f32,
    min_votes: usize,
}

impl Voting<TopNVotingElt, f32> for TopNVoting {
    fn winners(&self, distances: &[ObservationMetricResult<f32>]) -> Vec<TopNVotingElt> {
        let mut tracks: Vec<_> = distances
            .iter()
            .filter(
                |ObservationMetricResult(_, f_attr_dist, _)| match f_attr_dist {
                    Some(e) => *e >= self.min_distance,
                    _ => false,
                },
            )
            .map(|ObservationMetricResult(track, _, _)| track)
            .collect();
        tracks.sort_unstable();
        let mut counts = tracks
            .into_iter()
            .counts()
            .into_iter()
            .filter(|(_, count)| *count >= self.min_votes)
            .map(|(e, c)| TopNVotingElt {
                track_id: *e,
                votes: c,
            })
            .collect::<Vec<_>>();

        counts.sort_by(|l, r| r.votes.partial_cmp(&l.votes).unwrap());
        counts.truncate(self.topn);
        counts
    }
}

fn main() {
    let mut store: TrackStore<BBoxAttributes, BBoxAttributesUpdate, IOUMetric, Bbox> =
        TrackStore::default();

    let voting = TopNVoting {
        topn: 1,
        min_distance: 0.5,
        min_votes: 1,
    };

    let pos_drift = 1.0;
    let box_drift = 1.0;
    let mut b1 = BoxGen2::new(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);

    let mut b2 = BoxGen2::new(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let obj1b = b1.next();
        let obj2b = b2.next();

        let mut obj1t: Track<BBoxAttributes, IOUMetric, BBoxAttributesUpdate, Bbox> =
            Track::new(u64::try_from(current_time_ms()).unwrap(), None, None, None);

        obj1t
            .add_observation(
                FEAT0,
                obj1b.clone(),
                None,
                Some(BBoxAttributesUpdate {
                    bbox: obj1b.unwrap(),
                }),
            )
            .unwrap();

        let mut obj2t: Track<BBoxAttributes, IOUMetric, BBoxAttributesUpdate, Bbox> = Track::new(
            u64::try_from(current_time_ms()).unwrap() + 1,
            None,
            None,
            None,
        );

        obj2t
            .add_observation(
                FEAT0,
                obj2b.clone(),
                None,
                Some(BBoxAttributesUpdate {
                    bbox: obj2b.unwrap(),
                }),
            )
            .unwrap();

        thread::sleep(Duration::from_millis(2));

        for t in [obj1t, obj2t] {
            let search_track = Arc::new(t.clone());
            let (dists, errs) = store.foreign_track_distances(search_track, FEAT0, false, None);
            assert!(errs.is_empty());
            let winners = voting.winners(&dists);
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                store
                    .merge_external(winners[0].track_id, &t, None, false)
                    .unwrap();
            }
        }
    }

    let tracks = store.find_usable();
    for (t, _) in tracks {
        let t = store.fetch_tracks(&vec![t]);
        eprintln!("Track id: {}", t[0].get_track_id());
        eprintln!("Boxes: {:#?}", t[0].get_attributes().bboxes);
    }
}
