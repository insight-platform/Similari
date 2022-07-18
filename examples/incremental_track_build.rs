use similari::distance::euclidean;
use similari::store::TrackStore;
use similari::test_stuff::{current_time_ms, BBox, BoxGen2, FeatGen2};
use similari::track::{
    MetricOutput, ObservationAttributes, ObservationMetric, ObservationSpec, ObservationsDb, Track,
    TrackAttributes, TrackAttributesUpdate, TrackStatus,
};
use similari::voting::topn::TopNVoting;
use similari::voting::Voting;
use std::thread;
use std::time::Duration;

const FEAT0: u64 = 0;

#[derive(Debug, Clone, Default)]
struct NoopAttributes;

#[derive(Clone, Debug)]
struct NoopAttributesUpdate;

impl TrackAttributesUpdate<NoopAttributes> for NoopAttributesUpdate {
    fn apply(&self, _attrs: &mut NoopAttributes) -> anyhow::Result<()> {
        Ok(())
    }
}

impl TrackAttributes<NoopAttributes, BBox> for NoopAttributes {
    fn compatible(&self, _other: &NoopAttributes) -> bool {
        true
    }

    fn merge(&mut self, _other: &NoopAttributes) -> anyhow::Result<()> {
        Ok(())
    }

    fn baked(&self, _observations: &ObservationsDb<BBox>) -> anyhow::Result<TrackStatus> {
        Ok(TrackStatus::Ready)
    }
}

#[derive(Clone, Default)]
pub struct TrackMetric;

impl ObservationMetric<NoopAttributes, BBox> for TrackMetric {
    fn metric(
        _feature_class: u64,
        _attrs1: &NoopAttributes,
        _attrs2: &NoopAttributes,
        e1: &ObservationSpec<BBox>,
        e2: &ObservationSpec<BBox>,
    ) -> MetricOutput<f32> {
        // bbox information (.0) is not used but can be used
        // to implement additional IoU tracking
        // one can use None if low IoU
        // or implement weighted distance based on IoU and euclidean distance
        //
        Some((
            BBox::calculate_metric_object(&e1.0, &e2.0),
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
        _attrs: &mut NoopAttributes,
        _observations: &mut Vec<ObservationSpec<BBox>>,
        _prev_length: usize,
        _is_merge: bool,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

fn main() {
    let mut store: TrackStore<NoopAttributes, NoopAttributesUpdate, TrackMetric, BBox> =
        TrackStore::default();
    let voting: TopNVoting<BBox> = TopNVoting::new(1, 0.1, 1);
    let feature_drift = 0.01;
    let pos_drift = 5.0;
    let box_drift = 2.0;
    let mut p1 = FeatGen2::new(0.0, 0.0, feature_drift);
    let mut b1 = BoxGen2::new(100.0, 100.0, 10.0, 15.0, pos_drift, box_drift);

    let mut p2 = FeatGen2::new(1.0, 1.0, feature_drift);
    let mut b2 = BoxGen2::new(10.0, 10.0, 12.0, 18.0, pos_drift, box_drift);

    for _ in 0..10 {
        let (obj1f, obj1b) = (p1.next().unwrap().1, b1.next());
        let mut obj1t: Track<NoopAttributes, TrackMetric, NoopAttributesUpdate, BBox> =
            Track::new(u64::try_from(current_time_ms()).unwrap(), None, None, None);
        obj1t.add_observation(FEAT0, obj1b, obj1f, None).unwrap();

        let (obj2f, obj2b) = (p2.next().unwrap().1, b2.next());
        let mut obj2t: Track<NoopAttributes, TrackMetric, NoopAttributesUpdate, BBox> = Track::new(
            (u64::try_from(current_time_ms()).unwrap()) + 1,
            None,
            None,
            None,
        );
        obj2t.add_observation(FEAT0, obj2b, obj2f, None).unwrap();
        thread::sleep(Duration::from_millis(2));

        for t in [obj1t, obj2t] {
            let search_track = t.clone();
            let (dists, errs) = store.foreign_track_distances(vec![search_track], FEAT0, false);
            assert!(errs.is_empty());
            let winners = voting.winners(&dists);
            if winners.is_empty() {
                store.add_track(t).unwrap();
            } else {
                store
                    .merge_external(
                        winners.get(&t.get_track_id()).unwrap()[0].winner_track,
                        &t,
                        None,
                        false,
                    )
                    .unwrap();
            }
        }
    }

    let tracks = store.find_usable();
    for (t, _) in tracks {
        let t = store.fetch_tracks(&vec![t]);
        eprintln!("Track id: {}", t[0].get_track_id());
        eprintln!(
            "Boxes: {:#?}",
            t[0].get_observations(FEAT0)
                .unwrap()
                .iter()
                .map(|x| x.0.clone())
                .collect::<Vec<_>>()
        );
    }
}
