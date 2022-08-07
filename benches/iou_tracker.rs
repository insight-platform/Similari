#![feature(test)]

extern crate test;

use similari::examples::iou::{BBoxAttributes, BBoxAttributesUpdate, IOUMetric};
use similari::examples::BoxGen2;
use similari::prelude::{NoopNotifier, ObservationBuilder, TrackStoreBuilder};
use similari::utils::bbox::IOUTopNVoting;
use similari::voting::Voting;
use std::time::Instant;
use test::Bencher;

const FEAT0: u64 = 0;

#[bench]
fn bench_iou_00010_4cores(b: &mut Bencher) {
    bench_iou(10, b);
}

#[bench]
fn bench_iou_00100_4cores(b: &mut Bencher) {
    bench_iou(100, b);
}

#[bench]
fn bench_iou_00500_4cores(b: &mut Bencher) {
    bench_iou(500, b);
}

#[bench]
fn bench_iou_01000_4cores(b: &mut Bencher) {
    bench_iou(1000, b);
}

fn bench_iou(objects: usize, b: &mut Bencher) {
    let ncores = match objects {
        10 => 1,
        100 => 2,
        _ => num_cpus::get(),
    };

    let mut store = TrackStoreBuilder::new(ncores)
        .metric(IOUMetric::default())
        .default_attributes(BBoxAttributes::default())
        .notifier(NoopNotifier)
        .build();

    let voting = IOUTopNVoting {
        topn: 1,
        min_distance: 0.2,
        min_votes: 1,
    };

    let pos_drift = 1.0;
    let box_drift = 0.01;
    let mut iterators = Vec::default();

    for i in 0..objects {
        iterators.push(BoxGen2::new(
            1000.0 * i as f32,
            1000.0 * i as f32,
            50.0,
            50.0,
            pos_drift,
            box_drift,
        ))
    }

    let mut iteration = 0;
    b.iter(|| {
        let mut tracks = Vec::new();
        let tm = Instant::now();
        for i in &mut iterators {
            iteration += 1;
            let b = i.next().unwrap();
            let t = store
                .new_track(iteration)
                .observation(
                    ObservationBuilder::new(FEAT0)
                        .observation_attributes(b)
                        .track_attributes_update(BBoxAttributesUpdate)
                        .build(),
                )
                .build()
                .unwrap();
            tracks.push(t);
        }

        let search_tracks = tracks.clone();
        let elapsed = tm.elapsed();
        eprintln!("Construction time: {:?}", elapsed);

        let tm = Instant::now();
        let (dists, errs) = store.foreign_track_distances(search_tracks, FEAT0, false);
        let elapsed = tm.elapsed();
        eprintln!("Lookup time: {:?}", elapsed);

        let tm = Instant::now();
        let winners = voting.winners(dists);
        let elapsed = tm.elapsed();
        eprintln!("Voting time: {:?}", elapsed);
        assert!(errs.all().is_empty());

        let tm = Instant::now();
        for t in tracks {
            let winner = winners.get(&t.get_track_id());
            if let Some(winners) = winner {
                let _res = store
                    .merge_external_noblock(winners[0].winner_track, t, None, false)
                    .unwrap();
            } else {
                store.add_track(t).unwrap();
            }
        }
        let elapsed = tm.elapsed();
        eprintln!("Merging time: {:?}", elapsed);
    });
    eprintln!("Store stats: {:?}", store.shard_stats());
}
