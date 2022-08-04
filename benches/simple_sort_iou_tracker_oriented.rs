#![feature(test)]

extern crate test;

use similari::examples::BoxGen2;
use similari::trackers::sort::simple_iou::SORT;
use similari::trackers::sort::DEFAULT_SORT_IOU_THRESHOLD;
use similari::utils::bbox::Universal2DBox;
use test::Bencher;

#[bench]
fn bench_sort_iou_oriented_00010(b: &mut Bencher) {
    bench_sort(10, b);
}

#[bench]
fn bench_sort_iou_oriented_00100(b: &mut Bencher) {
    bench_sort(100, b);
}

#[bench]
fn bench_sort_iou_oriented_00500(b: &mut Bencher) {
    bench_sort(500, b);
}

#[bench]
fn bench_sort_iou_oriented_01000(b: &mut Bencher) {
    bench_sort(1000, b);
}

fn bench_sort(objects: usize, b: &mut Bencher) {
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
    let ncores = match objects {
        10 => 1,
        100 => 2,
        _ => num_cpus::get(),
    };

    let mut tracker = SORT::new(ncores, 10, 1, DEFAULT_SORT_IOU_THRESHOLD);

    b.iter(|| {
        let mut observations = Vec::new();
        for i in &mut iterators {
            iteration += 1;
            let b = Universal2DBox::from(i.next().unwrap())
                .rotate(tracker.current_epoch() as f32 / 10.0)
                .gen_vertices();
            observations.push(b);
        }
        let tracks = tracker.predict(&observations);
        assert_eq!(tracks.len(), objects);
    });
    let wasted = tracker.wasted();
    assert!(wasted.is_empty());

    eprintln!("Store stats: {:?}", tracker.shard_stats());
    assert_eq!(tracker.shard_stats().into_iter().sum::<usize>(), objects);

    tracker.skip_epochs(2);
    let wasted = tracker.wasted();
    assert_eq!(wasted.len(), objects);
}
