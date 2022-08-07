# Benchmarks

All benchmarks numbers received on Run on 4 cores of Intel(R) Core(TM) i5-7440HQ CPU @ 2.80GHz.

Version: v0.21.0

**Non-Maximum Suppression (non-oriented boxes)**. Benchmark for filtering out of bounding boxes without orientation. 

| Objects | Time (ns/iter) | FPS    |
|---------|----------------|--------|
| 10      | 1,586          | 632000 |
| 100     | 148,906        | 6711   |
| 500     | 4,082,791      | 250    |
| 1000    | 13,773,713     | 72     |

The benchmark is located at [benches/nms.rs](benches/nms.rs).

**Non-Maximum Suppression (oriented boxes)**. Benchmark for filtering out of bounding boxes with angular orientation. 

| Objects | Time (ns/iter)  | FPS    |
|---------|-----------------|--------|
| 10      | 2,169           | 461000 |
| 100     | 139,204         | 7100   |
| 300     | 1,752,410       | 570    |
| 500     | 4,571,784       | 218    |
| 1000    | 18,155,136      | 54     |


The benchmark is located at [benches/nms_oriented.rs](benches/nms_oriented.rs).

**SORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark doesn't use heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker.rs](benches/simple_sort_iou_tracker.rs).

| Objects | Time (ns/iter) | FPS   |
|---------|----------------|-------|
| 10      | 127,508        | 12048 |
| 100     | 1,959,395      | 433   |
| 500     | 24,170,165     | 41    |
| 1000    | 83,859,085     | 11    |


**Oriented SORT tracking (IoU)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark use 
heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker_oriented.rs](benches/simple_sort_iou_tracker_oriented.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 150,702        | 6600 |
| 100     | 2,532,766      | 390  |
| 500     | 26,976,983     | 37   |
| 1000    | 87,453,176     | 11   |

**SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark doesn't use heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker.rs](benches/simple_sort_maha_tracker.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 121,941        | 8100 |
| 100     | 2,397,841      | 470  |
| 500     | 27,444,708     | 36   |
| 1000    | 85,111,059     | 11   |

**Oriented SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark 
use heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker_oriented.rs](benches/simple_sort_maha_tracker_oriented.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 133,531        | 7400 |
| 100     | 2,565,572      | 389  |
| 500     | 30,443,250     | 32   |
| 1000    | 82,211,441     | 12   |

**Feature (256 @ f32, hist=3) tracking**. Benchmark for N simultaneously observed objects. The benchmark doesn't use 
heuristics that separate the observed objects based on object distances. The 3 last observations are used to select 
winning track.

The benchmark located at [benches/feature_tracker.rs](benches/feature_tracker.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 101,465        | 9900 |
| 100     | 4,020,673      | 250  |
| 500     | 61,716,729     | 16   |
