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

**SORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark uses the heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker.rs](benches/simple_sort_iou_tracker.rs).

| Objects | Time (ns/iter) | FPS   |
|---------|----------------|-------|
| 10      | 127,508        | 12048 |
| 100     | 1,959,395      | 433   |
| 500     | 20,265,114     | 49    |


**Oriented SORT tracking (IoU)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark uses 
the heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker_oriented.rs](benches/simple_sort_iou_tracker_oriented.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 150,702        | 6600 |
| 100     | 2,102,687      | 475  |
| 500     | 20,265,114     | 49   |

**SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark uses heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker.rs](benches/simple_sort_maha_tracker.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 121,941        | 8100 |
| 100     | 1,974,091      | 500  |
| 500     | 20,268,841     | 49   |

**Oriented SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark 
uses the heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker_oriented.rs](benches/simple_sort_maha_tracker_oriented.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 133,531        | 7400 |
| 100     | 1,859,690      | 500  |
| 500     | 20,594,887     | 49   |

**Visual (256 @ f32, hist=3) tracking**. Benchmark for N simultaneously observed objects. The benchmark doesn't use 
heuristics that separate the observed objects based on object distances. The 3 last observations are used to select 
winning track.

The benchmark located at [benches/feature_tracker.rs](benches/feature_tracker.rs).

| Objects | Time (ns/iter) | FPS  |
|---------|----------------|------|
| 10      | 101,465        | 9900 |
| 100     | 4,020,673      | 250  |
| 500     | 61,716,729     | 16   |

**Visual SORT (aka DeepSORT) tracking**. Benchmark for N simultaneously observed objects with feature vectors. The benchmark uses heuristics 
that separate the observed objects based on object distances. Every track holds 3 feature vectors for comparison with candidats.

The benchmark is located at [benches/simple_visual_sort_tracker.rs](benches/simple_visual_sort_tracker.rs).

| Objects | Vector Len | Time (ns/iter) | FPS  |
|---------|------------|----------------|------|
| 10      | 128        | 356,237        | 2800 |
| 10      | 256        | 404,416        | 2460 |
| 10      | 512        | 447,903        | 2230 |
| 10      | 1024       | 573,197        | 1740 |
| 10      | 2048       | 767,031        | 1300 |
| 50      | 128        | 1,923,861      | 519  |
| 50      | 256        | 2,105,886      | 474  |
| 50      | 512        | 2,249,694      | 444  |
| 50      | 1024       | 2,958,547      | 337  |
| 50      | 2048       | 4,563,691      | 218  |
| 100     | 128        | 3,807,716      | 262  |
| 100     | 256        | 4,717,401      | 211  |
| 100     | 512        | 5,775,469      | 173  |
| 100     | 1024       | 7,497,783      | 133  |
| 100     | 2048       | 10,527,237     | 94   |
