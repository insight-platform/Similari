# Benchmarks

All benchmarks numbers received on Run on 4 cores of Intel(R) Core(TM) i5-7440HQ CPU @ 2.80GHz.

Version: v0.22.5

**Non-Maximum Suppression (non-oriented boxes)**. Benchmark for filtering out of bounding boxes without orientation. 

| Objects |  Time (ns/iter) |     FPS |
|---------|----------------:|--------:|
| 10      |           1,586 |  632000 |
| 100     |         148,906 |    6711 |
| 500     |       4,082,791 |     250 |
| 1000    |      13,773,713 |      72 |

The benchmark is located at [/benches/nms.rs](/benches/nms.rs).

**Non-Maximum Suppression (oriented boxes)**. Benchmark for filtering out of bounding boxes with angular orientation. 

| Objects |   Time (ns/iter) |     FPS |
|---------|-----------------:|--------:|
| 10      |            2,169 |  461000 |
| 100     |          139,204 |    7100 |
| 300     |        1,752,410 |     570 |
| 500     |        4,571,784 |     218 |
| 1000    |       18,155,136 |      54 |


The benchmark is located at [/benches/nms_oriented.rs](/benches/nms_oriented.rs).

**SORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark uses the heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [/benches/simple_sort_iou_tracker.rs](/benches/simple_sort_iou_tracker.rs).

| Objects |   Time (ns/iter) |     FPS |
|---------|-----------------:|--------:|
| 10      |          100,931 |    9900 |
| 100     |        1,779,434 |     561 |
| 500     |       18,705,819 |      53 |


**Oriented SORT tracking (IoU)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark uses 
the heuristics that separate the observed objects based on object distances.

The benchmark is located at [/benches/simple_sort_iou_tracker_oriented.rs](/benches/simple_sort_iou_tracker_oriented.rs).

| Objects |   Time (ns/iter) |  FPS |
|---------|-----------------:|-----:|
| 10      |          108,414 | 9170 |
| 100     |        1,601,062 |  624 |
| 500     |       18,945,655 |   52 |

**SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark uses heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [/benches/simple_sort_maha_tracker.rs](/benches/simple_sort_maha_tracker.rs).

| Objects | Time (ns/iter) |  FPS |
|---------|---------------:|-----:|
| 10      |        105,311 | 9500 |
| 100     |      1,696,943 |  588 |
| 500     |     18,233,557 |   54 |

**Oriented SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark 
uses the heuristics that separate the observed objects based on object distances.

The benchmark is located at [/benches/simple_sort_maha_tracker_oriented.rs](/benches/simple_sort_maha_tracker_oriented.rs).

| Objects |  Time (ns/iter) |  FPS |
|---------|----------------:|-----:|
| 10      |         111,778 | 8900 |
| 100     |       1,567,771 |  636 |
| 500     |      17,762,559 |   56 |

**Visual (256 @ f32, hist=3) tracking**. Benchmark for N simultaneously observed objects. The benchmark doesn't use 
heuristics that separate the observed objects based on object distances. The 3 last observations are used to select 
winning track.

The benchmark located at [/benches/feature_tracker.rs](/benches/feature_tracker.rs).

| Objects |  Time (ns/iter) |   FPS |
|---------|----------------:|------:|
| 10      |         101,465 |  9900 |
| 100     |       4,020,673 |   250 |
| 500     |      61,716,729 |    16 |

**Visual SORT (aka DeepSORT) tracking**. Benchmark for N simultaneously observed objects with feature vectors. The benchmark uses heuristics 
that separate the observed objects based on object distances. Every track holds 3 feature vectors for comparison with candidats.

The benchmark is located at [/benches/simple_visual_sort_tracker.rs](/benches/simple_visual_sort_tracker.rs).

| Objects |  Vector Len |  Time (ns/iter) |   FPS |
|---------|------------:|----------------:|------:|
| 10      |         128 |         356,237 |  2800 |
| 10      |         256 |         404,416 |  2460 |
| 10      |         512 |         447,903 |  2230 |
| 10      |        1024 |         573,197 |  1740 |
| 10      |        2048 |         767,031 |  1300 |
| 50      |         128 |       1,923,861 |   519 |
| 50      |         256 |       2,105,886 |   474 |
| 50      |         512 |       2,249,694 |   444 |
| 50      |        1024 |       2,958,547 |   337 |
| 50      |        2048 |       4,563,691 |   218 |
| 100     |         128 |       3,807,716 |   262 |
| 100     |         256 |       4,717,401 |   211 |
| 100     |         512 |       5,775,469 |   173 |
| 100     |        1024 |       7,497,783 |   133 |
| 100     |        2048 |      10,527,237 |    94 |

**BatchSORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark uses the heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [/benches/batch_sort_iou_tracker.rs](/benches/batch_sort_iou_tracker.rs).

| Objects |   Time (ns/iter) |  FPS |
|---------|-----------------:|-----:|
| 10      |          106,876 | 9300 |
| 100     |        1,616,542 |  618 |
| 500     |       20,454,230 |   48 |

**BatchSORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark uses heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [/benches/batch_sort_maha_tracker.rs](/benches/batch_sort_maha_tracker.rs).

| Objects | Time (ns/iter) |  FPS |
|---------|---------------:|-----:|
| 10      |        114,592 | 8695 |
| 100     |      1,533,445 |  649 |
| 500     |     18,270,742 |   54 |

