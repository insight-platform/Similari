# Benchmarks

All benchmarks numbers received on Run on 4 cores of Intel(R) Core(TM) i5-7440HQ CPU @ 2.80GHz.

Version: v0.22.5

**Non-Maximum Suppression (non-oriented boxes)**. Benchmark for filtering out of bounding boxes without orientation. 

| Objects |          Time (ns/iter) |     FPS |
|---------|------------------------:|--------:|
| 10      |                     968 | 1000000 |
| 100     |                 101,904 |    9803 |
| 300     |               1,192,594 |     838 |
| 500     |               3,110,681 |     321 |
| 1000    |              10,978,617 |      90 |

The benchmark is located at [benches/nms.rs](benches/nms.rs).

**Non-Maximum Suppression (oriented boxes)**. Benchmark for filtering out of bounding boxes with angular orientation. 

| Objects | Time (ns/iter) |    FPS |
|---------|---------------:|-------:|
| 10      |          2,169 | 461000 |
| 100     |        124,306 |   8045 |
| 300     |      1,572,835 |    635 |
| 500     |      4,321,719 |    231 |
| 1000    |     14,887,268 |     67 |


The benchmark is located at [benches/nms_oriented.rs](benches/nms_oriented.rs).

**SORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark uses the heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker.rs](benches/simple_sort_iou_tracker.rs).

| Objects |   Time (ns/iter) |     FPS |
|---------|-----------------:|--------:|
| 10      |          100,931 |    9900 |
| 100     |        1,779,434 |     561 |
| 500     |       18,705,819 |      53 |


**Oriented SORT tracking (IoU)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark uses 
the heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_iou_tracker_oriented.rs](benches/simple_sort_iou_tracker_oriented.rs).

| Objects |   Time (ns/iter) |  FPS |
|---------|-----------------:|-----:|
| 10      |          108,414 | 9170 |
| 100     |        1,601,062 |  624 |
| 500     |       18,945,655 |   52 |

**SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark uses heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker.rs](benches/simple_sort_maha_tracker.rs).

| Objects | Time (ns/iter) |  FPS |
|---------|---------------:|-----:|
| 10      |        105,311 | 9500 |
| 100     |      1,696,943 |  588 |
| 500     |     18,233,557 |   54 |

**Oriented SORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed **oriented** objects. The benchmark 
uses the heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_maha_tracker_oriented.rs](benches/simple_sort_maha_tracker_oriented.rs).

| Objects |  Time (ns/iter) |  FPS |
|---------|----------------:|-----:|
| 10      |         111,778 | 8900 |
| 100     |       1,567,771 |  636 |
| 500     |      17,762,559 |   56 |

**Visual (256 @ f32, hist=3) tracking**. Benchmark for N simultaneously observed objects. The benchmark doesn't use 
heuristics that separate the observed objects based on object distances. The 3 last observations are used to select 
winning track.

The benchmark located at [benches/feature_tracker.rs](benches/feature_tracker.rs).

| Objects |  Time (ns/iter) |   FPS |
|---------|----------------:|------:|
| 10      |         101,465 |  9900 |
| 100     |       4,020,673 |   250 |
| 500     |      61,716,729 |    16 |

**Visual SORT (aka DeepSORT) tracking**. Benchmark for N simultaneously observed objects with feature vectors. The benchmark uses heuristics 
that separate the observed objects based on object distances. Every track holds 3 feature vectors for comparison with candidats.

The benchmark is located at [benches/simple_visual_sort_tracker.rs](benches/simple_visual_sort_tracker.rs).

| Objects |  Vector Len | Time (ns/iter) |  FPS |
|---------|------------:|---------------:|-----:|
| 10      |         128 |        214,740 | 4650 |
| 10      |         256 |        253,723 | 3940 |
| 10      |         512 |        305,891 | 3267 |
| 10      |        1024 |        368,687 | 2710 |
| 10      |        2048 |        488,885 | 2044 |
| 50      |         128 |      1,064,779 |  938 |
| 50      |         256 |      1,372,285 |  728 |
| 50      |         512 |      1,654,183 |  604 |
| 50      |        1024 |      2,203,557 |  453 |
| 50      |        2048 |      3,194,354 |  312 |
| 100     |         128 |      2,384,783 |  419 |
| 100     |         256 |      2,795,945 |  357 |
| 100     |         512 |      3,410,891 |  293 |
| 100     |        1024 |      3,994,822 |  250 |
| 100     |        2048 |      5,775,238 |  173 |

**BatchSORT tracking (IoU)**. Benchmark for N simultaneously observed objects. The benchmark uses the heuristics that 
separate the observed objects based on object distances.

The benchmark is located at [benches/batch_sort_iou_tracker.rs](benches/batch_sort_iou_tracker.rs).

| Objects |   Time (ns/iter) |  FPS |
|---------|-----------------:|-----:|
| 10      |          106,876 | 9300 |
| 100     |        1,616,542 |  618 |
| 500     |       18,204,723 |   54 |

**BatchSORT tracking (Mahalanobis)**. Benchmark for N simultaneously observed objects. The benchmark uses heuristics 
that separate the observed objects based on object distances.

The benchmark is located at [benches/batch_sort_maha_tracker.rs](benches/batch_sort_maha_tracker.rs).

| Objects | Time (ns/iter) |  FPS |
|---------|---------------:|-----:|
| 10      |        114,592 | 8695 |
| 100     |      1,533,445 |  649 |
| 500     |     17,905,566 |   55 |

