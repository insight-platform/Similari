# Python API

Python API is generated with PyO3 & Maturin.

## Functions
[nms](https://docs.rs/similari/0.21.3/similari/utils/nms/nms_py/fn.nms_py.html) - non-maximum suppression implementation for oriented or axis-aligned bounding boxes.

```python
bbox1 = (BoundingBox(10.0, 11.0, 3.0, 3.8).as_xyaah(), 1.0)
bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), 0.9)

res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
print(res[0].as_xywh())
```

[sutherland_hodgman_clip](https://docs.rs/similari/0.21.3/similari/utils/clipping/clipping_py/fn.sutherland_hodgman_clip_py.html) - calculates the resulting polygon for two oriented or axis-aligned bounding boxes.

```python
bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2 = BoundingBox(0.0, 0.0, 10.0, 5.0).as_xyaah()

clip = sutherland_hodgman_clip(bbox1, bbox2)
print(clip)
```

[intersection_area](https://docs.rs/similari/0.21.3/similari/utils/clipping/clipping_py/fn.intersection_area_py.html) - calculates the area of intersection for two oriented or axis-aligned bounding boxes.

```python
bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2.rotate(0.5)

clip = sutherland_hodgman_clip(bbox1, bbox2)
print(clip)

area = intersection_area(bbox1, bbox2)
print("Intersection area:", area)
```

## Classes

### Areas

* [Universal2DBox](https://docs.rs/similari/0.21.3/similari/utils/bbox/struct.Universal2DBox.html)
* [BoundingBox](https://docs.rs/similari/0.21.3/similari/utils/bbox/struct.BoundingBox.html)
* [Polygon](https://docs.rs/similari/0.21.3/similari/utils/clipping/clipping_py/struct.PyPolygon.html)

### Produced Tracks

* [SortTrack](https://docs.rs/similari/0.21.3/similari/trackers/sort/struct.SortTrack.html)
* [WastedSortTrack](https://docs.rs/similari/0.21.3/similari/trackers/sort/struct.PyWastedSortTrack.html)

### Kalman Filter

* [KalmanFilterState](https://docs.rs/similari/0.21.2/similari/utils/kalman/kalman_py/struct.PyKalmanFilterState.html)
* [KalmanFilter](https://docs.rs/similari/0.21.2/similari/utils/kalman/kalman_py/struct.PyKalmanFilter.html)

### Trackers

* [IoUSort](https://docs.rs/similari/0.21.3/similari/trackers/sort/simple_iou/struct.IoUSort.html)
* [MahaSort](https://docs.rs/similari/0.21.3/similari/trackers/sort/simple_maha/struct.MahaSort.html)
* [VisualSort](https://docs.rs/similari/0.21.2/similari/trackers/visual/simple_visual/struct.VisualSort.html)
* [VisualSortOptions](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/options/struct.VisualSortOptions.html)
* [VisualMetricType](https://docs.rs/similari/0.21.3/similari/trackers/visual/metric/struct.PyVisualMetricType.html)
* [PositionalMetricType](https://docs.rs/similari/0.21.3/similari/trackers/visual/metric/struct.PyPositionalMetricType.html)
* [VisualObservation](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/simple_visual_py/struct.PyVisualObservation.html)
* [VisualObservationSet](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/simple_visual_py/struct.PyVisualObservationSet.html)


