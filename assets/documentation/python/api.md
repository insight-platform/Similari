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

[Universal2DBox](https://docs.rs/similari/0.21.3/similari/utils/bbox/struct.Universal2DBox.html) - universal 2D bounding 
box format that represents oriented and axis-aligned bounding boxes.

```python
ubb = Universal2DBox(xc=3.0, yc=4.0, angle=0.0, aspect=1.5, height=5.0)
print(ubb)

ubb = Universal2DBox(3.0, 4.0, 0.0, 1.5, 5.0)
print(ubb)

ubb.rotate(0.5)
ubb.gen_vertices()
print(ubb)

print(ubb.area())
print(ubb.get_radius())
```

[BoundingBox](https://docs.rs/similari/0.21.3/similari/utils/bbox/struct.BoundingBox.html) - convenience class that must 
be transformed to Universal2DBox by calling `as_xyaah()` before passing to any methods.

```python
bb = BoundingBox(left=1.0, top=2.0, width=10.0, height=15.0)
print(bb)

bb = BoundingBox(1.0, 2.0, 10.0, 15.0)
print(bb.left, bb.top, bb.width, bb.height)

universal_bb = bb.as_xyaah()
print(universal_bb)
```

[Polygon](https://docs.rs/similari/0.21.3/similari/utils/clipping/clipping_py/struct.PyPolygon.html) - return type 
for [sutherland_hodgman_clip](https://docs.rs/similari/0.21.3/similari/utils/clipping/clipping_py/fn.sutherland_hodgman_clip_py.html). 
It cannot be created manually, but returned from the function:

```python
bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2.rotate(0.5)

clip = sutherland_hodgman_clip(bbox1, bbox2)
print(clip)
```

### Kalman Filter

[KalmanFilterState](https://docs.rs/similari/0.21.2/similari/utils/kalman/kalman_py/struct.PyKalmanFilterState.html) - predicted or updated 
state of the oriented bounding box for Kalman filter.

[KalmanFilter](https://docs.rs/similari/0.21.2/similari/utils/kalman/kalman_py/struct.PyKalmanFilter.html) - Kalman filter implementation.

```python
f = KalmanFilter()
state = f.initiate(BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah())
state = f.predict(state)
box_xywh = state.bbox()
print(box_xywh)
# if work with oriented box
# import Universal2DBox and use it
#
#box_xyaah = state.universal_bbox()
#print(box_xyaah)

state = f.update(state, BoundingBox(0.2, 0.2, 5.1, 9.9).as_xyaah())
state = f.predict(state)
box_xywh = state.bbox()
print(box_xywh)
```

### Produced Tracks

* [SortTrack](https://docs.rs/similari/0.21.3/similari/trackers/sort/struct.SortTrack.html)
* [WastedSortTrack](https://docs.rs/similari/0.21.3/similari/trackers/sort/struct.PyWastedSortTrack.html)

### Trackers

#### SORT Trackers
* [IoUSort](https://docs.rs/similari/0.21.3/similari/trackers/sort/simple_iou/struct.IoUSort.html)
* [MahaSort](https://docs.rs/similari/0.21.3/similari/trackers/sort/simple_maha/struct.MahaSort.html)

#### Visual SORT Tracker

##### Tracker Configuration

* [VisualSortOptions](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/options/struct.VisualSortOptions.html)
* [VisualMetricType](https://docs.rs/similari/0.21.3/similari/trackers/visual/metric/struct.PyVisualMetricType.html)
* [PositionalMetricType](https://docs.rs/similari/0.21.3/similari/trackers/visual/metric/struct.PyPositionalMetricType.html)

##### Tracker

* [VisualObservation](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/simple_visual_py/struct.PyVisualObservation.html)
* [VisualObservationSet](https://docs.rs/similari/0.21.3/similari/trackers/visual/simple_visual/simple_visual_py/struct.PyVisualObservationSet.html)
* [VisualSort](https://docs.rs/similari/0.21.2/similari/trackers/visual/simple_visual/struct.VisualSort.html)


