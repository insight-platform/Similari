# Python API

Python API is generated with PyO3 & Maturin.

## Functions
[nms](https://docs.rs/similari/0.22.0/similari/utils/nms/nms_py/fn.nms_py.html) - non-maximum suppression implementation for oriented or axis-aligned bounding boxes.

```python
bbox1 = (BoundingBox(10.0, 11.0, 3.0, 3.8).as_xyaah(), 1.0)
bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), 0.9)

res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
print(res[0].as_ltwh())
```

[sutherland_hodgman_clip](https://docs.rs/similari/0.22.0/similari/utils/clipping/clipping_py/fn.sutherland_hodgman_clip_py.html) - calculates the resulting polygon for two oriented or axis-aligned bounding boxes.

```python
bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
bbox2 = BoundingBox(0.0, 0.0, 10.0, 5.0).as_xyaah()

clip = sutherland_hodgman_clip(bbox1, bbox2)
print(clip)
```

[intersection_area](https://docs.rs/similari/0.22.0/similari/utils/clipping/clipping_py/fn.intersection_area_py.html) - calculates the area of intersection for two oriented or axis-aligned bounding boxes.

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

[Universal2DBox](https://docs.rs/similari/0.22.0/similari/utils/bbox/struct.Universal2DBox.html) - universal 2D bounding 
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

ubb = Universal2DBox.new_with_confidence(xc=3.0, yc=4.0, angle=0.0, aspect=1.5, height=5.0, confidence=0.85)
print(ubb)
```

[BoundingBox](https://docs.rs/similari/0.22.0/similari/utils/bbox/struct.BoundingBox.html) - convenience class that must 
be transformed to Universal2DBox by calling `as_xyaah()` before passing to any methods.

```python
bb = BoundingBox(left=1.0, top=2.0, width=10.0, height=15.0)
print(bb)

bb = BoundingBox(1.0, 2.0, 10.0, 15.0)
print(bb.left, bb.top, bb.width, bb.height)

universal_bb = bb.as_xyaah()
print(universal_bb)

bb = BoundingBox.new_with_confidence(1.0, 2.0, 10.0, 15.0, 0.95)
print(bb)
```

[Polygon](https://docs.rs/similari/0.22.0/similari/utils/clipping/clipping_py/struct.PyPolygon.html) - return type 
for [sutherland_hodgman_clip](https://docs.rs/similari/0.22.0/similari/utils/clipping/clipping_py/fn.sutherland_hodgman_clip_py.html). 
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
box_ltwh = state.bbox()
print(box_ltwh)
# if work with oriented box
# import Universal2DBox and use it
#
#box_xyaah = state.universal_bbox()
#print(box_xyaah)

state = f.update(state, BoundingBox(0.2, 0.2, 5.1, 9.9).as_xyaah())
state = f.predict(state)
box_ltwh = state.bbox()
print(box_ltwh)
```

### SORT Tracker

[PositionalMetricType](https://docs.rs/similari/0.22.0/similari/trackers/sort/struct.PyPositionalMetricType.html) - enum type that 
allows setting the positional metric used by a tracker. Two positional metrics are supported:
* IoU(threshold) - intersection over union with threshold that defines when the area is too low to merge the track candidate 
  with the track, and it is required to form a new track;
* Mahalanobis - Mahalanobis distance is used to compute the distance between track candidates and
  tracks kept in the store.

```python
metric = PositionalMetricType.iou(threshold=0.3)
metric = PositionalMetricType.maha()
```

#### Produced Tracks

[SortTrack](https://docs.rs/similari/0.22.0/similari/trackers/sort/struct.SortTrack.html) - the calling of the tracker's
`predict` causes the track candidates to be merged with the current tracks or form new tracks. The resulting information 
is returned for each track in the form of the structure `SortTrack`. Fields are accessible by their names.

```python
...
tracks = sort.predict([box])
for t in tracks:
    print(t)
```

Output:

```
SortTrack {
    id: 2862991017811132132,
    epoch: 1,
    predicted_bbox: Universal2DBox {
        xc: 13.5,
        yc: 8.5,
        angle: None,
        aspect: 1.0,
        height: 7.0,
        confidence: 1.0,
        _vertex_cache: None,
    },
    observed_bbox: Universal2DBox {
        xc: 13.5,
        yc: 8.5,
        angle: None,
        aspect: 1.0,
        height: 7.0,
        confidence: 1.0,
        _vertex_cache: None,
    },
    scene_id: 0,
    length: 1,
    voting_type: Positional,
    custom_object_id: None,
}
```

[WastedSortTrack](https://docs.rs/similari/0.22.0/similari/trackers/sort/struct.PyWastedSortTrack.html) - the trackers 
return the structure for the track when it is wasted from the track store. Fields are accessible by their names. Despite
the `SortTrack` the `WastedSortTrack` includes historical data for predictions and observations.

```python
sort.skip_epochs(10)
wasted = sort.wasted()
print(wasted[0])
```

```
PyWastedSortTrack {
    id: 3466318811797522494,
    epoch: 1,
    predicted_bbox: Universal2DBox {
        xc: 13.5,
        yc: 8.5,
        angle: None,
        aspect: 1.0,
        height: 7.0,
        confidence: 1.0,
        _vertex_cache: None,
    },
    observed_bbox: Universal2DBox {
        xc: 13.5,
        yc: 8.5,
        angle: None,
        aspect: 1.0,
        height: 7.0,
        confidence: 1.0,
        _vertex_cache: None,
    },
    scene_id: 0,
    length: 1,
    predicted_boxes: [
        Universal2DBox {
            xc: 13.5,
            yc: 8.5,
            angle: None,
            aspect: 1.0,
            height: 7.0,
            confidence: 1.0,
            _vertex_cache: None,
        },
    ],
    observed_boxes: [
        Universal2DBox {
            xc: 13.5,
            yc: 8.5,
            angle: None,
            aspect: 1.0,
            height: 7.0,
            confidence: 1.0,
            _vertex_cache: None,
        },
    ],
}
```

#### Tracker Usage

[SORT](https://docs.rs/similari/0.22.0/similari/trackers/sort/simple_api/struct.Sort.html) - basic tracker that uses 
only positional information for tracking. The SORT tracker is widely used in the environments with rare or no occlusions 
happen. SORT is a high-performance low-resource tracker. Despite the original SORT, Similari SORT supports
both axis-aligned and oriented (rotated) bounding boxes.

The Similari SORT is able to achieve the following speeds:

| Objects | Time (ms/prediction) | FPS  | CPU Cores |
|---------|----------------------|------|-----------|
| 10      | 0.149                | 6711 | 1         |
| 100     | 1.660                | 602  | 1         |        
| 200     | 4.895                | 204  | 2         |       
| 300     | 8.991                | 110  | 4         |      
| 500     | 17.432               | 57   | 4         |     
| 1000    | 53.098               | 18   | 5         |    

Comparing to a standard Python SORT from the original [repository](https://github.com/abewley/sort), 
the Similari SORT tracker works several times faster:

| Objects | Time (ms/prediction)  | FPS | CPU Cores (NumPy) | Similari Gain |
|---------|-----------------------|-----|-------------------|---------------|
| 10      | 1.588                 | 620 | ALL               | x10.8         |
| 100     | 11.976                | 83  | ALL               | x7.25         |
| 200     | 25.160                | 39  | ALL               | x5.23         |
| 300     | 40.922                | 24  | ALL               | x4.58         |
| 500     | 74.254                | 13  | ALL               | x4.38         |
| 1000    | 162.037               | 6   | ALL               | x3            |

The examples of the tracker usage are located at:
* [SORT IOU](/python/sort_iou.py) - IoU SORT tracker;
* [SORT_IOU_BENCH](/python/sort_iou_bench.py) - IoU SORT tracker benchmark;
* [SORT_MAHA](/python/sort_maha.py) - Mahalanobis SORT tracker;
* [SORT_IOU_ROTATED](/python/sort_iou_rotated.py) - IoU SORT with rotated boxes.

Also, with Similari SORT you can use custom `scene_id` that allows combining several trackers in 
one without the need to create a separate tracker for every object class. There are methods that support
`scene_id` passing:

* [SORT_IOU_SCENE_ID](/python/sort_iou_scene_id.py) - IoU tracker with several scenes.

To increase the performance of the SORT in scenes with large number of objects one can use 
[SpatioTemporalConstraints](https://docs.rs/similari/0.22.0/similari/trackers/spatio_temporal_constraints/struct.SpatioTemporalConstraints.html).

#### Visual SORT Tracker

Visual SORT tracker is DeepSORT flavour with improvements. It uses custom user ReID model and 
positional tracking based on IoU or Mahalanobis distance.

##### Tracker Configuration

* [VisualSortOptions](https://docs.rs/similari/0.22.0/similari/trackers/visual/simple_api/options/struct.VisualSortOptions.html)
* [VisualMetricType](https://docs.rs/similari/0.22.0/similari/trackers/visual/metric/struct.PyVisualMetricType.html)

##### Tracker Usage

* [VisualObservation](https://docs.rs/similari/0.22.0/similari/trackers/visual/simple_api/simple_visual_py/struct.PyVisualObservation.html)
* [VisualObservationSet](https://docs.rs/similari/0.22.0/similari/trackers/visual/simple_api/simple_visual_py/struct.PyVisualObservationSet.html)
* [VisualSort](https://docs.rs/similari/0.22.0/similari/trackers/visual/simple_api/struct.VisualSort.html)


