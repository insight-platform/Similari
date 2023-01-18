# Similari

[![Rust](https://img.shields.io/crates/d/similari.svg)](https://crates.io/crates/similari)
[![Rust](https://img.shields.io/crates/v/similari.svg)](https://img.shields.io/crates/v/similari.svg)
[![Rust](https://github.com/insight-platform/Similari/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/rust.yml)
[![Rust](https://img.shields.io/github/license/insight-platform/Similari.svg)](https://img.shields.io/github/license/insight-platform/Similari.svg)

[![Docker Rust 1.66](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-rust-1_65.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-rust-1_66.yml)
[![Docker Python 3.8](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_8.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_8.yml)
[![Docker Python 3.9](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_9.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_9.yml)
[![Docker Python 3.10](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_10.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_10.yml)
[![Docker Python 3.11](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_11.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/docker-maturin-python-3_11.yml)

Similari is a Rust framework with Python bindings that helps build sophisticated tracking systems. With Similari one 
can develop highly efficient parallelized [SORT](https://github.com/abewley/sort), [DeepSORT](https://github.com/nwojke/deep_sort), and 
other sophisticated single observer (e.g. Cam) or multi-observer tracking engines.

## Introduction

The primary purpose of Similari is to provide means to build sophisticated in-memory multiple object tracking engines.

The framework helps to build various kinds of tracking and similarity search engines - the simplest one that holds 
vector features and allows comparing new vectors against the ones kept in the database. More sophisticated engines 
operate over tracks - a series of observations for the same feature collected during the lifecycle. Such systems are 
often used in video processing or other systems where the observer receives fuzzy or changing observation results.

## Out-of-The-Box Stuff

Similari is a framework to build custom trackers, however it provides certain algorithms as an end-user functionality:

**Bounding Box Kalman filter**, that predicts rectangular bounding boxes axis-aligned to scene, supports the oriented (rotated) 
bounding boxes as well.

**2D Point Kalman filter**, that predicts 2D point motion.

**2D Point Vector Kalman filter**, that predicts the vector of independent 2D points motion (used in the Keypoint Tracker).

**Bounding box clipping**, that allows calculating the area of intersection for axis-aligned and oriented (rotated) 
bounding boxes.

**Non-Maximum Suppression (NMS)** - filters rectangular bounding boxes co-axial to scene, and supports the oriented 
  bounding boxes.

**SORT tracking** algorithm (axis-aligned and oriented boxes are supported) - IoU and Mahalanobis distances are 
  supported.

**Batch SORT tracking** algorithm (axis-aligned and oriented boxes are supported) - IoU and Mahalanobis distances are 
supported. Batch tracker allows passing multiple scenes to tracker in a single batch and get them back. If the platform
supports batching (like Nvidia DeepStream or Intel DL Streamer) the batch tracker is more beneficial to use.

**VisualSORT tracking** - a DeepSORT-like algorithm (axis-aligned and oriented boxes are supported) - IoU and 
Mahalanobis distances are supported for positional tracking, euclidean, cosine distances are used for visual tracking on 
feature vectors.

**Batch VisualSORT** - batched VisualSORT flavor;


## Applicability Notes

Although Similari allows building various tracking and similarity engines, there are competitive tools that sometime 
(or often) may be more desirable. The section will explain where it is applicable and what alternatives exist.

Similari fits best for the tracking tasks where objects are described by multiple observations for a certain feature 
class, not a single feature vector. Also, their behavior is dynamic - you remove them from the index or modify them as 
often as add new ones. This is a very important point - it is less efficient than tools that work with growing or static 
object spaces.

**Fit**: track the person across the room: person ReID, age/gender, and face features are collected multiple times during 
the tracking and used to merge tracks or provide aggregated results at the end of the track;

**Not fit**: plagiarism database, when a single document is described by a number (or just one) constant ReID vectors, 
documents are added but not removed. The task is to find the top X most similar documents to a checked.

If your task looks like **Not fit**, can use Similari, but you're probably looking for `HNSW` or `NMS` implementations:
* HNSW Rust - [Link](https://github.com/jean-pierreBoth/hnswlib-rs)
* HNSW C/Python - [link](https://github.com/nmslib/hnswlib)
* NMSLib - [link](https://github.com/nmslib/nmslib)

Similari objects support following features:

**Track lifecycle** - the object is represented by its lifecycle (track) - it appears, evolves, and disappears. During 
its lifetime object evolves according to its behavioral properties (attributes, and feature observations).

**Observations** - Similari assumes that an object is observed by an observer entity that collects its features 
(uniform vectors) and custom observation attributes (like GPS or screen box position)multiple times. Those 
features are presented by vectors of float numbers and observation attributes. When the observation happened, the 
track is updated with gathered features. Future observations are used to find similar tracks in the index and merge them.

**Track Attributes** - Arbitrary attributes describe additional track properties aside from feature observations. 
Track attributes is crucial part when you are comparing objects in the wild, because there may be attributes 
disposition when objects are incompatible, like `animal_type` that prohibits you from comparing `dogs` and `cats` 
between each other. Another popular use of attributes is a spatial or temporal characteristic of an object, e.g. objects 
that are situated at distant locations at the same time cannot be compared. Attributes in Similari are dynamic and 
evolve upon every feature observation addition and when objects are merged. They are used in both distance calculations 
and compatibility guessing (which decreases compute space by skipping incompatible objects).

If you plan to use Similari to search in a large index, consider object attributes to split the lookup space. If the 
attributes of the two tracks are not compatible, their distance calculations are skipped.

## Performance

The Similari is fast. It is usually faster than trackers built with Python and NumPy.

To run visual feature calculations performant the framework uses [ultraviolet](https://crates.io/crates/ultraviolet) - 
the library for fast SIMD computations.

Parallel computations are implemented with index sharding and parallel computations based on a dedicated thread workers 
pool.

Vector operations performance depends a lot on the optimization level defined for the build. On low or default 
optimization levels Rust may not use f32 vectorization, so when running benchmarks take care of proper optimization 
levels configured.

### Rust optimizations

Use `RUSTFLAGS="-C target-cpu=native"` to enable all cpu features like AVX, AVX2, etc. It is beneficial to ultraviolet.

Alternatively you can add build instructions to `.cargo/config`:

```
[build]
rustflags = "-C target-cpu=native"
```

Take a look at [benchmarks](benches) for numbers.

### Performance Benchmarks

Some benchmarks numbers are presented here: [Benchmarks](assets/benchmarks/benchmarks.md)

You can run your own benchmarks by:

```
rustup default nightly
cargo bench
```

## Apple Silicone Build Notes

You may need to add following lines into your `~/.cargo/config` to build the code on Apple Silicone:

```
[build]
rustflags = "-C target-cpu=native"

# Apple Silicone fix
[target.aarch64-apple-darwin]
rustflags = [
    "-C", "link-arg=-undefined",
    "-C", "link-arg=dynamic_lookup",
]
```

## Python API

Python interface exposes ready-to-use functions and classes of Similari. As for now, the Python interface provides:
* the Kalman filter for axis-aligned and oriented (rotated) boxes prediction;
* the Kalman filter for 2D point motion prediction;
* the 2D Point Vector Kalman filter, that predicts the vector of independent 2D points motion (used in the Keypoint Tracker);
* NMS (Non-maximum suppression);
* the Sutherland-Hodgman clipping, intersection area for oriented (rotated) boxes;
* SORT with IoU and Mahalanobis metric;
* BatchSORT with IoU and Mahalanobis metric;
* VisualSORT - DeepSORT-like tracker with euclidean/cosine metric for visual features and IoU/Mahalanobis metric 
  for positional tracking (VisualSort).
* BatchVisualSORT - batched VisualSORT flavor;

Python API classes and functions can be explored in the python 
[documentation](assets/documentation/python/api.md) and tiny [examples](python) provided.

There is also [MOTChallenge](python/motchallenge) evaluation kit provided which you can use to simply evaluate trackers 
performance and metrics.

### Build Python API in Docker

You can build the wheel in the Docker and if you want to install it in the host system, 
copy the resulting package to the host system as demonstrated by the following examples.

#### Rust 1.66 Base Image

If you use other rust libraries you may find it beneficial to build with base Rust 
container (and Python 3.8):  

```
docker build -t similari_py -f docker/rust_1.66/Dockerfile .

# optional: copy and install to host system
docker run --rm -it -v $(pwd)/distfiles:/tmp similari_py cp -R /opt/dist /tmp
pip3 install --force-reinstall distfiles/dist/*.whl
```

#### Python 3.8 Base Image

Python 3.8 is still a very frequently used. Here is how to build Similari with it:

```
docker build -t similari_py -f docker/python_3.8/Dockerfile .

# optional: copy and install to host system
docker run --rm -it -v $(pwd)/distfiles:/tmp similari_py cp -R /opt/dist /tmp
pip3 install --force-reinstall distfiles/dist/*.whl
```

#### Python 3.10 Base Image

If you use the most recent Python environment, you can build with base Python container:

```
docker build -t similari_py -f docker/python_3.10/Dockerfile .

# optional: copy and install to host system
docker run --rm -it -v $(pwd)/distfiles:/tmp similari_py cp -R /opt/dist /tmp
pip3 install --force-reinstall distfiles/dist/*.whl
```

**NOTE**: If you are getting the `pip3` error like:

```
ERROR: similari_py-0.22.5-cp38-cp38-manylinux_2_28_x86_64.whl is not a supported wheel on this platform.
```

It means that the Python version in the host system doesn't match to the one that is in the image used
to build the wheel.

### Build Python API in Host System

#### Linux Instruction

0. Install up-to-date Rust toolkit:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup update
```

1. Install build-essential tools `apt install build-essential -y`.

2. Install Python3 (>= 3.8) and the development files (`python3-dev`).

3. Install Maturin:
```
pip3 install --upgrade maturin~=0.13
```

4. **Not in VENV**. Build the python module: 

```
RUSTFLAGS=" -C target-cpu=native -C opt-level=3" maturin build --release --out dist
pip3 install --force-reinstall dist/*.whl
```

4. **In VENV**. Build the python module:

```
RUSTFLAGS=" -C target-cpu=native -C opt-level=3" maturin develop
```

5. Usage examples are located at [python](python).
6. MOT Challenge Docker image for Similari trackers and conventional trackers is [here](python/motchallenge). 
   You can easily build all-in-one Docker image and try ours trackers.

## Manuals and Articles
Collected articles about how the Similari can be used to solve specific problems.

#### Medium.com

* IoU object tracker [example](https://medium.com/@kudryavtsev_ia/high-performance-object-tracking-engine-with-rust-59ccbc79cdb0);
* Re-ID object tracker [example](https://medium.com/@kudryavtsev_ia/feature-based-object-tracker-with-similari-and-rust-25d72d01d2e2);
* SORT object tracker [example](https://medium.com/p/9a1dd18c259c);
* Python SORT object tracker [example](https://medium.com/@kudryavtsev_ia/high-performance-python-sort-tracker-225c2b507562).

## Usage Examples

Take a look at samples in the repo:
* [simple.rs](examples/simple.rs) - an idea of simple usage.
* [track_merging.rs](examples/track_merging.rs) - an idea of intra-cam track merging.
* [incremental_track_build.rs](examples/incremental_track_build.rs) - very simple feature-based tracker.
* [iou_tracker.rs](examples/iou_tracker.rs) - very simple IoU tracker (without Kalman filter).
* [simple_sort_iou_tracker.rs](examples/simple_sort_iou_tracker.rs) - SORT tracker (with Kalman filter, IoU).
* [simple_sort_iou_tracker_oriented.rs](examples/simple_sort_iou_tracker_oriented.rs) - Oriented (rotated) SORT tracker 
  (with Kalman filter, IoU).
* [simple_sort_maha_tracker.rs](examples/simple_sort_maha_tracker.rs) - SORT tracker (with Kalman filter, Mahalanobis).
* [simple_sort_maha_tracker_oriented.rs](examples/simple_sort_maha_tracker_oriented.rs) - Oriented SORT tracker (with Kalman filter, Mahalanobis).
* [middleware_sort_tracker.rs](examples/middleware_sort_tracker.rs) - SORT tracker (with Kalman filter, middleware implementation).
