# Similari

[![Rust](https://img.shields.io/crates/d/similari.svg)](https://crates.io/crates/similari)
[![Rust](https://img.shields.io/crates/v/similari.svg)](https://img.shields.io/crates/v/similari.svg)
[![Rust](https://github.com/insight-platform/Similari/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/rust.yml)
[![Rust](https://img.shields.io/github/license/insight-platform/Similari.svg)](https://img.shields.io/github/license/insight-platform/Similari.svg)

Similari is a framework that helps build sophisticated tracking systems. The operations that can be 
efficiently implemented with Similari are collecting of observable object features, looking for similar objects, and 
merging them into tracks based on features and attributes.

With Similari one can develop highly efficient parallelized [SORT](https://github.com/abewley/sort), 
[DeepSORT](https://github.com/nwojke/deep_sort), and other sophisticated single observer (e.g. Cam) or multi-observer 
tracking engines.

Language bindings:
* Rust - native support;
* Python3 - ready-to-use functions and objects.

## Introduction

The primary purpose of Similari is to provide means to build sophisticated in-memory object tracking engines.

The framework helps to build various kinds of tracking or similarity search engines - the simplest one that holds vector 
features and allows comparing new vectors against the ones kept in the database. More sophisticated engines operate over 
tracks - a series of observations for the same feature collected during the lifecycle. Such systems are often used in video 
processing or other systems where the observer receives fuzzy or changing observation results.

## Out-of-The-Box Stuff

* **Kalman filter**, that predicts rectangular bounding boxes co-axial to scene, supports the oriented bounding 
  boxes as well.
* **Non-Maximum Suppression (NMS)** - filters rectangular bounding boxes co-axial to scene, and supports the oriented 
  bounding 
  boxes.
* **SORT tracking** algorithm (non-oriented and oriented boxes are supported) - IoU and Mahalanobis distances are 
  supported.

## Applicability Notes

Although Similari allows building various similarity engines, there are competitive tools that sometime (or often) may 
be more desirable. The section will explain where it is applicable and what alternatives exist.

Similari fits best for the tasks where objects are described by multiple observations for a certain feature class, not a 
single feature vector. Also, their behavior is dynamic - you remove them from the index or modify them as often as add new ones. This is a very important point - it is less efficient than tools that work with growing or static object spaces.

**Fit**: track the person across the room: person ReID, age/gender, and face features are collected multiple times during 
the tracking and used to merge tracks or provide aggregated results at the end of the track;

**Not fit**: plagiarism database, when a single document is described by a number (or just one) constant ReID vectors, 
documents are added but not removed. The task is to find the top X most similar documents to a checked.

If your task looks like **Not fit**, can use Similari, but you're probably looking for `HNSW` or `NMS` implementations:
* HNSW Rust - [Link](https://github.com/jean-pierreBoth/hnswlib-rs)
* HNSW C/Python - [link](https://github.com/nmslib/hnswlib)
* NMS - [link](https://github.com/nmslib/nmslib)

Objects in Similari index support following features:

**Track lifecycle** - the object is represented by its lifecycle (track) - it appears, evolves, and disappears. During 
its lifetime object evolves according to its behavioral properties (attributes, and feature observations).

**Feature Observation** - Similari assumes that an object is observed by an observer entity that collects its features 
multiple times. Those features are presented by vectors of float numbers and observation attributes. When the observation 
happened, the track is updated with gathered features. Future observations are used to find similar tracks in the index 
and merge them.

**Track Attributes** - Arbitrary attributes describe additional track properties aside from feature observations. 
Attributes is crucial part when you are comparing objects in the wild, because there may be attributes disposition when 
objects are incompatible, like `animal_type` that prohibits you from comparing `dogs` and `cats` between each other. 
Another popular use of attributes is a spatial or temporal characteristic of an object, e.g. objects that are situated 
at distant locations at the same time cannot be compared. Attributes in Similari are dynamic and evolve upon every 
feature observation addition and when objects are merged. They are used in both distance calculations and compatibility 
guessing (which decreases compute space by skipping incompatible objects).

If you are planning to use Similari to search in a huge index, consider object attributes to decrease the lookup space. 
If the attributes of the two tracks are not compatible, their distance calculations are skipped.

## Performance

To keep the calculations performant the framework uses:
* [ultraviolet](https://crates.io/crates/ultraviolet) - fast SIMD computations.

Parallel computations are implemented with index sharding and parallel computations based on a dedicated thread workers pool.

The vector operations performance depends a lot on the optimization level defined for the build. On low or default 
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

## Build Similari Python Interface

Python interface is an PyO3 bindings to several ready-to-use functions and classes of Similari.

As for now, the Python interface exposes:
* Kalman filter for co-axial and oriented boxes prediction;
* NMS, parallel NMS;
* Sutherland-Hodgman clipping, intersection area for oriented boxes;
* SORT with IoU metric;
* SORT with Mahalanobis metric.

### Build in Docker

```
docker build -t similari_py -f python/Dockerfile .
```

### Build in Host System (Linux)

0. Install Rust 1.62:

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

## Manuals and Articles
Collected articles about how the Similari can be used to solve specific problems.

#### Medium.com

* IoU object tracker [example](https://medium.com/@kudryavtsev_ia/high-performance-object-tracking-engine-with-rust-59ccbc79cdb0);
* Re-ID object tracker [example](https://medium.com/@kudryavtsev_ia/feature-based-object-tracker-with-similari-and-rust-25d72d01d2e2);
* SORT object tracker [example](https://medium.com/p/9a1dd18c259c).

## Usage Examples

Take a look at samples in the repo:
* [simple.rs](examples/simple.rs) - an idea of simple usage.
* [track_merging.rs](examples/track_merging.rs) - an idea of intra-cam track merging.
* [incremental_track_build.rs](examples/incremental_track_build.rs) - very simple feature-based tracker.
* [iou_tracker.rs](examples/iou_tracker.rs) - very simple IoU tracker (without Kalman filter).
* [simple_sort_iou_tracker.rs](examples/simple_sort_iou_tracker.rs) - SORT tracker (with Kalman filter, IoU).
* [simple_sort_iou_tracker_oriented.rs](examples/simple_sort_iou_tracker_oriented.rs) - Oriented SORT tracker (with Kalman filter, IoU).
* [simple_sort_maha_tracker.rs](examples/simple_sort_maha_tracker.rs) - SORT tracker (with Kalman filter, Mahalanobis).
* [simple_sort_maha_tracker_oriented.rs](examples/simple_sort_maha_tracker_oriented.rs) - Oriented SORT tracker (with Kalman filter, Mahalanobis).
* [middleware_sort_tracker.rs](examples/middleware_sort_tracker.rs) - SORT tracker (with Kalman filter, middleware implementation).
