# Similari

[![Rust](https://github.com/insight-platform/Similari/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/rust.yml)
[![Rust](https://img.shields.io/crates/v/similari.svg)](https://img.shields.io/crates/v/similari.svg)
[![Rust](https://img.shields.io/github/license/insight-platform/Similari.svg)](https://img.shields.io/github/license/insight-platform/Similari.svg)

Similari is a framework that helps build sophisticated tracking systems. The most frequently met operations that can be efficiently implemented with Similari - collecting of observable object features, looking for similar objects, and merging them into tracks based on features and attributes.

With Similari one can develop highly efficient parallelized [SORT](https://github.com/abewley/sort), 
[DeepSORT](https://github.com/nwojke/deep_sort), and other sophisticated single observer (e.g. Cam) or multi-observer tracking engines.

## Introduction

The primary purpose of Similari is to provide means to build sophisticated in-memory object tracking engines.

The framework helps to build various kinds of tracking or similarity search engines - the simplest one that holds vector features and allows comparing new vectors against the ones kept in the database. More sophisticated engines operate over tracks - a series of observations for the same feature collected during the lifecycle. Such systems are often used in video processing or other systems where the observer receives fuzzy or changing observation results.

## Applicability Notes

Although Similari allows building various similarity engines, there are competitive tools that sometime (or often) may be more desirable. The section will explain where it is applicable and what alternatives exist.

Similari fits best for the tasks where objects are described by multiple observations for a certain feature class, not a single feature vector. Also, their behavior is dynamic - you remove them from the index or modify them as often as add new ones. This is a very important point - it is less efficient than tools that work with growing or static object spaces.

* **Fit**: track the person across the room: person ReID, age/gender, and face features are collected multiple times during the tracking and used to merge tracks or provide aggregated results at the end of the track;
* **Not fit**: plagiarism database, when a single document is described by a number (or just one) constant ReID vectors, documents are added but not removed. The task is to find the top X most similar documents to a checked.

If your task looks like **Not fit**, can use Similari, but you're probably looking for `HNSW` or `NMS` implementations:
* HNSW Rust - [Link](https://github.com/jean-pierreBoth/hnswlib-rs)
* HNSW C/Python - [link](https://github.com/nmslib/hnswlib)
* NMS - [link](https://github.com/nmslib/nmslib)

Objects in Similari index support following features:

* **Track lifecycle** - the object is represented by its lifecycle (track) - it appears, evolves, and disappears. During its lifetime object evolves according to its behavioral properties (attributes, and feature observations).
* **Feature Observation** - Similari assumes that an object is observed by an observer entity that collects its features multiple times. Those features are presented by vectors of float numbers and observation attributes. When the observation happened, the track is updated with gathered features. Future observations are used to find similar tracks in the index and merge them.
* **Track Attributes** - Arbitrary attributes describe additional track properties aside from feature observations. Attributes is crucial part when you are comparing objects in the wild, because there may be attributes disposition when objects are incompatible, like `animal_type` that prohibits you from comparing `dogs` and `cats` between each other. Another popular use of attributes is a spatial or temporal characteristic of an object, e.g. objects that are situated at distant locations at the same time cannot be compared. Attributes in Similari are dynamic and evolve upon every feature observation addition and when objects are merged. They are used in both distance calculations and compatibility guessing (which decreases compute space by skipping incompatible objects).

If you are planning to use Similari to search in a huge index, consider object attributes to decrease the lookup space. If the attributes of the two tracks are not compatible, their distance calculations are skipped.

## Performance

To keep the calculations performant the framework uses:
* [ultraviolet](https://crates.io/crates/ultraviolet) - fast SIMD computations.

Parallel computations are implemented with index sharding and parallel computations based on a dedicated thread workers pool.

The vector operations performance depends a lot on the optimization level defined for the build. On low or default optimization levels Rust may not use f32 vectorization, so when running benchmarks take care of proper optimization levels configured.

### Rust optimizations

Use `RUSTFLAGS="-C target-cpu=native"` to enable all cpu features like AVX, AVX2, etc. It is beneficial to ultraviolet.

Alternatively you can add build instructions to `.cargo/config`:

```
[build]
rustflags = "-C target-cpu=native"
```

Take a look at [benchmarks](benches) for numbers.

### Numbers

**IoU tracking**. Benchmark for N simultaneously observed objects run on 4 cores of Intel(R) Core(TM) i5-7440HQ CPU 
@ 2.80GHz. The benchmark doesn't use heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/iou_tracker.rs](benches/iou_tracker.rs).

```
10 objects:   261,184 ns/iter    [3800 FPS]
100 objects:  1,440,733 ns/iter  [ 694 FPS]
500 objects:  17,705,508 ns/iter [  57 FPS]
1000 objects: 58,834,824 ns/iter [  17 FPS]
```

**SORT tracking**. Benchmark for N simultaneously observed objects run on 4 cores of Intel(R) Core(TM) i5-7440HQ CPU 
@ 2.80GHz. The benchmark doesn't use heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_tracker.rs](benches/simple_sort_tracker.rs).

```
10 objects:   83,218 ns/iter     [12048 FPS]
100 objects:  2,305,982 ns/iter  [  433 FPS]
500 objects:  24,170,165 ns/iter [   41 FPS]
1000 objects: 83,859,085 ns/iter [   11 FPS]
```

**Oriented SORT tracking**. Benchmark for N simultaneously observed **oriented** objects run on 4 cores of Intel(R) Core
(TM) i5-7440HQ CPU @ 2.80GHz. The benchmark use heuristics that separate the observed objects based on object distances.

The benchmark is located at [benches/simple_sort_tracker_rotated.rs](benches/simple_sort_tracker_rotated.rs).

```
10 objects:   485,236 ns/iter     [2000 FPS]
100 objects:  5,578,176 ns/iter   [ 180 FPS]
500 objects:  47,809,027 ns/iter  [  20 FPS]
1000 objects: 131,859,818 ns/iter [   7 FPS]
```

**Feature (256 @ f32) tracking**. Benchmark for N simultaneously observed objects run on 4 cores of 
Intel(R) Core(TM) i5-7440HQ CPU @ 2.80GHz. The benchmark doesn't use heuristics that separate the observed objects 
based on object distances.

The benchmark located at [benches/feature_tracker.rs](benches/feature_tracker.rs).

```
10 objects:   101,465 ns/iter     [9900 FPS]
100 objects:  4,020,673 ns/iter   [ 250 FPS]
500 objects:  61,716,729 ns/iter  [  16 FPS]
1000 objects: 235,187,877 ns/iter [   4 FPS]
```

## Manuals and Articles
Collected articles about Similari:

* IoU object tracker [example](https://medium.com/@kudryavtsev_ia/high-performance-object-tracking-engine-with-rust-59ccbc79cdb0) at Medium
* Re-ID object tracker [example](https://medium.com/@kudryavtsev_ia/feature-based-object-tracker-with-similari-and-rust-25d72d01d2e2) at Medium

## Usage Examples

Take a look at samples in the repo:
* [examples/simple.rs](examples/simple.rs) for an idea of simple usage.
* [examples/track_merging.rs](examples/track_merging.rs) for an idea of intra-cam track merging.
* [examples/incremental_track_build.rs](examples/incremental_track_build.rs) very simple feature-based tracker.
* [examples/iou_tracker.rs](examples/iou_tracker.rs) very simple IoU tracker (without Kalman filter).
* [examples/simple_sort_tracker.rs](examples/simple_sort_tracker.rs) SORT tracker (with Kalman filter).
* [examples/middleware_sort_tracker.rs](examples/middleware_sort_tracker.rs) SORT tracker (with Kalman filter, 
  middleware implementation).
