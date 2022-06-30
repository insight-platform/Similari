# Similari

[![Rust](https://github.com/insight-platform/Similari/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/insight-platform/Similari/actions/workflows/rust.yml)

The purpose of crate is to provide tools to build in-memory vector similarity engines.

Similarity calculation is the important resource demanding task in machine learning and AI systems. Vectors in 
similarity engines are compared by calculating of n-dimensional distance - Euclidian, cosine or another one.
The distance is used to estimate how the vectors are close between each other.

The library helps to build various kinds of similarity engines - the simplest one is that holds vector features and 
allows comparing new vectors against the ones kept in the database. More sophisticated engines operates over tracks - 
series of observations for the same feature collected during the lifecycle. Such kind of systems are often used in video 
processing or other class of systems where observer receives fuzzy or changing observation results.

The crate provides the tools to collect tracks, build track storages, find similar tracks, and merge them. The crate 
doesn't provide any persistence layer yet.

## Performance

To keep the calculations performant the crate uses:
* [rayon](https://docs.rs/rayon/latest/rayon/) - parallel calculations are implemented within track storage operations;
* [nalgebra](https://nalgebra.org/) - fast linear algebra library.

The vector operations performance depends a lot on the optimization level defined for the build. On low or default 
optimization levels Rust may not use f32 vectorization, so when running benchmarks take care of proper 
optimization levels configured.

## Usage Examples

Take a look at [examples/simple.rs](examples/simple.rs) for an idea of simple usage.