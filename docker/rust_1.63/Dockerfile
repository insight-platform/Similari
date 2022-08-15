FROM rust:1.63-bullseye as chef

RUN cargo install cargo-chef --locked
RUN apt-get update && apt-get install -y python3 build-essential python3-dev python3-pip
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN /usr/bin/python3 -m pip install --upgrade maturin~=0.13

FROM chef AS planner
WORKDIR /opt
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
WORKDIR /opt
COPY --from=planner /opt/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .

ENV RUSTFLAGS=" -C target-cpu=native -C opt-level=3"

RUN maturin build --release --out dist
RUN pip3 install dist/*.whl




