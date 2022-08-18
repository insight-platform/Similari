FROM python:3.8-buster as base

WORKDIR /opt
COPY docker/common/install-basic-deps.sh .
RUN bash /opt/install-basic-deps.sh

FROM base as chef
ENV PATH="/root/.cargo/bin:$PATH"
RUN rustc -V

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
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install dist/*.whl

RUN python3 -m pip install -r /opt/python/motchallenge/requirements.txt

# add original Sort (https://github.com/abewley/sort)
RUN git clone https://github.com/abewley/sort.git /tmp/sort \
    && python3 -m pip install -r /tmp/sort/requirements.txt \
    && cp /tmp/sort/sort.py /opt/python/motchallenge/original_sort.py \
    && rm -r /tmp/sort

# MOTChallenge Official Evaluation Kit
# https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official
RUN git clone https://github.com/JonathonLuiten/TrackEval.git /opt/TrackEval
WORKDIR /opt/TrackEval
VOLUME /opt/TrackEval/data
#ENTRYPOINT ['python3', 'run.py']
#CMD ['det_path', 'res_path', 'IoUSort', '...']
