#!/usr/bin/env bash

set -e

apt-get update && apt-get install -y curl python3 build-essential python3-dev python3-pip

curl -o rustup.sh --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs
sh rustup.sh -y
source $HOME/.cargo/env
rustup update
rustc -V

cargo install cargo-chef --locked

/usr/bin/python3 -m pip install --upgrade pip
/usr/bin/python3 -m pip install --upgrade maturin~=0.14
