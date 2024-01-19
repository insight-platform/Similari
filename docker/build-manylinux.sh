#!/usr/bin/env bash

set -e

ARGS=-f

if [[ -z $PYTHON_INTERPRETER ]]; then
    ARGS=-f
else
    ARGS="-i $PYTHON_INTERPRETER"
fi

echo "Additional build args: $ARGS"

maturin build $ARGS --release --out /opt/dist

