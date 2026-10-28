#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'Usage: %s INSTALL_DIR WORK_DIR\n' "$0" >&2
    exit 2
fi

install_dir=$(realpath -m -- "$1")
work_dir=$(realpath -m -- "$2")
dealii_version=9.8.0

mkdir -p "$work_dir"
curl --fail --location --retry 3 \
    --output "$work_dir/dealii.tar.gz" \
    "https://dealii.org/downloads/dealii-${dealii_version}.tar.gz"
tar -xzf "$work_dir/dealii.tar.gz" -C "$work_dir"

cmake -S "$work_dir/dealii-${dealii_version}" \
    -B "$work_dir/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DDEAL_II_ALLOW_AUTODETECTION=OFF \
    -DDEAL_II_WITH_LAPACK=ON \
    -DDEAL_II_WITH_TASKFLOW=ON \
    -DDEAL_II_COMPONENT_EXAMPLES=OFF \
    -DDEAL_II_COMPONENT_DOCUMENTATION=OFF

cmake --build "$work_dir/build" --parallel 4
cmake --install "$work_dir/build"
