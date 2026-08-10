#!/usr/bin/env bash
# Environment for the ONNX Runtime backend (`--backend ort`) of the Rust engines.
#
# The `ort` crate is built with `load-dynamic`, so it finds libonnxruntime at
# *runtime* via ORT_DYLIB_PATH — we point it at the one in the project venv
# (`onnxruntime-gpu`). Its CUDA and TensorRT execution providers then dlopen
# cuBLAS/cuFFT/cuDNN/nvinfer, which on this box live in the venv's `nvidia/*`
# and `tensorrt_libs` wheels rather than in a system CUDA install — hence the
# LD_LIBRARY_PATH. Without them ORT silently falls back to the CPU provider
# (you'll see "An error occurred when attempting to register
# `CUDAExecutionProvider`" and throughput a few × worse than candle).
#
# Usage:  source scripts/ort-env.sh   (alongside /tmp/cuda_env.sh for a cuda build)
#
# Override VENV to point at a different environment.

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$_repo/python/.venv}"
_site="$VENV/lib/python3.12/site-packages"

if [[ ! -d "$_site/onnxruntime" ]]; then
  echo "ort-env: no onnxruntime in $_site (pip install onnxruntime-gpu)" >&2
  return 1 2>/dev/null || exit 1
fi

export ORT_DYLIB_PATH="$(ls "$_site"/onnxruntime/capi/libonnxruntime.so* 2>/dev/null | head -1)"

# Every CUDA library the CUDA/TensorRT providers dlopen, as shipped by the wheels.
_libs="$_site/onnxruntime/capi:$_site/tensorrt_libs"
for d in "$_site"/nvidia/*/lib; do
  [[ -d "$d" ]] && _libs="$_libs:$d"
done
export LD_LIBRARY_PATH="$_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "ort-env: ORT_DYLIB_PATH=$ORT_DYLIB_PATH"
