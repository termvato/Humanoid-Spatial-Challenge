#!/usr/bin/env bash
# S9 sweep: build 3D-first predictions for a scene list over configs, evaluate each with the official evaluator.
# Usage: tools/s9_sweep.sh <scenes_file> <tag_prefix> "<name>=<bench_predict args>;..."
# Predictions run in parallel; Modal evaluation runs one at a time (concurrent `modal run` lost results).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
scenes=$(cat "$1"); prefix=$2
IFS=';' read -ra configs <<< "$3"
for c in "${configs[@]}"; do
  name=${c%%=*}; args=${c#*=}; tag=${prefix}_${name}
  uv run --group gpu python tools/bench_predict.py --scenes "$scenes" $args --out outputs/s9/$tag > outputs/s9/$tag.log 2>&1 &
done
wait
for c in "${configs[@]}"; do
  name=${c%%=*}; tag=${prefix}_${name}
  uv run modal volume put --force scenediff outputs/s9/$tag/pred outputs/$tag/pred > /dev/null 2>&1
  uv run modal run infra/modal_scenediff.py::evalpred --tag "$tag" --scenes "$scenes" > outputs/s9/$tag.eval.log 2>&1 || true
  grep -aE "Metric" outputs/s9/$tag.eval.log | head -4 > outputs/s9/$tag.metrics.txt || true
  echo "$tag: $(sed 's/.*: //' outputs/s9/$tag.metrics.txt | paste -sd' ')"
done
