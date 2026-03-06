#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    bash "$REPO_ROOT/eval/demo_run_longeval.sh" --cuda 0 --model LoGeR --mode kitti --seq "$seq" --win 32
done

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
    bash "$REPO_ROOT/eval/demo_run_longeval.sh" --cuda 0 --model LoGeR_star --mode kitti --seq "$seq" --win 64
done
