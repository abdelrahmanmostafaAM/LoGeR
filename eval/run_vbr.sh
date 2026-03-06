#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seq in campus_train1 diag_train0 colosseo_train0 pincio_train0 ciampino_train1 spagna_train0 campus_train0; do
    bash "$REPO_ROOT/eval/demo_run_longeval.sh" --cuda 0 --model LoGeR --mode vbr --seq "$seq" --win 48
done

for seq in campus_train1 diag_train0 colosseo_train0 pincio_train0 ciampino_train1 spagna_train0 campus_train0; do
    bash "$REPO_ROOT/eval/demo_run_longeval.sh" --cuda 0 --model LoGeR_star --mode vbr --seq "$seq" --win 64
done
