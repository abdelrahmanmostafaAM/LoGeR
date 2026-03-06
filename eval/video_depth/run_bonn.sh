#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval/video_depth/run_bonn.sh pi3_5T_48f_ft_cam1_local1_global0_chunkloss_roe_gradclip1_xcloud_win12-4_overlap3-1_tttv8_l1_35_multi4_h280_pe3/run_20251102_0934 --num-processes 4 --window-size 48 --overlap-size 3

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage: eval/video_depth/run_bonn.sh <checkpoint_run_path> [options] [datasets...]

Required positional arguments:
  checkpoint_run_path    Remote run identifier used by download_ckpt_gcp.sh,
                         e.g. pi3_5T_48f_ft_cam1_local1_global0_chunkloss_roe_gradclip1_xcloud_win12-4_overlap3-1_tttv2_l1_35_multi4_h280_pe3/run_20251024_2155

Optional flags:
  --datasets list        Comma-separated dataset names (default: bonn_s1_500)
  --size INT             Input resolution short-side (default: 512)
  --revisit INT          Number of revisits (default: 1)
  --freeze-state         Disable state updates between revisits
  --solve-pose           Enable post-hoc pose solving
  --window-size INT      Override Pi3 window size
  --overlap-size INT     Override Pi3 overlap size
  --num-iterations INT   Override Pi3 decoding iterations
  --causal true|false    Override causal flag
    --sim3 [true|false]    Override Sim(3) merge flag (default true when provided without value)
    --sim3_mean [true|false] Override Sim(3) merge flag with trimmed mean scale (default true when provided without value)
        --se3 [true|false]     Override SE(3) merge flag (default true when provided without value)
    --pi3x [true|false]    Override Pi3X flag (default true when provided without value)
    --pi3x-metric [true|false] Override Pi3X Metric flag (default true when provided without value)
  --epoch19              Use checkpoint_epoch_19.pt instead of latest.pt
  --num-processes INT    Accelerate process count (default: 2)
  --port INT             Accelerate main process port (default: 29556)
  --output-root PATH     Base directory for evaluation outputs (default: ${REPO_ROOT}/eval_results/video_depth)
  --tag NAME             Sub-directory tag under each dataset (default: pi3)
    --weights-path PATH    Explicit checkpoint weights file (skip auto-download)
    --pi3-config PATH      Explicit Pi3 config file (defaults to run's original_config.yaml)
    --no-pi3-config        Disable Pi3 config loading entirely
  --skip-download        Assume checkpoint already downloaded
  -h, --help             Show this message and exit

Datasets can also be provided as additional positional arguments after the checkpoint path.
If no checkpoint path is specified, defaults to runs/run_pi3.
EOF
}

CKPT_RUN=""
DEFAULT_RUN_PATH="runs/run_pi3"
DATASETS=()
SIZE=512
REVISIT=1
FREEZE_STATE=false
SOLVE_POSE=false
WINDOW_SIZE="48"
OVERLAP_SIZE="3"
NUM_ITERATIONS=""
CAUSAL_OVERRIDE=""
SIM3_OVERRIDE=""
SIM3_MEAN_OVERRIDE=""
SE3_OVERRIDE=""
PI3X_OVERRIDE=""
PI3X_METRIC_OVERRIDE=""
NUM_PROCESSES=2
MAIN_PORT=29556
OUTPUT_ROOT=""
OUTPUT_ROOT_SPECIFIED=0
TAG="pi3"
SKIP_DOWNLOAD=0
WEIGHTS_PATH=""
PI3_CONFIG=""
PI3_CONFIG_SPECIFIED=0
PI3_CONFIG_ENABLED=1
IS_HF_MODEL=0
EPOCH19=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT_RUN="$2"; shift 2 ;;
        --datasets)
            IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --size)
            SIZE="$2"; shift 2 ;;
        --revisit)
            REVISIT="$2"; shift 2 ;;
        --freeze-state)
            FREEZE_STATE=true; shift ;;
        --solve-pose)
            SOLVE_POSE=true; shift ;;
        --window-size)
            WINDOW_SIZE="$2"; shift 2 ;;
        --overlap-size)
            OVERLAP_SIZE="$2"; shift 2 ;;
        --num-iterations)
            NUM_ITERATIONS="$2"; shift 2 ;;
        --causal)
            CAUSAL_OVERRIDE="$2"; shift 2 ;;
        --sim3)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                SIM3_OVERRIDE="$2"
                shift 2
            else
                SIM3_OVERRIDE="true"
                shift
            fi ;;
        --sim3_mean)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                SIM3_MEAN_OVERRIDE="$2"
                shift 2
            else
                SIM3_MEAN_OVERRIDE="true"
                shift
            fi ;;
        --se3)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                SE3_OVERRIDE="$2"
                shift 2
            else
                SE3_OVERRIDE="true"
                shift
            fi ;;
        --pi3x)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                PI3X_OVERRIDE="$2"
                shift 2
            else
                PI3X_OVERRIDE="true"
                shift
            fi ;;
        --pi3x-metric)
            if [[ $# -ge 2 && "$2" != -* ]]; then
                PI3X_METRIC_OVERRIDE="$2"
                shift 2
            else
                PI3X_METRIC_OVERRIDE="true"
                shift
            fi ;;
        --epoch19)
            EPOCH19=1; shift ;;
        --num-processes)
            NUM_PROCESSES="$2"; shift 2 ;;
        --port)
            MAIN_PORT="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$2"; OUTPUT_ROOT_SPECIFIED=1; shift 2 ;;
        --tag)
            TAG="$2"; shift 2 ;;
        --weights-path)
            WEIGHTS_PATH="$2"; shift 2 ;;
        --pi3-config)
            PI3_CONFIG="$2"; PI3_CONFIG_SPECIFIED=1; shift 2 ;;
        --no-pi3-config)
            PI3_CONFIG_ENABLED=0; PI3_CONFIG=""; PI3_CONFIG_SPECIFIED=0; shift ;;
        --skip-download)
            SKIP_DOWNLOAD=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            if [[ -z "$CKPT_RUN" ]]; then
                CKPT_RUN="$1"
            else
                DATASETS+=("$1")
            fi
            shift ;;
    esac
done

if [[ -z "$CKPT_RUN" ]]; then
    CKPT_RUN="$DEFAULT_RUN_PATH"
    echo "No checkpoint specified; defaulting to ${CKPT_RUN}."
fi

# Derive run identifier (e.g., run_20251024_2155) from checkpoint path for default outputs
RUN_NAME=$(basename "${CKPT_RUN%/}")
if [[ "$RUN_NAME" != run_* ]]; then
    RUN_NAME="run_${RUN_NAME}"
fi
if [[ $EPOCH19 -eq 1 ]]; then
    RUN_NAME="${RUN_NAME}_epoch19"
fi

PI3X_IS_TRUE=0
PI3X_METRIC_IS_TRUE=0

if [[ -n "$PI3X_OVERRIDE" ]]; then
    val_pi3x=$(echo "$PI3X_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$val_pi3x" == "true" || "$val_pi3x" == "1" || "$val_pi3x" == "yes" ]]; then
        PI3X_IS_TRUE=1
    fi
fi
if [[ -n "$PI3X_METRIC_OVERRIDE" ]]; then
    val_pi3x_metric=$(echo "$PI3X_METRIC_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$val_pi3x_metric" == "true" || "$val_pi3x_metric" == "1" || "$val_pi3x_metric" == "yes" ]]; then
        PI3X_METRIC_IS_TRUE=1
    fi
fi

if [[ $PI3X_METRIC_IS_TRUE -eq 1 ]]; then
    RUN_NAME="${RUN_NAME}_pi3x_metric"
elif [[ $PI3X_IS_TRUE -eq 1 ]]; then
    RUN_NAME="${RUN_NAME}_pi3x"
fi

SIM3_SUFFIX=""
SE3_SUFFIX=""
SIM3_IS_TRUE=0
SE3_IS_TRUE=0
if [[ -n "$SIM3_OVERRIDE" ]]; then
    sim3_lower=$(echo "$SIM3_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$sim3_lower" == "true" || "$sim3_lower" == "1" || "$sim3_lower" == "yes" ]]; then
        SIM3_IS_TRUE=1
        SIM3_SUFFIX="_sim3"
    fi
fi
if [[ -n "$SE3_OVERRIDE" ]]; then
    se3_lower=$(echo "$SE3_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$se3_lower" == "true" || "$se3_lower" == "1" || "$se3_lower" == "yes" ]]; then
        SE3_IS_TRUE=1
        SE3_SUFFIX="_se3"
    fi
fi
if [[ $SIM3_IS_TRUE -eq 1 && $SE3_IS_TRUE -eq 1 ]]; then
    echo "Error: --sim3 and --se3 cannot both be true simultaneously." >&2
    exit 1
fi

SIM3_MEAN_IS_TRUE=0
if [[ -n "$SIM3_MEAN_OVERRIDE" ]]; then
    sim3_mean_lower=$(echo "$SIM3_MEAN_OVERRIDE" | tr '[:upper:]' '[:lower:]')
    if [[ "$sim3_mean_lower" == "true" || "$sim3_mean_lower" == "1" || "$sim3_mean_lower" == "yes" ]]; then
        SIM3_MEAN_IS_TRUE=1
        SIM3_IS_TRUE=1
        SIM3_SUFFIX="_sim3_mean"
    fi
fi

WINDOW_TAG="win${WINDOW_SIZE}o${OVERLAP_SIZE}${SIM3_SUFFIX}${SE3_SUFFIX}"

if [[ "$CKPT_RUN" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    run_tail=$(basename "${CKPT_RUN%/}")
    if [[ "$run_tail" != run_* ]]; then
        IS_HF_MODEL=1
    fi
fi

if [[ $OUTPUT_ROOT_SPECIFIED -eq 0 ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/eval_results/${RUN_NAME}/${WINDOW_TAG}/video_depth"
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("bonn_s1_500")
fi

relative_ckpt_dir="ckpts/${CKPT_RUN}"
config_path="${REPO_ROOT}/${relative_ckpt_dir}/original_config.yaml"
node_dir="${REPO_ROOT}/${relative_ckpt_dir}/checkpoints/node_0"
if [[ $EPOCH19 -eq 1 ]]; then
    weights_path="${node_dir}/checkpoint_epoch_19.pt"
else
    weights_path="${node_dir}/latest.pt"
fi

if [[ $IS_HF_MODEL -eq 1 ]]; then
    weights_path="$CKPT_RUN"
    if [[ $PI3_CONFIG_SPECIFIED -eq 0 ]]; then
        PI3_CONFIG_ENABLED=0
        PI3_CONFIG=""
    fi
    SKIP_DOWNLOAD=1
fi

if [[ "$CKPT_RUN" == "$DEFAULT_RUN_PATH" && -z "$WEIGHTS_PATH" ]]; then
    echo "Mapping ${CKPT_RUN} to default Hugging Face weights yyfz233/Pi3."
    weights_path="yyfz233/Pi3"
    IS_HF_MODEL=1
    SKIP_DOWNLOAD=1
    if [[ $PI3_CONFIG_SPECIFIED -eq 0 ]]; then
        PI3_CONFIG_ENABLED=0
        PI3_CONFIG=""
    fi
fi

if [[ -n "$WEIGHTS_PATH" ]]; then
    weights_path="$WEIGHTS_PATH"
fi

if [[ $PI3_CONFIG_ENABLED -eq 1 && $PI3_CONFIG_SPECIFIED -eq 0 ]]; then
    if [[ -f "$config_path" ]]; then
        PI3_CONFIG="$config_path"
    else
        PI3_CONFIG=""
    fi
fi

download_checkpoint() {
    local run_path="$1"
    echo "Ensuring checkpoint assets for ${run_path}"

    mkdir -p "${REPO_ROOT}/ckpts/${run_path}/checkpoints/node_0"

    local commands=(
        "python ~/Code/Management/syncutil.py download Checkpoints \"${run_path}/checkpoints/node_0\" --force --viscam"
        "python ~/Code/Management/syncutil.py download Checkpoints \"${run_path}/original_config.yaml\" --force --viscam"
        "python ~/Code/Management/syncutil.py download Checkpoints \"${run_path}/checkpoints/node_0\" --force"
        "python ~/Code/Management/syncutil.py download Checkpoints \"${run_path}/original_config.yaml\" --force"
    )

    for cmd in "${commands[@]}"; do
        echo "Running: ${cmd}"
        if ! eval "${cmd}"; then
            echo "Warning: command failed -> ${cmd}" >&2
        fi
    done
}

if [[ $IS_HF_MODEL -eq 0 && $SKIP_DOWNLOAD -eq 0 && -z "$WEIGHTS_PATH" ]]; then
    if [[ -x "${REPO_ROOT}/download_ckpt_gcp.sh" ]]; then
        echo "Invoking download_ckpt_gcp.sh ${CKPT_RUN}"
        if ! "${REPO_ROOT}/download_ckpt_gcp.sh" "${CKPT_RUN}"; then
            echo "download_ckpt_gcp.sh returned a non-zero status, attempting manual sync..." >&2
        fi
    fi

    if [[ ! -f "$config_path" || ! -d "$node_dir" ]]; then
        download_checkpoint "$CKPT_RUN"
    fi
fi

if [[ $PI3_CONFIG_ENABLED -eq 1 && $PI3_CONFIG_SPECIFIED -eq 0 && -z "$PI3_CONFIG" && -f "$config_path" ]]; then
    PI3_CONFIG="$config_path"
fi

if [[ $IS_HF_MODEL -eq 0 ]]; then
    if [[ -z "$WEIGHTS_PATH" && ! -f "$weights_path" ]]; then
        if [[ -d "$node_dir" ]]; then
            candidate=$(find "$node_dir" -maxdepth 1 -type f -name '*.pt' | sort | head -n 1 || true)
            if [[ -n "${candidate:-}" ]]; then
                weights_path="$candidate"
                echo "Using checkpoint file ${weights_path}"
            fi
        fi
    fi

    if [[ ! -f "$weights_path" ]]; then
        echo "Error: checkpoint weights not found under ${node_dir}" >&2
        exit 1
    fi
else
    echo "Using default Pi3 weights from Hugging Face model '${weights_path}'."
fi

if [[ $PI3_CONFIG_ENABLED -eq 1 && -n "$PI3_CONFIG" && ! -f "$PI3_CONFIG" ]]; then
    echo "Warning: Pi3 config not found (${PI3_CONFIG}); continuing without it." >&2
    PI3_CONFIG=""
fi

mkdir -p "$OUTPUT_ROOT"

forward_args=()
if [[ -n "$WINDOW_SIZE" ]]; then forward_args+=(--window_size "$WINDOW_SIZE"); fi
if [[ -n "$OVERLAP_SIZE" ]]; then forward_args+=(--overlap_size "$OVERLAP_SIZE"); fi
if [[ -n "$NUM_ITERATIONS" ]]; then forward_args+=(--num_iterations "$NUM_ITERATIONS"); fi
if [[ -n "$CAUSAL_OVERRIDE" ]]; then forward_args+=(--causal "$CAUSAL_OVERRIDE"); fi
if [[ -n "$SIM3_OVERRIDE" ]]; then forward_args+=(--sim3 "$SIM3_OVERRIDE"); fi
if [[ -n "$SIM3_MEAN_OVERRIDE" ]]; then forward_args+=(--sim3_mean "$SIM3_MEAN_OVERRIDE"); fi
if [[ -n "$SE3_OVERRIDE" ]]; then forward_args+=(--se3 "$SE3_OVERRIDE"); fi
if [[ -n "$PI3X_OVERRIDE" ]]; then forward_args+=(--pi3x "$PI3X_OVERRIDE"); fi
if [[ -n "$PI3X_METRIC_OVERRIDE" ]]; then forward_args+=(--pi3x_metric "$PI3X_METRIC_OVERRIDE"); fi

pi3_flag=()
if [[ $PI3_CONFIG_ENABLED -eq 1 && -n "$PI3_CONFIG" ]]; then
    pi3_flag+=(--pi3_config "$PI3_CONFIG")
fi

for dataset in "${DATASETS[@]}"; do
    output_dir="${OUTPUT_ROOT%/}/${dataset}/${TAG}"
    mkdir -p "$output_dir"

    printf '\n>>> Evaluating %s with checkpoint %s\n' "$dataset" "$CKPT_RUN"
    echo "Results -> ${output_dir}"

    accelerate launch --num_processes "$NUM_PROCESSES" --main_process_port "$MAIN_PORT" \
        eval/video_depth/launch.py \
        --weights "$weights_path" \
        --output_dir "$output_dir" \
        --eval_dataset "$dataset" \
        --size "$SIZE" \
        "${pi3_flag[@]}" \
        "${forward_args[@]}"

    # scale&shift scale metric
    # python eval/video_depth/eval_depth.py \
    # --output_dir "$output_dir" \
    # --eval_dataset "$dataset" \
    # --align "metric"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$dataset" \
    --align "scale"

    # python eval/video_depth/eval_depth.py \
    # --output_dir "$output_dir" \
    # --eval_dataset "$dataset" \
    # --align "scale&shift"
done
