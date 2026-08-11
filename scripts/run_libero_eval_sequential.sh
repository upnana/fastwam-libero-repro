#!/bin/bash
# Sequential LIBERO eval (no tmux). Same worker args as run_libero_parallel_test.sh.
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}
cd "$ROOT_DIR"

CONFIG=${CONFIG:-libero_uncond_2cam224_1e-4}
CKPT=${CKPT:-./checkpoints/fastwam_release/libero_uncond_2cam224.pt}
NUM_TRIALS=${NUM_TRIALS:-50}
OUTPUT_DIR=${OUTPUT_DIR:-./evaluate_results/libero/${CONFIG}_sequential_$(date +%Y%m%d_%H%M%S)}
EXTRA_ARGS=${EXTRA_ARGS:-"EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"}
TASK_SUITES=${TASK_SUITES:-"libero_10 libero_goal libero_spatial libero_object"}

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$ROOT_DIR/checkpoints}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR/task_logs" "$OUTPUT_DIR/task_status"
TASK_FILE="$OUTPUT_DIR/tasks.txt"
: > "$TASK_FILE"
FAILED_FILE="$OUTPUT_DIR/failed_tasks.txt"
: > "$FAILED_FILE"

python - <<PY
from pathlib import Path
from libero.libero.benchmark import get_benchmark_dict
suites = "${TASK_SUITES}".split()
out = Path("${TASK_FILE}")
d = get_benchmark_dict()
with out.open("w") as f:
    for suite in suites:
        b = d[suite]()
        for i in range(int(b.n_tasks)):
            f.write(f"{suite},{i}\n")
print(f"Wrote {out}")
PY

total=$(wc -l < "$TASK_FILE")
idx=0
while IFS=, read -r suite task_id; do
  [ -z "$suite" ] && continue
  idx=$((idx + 1))
  log_file="$OUTPUT_DIR/task_logs/${suite}_task${task_id}.log"
  status_file="$OUTPUT_DIR/task_status/${suite}_task${task_id}.status"
  result_file="$OUTPUT_DIR/${suite}/task${task_id}_results.json"
  echo "[$(date '+%F %T')] ($idx/$total) Starting $suite task_id=$task_id"
  set +e
  CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py \
    task="$CONFIG" ckpt="$CKPT" \
    EVALUATION.task_suite_name="$suite" EVALUATION.task_id="$task_id" gpu_id=0 \
    EVALUATION.num_trials="$NUM_TRIALS" EVALUATION.output_dir="$OUTPUT_DIR" \
    $EXTRA_ARGS > "$log_file" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ] && [ -f "$result_file" ]; then
    echo "SUCCESS|0|$rc|$(date +%s)|$log_file" > "$status_file"
    echo "[$(date '+%F %T')] SUCCESS $suite task_id=$task_id"
  else
    echo "FAILED|0|$rc|$(date +%s)|$log_file" > "$status_file"
    echo "$suite,$task_id" >> "$FAILED_FILE"
    echo "[$(date '+%F %T')] FAILED $suite task_id=$task_id rc=$rc (see $log_file)"
  fi
done < "$TASK_FILE"

echo "Done. Output: $OUTPUT_DIR"
if [ -s "$FAILED_FILE" ]; then
  echo "Failed tasks:"
  cat "$FAILED_FILE"
  exit 1
fi
