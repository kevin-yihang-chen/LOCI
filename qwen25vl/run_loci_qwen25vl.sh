#!/bin/bash
# =============================================================================
# Qwen2.5-VL + LOCI on MIMIC-CXR (phrase-level finding lexicon variant)
#
#   sbatch run_loci_qwen25vl.sh         # SLURM
#   bash   run_loci_qwen25vl.sh         # direct shell
#
# Override via env: MODEL=  DATA=  TRIGGER=  OUT=  PYTHON_BIN=
#                   ALPHA= APC= REP= ES= LAYER= BIAS=
#                   MAX_TOKENS= NUM_GPUS=
#                   CANDIDATE_TOPK= MAX_FINDINGS=
# =============================================================================
#SBATCH --nodes=1 --gres=gpu:2 --mail-type=ALL
#SBATCH -t 12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# ---- Environment defaults (override via env) ----------------------------
export HF_HOME="${HF_HOME:-$HOME/Data/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATA="${DATA:-$REPO_ROOT/mimic_cxr_test.json}"
TRIGGER="${TRIGGER:-$SCRIPT_DIR/clinical_trigger_ids_mimic_qwen25vl.pt}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/}"

OUT="${OUT:-results/MIMIC_qwen25vl_loci_new}"
mkdir -p "$OUT"

# ---- Hyperparameters (Final L from sweeps; documented in README) --------
ALPHA="${ALPHA:-0.80}"
APC="${APC:-0.30}"
REP="${REP:-1.2}"
ES="${ES:-1.0}"
LAYER="${LAYER:-18}"
BIAS="${BIAS:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-150}"
NUM_GPUS="${NUM_GPUS:-2}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-30}"
MAX_FINDINGS="${MAX_FINDINGS:-3}"

# ---- Generation (sharded across NUM_GPUS) ------------------------------
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i "$PYTHON_BIN" loci_qwen25vl.py \
    --model-path "$MODEL" --data-path "$DATA" --image-folder "$IMAGE_FOLDER" \
    --trigger-file "$TRIGGER" \
    --output-file "$OUT/results.jsonl" \
    --max-tokens "$MAX_TOKENS" \
    --alpha "$ALPHA" --alpha-apc "$APC" \
    --repetition-penalty "$REP" --bias-weight "$BIAS" \
    --erasure-scale "$ES" --erasure-layer "$LAYER" \
    --concept-pooling last \
    --candidate-topk "$CANDIDATE_TOPK" --max-findings-per-step "$MAX_FINDINGS" \
    --num-chunks "$NUM_GPUS" --chunk-idx "$i" \
    --quiet \
    > "$OUT/runner_log_part${i}.txt" 2>&1 &
done
wait

# ---- Merge per-chunk outputs -------------------------------------------
if compgen -G "$OUT/results_part*.jsonl" > /dev/null; then
  cat "$OUT"/results_part*.jsonl > "$OUT/results.jsonl"
fi
echo "GENERATION DONE: $(wc -l < "$OUT/results.jsonl") samples"

# ---- Evaluate ----------------------------------------------------------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
"$PYTHON_BIN" "$REPO_ROOT/eval/evaluate_reports.py" \
  --result-file "$OUT/results.jsonl" \
  > "$OUT/result.txt" 2>&1

echo "ALL DONE — see $OUT/result.txt"
date
