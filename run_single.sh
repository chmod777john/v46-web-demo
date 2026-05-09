#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Single-instance launcher for the MiniCPM-V 4.6 Gradio demo.
#
# Loads BOTH the instruct and the thinking checkpoints onto ONE GPU, then
# starts the Gradio app on $PORT.  No load-balancer, no tmux: foreground
# process, Ctrl-C to stop.
#
# Edit the four variables below to match your machine, then run:
#     bash run_single.sh
# ----------------------------------------------------------------------------
set -euo pipefail

# -------- EDIT ME ------------------------------------------------------------
PYTHON=${PYTHON:-/cache/caitianchi/install/miniconda3/envs/v46/bin/python}
GPU=${GPU:-0}
PORT=${PORT:-8890}
INSTRUCT_PATH=${INSTRUCT_PATH:-/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-instruct}
THINKING_PATH=${THINKING_PATH:-/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-thinking}
# -----------------------------------------------------------------------------

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cd "$HERE/v46"

echo "[run_single] python  : $PYTHON"
echo "[run_single] gpu     : $GPU"
echo "[run_single] port    : $PORT"
echo "[run_single] instruct: $INSTRUCT_PATH"
echo "[run_single] thinking: $THINKING_PATH"
echo

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU \
  "$PYTHON" -u app.py \
  --port "$PORT" \
  --instruct_path "$INSTRUCT_PATH" \
  --thinking_path "$THINKING_PATH"
