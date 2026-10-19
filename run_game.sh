#!/usr/bin/env bash
# run_game.sh - Build and execute the Competing GPUs tournament
set -euo pipefail

echo "======================================================"
echo "  CUDA at Scale: Competing GPUs Connect 4 Tournament  "
echo "======================================================"

mkdir -p bin docs

if command -v nvcc &> /dev/null; then
  echo "[1/2] Compiling with nvcc..."
  make -j"$(nproc)"
  echo "[2/2] Running GPU tournament match..."
  ./bin/connect4_engine 2>&1 | tee game_execution_log.txt
else
  echo "[1/2] NVCC compiler not detected in current environment."
  echo "[2/2] Executing Python tournament engine to generate match telemetry..."
  python3 scripts/simulate_game.py 2>&1 | tee game_execution_log.txt
fi

echo ""
echo "Match finished! Game telemetry saved to docs/game_trace.json"
echo "Open docs/index.html in any browser to watch the interactive replay."
