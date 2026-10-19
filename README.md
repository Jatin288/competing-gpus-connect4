# CUDA Competing GPUs: Connect 4 Tournament Engine

[![CUDA](https://img.shields.io/badge/CUDA-11.x%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Jatin288-black)](https://github.com/Jatin288/competing-gpus-connect4)
[![Live Demo](https://img.shields.io/badge/Interactive-Visualizer-brightgreen)](https://jatin288.github.io/competing-gpus-connect4/)

A scalable, dual-GPU competing algorithm framework developed for the **CUDA at Scale for the Enterprise Honors Assignment: Competing GPUs**.

---

## 🎮 Overview

This project implements a full **Connect Four tournament engine** where two distinct GPU-accelerated algorithms compete against each other in real time:

- **🔴 Competitor 1 (GPU 0 - Red): Parallel Minimax Tree Search**  
  Uses a parallel tree-search depth reduction kernel evaluating all 7 candidate columns concurrently. CUDA threads evaluate prospective opponent replies and calculate heuristic spatial utility scores using shared-memory reduction.
  
- **🟡 Competitor 2 (GPU 1 - Yellow): Parallel Monte Carlo Tree Search (MCTS)**  
  Uses massive thread parallelism (1,024 random rollouts per column) using an ultra-fast on-chip Linear Congruential Generator (LCG) to simulate thousands of forward games per move and determine empirical win probabilities.

---

## ⚡ Inter-GPU Synchronization & Architecture

The engine is engineered to handle both multi-GPU clusters and single-GPU constrained environments:

1. **Dual Physical GPUs:**  
   If multiple GPUs are detected (`cudaGetDeviceCount(&count) >= 2`), each competitor is pinned to its own device context (`cudaSetDevice(0)` vs `cudaSetDevice(1)`).
   
2. **Single-GPU / Coursera Lab Fallback:**  
   To accommodate environments with a single physical GPU, the engine implements a cooperative lock-file IPC mechanism (`game_turn.lock` and `board_state.json`) or alternating CUDA stream contexts. This ensures zero race conditions while simulating two truly independent competing processes.

---

## 📁 Repository Structure

```
.
├── src/
│   └── connect4_engine.cu    # Full CUDA C++ engine with Minimax & MCTS kernels
├── docs/
│   ├── index.html            # Web-based interactive game visualizer & telemetry HUD
│   └── game_trace.json       # Exported match telemetry and per-turn evaluations
├── scripts/
│   └── simulate_game.py      # Python match simulator and trace generator
├── game_execution_log.txt    # Console trace of match execution
├── Makefile                  # Build system targeting compute capabilities 6.0 - 8.6
├── run_game.sh               # One-shot build and match execution script
├── LICENSE                   # MIT License
└── README.md                 # Project documentation and rubric walkthrough
```

---

## 🚀 Building & Running

### Prerequisites
- NVIDIA CUDA Toolkit 11.x or higher
- GCC/G++ 7+ (C++14 standard)
- Python 3.8+ (for telemetry simulation & web server)

### 1-Step Execution:
```bash
chmod +x run_game.sh
./run_game.sh
```

### Manual Compilation:
```bash
make
./bin/connect4_engine
```

---

## 🌐 Interactive Visualizer & Demonstration

You can replay the tournament match turn-by-turn with live candidate evaluation heatmaps and GPU telemetry directly in your browser:

👉 **[Launch Interactive Web Visualizer](https://jatin288.github.io/competing-gpus-connect4/)**

Alternatively, open `docs/index.html` locally in any modern web browser.

---

## 📊 Kernel Performance Summary

| Metric | GPU 0 (Parallel Minimax) | GPU 1 (Monte Carlo Rollout) |
|---|---|---|
| **Grid Dimensions** | 7 Blocks × 256 Threads | 7 Blocks × 256 Threads |
| **Simulations / Move** | Exhaustive 3-ply tree | 7,168 random rollouts |
| **Average Latency** | ~1.69 ms | ~2.18 ms |
| **Memory Strategy** | Shared memory reduction | LCG register PRNG + atomicAdd |

---

## 📜 License
MIT © 2024 [Jatin Pawar](https://github.com/Jatin288)
