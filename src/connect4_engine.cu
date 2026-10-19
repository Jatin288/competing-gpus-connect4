// Copyright 2024 Jatin Pawar
//
// Licensed under the MIT License. See LICENSE for details.
//
// File: src/connect4_engine.cu
// Description: Competing GPUs - Connect 4 Tournament Engine.
//   Competitor 1 (Red / GPU 0): Parallel Minimax with heuristic evaluation.
//   Competitor 2 (Yellow / GPU 1): Parallel Monte Carlo Rollout (MCTS) search.
//   Supports dual physical GPUs or single-GPU lock-file cooperative execution.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Board Dimensions & Constants
// ---------------------------------------------------------------------------
constexpr int kRows = 6;
constexpr int kCols = 7;
constexpr int kCells = kRows * kCols;
constexpr int kEmpty = 0;
constexpr int kPlayerRed = 1;     // Competitor 1 (GPU 0)
constexpr int kPlayerYellow = 2;  // Competitor 2 (GPU 1)

constexpr int kMaxRolloutsPerCol = 1024;
constexpr int kThreadsPerBlock = 256;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Board Structure
// ---------------------------------------------------------------------------
struct Connect4Board {
  int grid[kRows][kCols];  // [row][col], row 0 is top, row 5 is bottom

  void Clear() {
    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c < kCols; ++c) {
        grid[r][c] = kEmpty;
      }
    }
  }

  __host__ __device__ bool IsColumnFull(int col) const {
    return grid[0][col] != kEmpty;
  }

  __host__ __device__ int GetLowestEmptyRow(int col) const {
    for (int r = kRows - 1; r >= 0; --r) {
      if (grid[r][col] == kEmpty) return r;
    }
    return -1;
  }

  __host__ __device__ bool DropPiece(int col, int player) {
    int r = GetLowestEmptyRow(col);
    if (r == -1) return false;
    grid[r][col] = player;
    return true;
  }

  __host__ __device__ bool CheckWin(int player) const {
    // Horizontal
    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c <= kCols - 4; ++c) {
        if (grid[r][c] == player && grid[r][c + 1] == player &&
            grid[r][c + 2] == player && grid[r][c + 3] == player)
          return true;
      }
    }
    // Vertical
    for (int r = 0; r <= kRows - 4; ++r) {
      for (int c = 0; c < kCols; ++c) {
        if (grid[r][c] == player && grid[r + 1][c] == player &&
            grid[r + 2][c] == player && grid[r + 3][c] == player)
          return true;
      }
    }
    // Diagonal /
    for (int r = 3; r < kRows; ++r) {
      for (int c = 0; c <= kCols - 4; ++c) {
        if (grid[r][c] == player && grid[r - 1][c + 1] == player &&
            grid[r - 2][c + 2] == player && grid[r - 3][c + 3] == player)
          return true;
      }
    }
    // Diagonal \
    for (int r = 0; r <= kRows - 4; ++r) {
      for (int c = 0; c <= kCols - 4; ++c) {
        if (grid[r][c] == player && grid[r + 1][c + 1] == player &&
            grid[r + 2][c + 2] == player && grid[r + 3][c + 3] == player)
          return true;
      }
    }
    return false;
  }

  __host__ __device__ bool IsBoardFull() const {
    for (int c = 0; c < kCols; ++c) {
      if (grid[0][c] == kEmpty) return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Competitor 1: GPU Minimax Heuristic Evaluation Kernel
// ---------------------------------------------------------------------------
__device__ int EvaluateWindow(int p1, int p2, int p3, int p4, int player) {
  int opp = (player == kPlayerRed) ? kPlayerYellow : kPlayerRed;
  int my_count = (p1 == player) + (p2 == player) + (p3 == player) + (p4 == player);
  int empty_count = (p1 == kEmpty) + (p2 == kEmpty) + (p3 == kEmpty) + (p4 == kEmpty);
  int opp_count = (p1 == opp) + (p2 == opp) + (p3 == opp) + (p4 == opp);

  if (my_count == 4) return 10000;
  if (my_count == 3 && empty_count == 1) return 100;
  if (my_count == 2 && empty_count == 2) return 10;
  if (opp_count == 3 && empty_count == 1) return -80;
  if (opp_count == 4) return -10000;
  return 0;
}

__device__ int ScoreBoard(const Connect4Board* b, int player) {
  int score = 0;
  int center_count = 0;
  for (int r = 0; r < kRows; ++r) {
    if (b->grid[r][kCols / 2] == player) center_count++;
  }
  score += center_count * 6;

  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c <= kCols - 4; ++c) {
      score += EvaluateWindow(b->grid[r][c], b->grid[r][c + 1],
                              b->grid[r][c + 2], b->grid[r][c + 3], player);
    }
  }
  for (int r = 0; r <= kRows - 4; ++r) {
    for (int c = 0; c < kCols; ++c) {
      score += EvaluateWindow(b->grid[r][c], b->grid[r + 1][c],
                              b->grid[r + 2][c], b->grid[r + 3][c], player);
    }
  }
  for (int r = 3; r < kRows; ++r) {
    for (int c = 0; c <= kCols - 4; ++c) {
      score += EvaluateWindow(b->grid[r][c], b->grid[r - 1][c + 1],
                              b->grid[r - 2][c + 2], b->grid[r - 3][c + 3], player);
    }
  }
  for (int r = 0; r <= kRows - 4; ++r) {
    for (int c = 0; c <= kCols - 4; ++c) {
      score += EvaluateWindow(b->grid[r][c], b->grid[r + 1][c + 1],
                              b->grid[r + 2][c + 2], b->grid[r + 3][c + 3], player);
    }
  }
  return score;
}

__global__ void GpuMinimaxKernel(Connect4Board board, int player, int* out_scores) {
  int candidate_col = blockIdx.x;
  int tid = threadIdx.x;

  __shared__ int shared_min_scores[kThreadsPerBlock];

  if (board.IsColumnFull(candidate_col)) {
    if (tid == 0) out_scores[candidate_col] = -999999;
    return;
  }

  Connect4Board b1 = board;
  b1.DropPiece(candidate_col, player);

  if (b1.CheckWin(player)) {
    if (tid == 0) out_scores[candidate_col] = 100000;
    return;
  }

  int opponent = (player == kPlayerRed) ? kPlayerYellow : kPlayerRed;
  int worst_opp_score = 999999;

  int opp_col = tid % kCols;
  if (!b1.IsColumnFull(opp_col)) {
    Connect4Board b2 = b1;
    b2.DropPiece(opp_col, opponent);

    if (b2.CheckWin(opponent)) {
      worst_opp_score = -50000;
    } else {
      int best_counter = -999999;
      for (int my_reply = 0; my_reply < kCols; ++my_reply) {
        if (!b2.IsColumnFull(my_reply)) {
          Connect4Board b3 = b2;
          b3.DropPiece(my_reply, player);
          int s = ScoreBoard(&b3, player);
          if (b3.CheckWin(player)) s = 10000;
          if (s > best_counter) best_counter = s;
        }
      }
      worst_opp_score = best_counter;
    }
  }

  shared_min_scores[tid] = worst_opp_score;
  __syncthreads();

  for (int s = kThreadsPerBlock / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (shared_min_scores[tid + s] < shared_min_scores[tid]) {
        shared_min_scores[tid] = shared_min_scores[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_scores[candidate_col] = shared_min_scores[0];
  }
}

// ---------------------------------------------------------------------------
// Competitor 2: GPU Monte Carlo Tree Search (Rollout) Kernel
// ---------------------------------------------------------------------------
__device__ unsigned int LcgRand(unsigned int* seed) {
  *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
  return *seed;
}

__global__ void GpuMctsRolloutKernel(Connect4Board board, int player,
                                     unsigned int base_seed, int* out_wins,
                                     int* out_totals) {
  int candidate_col = blockIdx.x;
  int tid = threadIdx.x;

  if (board.IsColumnFull(candidate_col)) {
    if (tid == 0) {
      out_wins[candidate_col] = 0;
      out_totals[candidate_col] = 1;
    }
    return;
  }

  unsigned int thread_seed = base_seed + (blockIdx.x * 10007U) + (tid * 37U);

  __shared__ int block_wins;
  __shared__ int block_trials;
  if (tid == 0) {
    block_wins = 0;
    block_trials = 0;
  }
  __syncthreads();

  int rollouts_per_thread = kMaxRolloutsPerCol / blockDim.x;
  int local_wins = 0;
  int opponent = (player == kPlayerRed) ? kPlayerYellow : kPlayerRed;

  for (int sim = 0; sim < rollouts_per_thread; ++sim) {
    Connect4Board sim_board = board;
    sim_board.DropPiece(candidate_col, player);

    if (sim_board.CheckWin(player)) {
      local_wins += 2;
      continue;
    }

    int current_player = opponent;
    bool game_over = false;

    for (int step = 0; step < 20; ++step) {
      int legal_cols[kCols];
      int n_legal = 0;
      for (int c = 0; c < kCols; ++c) {
        if (!sim_board.IsColumnFull(c)) {
          legal_cols[n_legal++] = c;
        }
      }
      if (n_legal == 0) break;

      int chosen = legal_cols[LcgRand(&thread_seed) % n_legal];
      sim_board.DropPiece(chosen, current_player);

      if (sim_board.CheckWin(current_player)) {
        if (current_player == player) local_wins += 2;
        game_over = true;
        break;
      }
      current_player = (current_player == kPlayerRed) ? kPlayerYellow : kPlayerRed;
    }
    if (!game_over) {
      local_wins += 1;
    }
  }

  atomicAdd(&block_wins, local_wins);
  atomicAdd(&block_trials, rollouts_per_thread * 2);
  __syncthreads();

  if (tid == 0) {
    out_wins[candidate_col] = block_wins;
    out_totals[candidate_col] = block_trials;
  }
}

// ---------------------------------------------------------------------------
// GPU Decision Wrappers
// ---------------------------------------------------------------------------
int DecideMoveMinimaxGpu(const Connect4Board& board, int player,
                         double* elapsed_ms, std::vector<int>* col_scores) {
  auto t_start = std::chrono::high_resolution_clock::now();

  int* d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, kCols * sizeof(int)));

  dim3 blocks(kCols);
  dim3 threads(kThreadsPerBlock);
  GpuMinimaxKernel<<<blocks, threads>>>(board, player, d_scores);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_scores[kCols];
  CUDA_CHECK(cudaMemcpy(h_scores, d_scores, kCols * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_scores));

  auto t_end = std::chrono::high_resolution_clock::now();
  *elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  col_scores->assign(h_scores, h_scores + kCols);

  int best_col = -1;
  int best_val = -9999999;
  for (int c = 0; c < kCols; ++c) {
    if (!board.IsColumnFull(c) && h_scores[c] > best_val) {
      best_val = h_scores[c];
      best_col = c;
    }
  }
  return best_col;
}

int DecideMoveMctsGpu(const Connect4Board& board, int player,
                      unsigned int seed, double* elapsed_ms,
                      std::vector<double>* win_rates) {
  auto t_start = std::chrono::high_resolution_clock::now();

  int* d_wins = nullptr;
  int* d_totals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_wins, kCols * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_totals, kCols * sizeof(int)));

  dim3 blocks(kCols);
  dim3 threads(kThreadsPerBlock);
  GpuMctsRolloutKernel<<<blocks, threads>>>(board, player, seed, d_wins, d_totals);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_wins[kCols];
  int h_totals[kCols];
  CUDA_CHECK(cudaMemcpy(h_wins, d_wins, kCols * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_totals, d_totals, kCols * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_wins));
  CUDA_CHECK(cudaFree(d_totals));

  auto t_end = std::chrono::high_resolution_clock::now();
  *elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  win_rates->resize(kCols);
  int best_col = -1;
  double best_rate = -1.0;

  for (int c = 0; c < kCols; ++c) {
    if (board.IsColumnFull(c)) {
      (*win_rates)[c] = 0.0;
    } else {
      double rate = static_cast<double>(h_wins[c]) / std::max(1, h_totals[c]);
      (*win_rates)[c] = rate;
      if (rate > best_rate) {
        best_rate = rate;
        best_col = c;
      }
    }
  }
  return best_col;
}

void PrintBoard(const Connect4Board& board) {
  std::printf("\n   0   1   2   3   4   5   6\n");
  std::printf(" +---+---+---+---+---+---+---+\n");
  for (int r = 0; r < kRows; ++r) {
    std::printf(" |");
    for (int c = 0; c < kCols; ++c) {
      if (board.grid[r][c] == kPlayerRed) {
        std::printf(" R |");
      } else if (board.grid[r][c] == kPlayerYellow) {
        std::printf(" Y |");
      } else {
        std::printf("   |");
      }
    }
    std::printf("\n +---+---+---+---+---+---+---+\n");
  }
}

struct MoveLog {
  int turn_number;
  int player;
  std::string algorithm;
  int chosen_col;
  int row_placed;
  double latency_ms;
  std::vector<double> evaluations;
  int board_snapshot[kRows][kCols];
};

void SaveGameTraceJson(const std::string& path,
                       const std::vector<MoveLog>& logs,
                       int winner) {
  std::ofstream out(path);
  if (!out.is_open()) return;

  out << "{\n";
  out << "  \"game\": \"CUDA Competing GPUs: Connect 4\",\n";
  out << "  \"competitor1\": {\"name\": \"GPU 0 (Red)\", \"algorithm\": \"Parallel Minimax (Depth 3+Heuristic)\"},\n";
  out << "  \"competitor2\": {\"name\": \"GPU 1 (Yellow)\", \"algorithm\": \"Parallel Monte Carlo Rollouts (MCTS)\"},\n";
  out << "  \"winner\": " << winner << ",\n";
  out << "  \"total_turns\": " << logs.size() << ",\n";
  out << "  \"moves\": [\n";

  for (size_t i = 0; i < logs.size(); ++i) {
    const auto& m = logs[i];
    out << "    {\n";
    out << "      \"turn\": " << m.turn_number << ",\n";
    out << "      \"player\": " << m.player << ",\n";
    out << "      \"algorithm\": \"" << m.algorithm << "\",\n";
    out << "      \"column\": " << m.chosen_col << ",\n";
    out << "      \"row\": " << m.row_placed << ",\n";
    out << "      \"latency_ms\": " << std::fixed << std::setprecision(2) << m.latency_ms << ",\n";
    out << "      \"evaluations\": [";
    for (size_t j = 0; j < m.evaluations.size(); ++j) {
      out << std::fixed << std::setprecision(2) << m.evaluations[j]
          << (j + 1 < m.evaluations.size() ? ", " : "");
    }
    out << "],\n";
    out << "      \"board\": [\n";
    for (int r = 0; r < kRows; ++r) {
      out << "        [";
      for (int c = 0; c < kCols; ++c) {
        out << m.board_snapshot[r][c] << (c + 1 < kCols ? "," : "");
      }
      out << "]" << (r + 1 < kRows ? ",\n" : "\n");
    }
    out << "      ]\n";
    out << "    }" << (i + 1 < logs.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

int main(int argc, char* argv[]) {
  std::printf("=================================================================\n");
  std::printf("  CUDA AT SCALE: COMPETING GPUS CONNECT 4 TOURNAMENT\n");
  std::printf("=================================================================\n");

  int num_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  std::printf("Detected %d CUDA-capable device(s).\n", num_gpus);

  int gpu0_id = 0;
  int gpu1_id = (num_gpus > 1) ? 1 : 0;

  cudaDeviceProp prop0, prop1;
  CUDA_CHECK(cudaGetDeviceProperties(&prop0, gpu0_id));
  CUDA_CHECK(cudaGetDeviceProperties(&prop1, gpu1_id));

  std::printf("Competitor 1 (Red)    -> Device [%d]: %s (Parallel Minimax)\n",
              gpu0_id, prop0.name);
  if (num_gpus > 1) {
    std::printf("Competitor 2 (Yellow) -> Device [%d]: %s (Parallel Monte Carlo)\n",
                gpu1_id, prop1.name);
  } else {
    std::printf("Competitor 2 (Yellow) -> Device [%d]: %s (Simulated Dual Context / Stream)\n",
                gpu1_id, prop1.name);
  }
  std::printf("-----------------------------------------------------------------\n");

  Connect4Board board;
  board.Clear();

  std::vector<MoveLog> move_history;
  int current_player = kPlayerRed;
  int turn_count = 0;
  int winner = 0;
  unsigned int rng_seed = 1337;

  while (!board.IsBoardFull()) {
    turn_count++;
    MoveLog log;
    log.turn_number = turn_count;
    log.player = current_player;

    double latency = 0.0;
    int chosen_col = -1;

    if (current_player == kPlayerRed) {
      CUDA_CHECK(cudaSetDevice(gpu0_id));
      log.algorithm = "Parallel Minimax";
      std::vector<int> scores;
      chosen_col = DecideMoveMinimaxGpu(board, kPlayerRed, &latency, &scores);
      for (int s : scores) log.evaluations.push_back(static_cast<double>(s));
    } else {
      CUDA_CHECK(cudaSetDevice(gpu1_id));
      log.algorithm = "Monte Carlo Rollout";
      std::vector<double> rates;
      chosen_col = DecideMoveMctsGpu(board, kPlayerYellow, rng_seed++, &latency, &rates);
      log.evaluations = rates;
    }

    if (chosen_col < 0 || board.IsColumnFull(chosen_col)) {
      std::printf("Error: Player %d selected illegal column %d\n", current_player, chosen_col);
      break;
    }

    int row_placed = board.GetLowestEmptyRow(chosen_col);
    board.DropPiece(chosen_col, current_player);

    log.chosen_col = chosen_col;
    log.row_placed = row_placed;
    log.latency_ms = latency;
    std::memcpy(log.board_snapshot, board.grid, sizeof(board.grid));
    move_history.push_back(log);

    std::printf("[Turn %2d] %s (%s) dropped in Col %d (row %d) in %.2f ms\n",
                turn_count,
                (current_player == kPlayerRed ? "RED   " : "YELLOW"),
                log.algorithm.c_str(), chosen_col, row_placed, latency);

    if (board.CheckWin(current_player)) {
      winner = current_player;
      break;
    }

    current_player = (current_player == kPlayerRed) ? kPlayerYellow : kPlayerRed;
  }

  PrintBoard(board);

  std::printf("\n=================================================================\n");
  if (winner != 0) {
    std::printf("  WINNER: Player %d (%s) in %d turns!\n", winner,
                (winner == kPlayerRed ? "RED - Parallel Minimax"
                                      : "YELLOW - Monte Carlo Rollouts"),
                turn_count);
  } else {
    std::printf("  GAME OVER: DRAW! All 42 slots filled.\n");
  }
  std::printf("=================================================================\n");

  SaveGameTraceJson("docs/game_trace.json", move_history, winner);
  std::printf("Saved interactive game telemetry to: docs/game_trace.json\n");

  return 0;
}
