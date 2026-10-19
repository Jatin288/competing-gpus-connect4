# =============================================================================
# Makefile - Competing GPUs Connect 4 Engine
# =============================================================================

CUDA_PATH   ?= /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc
CXX_FLAGS   := -std=c++14 -O2
NVCC_FLAGS  := $(CXX_FLAGS) -Xcompiler -Wall
ARCH        := -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_75,code=sm_75 \
               -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86
CUDA_LIBS   := -lcudart

SRC_DIR     := src
BIN_DIR     := bin
TARGET      := $(BIN_DIR)/connect4_engine

SRCS        := $(wildcard $(SRC_DIR)/*.cu)

.PHONY: all clean dirs run

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BIN_DIR) docs

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $^ -o $@ $(CUDA_LIBS)
	@echo "Build successful: $@"

clean:
	rm -rf $(BIN_DIR)

run: all
	./bin/connect4_engine
