# Makefile for parallel-mlp-mnist project.
# Builds three training variants (sequential, OpenMP, MPI) and unit tests.
#
# Usage:
#   make          - Build all three training binaries
#   make seq      - Build sequential-only binary
#   make omp      - Build OpenMP-parallelized binary
#   make mpi      - Build MPI-parallelized binary
#   make test     - Build and run all unit tests
#   make clean    - Remove all build artifacts

CXX      = g++
MPICXX   = mpicxx
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS  =

SRC_DIR   = src
INC_DIR   = include
BUILD_DIR = build

# Common source files shared by all build targets (excludes train_*.cpp entry points)
COMMON_SRC = $(SRC_DIR)/utils.cpp \
             $(SRC_DIR)/dataset.cpp \
             $(SRC_DIR)/activations.cpp \
             $(SRC_DIR)/loss.cpp \
             $(SRC_DIR)/optimizer.cpp \
             $(SRC_DIR)/mlp.cpp \
             $(SRC_DIR)/metrics.cpp \
             $(SRC_DIR)/timer.cpp

# Object files: separate sets for seq, omp, and mpi to allow different compiler flags
COMMON_OBJ     = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(COMMON_SRC))
COMMON_OBJ_OMP = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%_omp.o, $(COMMON_SRC))
COMMON_OBJ_MPI = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%_mpi.o, $(COMMON_SRC))

# Output binaries
SEQ_BIN = $(BUILD_DIR)/train_seq
OMP_BIN = $(BUILD_DIR)/train_omp
MPI_BIN = $(BUILD_DIR)/train_mpi

.PHONY: all seq omp mpi test clean

all: seq omp mpi

# --- Sequential build (standard g++ compilation) ---
seq: $(SEQ_BIN)

$(SEQ_BIN): $(COMMON_OBJ) $(BUILD_DIR)/train_sequential.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# --- OpenMP build (adds -fopenmp for thread parallelism) ---
omp: $(OMP_BIN)

$(OMP_BIN): $(COMMON_OBJ_OMP) $(BUILD_DIR)/train_openmp_omp.o
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%_omp.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fopenmp -c -o $@ $<

# --- MPI build (uses mpicxx wrapper for MPI-distributed parallelism) ---
mpi: $(MPI_BIN)

$(MPI_BIN): $(COMMON_OBJ_MPI) $(BUILD_DIR)/train_mpi_mpi.o
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%_mpi.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -c -o $@ $<

# --- Tests (linked against sequential common objects) ---
TEST_SRC = $(wildcard tests/test_*.cpp)
TEST_BIN = $(patsubst tests/%.cpp, $(BUILD_DIR)/%, $(TEST_SRC))

test: $(TEST_BIN)
	@for t in $(TEST_BIN); do echo "Running $$t ..."; $$t || exit 1; done

$(BUILD_DIR)/test_%: tests/test_%.cpp $(COMMON_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# --- Utilities ---
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*
