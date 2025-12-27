// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <utility> // for std::move
#include "dense_nn.h"
#include "def.h"
#include "utils.h"
#include "gpu_utils.h"
#include "gpu_dense_nn.cuh"

#include "gpu_sparse_nn.cuh"


// Utility: show GPU summaries only
void report_summary(nvbench::state &state)
{
  // Show GPU time summaries
  state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/stdev/relative").remove_value("hide");

  // Hide CPU summaries
  state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
}

// Benchmark wrapper for GPU GEMM NN
void nvbench_nn_gemm(nvbench::state &state)
{
  state.set_blocking_kernel_timeout(-1);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
  state.set_run_once(true);

  const int tile1 = static_cast<int>(state.get_int64("tile1"));
  const int tile2 = static_cast<int>(state.get_int64("tile2"));
/*
  // Load MNIST data
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels    = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features  = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * (mnistData->n - 1) + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0f;
    }
  }
*/

  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels   = new swiftware::hpp::DenseMatrix(10, 1);
  auto *features = new swiftware::hpp::DenseMatrix(10, mnistData->n - 1);

  for (int i = 0; i < 10; ++i) {  // limit to first 10 samples
    // First column is the label
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    // Remaining columns are pixel values
    for (int j = 1; j < mnistData->n; ++j) {
      features->data[i * (mnistData->n - 1) + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0f;
    }
  }


  auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
  auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
  auto *biasesHidden  = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput  = swiftware::hpp::readCSV("./data/model/biases_output.csv");

  swiftware::hpp::ScheduleParams sp(tile1, tile2);

  swiftware::hpp::DenseMatrix *pred = nullptr;

  state.exec([&](nvbench::launch &launch) {
    cudaStream_t s = launch.get_stream();
    pred = swiftware::hpp::dense_nn_gemm(features, weightsHidden,
                                         weightsOutput, biasesHidden,
                                         biasesOutput, sp);
  });

  if (pred) {
    int correct = 0;
    for (int i = 0; i < labels->m; i++) {
      int predicted = static_cast<int>(pred->data[i]);
      int actual    = static_cast<int>(labels->data[i]);
      std::cout << "Sample " << i  << " -> Predicted: " << predicted  << ", Label: " << actual << std::endl;
      if (predicted == actual) {
        correct++;
      }
    }
    double accuracy = static_cast<double>(correct) / labels->m;

    nvbench::summary acc_summary;
    acc_summary.set_string("name", "Accuracy");
    acc_summary.set_float64("value", accuracy);
    state.add_summary(std::move(acc_summary));

    delete pred;
  }

  delete mnistData;
  delete labels;
  delete features;
  delete weightsOutput;
  delete weightsHidden;
  delete biasesHidden;
  delete biasesOutput;

  report_summary(state);
}

// Benchmark wrapper for GPU GEMV NN
void nvbench_nn_gemv(nvbench::state &state)
{
  state.set_blocking_kernel_timeout(-1);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
  state.set_run_once(true);

  const int vecTile = static_cast<int>(state.get_int64("vecTile"));
/*
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels    = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features  = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * (mnistData->n - 1) + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0f;
    }
  }
*/
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels   = new swiftware::hpp::DenseMatrix(10, 1);
  auto *features = new swiftware::hpp::DenseMatrix(10, mnistData->n - 1);

  for (int i = 0; i < 10; ++i) {  // limit to first 10 samples
    // First column is the label
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    // Remaining columns are pixel values
    for (int j = 1; j < mnistData->n; ++j) {
      features->data[i * (mnistData->n - 1) + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0f;
    }
  }

  auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
  auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
  auto *biasesHidden  = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput  = swiftware::hpp::readCSV("./data/model/biases_output.csv");

  swiftware::hpp::ScheduleParams sp(1, 1, vecTile);

  swiftware::hpp::DenseMatrix *pred = nullptr;

  state.exec([&](nvbench::launch &launch) {
    cudaStream_t s = launch.get_stream();
    pred = swiftware::hpp::dense_nn_gemv(features, weightsHidden,
                                         weightsOutput, biasesHidden,
                                         biasesOutput, sp);
  });

  if (pred) {
    int correct = 0;
    for (int i = 0; i < labels->m; i++) {
      int predicted = static_cast<int>(pred->data[i]);
      int actual    = static_cast<int>(labels->data[i]);
      std::cout << "Sample " << i  << " -> Predicted: " << predicted  << ", Label: " << actual << std::endl;
      if (predicted == actual) {
        correct++;
      }
    }
    double accuracy = static_cast<double>(correct) / labels->m;

    nvbench::summary acc_summary;
    acc_summary.set_string("name", "Accuracy");
    acc_summary.set_float64("value", accuracy);
    state.add_summary(std::move(acc_summary));

    delete pred;
  }

  delete mnistData;
  delete labels;
  delete features;
  delete weightsOutput;
  delete weightsHidden;
  delete biasesHidden;
  delete biasesOutput;

  report_summary(state);
}

 
// =========================================================
// GPU Sparse SPMV Benchmark
// =========================================================

void nvbench_spmv(nvbench::state &state)
{
  state.set_blocking_kernel_timeout(-1);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
  state.set_run_once(true);

  const int m = static_cast<int>(state.get_int64("m"));
  const int n = static_cast<int>(state.get_int64("n"));

  // -----------------------------
  // Build synthetic CSR matrix
  // -----------------------------
  std::vector<int> Ap(m + 1);
  std::vector<int> Ai;
  std::vector<float> Ax;

  int nnz = 0;
  for (int row = 0; row < m; ++row)
  {
    Ap[row] = nnz;
    for (int col = 0; col < n; ++col)
    {
      if ((row + col) % 17 == 0)  // sparse pattern
      {
        Ai.push_back(col);
        Ax.push_back(1.0f);
        nnz++;
      }
    }
  }
  Ap[m] = nnz;

  // vectors
  std::vector<float> x(n, 1.0f);
  std::vector<float> y(m, 0.0f);
  std::vector<float> b(m, 0.0f);

  // -----------------------------
  // GPU allocation
  // -----------------------------
  int *Ap_d, *Ai_d;
  float *Ax_d, *x_d, *y_d, *b_d;

  CUDA_CHECK(cudaMalloc(&Ap_d, (m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ai_d, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ax_d, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_d, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_d, m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, m * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x_d, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y_d, y.data(), m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b.data(), m * sizeof(float), cudaMemcpyHostToDevice));

  // launch params
  int block = 256;
  int grid = (m + block - 1) / block;

  // -----------------------------
  // NVBench timed execution
  // -----------------------------
  state.exec([&](nvbench::launch &launch) {
   swiftware::hpp::spmvKernel<<<grid, block, 0, launch.get_stream()>>>(Ap_d, Ai_d, Ax_d, x_d, y_d, b_d, m);

    CUDA_CHECK(cudaGetLastError());
  });

  cudaFree(Ap_d);
  cudaFree(Ai_d);
  cudaFree(Ax_d);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(b_d);

  report_summary(state);
}


// =========================================================
// GPU Sparse SPMM Benchmark  (1 thread per output row)
// =========================================================


void nvbench_spmm(nvbench::state &state)
{
  state.set_blocking_kernel_timeout(-1);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
  state.set_run_once(true);

  const int m = static_cast<int>(state.get_int64("m"));
  const int k = static_cast<int>(state.get_int64("k"));
  const int n = static_cast<int>(state.get_int64("n"));

  // -----------------------------
  // Build synthetic CSR
  // -----------------------------
  std::vector<int> Ap(m + 1);
  std::vector<int> Ai;
  std::vector<float> Ax;

  int nnz = 0;
  for (int row = 0; row < m; ++row)
  {
    Ap[row] = nnz;
    for (int col = 0; col < k; ++col)
    {
      if ((row + col) % 23 == 0)
      {
        Ai.push_back(col);
        Ax.push_back(1.0f);
        nnz++;
      }
    }
  }
  Ap[m] = nnz;

  std::vector<float> B(k * n, 1.0f);
  std::vector<float> C(m * n, 0.0f);

  // GPU alloc
  int *Ap_d, *Ai_d;
  float *Ax_d, *B_d, *C_d;

  CUDA_CHECK(cudaMalloc(&Ap_d, (m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ai_d, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ax_d, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B_d,  k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C_d,  m * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d,  B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(C_d,  C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (m + block - 1) / block;

  // timed execution
  state.exec([&](nvbench::launch &launch) {
    swiftware::hpp::spmmKernel<<<grid, block, 0, launch.get_stream()>>>(
        Ap_d, Ai_d, Ax_d,
        B_d, C_d,
        m, n, k
    );
    CUDA_CHECK(cudaGetLastError());
  });

  cudaFree(Ap_d);
  cudaFree(Ai_d);
  cudaFree(Ax_d);
  cudaFree(B_d);
  cudaFree(C_d);

  report_summary(state);
}


// =========================================================
// GPU Sparse NN Benchmark (SPMV-based) with Sparsity Sweep
// =========================================================

void nvbench_nn_sparse_spmv(nvbench::state &state)
{
  state.set_blocking_kernel_timeout(-1);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudaStreamDefault));
  state.set_run_once(true);

  const int sparsity = static_cast<int>(state.get_int64("sparsity"));

  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels   = new swiftware::hpp::DenseMatrix(10, 1);
  auto *features = new swiftware::hpp::DenseMatrix(10, mnistData->n - 1);

  for (int i = 0; i < 10; ++i) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    for (int j = 1; j < mnistData->n; ++j) {
      features->data[i * (mnistData->n - 1) + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0f;
    }
  }

  // Try to load pruned weights at target sparsity level
  std::string w1_file = "./data/model/" + std::to_string(sparsity) + "_W1.csv";
  std::string w2_file = "./data/model/" + std::to_string(sparsity) + "_W2.csv";
  
  swiftware::hpp::DenseMatrix *weightsHidden = nullptr;
  swiftware::hpp::DenseMatrix *weightsOutput = nullptr;
  
  // Try to load pruned weights if available, otherwise use full weights
  try {
    weightsHidden = swiftware::hpp::readCSV(w1_file);
    weightsOutput = swiftware::hpp::readCSV(w2_file);
  } catch (...) {
    // Fallback to unpruned weights
    weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
  }

  auto *biasesHidden  = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput  = swiftware::hpp::readCSV("./data/model/biases_output.csv");

  // Convert dense weights to sparse CSR
  swiftware::hpp::CSR *W1_sparse = new swiftware::hpp::CSR(swiftware::hpp::denseToCSR(*weightsHidden));
  swiftware::hpp::CSR *W2_sparse = new swiftware::hpp::CSR(swiftware::hpp::denseToCSR(*weightsOutput));

  swiftware::hpp::DenseMatrix *pred = nullptr;

  // Benchmark loop: process each sample
  for (int sample = 0; sample < 10; ++sample) {
    auto *sample_features = new swiftware::hpp::DenseMatrix(1, features->n);
    for (int j = 0; j < features->n; ++j) {
      sample_features->data[j] = features->data[sample * features->n + j];
    }

    state.exec([&](nvbench::launch &launch) {
      pred = swiftware::hpp::sparseNNGPU(sample_features, W1_sparse, W2_sparse, biasesHidden, biasesOutput);
    });

    delete sample_features;
    if (pred) delete pred;
  }

  // Calculate accuracy
  int correct = 0;
  for (int i = 0; i < 10; ++i) {
    auto *sample_features = new swiftware::hpp::DenseMatrix(1, features->n);
    for (int j = 0; j < features->n; ++j) {
      sample_features->data[j] = features->data[i * features->n + j];
    }
    pred = swiftware::hpp::sparseNNGPU(sample_features, W1_sparse, W2_sparse, biasesHidden, biasesOutput);
    
    int predicted = static_cast<int>(pred->data[0]);
    int actual    = static_cast<int>(labels->data[i]);
    if (predicted == actual) correct++;
    
    delete sample_features;
    delete pred;
  }
  double accuracy = static_cast<double>(correct) / 10.0;
  
  // Calculate sparsity of actual sparse matrices
  int nnz1 = W1_sparse->Ax.size();
  int nnz2 = W2_sparse->Ax.size();
  double actual_sparsity_w1 = 100.0 * (1.0 - static_cast<double>(nnz1) / (W1_sparse->m * W1_sparse->n));
  double actual_sparsity_w2 = 100.0 * (1.0 - static_cast<double>(nnz2) / (W2_sparse->m * W2_sparse->n));
  
  nvbench::summary acc_summary;
  acc_summary.set_string("name", "Accuracy");
  acc_summary.set_float64("value", accuracy);
  state.add_summary(std::move(acc_summary));
  
  nvbench::summary sp1_summary;
  sp1_summary.set_string("name", "W1_Sparsity");
  sp1_summary.set_float64("value", actual_sparsity_w1);
  state.add_summary(std::move(sp1_summary));
  
  nvbench::summary sp2_summary;
  sp2_summary.set_string("name", "W2_Sparsity");
  sp2_summary.set_float64("value", actual_sparsity_w2);
  state.add_summary(std::move(sp2_summary));

  delete mnistData;
  delete labels;
  delete features;
  delete weightsOutput;
  delete weightsHidden;
  delete biasesHidden;
  delete biasesOutput;
  delete W1_sparse;
  delete W2_sparse;

  report_summary(state);
}


// =========================================================
// GPU Raw Sparse Kernel Benchmarks with Sparsity Sweep (Updated from main_bench.cu)
// =========================================================

void nvbench_spmv_sparse_sweep(nvbench::state &state)
{
  const int m = static_cast<int>(state.get_int64("m"));
  const int n = static_cast<int>(state.get_int64("n"));
  const int sparsity = static_cast<int>(state.get_int64("sparsity"));

  std::vector<int> Ap(m + 1);
  std::vector<int> Ai;
  std::vector<float> Ax;

  int nnz = 0;
  float sparsity_ratio = sparsity / 100.0f;
  
  for (int row = 0; row < m; ++row) {
    Ap[row] = nnz;
    for (int col = 0; col < n; ++col) {
      if ((static_cast<float>(rand()) / RAND_MAX) > sparsity_ratio) {
        Ai.push_back(col);
        Ax.push_back(1.0f);
        nnz++;
      }
    }
  }
  Ap[m] = nnz;

  std::vector<float> x(n, 1.0f);
  std::vector<float> y(m, 0.0f);
  std::vector<float> b(m, 0.0f);

  int *Ap_d, *Ai_d;
  float *Ax_d, *x_d, *y_d, *b_d;

  CUDA_CHECK(cudaMalloc(&Ap_d, (m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ai_d, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ax_d, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&x_d, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_d, m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b_d, m * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(x_d, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(y_d, y.data(), m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b.data(), m * sizeof(float), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (m + block - 1) / block;

  state.exec([&](nvbench::launch &launch) {
    swiftware::hpp::spmvKernel<<<grid, block, 0, launch.get_stream()>>>(Ap_d, Ai_d, Ax_d, x_d, y_d, b_d, m);
    CUDA_CHECK(cudaGetLastError());
  });

  CUDA_CHECK(cudaFree(Ap_d));
  CUDA_CHECK(cudaFree(Ai_d));
  CUDA_CHECK(cudaFree(Ax_d));
  CUDA_CHECK(cudaFree(x_d));
  CUDA_CHECK(cudaFree(y_d));
  CUDA_CHECK(cudaFree(b_d));

  report_summary(state);
}


void nvbench_spmm_sparse_sweep(nvbench::state &state)
{
  const int m = static_cast<int>(state.get_int64("m"));
  const int k = static_cast<int>(state.get_int64("k"));
  const int n = static_cast<int>(state.get_int64("n"));
  const int sparsity = static_cast<int>(state.get_int64("sparsity"));

  std::vector<float> B(k * n, 1.0f);
  std::vector<float> C(m * n, 0.0f);

  std::vector<int> Ap(m + 1);
  std::vector<int> Ai;
  std::vector<float> Ax;

  int nnz = 0;
  float sparsity_ratio = sparsity / 100.0f;
  
  for (int row = 0; row < m; ++row) {
    Ap[row] = nnz;
    for (int col = 0; col < k; ++col) {
      if ((static_cast<float>(rand()) / RAND_MAX) > sparsity_ratio) {
        Ai.push_back(col);
        Ax.push_back(1.0f);
        nnz++;
      }
    }
  }
  Ap[m] = nnz;

  int *Ap_d, *Ai_d;
  float *Ax_d, *B_d, *C_d;

  CUDA_CHECK(cudaMalloc(&Ap_d, (m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ai_d, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&Ax_d, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B_d, k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C_d, m * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(C_d, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (m + block - 1) / block;

  state.exec([&](nvbench::launch &launch) {
    swiftware::hpp::spmmKernel<<<grid, block, 0, launch.get_stream()>>>(
        Ap_d, Ai_d, Ax_d,
        B_d, C_d,
        m, n, k
    );
    CUDA_CHECK(cudaGetLastError());
  });

  CUDA_CHECK(cudaFree(Ap_d));
  CUDA_CHECK(cudaFree(Ai_d));
  CUDA_CHECK(cudaFree(Ax_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  report_summary(state);
}


NVBENCH_BENCH(nvbench_spmv)
  .set_name("gpu_spmv")
  .add_int64_axis("m", {256})
  .add_int64_axis("n", {256});

NVBENCH_BENCH(nvbench_spmm)
  .set_name("gpu_spmm")
  .add_int64_axis("m", {128})
  .add_int64_axis("k", {128})
  .add_int64_axis("n", {128});

NVBENCH_BENCH(nvbench_spmv_sparse_sweep)
  .set_name("gpu_spmv_sparse_sweep")
  .add_int64_axis("m", {512})
  .add_int64_axis("n", {512})
  .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});

NVBENCH_BENCH(nvbench_spmm_sparse_sweep)
  .set_name("gpu_spmm_sparse_sweep")
  .add_int64_axis("m", {256})
  .add_int64_axis("k", {256})
  .add_int64_axis("n", {256})
  .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});

// Register NVBench benchmarks
NVBENCH_BENCH(nvbench_nn_gemm)
  .set_name("nn_gpu_gemm")
  .add_int64_axis("tile1", {32})
  .add_int64_axis("tile2", {32});

NVBENCH_BENCH(nvbench_nn_gemv)
  .set_name("nn_gpu_gemv")
  .add_int64_axis("vecTile", {32});

NVBENCH_BENCH(nvbench_nn_sparse_spmv)
  .set_name("nn_gpu_sparse_spmv")
  .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});