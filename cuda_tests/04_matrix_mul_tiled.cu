#include "common.hpp"

#include <cmath>
#include <vector>

constexpr int TILE_DIM = 16;

__global__ void matMulTiled(const float *a, const float *b, float *c, int m,
                            int n, int k) {
  __shared__ float tile_a[TILE_DIM][TILE_DIM];
  __shared__ float tile_b[TILE_DIM][TILE_DIM];

  const int row = blockIdx.y * TILE_DIM + threadIdx.y;
  const int col = blockIdx.x * TILE_DIM + threadIdx.x;

  float accum = 0.0f;
  for (int tile = 0; tile < (k + TILE_DIM - 1) / TILE_DIM; ++tile) {
    const int tiled_col = tile * TILE_DIM + threadIdx.x;
    const int tiled_row = tile * TILE_DIM + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && tiled_col < k) ? a[row * k + tiled_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (tiled_row < k && col < n) ? b[tiled_row * n + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE_DIM; ++i) {
      accum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = accum;
  }
}

int main() {
  const std::string test_name = "Tiled Matrix Multiply";
  printBanner(test_name);

  constexpr int m = 128;
  constexpr int n = 128;
  constexpr int k = 128;

  std::vector<float> host_a(m * k);
  std::vector<float> host_b(k * n);
  std::vector<float> host_c(m * n);
  std::vector<float> reference_c(m * n, 0.0f);

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < k; ++col) {
      host_a[row * k + col] = static_cast<float>((row + col) % 13) * 0.25f;
    }
  }
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; ++col) {
      host_b[row * n + col] = static_cast<float>((row - col) % 17) * 0.1f;
    }
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float accum = 0.0f;
      for (int i = 0; i < k; ++i) {
        accum += host_a[row * k + i] * host_b[i * n + col];
      }
      reference_c[row * n + col] = accum;
    }
  }

  float *dev_a = nullptr;
  float *dev_b = nullptr;
  float *dev_c = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_a, host_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_b, host_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev_c, host_c.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 threads(TILE_DIM, TILE_DIM);
  dim3 blocks((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);
  matMulTiled<<<blocks, threads>>>(dev_a, dev_b, dev_c, m, n, k);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool passed = true;
  for (size_t idx = 0; idx < host_c.size(); ++idx) {
    if (std::fabs(host_c[idx] - reference_c[idx]) > 1e-1f) {
      std::cerr << "Mismatch at " << idx << ": got " << host_c[idx]
                << " expected " << reference_c[idx] << std::endl;
      passed = false;
      break;
    }
  }

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));

  reportValidation(test_name, passed);
  return 0;
}
