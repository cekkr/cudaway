#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const std::string test_name = "Vector Addition";
  printBanner(test_name);

  constexpr int num_elements = 1 << 18;
  constexpr size_t bytes = num_elements * sizeof(float);

  std::vector<float> host_a(num_elements);
  std::vector<float> host_b(num_elements);
  std::vector<float> host_c(num_elements);

  std::generate(host_a.begin(), host_a.end(),
                [i = 0]() mutable { return static_cast<float>(i++); });
  std::generate(host_b.begin(), host_b.end(),
                [i = 0]() mutable { return static_cast<float>(2 * i++); });

  float *dev_a = nullptr;
  float *dev_b = nullptr;
  float *dev_c = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_a, bytes));
  CUDA_CHECK(cudaMalloc(&dev_b, bytes));
  CUDA_CHECK(cudaMalloc(&dev_c, bytes));

  CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (num_elements + threads - 1) / threads;
  vectorAdd<<<blocks, threads>>>(dev_a, dev_b, dev_c, num_elements);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost));

  bool passed = true;
  for (int i = 0; i < num_elements; ++i) {
    const float expected = host_a[i] + host_b[i];
    if (std::fabs(host_c[i] - expected) > 1e-5f) {
      std::cerr << "Mismatch at " << i << ": got " << host_c[i]
                << " expected " << expected << std::endl;
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
