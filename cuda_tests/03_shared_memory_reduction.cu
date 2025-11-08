#include "common.hpp"

#include <cmath>
#include <vector>

__global__ void reduceKernel(const float *input, float *partials, int n) {
  extern __shared__ float sdata[];
  const int tid = threadIdx.x;
  int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float sum = 0.0f;
  if (idx < n) {
    sum += input[idx];
  }
  if (idx + blockDim.x < n) {
    sum += input[idx + blockDim.x];
  }
  sdata[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = sdata[0];
  }
}

int main() {
  const std::string test_name = "Shared Memory Reduction";
  printBanner(test_name);

  constexpr int num_elements = 1 << 20;
  constexpr size_t bytes = num_elements * sizeof(float);

  std::vector<float> host_input(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    host_input[i] = static_cast<float>((i % 1024) - 512) * 0.5f;
  }

  float reference_sum = 0.0f;
  for (float value : host_input) {
    reference_sum += value;
  }

  float *dev_input = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_input, bytes));
  CUDA_CHECK(
      cudaMemcpy(dev_input, host_input.data(), bytes, cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (num_elements + threads * 2 - 1) / (threads * 2);
  float *dev_partials = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_partials, blocks * sizeof(float)));

  reduceKernel<<<blocks, threads, threads * sizeof(float)>>>(
      dev_input, dev_partials, num_elements);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> host_partials(blocks);
  CUDA_CHECK(cudaMemcpy(host_partials.data(), dev_partials,
                        blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float device_sum = 0.0f;
  for (float value : host_partials) {
    device_sum += value;
  }

  CUDA_CHECK(cudaFree(dev_partials));
  CUDA_CHECK(cudaFree(dev_input));

  const bool passed = std::fabs(device_sum - reference_sum) < 1e-2f;
  if (!passed) {
    std::cerr << "Reference sum: " << reference_sum
              << " device sum: " << device_sum << std::endl;
  }

  reportValidation(test_name, passed);
  return 0;
}
