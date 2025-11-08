#include "common.hpp"

#include <iomanip>

int main() {
  const std::string test_name = "Device Query";
  printBanner(test_name);

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaErrorNoDevice) {
    std::cout << "No CUDA-capable devices detected." << std::endl;
    reportValidation(test_name, true);
    return 0;
  }
  CUDA_CHECK(err);

  std::cout << "Detected " << device_count << " CUDA device(s)." << std::endl;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Device " << device << ": " << props.name << std::endl;
    std::cout << "  Compute capability : " << props.major << "." << props.minor
              << std::endl;
    std::cout << "  Multi-processor(s) : " << props.multiProcessorCount
              << std::endl;
    std::cout << "  Global memory      : "
              << static_cast<double>(props.totalGlobalMem) / (1 << 30)
              << " GiB" << std::endl;
    std::cout << "  Max threads/block  : " << props.maxThreadsPerBlock
              << std::endl;
    std::cout << "  Warp size          : " << props.warpSize << std::endl;
  }

  reportValidation(test_name, true);
  return 0;
}
