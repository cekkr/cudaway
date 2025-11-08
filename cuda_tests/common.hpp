#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>

// Simple error checking helper so test programs fail fast with context.
#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t _err = (expr);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "    \
                << cudaGetErrorName(_err) << " (" << cudaGetErrorString(_err)   \
                << ")" << std::endl;                                            \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

inline void printBanner(const std::string &name) {
  std::cout << "===== " << name << " =====" << std::endl;
}

inline void reportValidation(const std::string &name, bool passed) {
  std::cout << (passed ? "[PASS] " : "[FAIL] ") << name << std::endl;
  if (!passed) {
    std::exit(EXIT_FAILURE);
  }
}
