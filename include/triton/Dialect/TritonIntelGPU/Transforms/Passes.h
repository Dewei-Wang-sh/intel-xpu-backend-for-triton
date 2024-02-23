#ifndef TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITON_INTEL_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"
#include <any>

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

enum class DeviceArch {
  ATS = 0,
  PVC = 1,
  UNKNOWN,
};

} // namespace intel
} // namespace gpu
} // namespace triton

std::unique_ptr<Pass> createTritonIntelGPUAccelerateMatmulPass(
    triton::gpu::intel::DeviceArch arch = triton::gpu::intel::DeviceArch::PVC);

std::unique_ptr<Pass> createTritonIntelGPUPipelinePass(
    int numStages = 2);


/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
