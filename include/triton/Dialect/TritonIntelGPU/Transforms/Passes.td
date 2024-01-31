#ifndef TRITON_INTEL_GPU_PASSES
#define TRITON_INTEL_GPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonIntelGPUAccelerateMatmul : Pass<"tritonintelgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "intel accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
  }];

  let constructor = "mlir::createTritonIntelGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"deviceArch", "device architecture",
            "mlir::triton::gpu::intel::DeviceArch", /*default*/" mlir::triton::gpu::intel::DeviceArch::UNKNOWN",
            "device architecture">
  ];
}

def TritonIntelGPUDecomposeConversions: Pass<"tritonintelgpu-decompose-conversions", "mlir::ModuleOp"> {
  let summary = "Decompose convert[mma -> blocked] into convert[mma -> shared -> blocked]";

  let description = "Because of the JointMatrix Type is opaque. store it to memory and load it back ";

  let constructor = "mlir::createTritonIntelGPUDecomposeConversionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonIntelGPUBufferDotOperandsInCacheConversions: Pass<"tritonintelgpu-buffer-cache", "mlir::ModuleOp"> {
  let summary = "Use the cache to buffer the dot operands";

  let description = ".....";

  let constructor = "mlir::createTritonIntelGPUBufferDotOperandsInCachePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonIntelGPUPipeline : Pass<"tritonintelgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Intel GPU pipeline";

  let description = [{
    Use the cache prefetch to buffer the `LoadOp` in loops that needed at next iteration
    Replace `LoadOp` in loops by `InsertSliceAsyncOp` instructions that asynchronously construct the data
    needed at the next iteration
  }];

  let constructor = "mlir::createTritonIntelGPUPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::gpu::intel::TritonIntelGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"numWarps", "num-warps",
           "int32_t", /*default*/"4",
           "number of warps per block">,
    Option<"numCTAs", "num-ctas",
           "int32_t", /*default*/"1",
           "number of CTAs per CGA">,
    Option<"computeCapability", "compute-capability",
                "std::map<std::string, int>", /*default*/"std::map<std::string, int>{}",
                "device compute capability">
  ];
}

def TritonIntelGPURewriteTensorPointer : Pass</*cli-arg*/"tritonintelgpu-rewrite-tensor-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores with tensor pointers into legacy load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_ptr` and `tt.advance` into legacy
    semantics. After this pass, `tt.make_tensor_ptr` and `tt.advance` will disappear, and it generates logics to compute
    the pointer/mask/other for each load/store.
  }];

  let constructor = "mlir::createTritonIntelGPURewriteTensorPointerPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    //Option<"computeCapability", "compute-capability",
    //            "std::map<std::string, int>", /*default*/"std::map<std::string, int>{}",
    //            "device compute capability">
  ];
}

#endif
