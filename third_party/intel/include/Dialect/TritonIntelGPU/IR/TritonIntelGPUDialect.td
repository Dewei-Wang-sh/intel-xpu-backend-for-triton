#ifndef TTIG_DIALECT
#define TTIG_DIALECT

include "mlir/IR/OpBase.td"

def TritonIntelGPU_Dialect : Dialect {
  let name = "ttig";

  let cppNamespace = "::mlir::triton::gpu::intel";

  let description = [{
    Triton Intel GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
  ];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to indicate the minimum subgroup
    /// size supported by the target device.
    static constexpr llvm::StringRef getMinSGSizeAttrName() {
      return "ttig.min_sg_size";
    }

    /// Get the name of the attribute used to indicate whether subgroup 2D block
    /// operations (e.g., 2D block read/write) are available.
    static constexpr llvm::StringRef getSupportSG2DBlockAttrName() {
      return "ttig.support_sg_2d_block";
    }

    /// Get the name of the attribute used to indicate whether the DPAS
    /// instruction is available.
    static constexpr llvm::StringRef getSupportDPASAttrName() {
      return "ttig.support_dpas";
    }

    /// Get the name of the attribute used to indicate whether the BF16 conversion
    /// instruction is available.
    static constexpr llvm::StringRef getSupportBF16ConversionAttrName() {
      return "ttig.support_bf16_conversion";
    }

    /// Get the name of the attribute used to convay information required for lowering
    /// memory operations (e.g. load, prefetches) to 2D block HW instructions.
    static constexpr llvm::StringRef getBlockIOAttrName() {
      return "ttig.block_io";
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
