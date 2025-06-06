#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

//===----------------------------------------------------------------------===//
// Intel DPAS Layout Encoding
//===----------------------------------------------------------------------===//

def DpasEncodingAttr : DistributedEncoding<"DpasEncoding", "intel_dpas_encoding",
                                          [MmaEncodingTrait], TritonIntelGPU_Dialect> {
  let mnemonic = "dpas";

  let description = [{
An encoding for the tensors distributed across the threads for the C and D operands of XMX tensor core operation.
The XMX tensor core operation is defined for matrix matmul as: D=A*B+C
The shape of the of XMX tensor core operation is defined by systolic depth, repeat count, execution size and operations per channel.

The encoding is characterized by parameters:
        - `repeatCount` which shall be in the range [1, 8]
        - `systolicDepth` For PVC/ATSM, the size is 8.
        - `executionSize` For PVC, the size is 16. For ATSM, the size is 8.
        - `opsPerChannel` 4 for 8 bit scalar type, 2 for 16 bit scalar type, 1 for 32 bit scalar type.
        - `warpsPerCTA`
        - `sugGroupSize` valid sub group size is 8/16/32


The layout example repeat_count=8, systolic_depth=8, execution_size=16 and operands_per_chan=2 for warp size 32.
For A operand:
                                   systolic depth = 8
<------------------------------------------------------------------------------------------------->
opsPerChan=2
<--------->
t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    ^
t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   | repeat count <= 8
t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    |
t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   v

For B operand:
               execution size = 16
<------------------------------------------------------------->
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     ^             ^
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .       | opsPerChan=2|
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     v             |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |  systolic depth = 8
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   v

This pattern repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
along the row (resp. col) dimension.
}];

  let parameters = (
    ins
    "unsigned":$repeatCount,
    "unsigned":$systolicDepth,
    "unsigned":$executionSize,
    "unsigned":$opsPerChannel,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    ArrayRefParameter<"unsigned">:$repCluster,
    "unsigned":$subGroupSize
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    enum class OpIdx : unsigned {
      OperandA = 0u,
      OperandB = 1u,
      OperandC = 2u
    };

    SmallVector<unsigned> getDPASInstShapeA() const;
    SmallVector<unsigned> getDPASInstShapeB() const;
    SmallVector<unsigned> getDPASInstShapeC() const;
    SmallVector<unsigned> getShapeA() const;
    SmallVector<unsigned> getShapeB() const;
    SmallVector<unsigned> getShapeC() const;

    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, OpIdx opIdx) const;
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth, OpIdx opIdx) const;
    SmallVector<unsigned> getElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, OpIdx opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(OpIdx opIdx) const;
    SmallVector<unsigned> getThreadsPerWarpForOperand(int opIdx) const;
    unsigned getTotalElemsPerThreadForOperand(ArrayRef<int64_t> shape, Type eltTy, int kWidth, OpIdx opIdx) const;

    // Forwarder functions for casting unsigned to OpIdx.
    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, unsigned opIdx) const {
      return getDPASRepetitions(shape, static_cast<OpIdx>(opIdx));
    }
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth, unsigned opIdx) const {
      return getSizePerThreadForOperand(kWidth, static_cast<OpIdx>(opIdx));
    }
    SmallVector<unsigned> getRepOrderForOperand(unsigned opIdx) const {
      return getRepOrderForOperand(static_cast<OpIdx>(opIdx));
    }
    unsigned getTotalElemsPerThreadForOperand(ArrayRef<int64_t> shape, Type eltTy, int kWidth, unsigned opIdx) const {
      return getTotalElemsPerThreadForOperand(shape, eltTy, kWidth, static_cast<OpIdx>(opIdx));
    }

    bool supportReduction() const {
      return true;
    }

    SmallVector<unsigned> getContigPerThread() const;

    struct DPASCapability {
      explicit DPASCapability(unsigned minSGSize) : executionSize(minSGSize) {}
      DPASCapability() = default;

      bool isPVC() const {
        return executionSize == 16;
      }
      bool isFalconShore() const {
        return executionSize == 16;
      }
      bool isATSM() const {
        return executionSize == 8;
      }

      static constexpr unsigned systolicDepth = 8u;
      static constexpr unsigned repeatCount = 8u;
      static constexpr unsigned opsChanBitWidths = 32u;
      unsigned executionSize = 0u;
    };

    static DPASCapability getDPASCapability(mlir::ModuleOp mod);
    static unsigned getOpsPerChannel(Type elemType);
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Intel Warp Encoding
//===----------------------------------------------------------------------===//

def WarpEncodingAttr : TritonGPU_Attr<"WarpEncoding", "intel_warp_encoding",
                                     [], TritonIntelGPU_Dialect> {
  let mnemonic = "warp";

  let description = [{
   An encoding characterized by two tuples -- thread tile size and warp tile size
   which specify the amount of elements owned by each thread and warp respectively.
   currently all their meaning remain the same as above blocked encoding.
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$order // the fastest-changing axis first
  );

  let extraClassDeclaration = [{
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif
