#include "TypeConverter.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

TritonIntelGPUToLLVMTypeConverter::TritonIntelGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
}

Type TritonIntelGPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (pointeeType.isa<RankedTensorType>()) {
    auto rankedTensorType = pointeeType.cast<RankedTensorType>();
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, type.getAddressSpace()));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
}

Type TritonIntelGPUToLLVMTypeConverter::getElementTypeForStruct(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  auto dotOpLayout = layout.dyn_cast<DotOperandEncodingAttr>();
  if (!dotOpLayout)
    return elemTy;

  if (auto dpasParent = dotOpLayout.getParent().dyn_cast<DpasEncodingAttr>()) {
    unsigned RC = dpasParent.getRepeatCount();
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();

    if (dotOpLayout.getOpIdx() == 0) {
      //  Elem. Type | vector size
      // ------------------------------
      //  f16/bf16   | vector<8xf16/bf16>
      //     i8      | vector<16xi8>
      //     tf32    | vector<4xf32>
      return vec_ty(elemTy, RC * ((8 * (float)(sizeof(int16_t)) / bitWidth)));
    } else {
      //  Elem. Type | vector size
      // ------------------------------
      //  f16/bf16   | vector<16xf16/bf16>
      //     i8      | vector<32xi8>
      //     tf32    | vector<8xf32>
      assert(dotOpLayout.getOpIdx() == 1);
      return vec_ty(elemTy, RC * ((8 * sizeof(int32_t)) / bitWidth));
    }
  }

  auto mmaParent = dotOpLayout.getParent().dyn_cast<NvidiaMmaEncodingAttr>();
  if (!mmaParent || mmaParent.isHopper())
    return elemTy;
  int bitwidth = elemTy.getIntOrFloatBitWidth();
  assert(bitwidth <= 32);
  return IntegerType::get(ctx, 32);
}

Type TritonIntelGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(type);

  if (auto shared_layout = layout.dyn_cast<SharedEncodingAttr>()) {
    SmallVector<Type, 4> types;
    // base ptr
    auto ptrType = LLVM::LLVMPointerType::get(ctx, 3);
    types.push_back(ptrType);
    // shape dims
    auto rank = type.getRank();
    // offsets + strides
    for (auto i = 0; i < rank * 2; i++) {
      types.push_back(IntegerType::get(ctx, 32));
    }
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}