
#include "PatternTritonGPUOpToLLVM.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttgi = mlir::triton::gpu::intel;

namespace {
struct BlockIdConversion : public ConvertOpToLLVMPattern<gpu::BlockIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(gpu::BlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewritter &rewritter) const override {
    Location loc = op->getLoc();
    unsigned dim = op.getDimension();
    llvm::Intrinsic::ID id;
    switch (dim) {
    case 1:
      id = llvm::Intrinsic::pisa_groupid_x;
      break;
    case 2:
      id = llvm::Intrinsic::pisa_groupid_y;
      break;
    case 3:
      id = llvm::Intrinsic::pisa_groupid_z;
      break;
    default:
      llvm_unreachable("invalid dimension");
    }
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    auto call = LLVM::detail::createIntrinsicCall(rewritter, id, {}, resTy);
    rewritter.replaceOp(op, call);
    return success();
  }
};

struct BlockDimConversion : public ConvertOpToLLVMPattern<gpu::BlockDimOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(gpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewritter &rewritter) const override {
    Location loc = op->getLoc();
    unsigned dim = op.getDimension();
    llvm::Intrinsic::ID id;
    switch (dim) {
    case 1:
      id = llvm::Intrinsic::pisa_localsize_x;
      break;
    case 2:
      id = llvm::Intrinsic::pisa_localsize_y;
      break;
    case 3:
      id = llvm::Intrinsic::pisa_localsize_z;
      break;
    default:
      llvm_unreachable("invalid dimension");
    }
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    auto call = LLVM::detail::createIntrinsicCall(rewritter, id, {}, resTy);
    rewritter.replaceOp(op, call);
    return success();
  }
};

struct ThreadIdConversion : public ConvertOpToLLVMPattern<gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewritter &rewritter) const override {
    Location loc = op->getLoc();
    unsigned dim = op.getDimension();
    llvm::Intrinsic::ID id;
    switch (dim) {
    case 1:
      id = llvm::Intrinsic::pisa_localid_x;
      break;
    case 2:
      id = llvm::Intrinsic::pisa_localid_y;
      break;
    case 3:
      id = llvm::Intrinsic::pisa_localid_z;
      break;
    default:
      llvm_unreachable("invalid dimension");
    }
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    auto call = LLVM::detail::createIntrinsicCall(rewritter, id, {}, resTy);
    rewritter.replaceOp(op, call);
    return success();
  }
};
} // namespace

void mlir::triton::intel::populateXe4ToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<BlockIdConversion>(typeConverter, benefit);
  patterns.add<BlockDimConversion>(typeConverter, benefit);
  patterns.add<ThreadIdConversion>(typeConverter, benefit);
}
