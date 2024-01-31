#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttig = mlir::triton::gpu::intel;
namespace {
using tt::DotOp;
using ttg::BlockedEncodingAttr;
using ttg::ConvertLayoutOp;
using ttg::DotOperandEncodingAttr;
using ttg::SliceEncodingAttr;
using ttig::DeviceArch;
using ttig::DpasEncodingAttr;

struct IntelXMXCapability {
  uint32_t systolicDepth;
  uint32_t repeatCount;
  uint32_t executionSize;
  uint32_t opsChanBitWidths;
};

static IntelXMXCapability caps[] = {
    [(uint32_t)DeviceArch::ATS] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 8,
            .opsChanBitWidths = 32,
        },

    [(uint32_t)DeviceArch::PVC] =
        {
            .systolicDepth = 8,
            .repeatCount = 8,
            .executionSize = 16,
            .opsChanBitWidths = 32,
        },
};

IntelXMXCapability getXMXCapability(DeviceArch arch) {
  assert(arch <= DeviceArch::UNKNOWN && "Unknown Intel GPU archs");
  return caps[(uint32_t)arch];
}

bool supportXMX(Value value, DeviceArch arch) {
  if (arch == DeviceArch::UNKNOWN)
    return false;
  assert((arch == DeviceArch::ATS || arch == DeviceArch::PVC) &&
         "Unexpected MMA layout version found");
  auto elemTy = value.getType().cast<RankedTensorType>().getElementType();
  return elemTy.isF16() || elemTy.isBF16(); /* ||
           (elemTy.isF32() && version >= 2) ||
           (elemTy.isInteger(8) && version >= 2);*/
}

bool supportXMX(DotOp op, DeviceArch arch) {
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  if (aElemTy.isF32() && bElemTy.isF32()) {
    // The FP32-FP32-FP32 data type result
    // incorrect:https://github.com/intel/intel-xpu-backend-for-triton/issues/402
    return false;
  }
  auto dElemTy = op.getD().getType().cast<RankedTensorType>().getElementType();
  if (dElemTy.isF16()) {
    // The FP16-FP16-FP16 data type result
    // incorrect:https://github.com/intel/intel-xpu-backend-for-triton/issues/400
    return false;
  }
  return supportXMX(op.getA(), arch) && supportXMX(op.getB(), arch);
}

SmallVector<unsigned, 2> getWarpsPerTile(DotOp dotOp,
                                         struct IntelXMXCapability xmxCap,
                                         const ArrayRef<int64_t> tensorShape,
                                         int numWarps) {
  auto filter = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  auto slices = mlir::getSlice(dotOp, {filter});
  for (Operation *op : slices)
    if (isa<DotOp>(op) && (op != dotOp))
      return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {xmxCap.repeatCount,
                                          xmxCap.executionSize};
  uint32_t rowColRatio =
      mlir::ceil<uint32_t>(xmxCap.repeatCount, xmxCap.executionSize);
  uint32_t colRowRatio =
      mlir::ceil<uint32_t>(xmxCap.executionSize, xmxCap.repeatCount);
  bool changed = false;
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (tensorShape[0] / (shapePerWarp[0] * rowColRatio) / ret[0] >=
        tensorShape[1] / (shapePerWarp[1] * colRowRatio) / ret[1]) {
      if (ret[0] < tensorShape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

class BlockedToMMA : public mlir::RewritePattern {

public:
  BlockedToMMA(mlir::MLIRContext *context, DeviceArch arch)
      : mlir::RewritePattern(DotOp::getOperationName(), 2, context),
        arch(arch) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<DpasEncodingAttr>())
      return failure();

    // for FMA, should retain the blocked layout.
    if (!supportXMX(dotOp, arch))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();

    auto xmxCap = getXMXCapability(arch);
    unsigned mmaElemBitWidths =
        oldAType.getElementType().getIntOrFloatBitWidth();
    unsigned opsPerChan = xmxCap.opsChanBitWidths / mmaElemBitWidths;

    auto warpsPerTile = getWarpsPerTile(dotOp, xmxCap, retShape, numWarps);

    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    DpasEncodingAttr mmaEnc = DpasEncodingAttr::get(
        oldRetType.getContext(), xmxCap.repeatCount, xmxCap.systolicDepth,
        xmxCap.executionSize, opsPerChan, warpsPerTile, threadsPerWarp);

    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<ttg::ConvertLayoutOp>(oldAcc.getLoc(),
                                                        newRetType, oldAcc);

    auto newAEncoding = ttg::DotOperandEncodingAttr::get(
        oldAType.getContext(), 0, newRetType.getEncoding(), opsPerChan);
    auto newBEncoding = ttg::DotOperandEncodingAttr::get(
        oldBType.getContext(), 1, newRetType.getEncoding(), opsPerChan);

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(), newAEncoding);
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(), newBEncoding);

    a = rewriter.create<ttg::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<ttg::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<DotOp>(dotOp.getLoc(), newRetType, a, b,
                                         newAcc, dotOp.getAllowTF32(),
                                         dotOp.getMaxNumImpreciseAcc());

    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType,
                                                      newDot.getResult());
    return success();
  }

private:
  DeviceArch arch;
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  auto tensorPromotedType =
      operand.getType().cast<RankedTensorType>().cloneWith(std::nullopt,
                                                           promotedType);
  Type elemType = tensorPromotedType.getElementType();
  return llvm::TypeSwitch<Type, Value>(elemType)
      .Case<FloatType>([&](auto) {
        return builder.create<tt::FpToFpOp>(loc, tensorPromotedType, operand);
      })
      .Case<IntegerType>([&](auto) {
        unsigned tgtBitWidth = elemType.getIntOrFloatBitWidth(),
                 valBitWidth = operand.getType()
                                   .cast<RankedTensorType>()
                                   .getElementTypeBitWidth();
        Operation *castOp = (valBitWidth <= tgtBitWidth)
                                ? builder.create<arith::ExtSIOp>(
                                      loc, tensorPromotedType, operand)
                                : builder.create<arith::TruncIOp>(
                                      loc, tensorPromotedType, operand);
        return castOp->getResult(0);
      });
}

// promote operands of dot op if the existing combination is not natively
// supported.
static void decomposeMixedModeDotOp(ModuleOp mod) {
  mod.walk([](DotOp dotOp) -> void {
    Value D = dotOp.getResult();
    OpBuilder builder(dotOp);
    Type AElType =
        dotOp.getA().getType().cast<RankedTensorType>().getElementType();
    Type promoteType;
    DpasEncodingAttr mmaLayout = D.getType()
                                     .cast<RankedTensorType>()
                                     .getEncoding()
                                     .dyn_cast<DpasEncodingAttr>();
    if (mmaLayout) {
      // No operands promotion because of DPAS using different packing layout
      // for MMA.
      return;
    } else {
      // FMA case.
      Type AElType =
          dotOp.getA().getType().cast<RankedTensorType>().getElementType();
      Type DElType = D.getType().cast<RankedTensorType>().getElementType();
      if (AElType == DElType)
        return;
      promoteType = DElType;
    }
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
}

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class TritonIntelGPUAccelerateMatmulPass
    : public TritonIntelGPUAccelerateMatmulBase<
          TritonIntelGPUAccelerateMatmulPass> {
public:
  TritonIntelGPUAccelerateMatmulPass() = default;
  TritonIntelGPUAccelerateMatmulPass(ttig::DeviceArch arch) {
    this->deviceArch = arch;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<::BlockedToMMA>(context, deviceArch);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
    // now that we pick the mma type decompose dot that are not natively
    // supported.
    decomposeMixedModeDotOp(m);
  }
};

std::unique_ptr<Pass>
mlir::createTritonIntelGPUAccelerateMatmulPass(ttig::DeviceArch arch) {
  return std::make_unique<TritonIntelGPUAccelerateMatmulPass>(arch);
}
