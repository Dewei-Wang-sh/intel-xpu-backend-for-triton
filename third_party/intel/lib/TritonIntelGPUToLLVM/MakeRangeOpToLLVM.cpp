#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
struct MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp> {
  MakeRangeOpConversion(LLVMTypeConverter &converter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp>(converter,
                                                             benefit),
        targetInfo(targetInfo) {}
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType ty = op.getType();
    auto shape = ty.getShape();
    auto layout = ty.getEncoding();
    auto elemTy = ty.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    auto idxs =
        ::intel::emitIndices(loc, rewriter, targetInfo, layout, ty, true);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    auto typeConverter = getTypeConverter();
    Value result = packLLElements(loc, typeConverter, retVals, rewriter, ty);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::intel::populateMakeRangeOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<MakeRangeOpConversion>(typeConverter, targetInfo, benefit);
}
