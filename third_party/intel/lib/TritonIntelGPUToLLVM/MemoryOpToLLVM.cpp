#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

// Forward declarations
namespace SharedToDotOperandDPAS::intel {
Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor,
                    DotOperandEncodingAttr bEncoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value thread);

} // namespace SharedToDotOperandDPAS::intel

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// blocked -> shared.
// Swizzling in shared memory to avoid bank conflict. Normally used for
// A/B operands of dots.
void lowerDistributedToShared(
    Location loc, Value src, Value dst, Value adaptorSrc,
    const SharedMemoryObject &smemObj, const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, const TargetInfoBase &targetInfo,
    std::pair<size_t, Type> *const llvmOpCount = nullptr) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto dstTy = cast<MemDescType>(dst.getType());
  auto outOrd = mlir::cast<SharedEncodingAttr>(dstTy.getEncoding()).getOrder();
  auto elemTy = typeConverter->convertType(srcTy.getElementType());

  auto inVals = unpackLLElements(loc, adaptorSrc, rewriter);
  storeDistributedToShared(dstTy, srcTy, elemTy, inVals, smemObj, loc, rewriter,
                           targetInfo, llvmOpCount);
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr =
        LLVM::getGlobalScratchPtr(loc, rewriter, funcOp, i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter,
                                                                   benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto resultTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();
    auto sharedLayout =
        cast<triton::gpu::SharedEncodingAttr>(resultTy.getEncoding());

    auto llvmElemTy = typeConverter->convertType(resultTy.getElementType());
    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, shapePerCTA,
                                      sharedLayout, loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      lowerDistributedToShared(loc, op.getSrc(), op.getResult(),
                               adaptor.getSrc(), smemObj, typeConverter,
                               rewriter, targetInfo);
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<SharedEncodingAttr>(srcLayout) &&
        isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
            dstLayout)) {
      return lowerSharedToDistributed(op, adaptor, getTypeConverter(),
                                      rewriter);
    }
    if (isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerSharedToDotOperand(op, adaptor, getTypeConverter(), rewriter);
    }
    return failure();
  }

private:
  // shared -> dot_operand if the result layout is dpas
  Value lowerSharedToDotOperandDPAS(
      LocalLoadOp op, LocalLoadOpAdaptor adaptor,
      const LLVMTypeConverter *typeConverter,
      ConversionPatternRewriter &rewriter, const DpasEncodingAttr &dpasLayout,
      const DotOperandEncodingAttr &dotOperandLayout, bool isOuter) const {
    auto loc = op.getLoc();
    auto src = op.getSrc();
    Value dst = op.getResult();

    auto llvmElemTy =
        typeConverter->convertType(src.getType().getElementType());

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    Value res;
    if (!isOuter) {
      res = SharedToDotOperandDPAS::intel::convertLayout(
          dotOperandLayout.getOpIdx(), rewriter, loc, src, dotOperandLayout,
          smemObj, typeConverter, tid_val());
    } else {
      assert(false && "unsupported DPAS layout found");
    }
    return res;
  }

  LogicalResult
  lowerSharedToDotOperand(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    RankedTensorType dstTy = op.getType();
    Attribute dstLayout = dstTy.getEncoding();
    auto dotLayout = cast<DotOperandEncodingAttr>(dstLayout);
    auto sharedLayout =
        cast<SharedEncodingAttr>(op.getSrc().getType().getEncoding());

    int K;
    if (dotLayout.getOpIdx() == 0) // $a
      K = op.getType().getShape()[sharedLayout.getOrder()[0]];
    else // $b
      K = op.getType().getShape()[sharedLayout.getOrder()[1]];
    bool isOuter = K == 1;

    Value res;
    if (auto dpasLayout =
            dyn_cast_or_null<DpasEncodingAttr>(dotLayout.getParent())) {
      res = lowerSharedToDotOperandDPAS(op, adaptor, typeConverter, rewriter,
                                        dpasLayout, dotLayout, isOuter);
    } else if (auto blockedLayout = dyn_cast_or_null<BlockedEncodingAttr>(
                   dotLayout.getParent())) {
      auto thread = getThreadId(rewriter, loc);
      res = SharedToDotOperandFMA::convertLayout(
          dotLayout.getOpIdx(), op.getSrc(), adaptor.getSrc(), blockedLayout,
          thread, loc, getTypeConverter(), rewriter);
    } else {
      assert(false && "Unsupported dot operand layout found");
    }

    rewriter.replaceOp(op, res);
    return success();
  }
  LogicalResult
  lowerSharedToDistributed(LocalLoadOp op, LocalLoadOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    auto srcSharedLayout = cast<SharedEncodingAttr>(srcTy.getEncoding());
    assert(!isa<DotOperandEncodingAttr>(dstTy.getEncoding()) &&
           "Unexpected rank of ConvertLayout(shared->blocked)");

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    SmallVector<Value> outVals = loadSharedToDistributed(
        dstTy, srcTy, elemLlvmTy, smemObj, loc, rewriter, targetInfo);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertTritonGPUOpToLLVMPattern;
  using BackendCallbackType =
      decltype(BackendCallbacks::localStoreOpConversion);

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         BackendCallbackType backendCallback,
                         PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter,
                                                                   benefit),
        targetInfo(targetInfo), backendCallback(backendCallback) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memDescVal = op.getDst();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);

    std::pair<size_t, Type> llvmOpCount;
    lowerDistributedToShared(op.getLoc(), op.getSrc(), op.getDst(),
                             adaptor.getSrc(), smemObj, getTypeConverter(),
                             rewriter, targetInfo, &llvmOpCount);

    if (backendCallback)
      (backendCallback)(op, llvmOpCount.first, llvmOpCount.second);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
  BackendCallbackType backendCallback;
};

} // namespace

void mlir::triton::intel::populateMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit,
    std::optional<BackendCallbacks> backendCallbacks) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);

  auto backendCall =
      backendCallbacks ? backendCallbacks->localStoreOpConversion : nullptr;
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, backendCall,
                                       benefit);
}
