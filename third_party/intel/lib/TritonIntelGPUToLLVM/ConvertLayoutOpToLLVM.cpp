#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

#include "llvm/ADT/TypeSwitch.h"

#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getShapePerCTATile;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace mlir::triton::gpu {
namespace {

struct ConvertLayoutOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ConvertLayoutOp> {
public:
  ConvertLayoutOpConversion(const LLVMTypeConverter &typeConverter,
                            const triton::intel::TargetInfo &targetInfo,
                            PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
            srcLayout) &&
        isa<BlockedEncodingAttr, MmaEncodingTrait, SliceEncodingAttr>(
            dstLayout)) {
      return lowerDistributedToDistributed(op, adaptor, rewriter);
    }
    if (isa<DpasEncodingAttr>(srcLayout) &&
        isa<DotOperandEncodingAttr>(dstLayout)) {
      return lowerDpasToDotOperand(op, adaptor, rewriter);
    }

    return failure();
  }

private:
  SmallVector<Value>
  getMultiDimOffset(Attribute layout, Location loc,
                    ConversionPatternRewriter &rewriter, unsigned elemId,
                    RankedTensorType type,
                    ArrayRef<unsigned> multiDimCTAInRepId,
                    ArrayRef<unsigned> shapePerCTATile) const {
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto blockedLayout = dyn_cast<BlockedEncodingAttr>(layout)) {
      auto multiDimOffsetFirstElem = ::intel::emitBaseIndexForLayout(
          loc, rewriter, targetInfo, blockedLayout, type, false);
      SmallVector<Value> multiDimOffset(rank);
      SmallVector<unsigned> multiDimElemId = getMultiDimIndex<unsigned>(
          elemId, getSizePerThread(layout), getOrder(layout));
      for (unsigned d = 0; d < rank; ++d) {
        multiDimOffset[d] =
            add(multiDimOffsetFirstElem[d],
                i32_val(multiDimCTAInRepId[d] * shapePerCTATile[d] +
                        multiDimElemId[d]));
      }
      return multiDimOffset;
    }
    if (auto sliceLayout = dyn_cast<SliceEncodingAttr>(layout)) {
      unsigned dim = sliceLayout.getDim();
      auto parentEncoding = sliceLayout.getParent();
      auto parentSizePerThread = getSizePerThread(parentEncoding);
      auto parentShape = sliceLayout.paddedShape(shape);
      auto parentTy = RankedTensorType::get(parentShape, type.getElementType(),
                                            parentEncoding);
      auto offsets = ::intel::emitOffsetForLayout(layout, type);
      auto parentOffset =
          ::intel::emitOffsetForLayout(parentEncoding, parentTy);
      SmallVector<int> idxs;
      for (SmallVector<unsigned> off : offsets) {
        off.insert(off.begin() + dim, 0);
        auto it = std::find(parentOffset.begin(), parentOffset.end(), off);
        idxs.push_back(std::distance(parentOffset.begin(), it));
      }
      auto multiDimOffsetParent = getMultiDimOffset(
          parentEncoding, loc, rewriter, idxs[elemId], parentTy,
          sliceLayout.paddedShape(multiDimCTAInRepId),
          sliceLayout.paddedShape(shapePerCTATile));
      SmallVector<Value> multiDimOffset(rank);
      for (unsigned d = 0; d < rank + 1; ++d) {
        if (d == dim)
          continue;
        unsigned slicedD = d < dim ? d : (d - 1);
        multiDimOffset[slicedD] = multiDimOffsetParent[d];
      }
      return multiDimOffset;
    }
    if (auto dpasLayout = dyn_cast<DpasEncodingAttr>(layout)) {
      assert((rank == 2 || rank == 3) &&
             "unexpected rank number for Dpas layout");
      auto multiDimBase = ::intel::emitBaseIndexForLayout(
          loc, rewriter, targetInfo, layout, type, false);
      SmallVector<SmallVector<unsigned>> offsets;
      ::emitOffsetForDpasLayoutPerCTA(
          dpasLayout, offsets,
          multiDimCTAInRepId[rank - 2] * shapePerCTATile[rank - 2],
          multiDimCTAInRepId[rank - 1] * shapePerCTATile[rank - 1]);

      SmallVector<Value> multiDimOffset(rank);
      if (rank == 3)
        multiDimOffset[0] = add(multiDimBase[0], i32_val(multiDimCTAInRepId[0] *
                                                         shapePerCTATile[0]));
      multiDimOffset[rank - 2] =
          add(multiDimBase[rank - 2], i32_val(offsets[elemId][rank - 2]));
      multiDimOffset[rank - 1] =
          add(multiDimBase[rank - 1], i32_val(offsets[elemId][rank - 1]));

      return multiDimOffset;
    }
    llvm_unreachable("unexpected layout in getMultiDimOffset");
  }

  // shared memory rd/st for blocked or dpas layout with data padding
  void processReplica(Location loc, ConversionPatternRewriter &rewriter,
                      bool stNotRd, RankedTensorType type,
                      ArrayRef<unsigned> numCTAsEachRep,
                      ArrayRef<unsigned> multiDimRepId, unsigned vec,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> origRepShape,
                      ArrayRef<unsigned> outOrd, SmallVector<Value> &vals,
                      Value smemBase) const {
    auto accumNumCTAsEachRep = product<unsigned>(numCTAsEachRep);
    auto layout = type.getEncoding();
    auto rank = type.getRank();
    auto sizePerThread = getSizePerThread(layout);
    auto accumSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<unsigned> numCTATiles(rank);
    auto shapePerCTATile = getShapePerCTATile(layout);
    auto shapePerCTA = getShapePerCTA(layout, type.getShape());
    auto order = getOrder(layout);
    for (unsigned d = 0; d < rank; ++d) {
      numCTATiles[d] = ceil<unsigned>(shapePerCTA[d], shapePerCTATile[d]);
    }
    auto elemTy = type.getElementType();
    bool isInt1 = elemTy.isInteger(1);
    bool isPtr = isa<triton::PointerType>(elemTy);
    auto llvmElemTyOrig = getTypeConverter()->convertType(elemTy);
    if (isInt1)
      elemTy = IntegerType::get(elemTy.getContext(), 8);
    else if (isPtr)
      elemTy = IntegerType::get(elemTy.getContext(), 64);

    auto llvmElemTy = getTypeConverter()->convertType(elemTy);

    for (unsigned ctaId = 0; ctaId < accumNumCTAsEachRep; ++ctaId) {
      auto multiDimCTAInRepId =
          getMultiDimIndex<unsigned>(ctaId, numCTAsEachRep, order);
      SmallVector<unsigned> multiDimCTAId(rank);
      for (const auto &it : llvm::enumerate(multiDimCTAInRepId)) {
        auto d = it.index();
        multiDimCTAId[d] = multiDimRepId[d] * numCTAsEachRep[d] + it.value();
      }

      auto linearCTAId =
          getLinearIndex<unsigned>(multiDimCTAId, numCTATiles, order);
      // TODO: This is actually redundant index calculation, we should
      //       consider of caching the index calculation result in case
      //       of performance issue observed.
      for (unsigned elemId = 0; elemId < accumSizePerThread; elemId += vec) {
        SmallVector<Value> multiDimOffset =
            getMultiDimOffset(layout, loc, rewriter, elemId, type,
                              multiDimCTAInRepId, shapePerCTATile);
        SmallVector<Value> multiDimOffsetWrapped =
            mlir::LLVM::getWrappedMultiDimOffset(rewriter, loc, multiDimOffset,
                                                 origRepShape, shapePerCTATile,
                                                 shapePerCTA);
        Value offset = linearize(rewriter, loc, multiDimOffsetWrapped,
                                 paddedRepShape, outOrd);
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value ptr = gep(elemPtrTy, llvmElemTy, smemBase, offset);
        auto vecTy = vec_ty(llvmElemTy, vec);
        ptr = bitcast(ptr, ptr_ty(rewriter.getContext(), 3));
        if (stNotRd) {
          Value valVec = undef(vecTy);
          for (unsigned v = 0; v < vec; ++v) {
            auto currVal = vals[elemId + linearCTAId * accumSizePerThread + v];
            if (isInt1)
              currVal = zext(llvmElemTy, currVal);
            else if (isPtr)
              currVal = ptrtoint(llvmElemTy, currVal);
            valVec = insert_element(vecTy, valVec, currVal, i32_val(v));
          }
          store(valVec, ptr);
        } else {
          Value valVec = load(vecTy, ptr);
          for (unsigned v = 0; v < vec; ++v) {
            Value currVal = extract_element(llvmElemTy, valVec, i32_val(v));
            if (isInt1)
              currVal = icmp_ne(currVal,
                                rewriter.create<LLVM::ConstantOp>(
                                    loc, i8_ty, rewriter.getI8IntegerAttr(0)));
            else if (isPtr)
              currVal = inttoptr(llvmElemTyOrig, currVal);
            vals[elemId + linearCTAId * accumSizePerThread + v] = currVal;
          }
        }
      }
    }
  }

  // blocked/dpas -> blocked/dpas.
  // Data padding in shared memory to avoid bank conflict.
  LogicalResult
  lowerDistributedToDistributed(triton::gpu::ConvertLayoutOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto shape = dstTy.getShape();
    unsigned rank = dstTy.getRank();
    SmallVector<unsigned> numReplicates(rank);
    SmallVector<unsigned> inNumCTAsEachRep(rank);
    SmallVector<unsigned> outNumCTAsEachRep(rank);
    SmallVector<unsigned> inNumCTAs(rank);
    SmallVector<unsigned> outNumCTAs(rank);
    auto srcShapePerCTATile = getShapePerCTATile(srcLayout);
    auto dstShapePerCTATile = getShapePerCTATile(dstLayout);
    auto shapePerCTA = getShapePerCTA(srcLayout, shape);

    for (unsigned d = 0; d < rank; ++d) {
      unsigned inPerCTA =
          std::min<unsigned>(shapePerCTA[d], srcShapePerCTATile[d]);
      unsigned outPerCTA =
          std::min<unsigned>(shapePerCTA[d], dstShapePerCTATile[d]);
      unsigned maxPerCTA = std::max(inPerCTA, outPerCTA);
      numReplicates[d] = ceil<unsigned>(shapePerCTA[d], maxPerCTA);
      inNumCTAsEachRep[d] = maxPerCTA / inPerCTA;
      outNumCTAsEachRep[d] = maxPerCTA / outPerCTA;
      assert(maxPerCTA % inPerCTA == 0 && maxPerCTA % outPerCTA == 0);
      inNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], inPerCTA);
      outNumCTAs[d] = ceil<unsigned>(shapePerCTA[d], outPerCTA);
    }
    // Potentially we need to store for multiple CTAs in this replication
    auto accumNumReplicates = product<unsigned>(numReplicates);
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto scratchConfig = getScratchConfigForCvt(srcTy, dstTy);
    unsigned inVec = scratchConfig.inVec;
    unsigned outVec = scratchConfig.outVec;
    const auto &paddedRepShape = scratchConfig.paddedRepShape;
    const auto &origRepShape = scratchConfig.repShape;
    if (isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNType>(
            getElementTypeOrSelf(op.getType()))) {
      assert(inVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
      assert(outVec % 4 == 0 && "conversion not supported for FP8E4M3B15");
    }

    unsigned outElems = getTotalElemsPerThread(dstTy);
    auto outOrd = getOrder(dstLayout);
    SmallVector<Value> outVals(outElems);

    for (unsigned repId = 0; repId < accumNumReplicates; ++repId) {
      auto multiDimRepId =
          getMultiDimIndex<unsigned>(repId, numReplicates, outOrd);
      if (repId != 0) {
        barrier();
      }
      if (isa<BlockedEncodingAttr>(srcLayout) ||
          isa<SliceEncodingAttr>(srcLayout) ||
          isa<DpasEncodingAttr>(srcLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ true, srcTy, inNumCTAsEachRep,
                       multiDimRepId, inVec, paddedRepShape, origRepShape,
                       outOrd, vals, smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with input layout not implemented");
        return failure();
      }

      barrier();
      if (isa<BlockedEncodingAttr>(dstLayout) ||
          isa<SliceEncodingAttr>(dstLayout) ||
          isa<DpasEncodingAttr>(dstLayout)) {
        processReplica(loc, rewriter, /*stNotRd*/ false, dstTy,
                       outNumCTAsEachRep, multiDimRepId, outVec, paddedRepShape,
                       origRepShape, outOrd, outVals, smemBase);
      } else {
        llvm::report_fatal_error(
            "ConvertLayout with output layout not implemented");
        return failure();
      }
    }

    Value result =
        packLLElements(loc, getTypeConverter(), outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

  using ValueTable = std::map<std::array<unsigned, 3>, Value>;

  ValueTable getValuesFromDpasLayoutStruct(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Value vals,
                                           RankedTensorType srcType) const {
    SmallVector<Value> elems = unpackLLElements(loc, vals, rewriter);
    auto dpasLayout = dyn_cast<DpasEncodingAttr>(srcType.getEncoding());

    size_t totalElems = elems.size();
    auto numElemsPerOperand =
        product<unsigned>(dpasLayout.getDPASInstShapeC()) /
        dpasLayout.getSubGroupSize();
    Type elemTy =
        this->getTypeConverter()->convertType(srcType.getElementType());
    VectorType dotOpTy = vec_ty(elemTy, numElemsPerOperand);
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(srcType.getShape(), 2 /*operand C*/);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    size_t outerDim = rank - 2;
    size_t innerDim = rank - 1;

    int offset = 0;
    ValueTable result;
    for (unsigned b = 0; b < repetitions[0]; ++b) {
      for (int i = 0; i < repetitions[1]; ++i) {
        for (int j = 0; j < repetitions[2]; ++j) {
          for (int repOuter = 0; repOuter < repCluster[outerDim]; ++repOuter) {
            for (int repInner = 0; repInner < repCluster[innerDim];
                 ++repInner) {
              Value matVal = rewriter.create<LLVM::UndefOp>(loc, dotOpTy);
              for (int k = 0; k < numElemsPerOperand; ++k) {
                matVal = insert_element(dotOpTy, matVal, elems[offset++],
                                        i32_val(k));
              }
              result[{b, i * repCluster[outerDim] + repOuter,
                      j * repCluster[innerDim] + repInner}] = matVal;
            }
          }
        }
      }
    }

    return result;
  }

  Value composeValuesToDotOperandLayoutStruct(
      Location loc, ConversionPatternRewriter &rewriter, const ValueTable &vals,
      RankedTensorType dstType) const {
    auto dotLayout = dyn_cast<DotOperandEncodingAttr>(dstType.getEncoding());
    auto dpasLayout = dyn_cast<DpasEncodingAttr>(dotLayout.getParent());

    auto opIdx = static_cast<DpasEncodingAttr::OpIdx>(dotLayout.getOpIdx());
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(dstType.getShape(), opIdx);
    ArrayRef<unsigned> repCluster = dpasLayout.getRepCluster();
    size_t rank = repCluster.size();
    unsigned repBatch = repetitions[0];
    unsigned repOuter = 0u;
    unsigned repInner = 0u;
    unsigned repClusterOuter = 0u;

    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA: {
      // operand A
      repOuter = repetitions[1];
      repInner = repetitions[2];
      repClusterOuter = repCluster[rank - 2];
    } break;
    case DpasEncodingAttr::OpIdx::OperandB: {
      // operand B
      repOuter = repetitions[2];
      repInner = repetitions[1];
      repClusterOuter = repCluster[rank - 1];
    } break;
    }

    // TODO: Operands B requires extra steps to combine [8, 16] to [16, 16].
    SmallVector<Value> elems;
    for (unsigned b = 0; b < repBatch; ++b) {
      for (int m = 0; m < repOuter; ++m) {
        for (int k = 0; k < repInner; ++k) {
          for (int repOuterIdx = 0; repOuterIdx < repClusterOuter;
               ++repOuterIdx) {
            unsigned offsetM = m * repClusterOuter + repOuterIdx;
            unsigned offsetN = k;
            Value matVal = vals.at({b, offsetM, offsetN});
            auto vecType = cast<VectorType>(matVal.getType());
            Type valTy = vecType.getElementType();
            for (int i = 0; i < vecType.getNumElements(); ++i) {
              Value val = extract_element(valTy, matVal, i32_val(i));
              elems.push_back(val);
            }
          }
        }
      }
    }

    Type elemTy = getTypeConverter()->convertType(dstType.getElementType());
    Type structTy = LLVM::LLVMStructType::getLiteral(
        getContext(), SmallVector<Type>(elems.size(), elemTy));
    return packLLElements(loc, this->getTypeConverter(), elems, rewriter,
                          structTy);
  }

  // dpas -> dot_operand
  LogicalResult
  lowerDpasToDotOperand(triton::gpu::ConvertLayoutOp op, OpAdaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    RankedTensorType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();

    if (!intel::isDpasToDotShortcut(srcTy, dstTy))
      return failure();

    // reorder the elements to match the dot_operand layout.
    ValueTable values =
        getValuesFromDpasLayoutStruct(loc, rewriter, adaptor.getSrc(), srcTy);
    Value view =
        composeValuesToDotOperandLayoutStruct(loc, rewriter, values, dstTy);

    rewriter.replaceOp(op, view);
    return success();
  }

private:
  const triton::intel::TargetInfo &targetInfo;
};

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  ConvertLayoutOpUsingLinearLayoutsConversion(LLVMTypeConverter &typeConverter,
                                              const TargetInfoBase &targetInfo,
                                              PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    const auto &shape = op.getType().getShape();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    auto conversion = minimalCvtLayout(srcTy, dstTy);
    if (!conversion.has_value()) {
      return rewriter.notifyMatchFailure(
          op, "NYI. srcTy and/or dstTy don't implement LLs yet");
    }
    LinearLayout srcLayout =
        *toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    LinearLayout dstLayout =
        *toLinearLayout(dstTy.getShape(), dstTy.getEncoding());

    StringAttr kBlock = str_attr("block");
    StringAttr kWarp = str_attr("warp");
    StringAttr kLane = str_attr("lane");
    StringAttr kRegister = str_attr("register");

    assert(to_vector(conversion->getInDimNames()) ==
           to_vector(conversion->getOutDimNames()));
    auto dims = conversion->getInDimNames();
    if (llvm::is_contained(dims, kBlock)) {
      // Case 1: Transfer between values in different CTAs.
      //          This requires moving values through distributed shared memory.
      return rewriter.notifyMatchFailure(
          op, "NYI: Transfer between different CTAs");
    } else if (llvm::is_contained(dims, kWarp)) {
      // Case 2: Transfer between values in the same CTA, in which case we move
      //         values through shared memory.
      // TODO: Implement
      return failure();
    } else if (llvm::is_contained(dims, kLane)) {
      // Case 3. Transfer between values in the same warp, in which case we try
      //         to move values using warp shuffles, though if the pattern is
      //         complicated enough we may fall back to using shared memory
      // If the operation is a supported sub-group shuffle, perform via shuffle
      // operations.
      if (intel::cvtIsSubGroupShuffle(srcTy, dstTy)) {
        performSubGroupShuffle(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
      // If the operation is a supported sub-group transposition, perform via
      // SLM.
      if (intel::cvtIsSubGroupTranspose(srcTy, dstTy)) {
        performSubGroupTranspose(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
      // TODO(jlebar): Implement me.
      return failure();
    } else if (llvm::is_contained(dims, kRegister)) {
      // Case 4. Transfer between values in the same thread, in which case we
      //         simply reorder the elements of adaptor.getSrc().
      return transferWithinThread(op, *conversion, adaptor, rewriter);
    } else {
      // Cast 5. The two layouts are equivalent. We should probably remove
      // these in RemoveLayoutConversion.
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
  }

  LogicalResult
  transferWithinThread(ConvertLayoutOp op, const LinearLayout &conversion,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    StringAttr kRegister = str_attr("register");
    assert(!cvtNeedsSharedMemory(op.getSrc().getType(), op.getType()));

    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    SmallVector<Value> outVals(conversion.getInDimSize(kRegister));
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = conversion.apply({{kRegister, i}}).begin()->second;
      outVals[i] = inVals[srcIdx];
    }
    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  int getNumContiguousRowsForShuffle(const LinearLayout &srcLayout,
                                     const LinearLayout &dstLayout) const {
    MLIRContext *ctx = getContext();

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp =
        *dstLayout.invertAndCompose(srcLayout).quotient({kWarp, kBlock});
    // Basic case: the number of contiguous rows is 1.
    if (comp.getBasis(kRegister, 0)[1] == 1)
      return 1;
    // In other case, we only allow all threads handled by a single element to
    // be contiguous, so we can simply:
    return comp.getOutDimSize(kRegister);
  }

  void performSubGroupShuffle(ConvertLayoutOp op, const LinearLayout &srcLayout,
                              const LinearLayout &dstLayout, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    assert(intel::cvtIsSubGroupShuffle(op.getSrc().getType(), op.getType()) &&
           "Expecting sub-group shuffle");

    MLIRContext *ctx = op->getContext();
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    LinearLayout conversion = *comp.quotient(kBlock)->quotient(kWarp);

    Location loc = op.getLoc();
    // FIXME: This workaround addresses the incorrect sgsize and SLM offset in
    // ReduceOp and ConvertLayoutOp, which prevents a segmentation fault.
    // However, this is a temporary solution. Once the OutDimSize computation
    // issue in LinearLayout is resolved, this workaround should be removed.
    int32_t subGroupSize = std::min((int32_t)op.getType().getNumElements(),
                                    conversion.getOutDimSize(kLane));
    if (!op->hasAttr("allocation.offset")) {
      op->setAttr("allocation.offset",
                  rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    }

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    Type origElemTy = inVals.front().getType();
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
          auto intTy = i16_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return bitcast(val, intTy);
          });
        })
        .Case([&](IntegerType intTy) {
          constexpr unsigned minWidth = 8;
          if (intTy.getWidth() >= minWidth)
            return;
          auto dstTy = i8_ty;
          llvm::transform(inVals, std::begin(inVals),
                          [&](Value val) -> Value { return zext(dstTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return ptrtoint(dstType, val);
          });
        });

    SmallVector<Value> outVals = performSubGroupShuffle(
        loc, inVals, subGroupSize, rewriter,
        getNumContiguousRowsForShuffle(srcLayout, dstLayout));

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return bitcast(val, origElemTy); });
        })
        .Case([&](IntegerType intTy) {
          // Check whether conversion took place.
          if (intTy == outVals.front().getType())
            return;
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return trunc(origElemTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType ptrTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return inttoptr(ptrTy, val); });
        });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  SmallVector<Value> performSubGroupShuffle(Location loc,
                                            ArrayRef<Value> inVals,
                                            int32_t subGroupSize,
                                            ConversionPatternRewriter &rewriter,
                                            int numContiguousRows) const {
    SmallVector<Value> res;
    Value width = i32_val(subGroupSize);
    // A work-item may handle more than one element. There are two cases we
    // support:
    if (numContiguousRows == 1) {
      // 1. Elements held by a work-item are strided rows in the abstract slice
      // matrix: Output element `i` will take the `i / 16`th value from the `i %
      // 16`th thread.
      for (Value val : inVals) {
        for (int32_t i = 0; i < subGroupSize; ++i) {
          res.push_back(
              rewriter
                  .create<mlir::gpu::ShuffleOp>(loc, val, i32_val(i), width,
                                                mlir::gpu::ShuffleMode::IDX)
                  .getShuffleResult());
        }
      }
    } else {
      // 2. Elements held by a work-item are contiguous rows in the abstract
      // slice matrix: Output element `i` will take the `i % 16`th value from
      // the `i / 16`th thread.
      for (int32_t i = 0; i < subGroupSize; ++i) {
        for (Value val : inVals) {
          res.push_back(
              rewriter
                  .create<mlir::gpu::ShuffleOp>(loc, val, i32_val(i), width,
                                                mlir::gpu::ShuffleMode::IDX)
                  .getShuffleResult());
        }
      }
    }
    return res;
  }

  int getNumContiguousRowsForTranspose(const LinearLayout &srcLayout,
                                       const LinearLayout &dstLayout) const {
    MLIRContext *ctx = getContext();

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp =
        *dstLayout.invertAndCompose(srcLayout).quotient({kWarp, kBlock});
    // Basic case: the number of contiguous rows is 0.
    if (comp.getBasis(kLane, 0)[0] == 1)
      return 1;
    // In other case, we only allow all threads handled by a single element to
    // be contiguous, so we can simply:
    int32_t sizePerThread = comp.getOutDimSize(kRegister);
    int32_t threadsPerWarp = comp.getOutDimSize(kLane);
    assert(sizePerThread % threadsPerWarp == 0 && "Invalid transpose");
    return sizePerThread / threadsPerWarp;
  }

  void performSubGroupTranspose(ConvertLayoutOp op,
                                const LinearLayout &srcLayout,
                                const LinearLayout &dstLayout,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    assert(intel::cvtIsSubGroupTranspose(op.getSrc().getType(), op.getType()) &&
           "Expecting sub-group transpose");

    Location loc = op.getLoc();

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    Type origElemTy = inVals.front().getType();

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          // TODO: Support FP4.
          Type dstType = int_ty(floatTy.getWidth());
          assert(intel::isValidElementTypeForSubGroupTranspose(dstType) &&
                 "Expecting valid type");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return bitcast(val, dstType);
          });
        })
        .Case([&](IntegerType intTy) {
          if (intel::isValidElementTypeForSubGroupTranspose(intTy))
            return;
          Type dstType = i8_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return zext(dstType, val);
          });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          assert(intel::isValidElementTypeForSubGroupTranspose(dstType) &&
                 "i64 type should be supported");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return ptrtoint(dstType, val);
          });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    SmallVector<Value> outVals = performSubGroupTranspose(
        loc, inVals, rewriter,
        getNumContiguousRowsForTranspose(srcLayout, dstLayout));

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return bitcast(val, origElemTy); });
        })
        .Case([&](IntegerType intTy) {
          // Check whether conversion took place.
          if (intTy == outVals.front().getType())
            return;
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return trunc(origElemTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType ptrTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return inttoptr(ptrTy, val); });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  SmallVector<Value>
  unwrapFromVectors(Location loc, ArrayRef<Value> vecs,
                    ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> res;
    for (Value vec : vecs) {
      for (unsigned i = 0, n = cast<VectorType>(vec.getType()).getShape()[0];
           i < n; ++i)
        res.push_back(extract_element(vec, i32_val(i)));
    }
    return res;
  }

  static unsigned getVecLoadWidth(unsigned threadsPerWarp) {
    assert(llvm::isPowerOf2_32(threadsPerWarp) &&
           "Expecting power of 2 sub-group size");
    constexpr unsigned maxVecWidth = 16;
    return std::min(maxVecWidth, threadsPerWarp);
  }

  SmallVector<Value>
  performSubGroupTranspose(Location loc, ArrayRef<Value> inVals,
                           ConversionPatternRewriter &rewriter,
                           int numContiguousRows) const {
    Type elementType = inVals.front().getType();
    auto mod = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                               &*rewriter.getInsertionPoint());
    Type ptrType = smemBase.getType();

    int numRows = inVals.size();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    // Add an element that won't be accessed at the end of the row to avoid bank
    // conflicts.
    int rowLength = threadsPerWarp + 1;
    Type offsetType = getTypeConverter()->getIndexType();
    unsigned offsetBitWidth = offsetType.getIntOrFloatBitWidth();
    Value subGroupId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::SubgroupIdOp>(
            loc, /*upper_bound=*/IntegerAttr{}));
    Value subGroupLocalId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::LaneIdOp>(loc,
                                             /*upper_bound=*/IntegerAttr{}));
    Value subGroupOffset =
        mul(subGroupId, int_val(offsetBitWidth, rowLength * numRows));
    Value subGroupBasePtr = gep(ptrType, elementType, smemBase,
                                ValueRange{subGroupOffset}, /*inbounds=*/true);
    Value base = subGroupBasePtr;
    // Store in matrix, transposed
    for (Value val : inVals) {
      rewriter.create<TritonGEN::SubGroupBlockWriteOp>(loc, base, val);
      base = gep(base.getType(), elementType, base,
                 ArrayRef<LLVM::GEPArg>{rowLength},
                 /*inbounds=*/true);
    }

    // Load from matrix, non-trasposed.

    // Each work-item will load a row (but the last garbage element) and go to
    // the next row it needs to handle.

    int32_t workItemStride =
        numContiguousRows == 1 ? rowLength * threadsPerWarp : rowLength;
    Value workItemOffset =
        mul(subGroupLocalId,
            int_val(offsetBitWidth, numContiguousRows * rowLength));
    Value workItemBasePtr = gep(ptrType, elementType, subGroupBasePtr,
                                ValueRange{workItemOffset}, /*inbounds=*/true);
    int32_t rowsPerThread = numRows / threadsPerWarp;
    assert((numContiguousRows == 1 || numContiguousRows == rowsPerThread) &&
           "In case of more than one contiguous rows per thread, these must be "
           "consecutive");
    // We may not be able to load rows in a single operation if the sub-group
    // size exceeds a given threshold (16):
    unsigned vecLoadWidth = getVecLoadWidth(threadsPerWarp);
    SmallVector<Value> transposedVecs;
    VectorType vecType = vec_ty(elementType, vecLoadWidth);
    assert(threadsPerWarp % vecLoadWidth == 0 &&
           "Column must be loadable with N loads");
    for (unsigned i = 0; i < rowsPerThread; ++i) {
      for (unsigned j = 0; j < threadsPerWarp; j += vecLoadWidth) {
        transposedVecs.push_back(load(vecType, workItemBasePtr));
        workItemBasePtr = gep(workItemBasePtr.getType(), vecType,
                              workItemBasePtr, ArrayRef<LLVM::GEPArg>{1},
                              /*inbounds=*/true);
      }
      workItemBasePtr =
          gep(workItemBasePtr.getType(), elementType, workItemBasePtr,
              // "Go back" to the first column and increment by the stride.
              ArrayRef<LLVM::GEPArg>{workItemStride - threadsPerWarp},
              /*inbounds=*/true);
    }
    return unwrapFromVectors(loc, transposedVecs, rewriter);
  }
};

} // namespace
} // namespace mlir::triton::gpu

void mlir::triton::intel::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the linear layout conversion, so it gets a higher benefit.
  // Eventually the LL conversion will subsume all of the others and be the only
  // one left.
  patterns.add<gpu::ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 1);
  patterns.add<gpu::ConvertLayoutOpConversion>(typeConverter, targetInfo,
                                               benefit);
}
