#include "../TritonGPUToLLVMBase.h"
#include "../Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/ErrorHandling.h"

using ValueTable = std::map<std::array<int, 3>, Value>;
using mlir::triton::gpu::getShapePerCTA;
using mlir::triton::gpu::MemDescType;
using mlir::triton::gpu::SharedEncodingAttr;
using mlir::triton::gpu::intel::DpasEncodingAttr;

namespace {

// Data loader for DPAS instruction.
template <DpasEncodingAttr::OpIdx opIdx> class DpasMatmulLoader {
public:
  DpasMatmulLoader(DpasEncodingAttr dpasLayout, MemDescType descTy,
                   unsigned warpsPerTile, ArrayRef<Value> smemStrides,
                   const SmallVector<unsigned> &warpShape,
                   SmallVector<Value> multiDimWarpId,
                   ConversionPatternRewriter &rewriter,
                   const LLVMTypeConverter *typeConverter, Location loc)
      : dpasLayout(dpasLayout), descTy(descTy), smemStrides(smemStrides),
        multiDimWarpId(std::move(multiDimWarpId)), rewriter(rewriter),
        loc(loc) {
    size_t rank = warpShape.size();
    unsigned kDim = bool(opIdx) ? rank - 2 : rank - 1;
    unsigned nonKDim = bool(opIdx) ? rank - 1 : rank - 2;
    // Assume that smem is created with layout offset {2, 1, 0}
    repBatchDimStride = smemStrides[0];
    repKDimStride = mul(i32_val(warpShape[kDim]), smemStrides[kDim]);
    repNonKDimStride =
        mul(i32_val(warpShape[nonKDim] * warpsPerTile), smemStrides[nonKDim]);
    warpMatStride = mul(i32_val(warpShape[nonKDim]), smemStrides[nonKDim]);

    unsigned threadsPerWarp = getThreadsPerWarp();

    numPtrs = product<unsigned>(warpShape) / threadsPerWarp;
  }

  SmallVector<Value> computeOffsets(Value warpId, Value laneId,
                                    Value cSwizzleOffset) {
    return computeLdsMatOffs(warpId, laneId, cSwizzleOffset);
  }

  int getNumPtrs() const { return numPtrs; }

  // Compute matrix load offsets.
  SmallVector<Value> computeLdsMatOffs(Value warpOff, Value lane,
                                       Value cSwizzleOffset);
  // Load the matrix value.
  Value loadMatrix(int repBatch, int repOuter, int repInner,
                   const ArrayRef<Value> ptrs, LLVM::LLVMStructType structTy,
                   Type smemTy, Value cSwizzleOffset) const;

private:
  unsigned getThreadsPerWarp() const {
    return product<unsigned>(triton::gpu::getThreadsPerWarp(dpasLayout));
  }

  DpasEncodingAttr dpasLayout;
  MemDescType descTy;

  SmallVector<Value> smemStrides;
  SmallVector<Value> multiDimWarpId;
  Value repBatchDimStride;
  Value repNonKDimStride;
  Value repKDimStride;

  // Stride in number of matrices to increment on non-k dim across warps
  Value warpMatStride;
  int numPtrs;

  ConversionPatternRewriter &rewriter;
  Location loc;
};

template <DpasEncodingAttr::OpIdx opIdx>
SmallVector<Value>
DpasMatmulLoader<opIdx>::computeLdsMatOffs(Value warpId, Value laneId,
                                           Value cSwizzleOffset) {
  unsigned systolicDepth = dpasLayout.getSystolicDepth();
  unsigned repeatCount = dpasLayout.getRepeatCount();
  unsigned executionSize = dpasLayout.getExecutionSize();
  unsigned opsPerChannel = dpasLayout.getOpsPerChannel();
  unsigned threadsPerWarp = getThreadsPerWarp();
  unsigned rank = dpasLayout.getRepCluster().size();

  Value laneRowIndex, laneColIndex;
  unsigned rowsPerInst = 0u, rowsPerWarp = 0u, packedOpsPerLane = 0u;
  switch (opIdx) {
  case DpasEncodingAttr::OpIdx::OperandA: {
    assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
           "invalid opsPerChannel number.");
    SmallVector<unsigned> shapeA = dpasLayout.getShapeA();
    // Unlike the operand B, to pack the value to i16 for scalar bit width <=16.
    packedOpsPerLane = opsPerChannel == 4 ? 2 : 1;
    unsigned packedColNum = shapeA[rank - 1] / packedOpsPerLane;
    assert(threadsPerWarp >= packedColNum &&
           "DpasEncodingAttr sub-group size could not "
           "be smaller than the threads required per row for A operand.");
    rowsPerWarp = threadsPerWarp / packedColNum;
    rowsPerInst = repeatCount / rowsPerWarp;
    laneRowIndex = udiv(laneId, i32_val(packedColNum));
    laneColIndex = urem(laneId, i32_val(packedColNum));
    laneColIndex = mul(laneColIndex, i32_val(packedOpsPerLane));
  } break;
  case DpasEncodingAttr::OpIdx::OperandB: {
    assert(threadsPerWarp >= executionSize &&
           "DpasEncodingAttr sub-group size could not "
           "be smaller than the execution size for B operand.");
    packedOpsPerLane = opsPerChannel;
    rowsPerWarp = threadsPerWarp / executionSize;
    rowsPerInst = systolicDepth / rowsPerWarp;
    rowsPerWarp = rowsPerWarp * opsPerChannel;
    laneRowIndex = udiv(laneId, i32_val(executionSize));
    laneRowIndex = mul(laneRowIndex, i32_val(opsPerChannel));
    laneColIndex = urem(laneId, i32_val(executionSize));
  } break;
  }

  // outer index offset
  Value iOff = mul(warpId, warpMatStride);

  SharedEncodingAttr sharedLayout =
      cast<SharedEncodingAttr>(descTy.getEncoding());
  const int perPhase = sharedLayout.getPerPhase();
  const int maxPhase = sharedLayout.getMaxPhase();
  const int vec = sharedLayout.getVec();

  Value rowsPerWarpVal = i32_val(rowsPerWarp);
  Value zeroVal = i32_val(0);
  Value perPhaseVal = i32_val(perPhase);
  Value maxPhaseVal = i32_val(maxPhase);
  Value vecVal = i32_val(vec);
  SmallVector<unsigned> instShape = (opIdx == DpasEncodingAttr::OpIdx::OperandA)
                                        ? dpasLayout.getDPASInstShapeA()
                                        : dpasLayout.getDPASInstShapeB();
  ArrayRef<int64_t> shareMemoryShape = descTy.getShape();
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(descTy);

  SmallVector<Value> offs(numPtrs);
  const unsigned repClusterSize =
      dpasLayout.getRepCluster()[bool(opIdx) ? rank - 1 : rank - 2];
  unsigned index = 0u;
  for (unsigned repIdx = 0; repIdx < repClusterSize; ++repIdx) {
    unsigned repIndex = repIdx * instShape[unsigned(opIdx)];
    for (int rowIdx = 0; rowIdx < rowsPerInst; ++rowIdx) {
      Value rowIndex = mul(i32_val(rowIdx), rowsPerWarpVal);
      for (unsigned opsIdx = 0; opsIdx < packedOpsPerLane; ++opsIdx) {
        // inner index base
        Value jBase = laneColIndex;
        // outer index base
        Value iBase = add(rowIndex, laneRowIndex);
        switch (opIdx) {
        case DpasEncodingAttr::OpIdx::OperandA:
          iBase = add(iBase, i32_val(repIndex));
          jBase = add(jBase, i32_val(opsIdx));
          break;
        case DpasEncodingAttr::OpIdx::OperandB:
          iBase = add(iBase, i32_val(opsIdx));
          jBase = add(jBase, i32_val(repIndex));
          break;
        }

        // round the offset into the tensor's shape limitation. (Rounded
        // broadcast)
        iBase = urem(iBase, i32_val(shareMemoryShape[rank - 2]));
        jBase = urem(jBase, i32_val(shareMemoryShape[rank - 1]));

        // inner index offset
        Value jOff = zeroVal;
        // swizzle: col_swizzled = (col / vec) ^ phase * vec
        Value phase = urem(udiv(iBase, perPhaseVal), maxPhaseVal);
        jOff = add(jOff, udiv(cSwizzleOffset, vecVal));
        jOff = mul(xor_(jOff, phase), vecVal);

        Value i = add(mul(iBase, smemStrides[rank - 2]), iOff);
        Value j = add(mul(jBase, smemStrides[rank - 1]), jOff);

        Value baseOff;
        if (shapePerCTA.size() == 3 && shapePerCTA[0] > 1) {
          Value batchOffset =
              mul(multiDimWarpId[0], i32_val(shapePerCTA[1] * shapePerCTA[2]));
          offs[index++] = add(batchOffset, add(i, j));
        } else {
          offs[index++] = add(i, j);
        }
      }
    }
  }

  return offs;
}

template <DpasEncodingAttr::OpIdx opIdx>
Value DpasMatmulLoader<opIdx>::loadMatrix(
    int repBatch, int repOuter, int repInner, const ArrayRef<Value> ptrs,
    LLVM::LLVMStructType structTy, Type smemTy, Value cSwizzleOffset) const {
  Type elemTy = structTy.getBody()[0];
  assert(
      llvm::any_of(structTy.getBody(), [&](Type ty) { return ty == elemTy; }) &&
      "The struct should have the same element types.");

  Value offsetOuter = mul(i32_val(repOuter), repNonKDimStride);
  Value offsetInner = mul(i32_val(repInner), repKDimStride);
  Value offset = add(offsetOuter, offsetInner);
  if (repBatch > 0) {
    SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();
    Value offsetBatch =
        mul(i32_val(repBatch * warpsPerCTA[0]), repBatchDimStride);
    offset = add(offset, offsetBatch);
  }

  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  size_t elemNum = structTy.getBody().size();
  for (int i = 0; i < elemNum; i++) {
    Value readPtr =
        gep(ptr_ty(rewriter.getContext(), 3), smemTy, ptrs[i], offset);
    Value val = rewriter.create<LLVM::LoadOp>(loc, elemTy, readPtr);
    llvmStruct = insert_val(structTy, llvmStruct, val, i);
  }

  return llvmStruct;
}

Value composeValuesToDotOperandLayoutStruct(
    const ValueTable &vals, int batch, int n0, int n1,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {
  std::vector<Value> elems;
  for (int b = 0; b < batch; ++b) {
    for (int m = 0; m < n0; ++m) {
      for (int k = 0; k < n1; ++k) {
        Value matVal = vals.at({b, m, k});
        auto matType = cast<LLVM::LLVMStructType>(matVal.getType());
        Type valTy = matType.getBody()[0];
        for (int i = 0; i < matType.getBody().size(); ++i) {
          auto val = extract_val(valTy, matVal, i);
          elems.push_back(val);
        }
      }
    }
  }
  assert(!elems.empty() && "Expecting non-empty vector");

  Type elemTy = elems[0].getType();
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(elems.size(), elemTy));

  return packLLElements(loc, typeConverter, elems, rewriter, structTy);
}

Type getSharedMemTy(Type argType) {
  MLIRContext *ctx = argType.getContext();
  if (argType.isF16())
    return type::f16Ty(ctx);
  if (argType.isBF16())
    return type::i16Ty(ctx);
  if (argType.isF32())
    return type::f32Ty(ctx);
  if (argType.getIntOrFloatBitWidth() == 8)
    return type::i8Ty(ctx);
  if (argType.isF64())
    return type::f64Ty(ctx);

  llvm::report_fatal_error("unsupported data type for the dot layout of DPAS");
}

template <DpasEncodingAttr::OpIdx opIdx>
std::function<void(int, int, int)>
getLoadMatrixFn(MemDescType descTy, const SharedMemoryObject &smemObj,
                DpasEncodingAttr dpasLayout, unsigned warpsPerTile,
                SmallVector<unsigned> shapePerWarp,
                SmallVector<Value> multiDimWarpId, Value warpId,
                Value outerWarpDim, Value laneId, ValueTable &vals,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, Location loc) {
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(descTy);
  Type eltTy = descTy.getElementType();

  auto sharedLayout = cast<SharedEncodingAttr>(descTy.getEncoding());
  ArrayRef<unsigned> order = sharedLayout.getOrder();
  size_t rank = order.size();

  // (a, b) is the coordinate.
  auto load = [=, &rewriter, &smemObj, &shapePerWarp, &multiDimWarpId,
               &vals](int batch, int outer, int inner) {
    DpasMatmulLoader<opIdx> loader(
        dpasLayout, descTy, warpsPerTile, smemObj.strides, shapePerWarp,
        multiDimWarpId, rewriter, typeConverter, loc);

    // Offset of a slice within the original tensor in shared memory.
    Value cSwizzleOffset = smemObj.getCSwizzleOffset(order[0]);
    SmallVector<Value> offs =
        loader.computeOffsets(outerWarpDim, laneId, cSwizzleOffset);

    // Initialize pointers.
    const int numPtrs = loader.getNumPtrs();
    SmallVector<Value> ptrs(numPtrs);

    Value smemBase = smemObj.getBaseBeforeSlice(order[0], loc, rewriter);
    Type smemTy = getSharedMemTy(eltTy);
    for (int i = 0; i < numPtrs; ++i)
      ptrs[i] =
          gep(ptr_ty(rewriter.getContext(), 3), smemTy, smemBase, offs[i]);

    // Load from shared memory.
    unsigned totalElem = product<unsigned>(shapePerWarp);
    unsigned threadsPerWarp = product<unsigned>(getThreadsPerWarp(dpasLayout));
    auto matTy = LLVM::LLVMStructType::getLiteral(
        eltTy.getContext(),
        SmallVector<Type>(totalElem / threadsPerWarp,
                          typeConverter->convertType(eltTy)));

    vals[{batch, outer, inner}] = loader.loadMatrix(
        batch, outer, inner, ptrs, matTy, smemTy, cSwizzleOffset);
  };

  return load;
}

template <DpasEncodingAttr::OpIdx opIdx>
Value loadOperand(ConversionPatternRewriter &rewriter, Location loc,
                  MemDescType descTy, DotOperandEncodingAttr encoding,
                  const SharedMemoryObject &smemObj,
                  const LLVMTypeConverter *typeConverter, Value threadId) {
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(descTy);
  auto dpasLayout = cast<DpasEncodingAttr>(encoding.getParent());
  const SmallVector<unsigned> warpsPerCTA = dpasLayout.getWarpsPerCTA();

  SmallVector<unsigned> order = triton::gpu::getOrder(dpasLayout);

  SmallVector<unsigned> shape;
  if constexpr (opIdx == DpasEncodingAttr::OpIdx::OperandA)
    shape = dpasLayout.getShapeA();
  else
    shape = dpasLayout.getShapeB();

  SmallVector<int64_t> numReps =
      dpasLayout.getDPASRepetitions(shapePerCTA, opIdx);

  Value warpSize = i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  SmallVector<Value> multiDimWarpId =
      LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  unsigned rank = shape.size();
  unsigned dimOuter = bool(opIdx) ? (rank - 1) : (rank - 2);
  unsigned ceilRes =
      mlir::ceil<unsigned>(shapePerCTA[dimOuter], shape[dimOuter]);
  Value outerWarpDim = urem(multiDimWarpId[dimOuter], i32_val(ceilRes));
  unsigned warpsPerTile = std::min<unsigned>(warpsPerCTA[dimOuter], ceilRes);

  // Get the function to use to load the operand.
  ValueTable vals;
  std::function<void(int, int, int)> loadFn = getLoadMatrixFn<opIdx>(
      descTy, smemObj, dpasLayout, warpsPerTile, std::move(shape),
      std::move(multiDimWarpId), warpId, outerWarpDim, laneId, vals,
      typeConverter, rewriter, loc);

  // Load the operand.
  int64_t numRepBatch = numReps[0];
  int64_t numRepOuter = numReps[unsigned(opIdx) ? 2 : 1];
  int64_t numRepK = numReps[unsigned(opIdx) ? 1 : 2];

  for (int b = 0; b < numRepBatch; ++b)
    for (int m = 0; m < numRepOuter; ++m)
      for (int k = 0; k < numRepK; ++k)
        loadFn(b, m, k);

  // Format the values into an LLVM::Struct.
  return composeValuesToDotOperandLayoutStruct(
      vals, numRepBatch, numRepOuter, numRepK, typeConverter, loc, rewriter);
}

} // namespace

namespace SharedToDotOperandDPAS::intel {

Value convertLayout(int opIdx, ConversionPatternRewriter &rewriter,
                    Location loc, Value tensor, DotOperandEncodingAttr encoding,
                    const SharedMemoryObject &smemObj,
                    const LLVMTypeConverter *typeConverter, Value threadId) {
  auto descTy = cast<MemDescType>(tensor.getType());

  switch (opIdx) {
  case 0:
    return loadOperand<DpasEncodingAttr::OpIdx::OperandA>(
        rewriter, loc, descTy, encoding, smemObj, typeConverter, threadId);
  case 1:
    return loadOperand<DpasEncodingAttr::OpIdx::OperandB>(
        rewriter, loc, descTy, encoding, smemObj, typeConverter, threadId);
  default:
    llvm_unreachable("unexpected operand idx");
  }
  return Value();
}

} // namespace SharedToDotOperandDPAS::intel
