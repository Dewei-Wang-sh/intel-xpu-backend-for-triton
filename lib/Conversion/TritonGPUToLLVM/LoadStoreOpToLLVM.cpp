#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <numeric>

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {
// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::LoadOp>::ConvertTritonGPUOpToLLVMPattern;

  LoadOpConversion(TritonGPUToLLVMTypeConverter &converter,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   triton::Target target, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, target,
                                                        benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto *ctx = rewriter.getContext();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && valueElemTy.isa<IntegerType>() &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        constAttr.getElementType().isa<IntegerType>()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      // TODO: optimization when ptr is GEP with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      if (target == triton::Target::GENX) {
        SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
        Type retTy = retTys.size() > 1
                         ? vec_ty(IntegerType::get(ctx, width), nWords)
                         : retTys[0];

        Value other_ = undef(retTy);
        if (other) {
          for (size_t ii = 0; ii < nWords; ++ii) {
            size_t size = width / valueElemNBits;

            auto vecTy = vec_ty(valueElemTy, size);
            Value v = undef(vecTy);
            for (size_t s = 0; s < size; ++s) {
              Value falseVal = otherElems[vecStart + ii * size + s];
              Value sVal = createIndexAttrConstant(
                  rewriter, loc, this->getTypeConverter()->getIndexType(), s);
              v = insert_element(vecTy, v, falseVal, sVal);
            }
            v = bitcast(v, IntegerType::get(ctx, width));

            if (otherIsSplatConstInt) {
              for (size_t s = 0; s < 32; s += valueElemNBits)
                splatVal |= splatVal << valueElemNBits;
              v = int_val(width, splatVal);
            }

            Value iiVal = createIndexAttrConstant(
                rewriter, loc, this->getTypeConverter()->getIndexType(), ii);
            if (nWords > 1) {
              other_ = insert_element(retTy, other_, v, iiVal);
            } else {
              other_ = v;
            }
          }
        }

        // Create a predicated load operation.
        Block &endBlock = LLVM::createPredicatedBlock(
            rewriter, loc, pred, SmallVector<Value, 1>{other_}, [&]() {
              Value addrElem =
                  bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
              Value ret = load(retTy, addrElem);
              return SmallVector<Value, 1>{ret};
            });
        Value ret = *endBlock.args_begin();

        // Extract and store return values
        SmallVector<Value> rets;
        for (unsigned int ii = 0; ii < nWords; ++ii) {
          Value curr;
          if (retTy.isa<VectorType>()) {
            curr =
                extract_element(IntegerType::get(ctx, width), ret, i32_val(ii));
          } else {
            curr = ret;
          }
          curr = bitcast(curr, LLVM::getFixedVectorType(
                                   valueElemTy, width / valueElemNBits));
          rets.push_back(curr);
        }
        int tmp = width / valueElemNBits;
        for (size_t ii = 0; ii < vec; ++ii) {
          Value loaded =
              extract_element(valueElemTy, rets[ii / tmp], i32_val(ii % tmp));
          loadedVals.push_back(loaded);
        }
      } else { // Not target::SPIRV || target::GENX

        // TODO(Superjomn) Add cache policy fields to StoreOp.
        // TODO(Superjomn) Deal with cache policy here.
        const bool hasL2EvictPolicy = false;

        PTXBuilder ptxBuilder;

        const std::string readConstraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        const std::string writeConstraint =
            (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");

        // prepare asm operands
        auto *dstsOpr = ptxBuilder.newListOperand();
        for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
          auto *opr = ptxBuilder.newOperand(writeConstraint,
                                            /*init=*/true); // =r operations
          dstsOpr->listAppend(opr);
        }

        auto *addrOpr =
            ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

        // Define the instruction opcode
        auto &ld = ptxBuilder.create<>("ld")
                       ->o("volatile", op.getIsVolatile())
                       .global()
                       .o("ca", op.getCache() == triton::CacheModifier::CA)
                       .o("cg", op.getCache() == triton::CacheModifier::CG)
                       .o("L1::evict_first",
                          op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
                       .o("L1::evict_last",
                          op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
                       .o("L1::cache_hint", hasL2EvictPolicy)
                       .v(nWords)
                       .b(width);

        PTXBuilder::Operand *evictOpr{};

        // Here lack a mlir::Value to bind to this operation, so disabled.
        // if (has_l2_evict_policy)
        //   evictOpr = ptxBuilder.newOperand(l2Evict, "l");

        if (!evictOpr)
          ld(dstsOpr, addrOpr).predicate(pred, "b");
        else
          ld(dstsOpr, addrOpr, evictOpr).predicate(pred, "b");

        if (other) {
          for (size_t ii = 0; ii < nWords; ++ii) {
            // PTX doesn't support mov.u8, so we need to use mov.u16
            PTXInstr &mov =
                ptxBuilder.create<>("mov")->o("u" + std::to_string(movWidth));

            size_t size = width / valueElemNBits;

            auto vecTy = LLVM::getFixedVectorType(valueElemTy, size);
            Value v = undef(vecTy);
            for (size_t s = 0; s < size; ++s) {
              Value falseVal = otherElems[vecStart + ii * size + s];
              Value sVal = createIndexAttrConstant(
                  rewriter, loc, this->getTypeConverter()->getIndexType(), s);
              v = insert_element(vecTy, v, falseVal, sVal);
            }
            v = bitcast(v, IntegerType::get(getContext(), width));

            PTXInstr::Operand *opr{};

            if (otherIsSplatConstInt) {
              for (size_t s = 0; s < 32; s += valueElemNBits)
                splatVal |= splatVal << valueElemNBits;
              opr = ptxBuilder.newConstantOperand(splatVal);
            } else
              opr = ptxBuilder.newOperand(v, readConstraint);

            mov(dstsOpr->listGet(ii), opr).predicateNot(pred, "b");
          }
        }

        // Create inline ASM signature
        SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
        Type retTy =
            retTys.size() > 1
                ? LLVM::LLVMStructType::getLiteral(getContext(), retTys)
                : retTys[0];

        // TODO: if (has_l2_evict_policy)
        // auto asmDialectAttr =
        // LLVM::AsmDialectAttr::get(rewriter.getContext(),
        //                                                 LLVM::AsmDialect::AD_ATT);
        Value ret = ptxBuilder.launch(rewriter, loc, retTy);

        // Extract and store return values
        SmallVector<Value> rets;
        for (unsigned int ii = 0; ii < nWords; ++ii) {
          Value curr;
          if (retTy.isa<LLVM::LLVMStructType>()) {
            curr = extract_val(IntegerType::get(getContext(), width), ret, ii);
          } else {
            curr = ret;
          }
          curr = bitcast(curr, LLVM::getFixedVectorType(
                                   valueElemTy, width / valueElemNBits));
          rets.push_back(curr);
        }
        int tmp = width / valueElemNBits;
        for (size_t ii = 0; ii < vec; ++ii) {
          Value vecIdx = createIndexAttrConstant(
              rewriter, loc, this->getTypeConverter()->getIndexType(),
              ii % tmp);
          Value loaded = extract_element(valueElemTy, rets[ii / tmp], vecIdx);
          loadedVals.push_back(loaded);
        }
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(TritonGPUToLLVMTypeConverter &converter,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    triton::Target target, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, target,
                                                         benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    Value mask = getMask(valueTy, rewriter, loc);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = sext(i8_ty, elem);
          elem = bitcast(elem, valueElemTy);

          llWord = insert_element(wordTy, llWord, elem, i32_val(elemIdx));
        }
        llWord = bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      Value maskVal = llMask ? and_(mask, maskElems[vecStart]) : mask;

      if (target == triton::Target::GENX) {
        auto vecTy = vec_ty(valArgTy, nWords);
        Value vecWord = undef(vecTy);
        for (int index = 0; index < asmArgs.size(); ++index) {
          auto llWord = asmArgs[index].first;
          vecWord = insert_element(vecTy, vecWord, llWord, i32_val(index));
        }

        // Create a predicated store operation.
        mlir::LLVM::createPredicatedBlock(rewriter, loc, maskVal, [&] {
          Value addrElem =
              bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
          store(vecWord, addrElem);
          return ArrayRef<Value>();
        });
      } else {
        // Prepare the PTX inline asm.
        PTXBuilder ptxBuilder;
        auto *asmArgList = ptxBuilder.newListOperand(asmArgs);

        auto *asmAddr =
            ptxBuilder.newAddrOperand(ptrElems[vecStart], "l", in_off);

        auto &ptxStoreInstr =
            ptxBuilder.create<>("st")
                ->global()
                .o("wb", op.getCache() == triton::CacheModifier::WB)
                .o("cg", op.getCache() == triton::CacheModifier::CG)
                .o("cs", op.getCache() == triton::CacheModifier::CS)
                .o("wt", op.getCache() == triton::CacheModifier::WT)
                .o("L1::evict_first",
                   op.getEvict() == triton::EvictionPolicy::EVICT_FIRST)
                .o("L1::evict_last",
                   op.getEvict() == triton::EvictionPolicy::EVICT_LAST)
                .v(nWords)
                .b(width);
        ptxStoreInstr(asmAddr, asmArgList).predicate(maskVal, "b");

        Type boolTy =
            getTypeConverter()->convertType(rewriter.getIntegerType(1));
        llvm::SmallVector<Type> argTys({boolTy, ptr.getType()});
        argTys.insert(argTys.end(), nWords, valArgTy);

        auto asmReturnTy = void_ty(ctx);

        ptxBuilder.launch(rewriter, loc, asmReturnTy);
      }
    } // for
    rewriter.eraseOp(op);
    return success();
  }
};
void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  if (numCTAs == 1) {
    barrier();
  } else {
    rewriter.create<triton::nvidia_gpu::ClusterArriveOp>(loc, false);
    rewriter.create<triton::nvidia_gpu::ClusterWaitOp>(loc);
  }
}

struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(TritonGPUToLLVMTypeConverter &converter,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        triton::Target target, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(converter, target,
                                                             benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr());
    // tensor
    if (tensorTy) {
      auto valTy = op.getVal().getType().cast<RankedTensorType>();
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    Value mask = getMask(valueTy, rewriter, loc);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      switch (target) {
      case triton::Target::ROCDL:
      case triton::Target::NVVM: {
        PTXBuilder ptxBuilderAtomicCAS;
        std::string tyId = valueElemNBits * vec == 64
                               ? "l"
                               : (valueElemNBits * vec == 32 ? "r" : "h");
        auto *dstOpr =
            ptxBuilderAtomicCAS.newOperand("=" + tyId, /*init=*/true);
        auto *ptrOpr = ptxBuilderAtomicCAS.newAddrOperand(casPtr, "l");
        auto *cmpOpr = ptxBuilderAtomicCAS.newOperand(casCmp, tyId);
        auto *valOpr = ptxBuilderAtomicCAS.newOperand(casVal, tyId);
        auto &atom = *ptxBuilderAtomicCAS.create<PTXInstr>("atom");
        auto sTy = "b" + std::to_string(valueElemNBits);
        std::string semStr;
        llvm::raw_string_ostream os(semStr);
        os << op.getSem();
        auto scope = stringifyMemSyncScope(op.getScope()).str();
        atom.global().o(semStr).o(scope).o("cas").o(sTy);
        atom(dstOpr, ptrOpr, cmpOpr, valOpr).predicate(mask);

        if (tensorTy) {
          auto retType = vec == 1 ? valueElemTy : vecTy;
          auto ret = ptxBuilderAtomicCAS.launch(rewriter, loc, retType);
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
          }
        } else {
          auto old = ptxBuilderAtomicCAS.launch(rewriter, loc, valueElemTy);
          createBarrier(rewriter, loc, numCTAs);
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter,
                                                    op.getOperation(), target);
          atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
          // Only threads with mask = True store the result
          PTXBuilder ptxBuilderStore;
          auto *dstOprStore = ptxBuilderStore.newAddrOperand(atomPtr, "r");
          auto *valOprStore = ptxBuilderStore.newOperand(old, "r");
          auto &st = *ptxBuilderStore.create<PTXInstr>("st");
          st.shared().o(sTy);
          st(dstOprStore, valOprStore).predicate(mask);
          auto ASMReturnTy = void_ty(ctx);
          ptxBuilderStore.launch(rewriter, loc, ASMReturnTy);
          createBarrier(rewriter, loc, numCTAs);
          Value ret = load(valueElemTy, atomPtr);
          createBarrier(rewriter, loc, numCTAs);
          rewriter.replaceOp(op, {ret});
        }
      } break;
      case triton::Target::GENX: {
        assert((valueElemNBits == 32 || valueElemNBits == 64) &&
               "Unexpected width");

        Value zero = (valueElemNBits == 32) ? i32_val(0) : i64_val(0);
        Block &endBlock =
            mlir::LLVM::createPredicatedBlock(rewriter, loc, mask, {zero}, [&] {
              // casPtr = bitcast(casPtr, ptr_ty(ctx, 1));
              casCmp = bitcast(casCmp, zero.getType());
              casVal = bitcast(casVal, zero.getType());

              auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
                  loc, casPtr, casCmp, casVal, LLVM::AtomicOrdering::acq_rel,
                  LLVM::AtomicOrdering::monotonic);
              Value newLoaded =
                  rewriter.create<LLVM::ExtractValueOp>(loc, cmpxchg, 0);
              return SmallVector<Value, 1>{newLoaded};
            });

        Value ret = endBlock.getArgument(0);
        Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
        ret = bitcast(ret, retType);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
          }
        } else {
          createBarrier(rewriter, loc, numCTAs);
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter,
                                                    op.getOperation(), target);
          atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
          mlir::LLVM::storeShared(rewriter, loc, atomPtr, ret, mask, target);
          createBarrier(rewriter, loc, numCTAs);
          Value ret = load(valueElemTy, atomPtr);
          createBarrier(rewriter, loc, numCTAs);
          rewriter.replaceOp(op, {ret});
        }
      } break;
      default:
        llvm_unreachable("Unhandled target");
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct AtomicRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicRMWOpConversion(TritonGPUToLLVMTypeConverter &converter,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        triton::Target target, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(converter, target,
                                                             benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    auto vec = getVectorSize(ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = val.getType().cast<RankedTensorType>();
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    Value mask = getMask(valueTy, rewriter, loc);

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwVal = undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = llMask ? and_(mask, maskElements[i]) : mask;

      switch (target) {
      case triton::Target::ROCDL:
      case triton::Target::NVVM: {
        std::string sTy;
        PTXBuilder ptxBuilderAtomicRMW;
        std::string tyId = valueElemNBits * vec == 64
                               ? "l"
                               : (valueElemNBits * vec == 32 ? "r" : "h");
        auto *dstOpr =
            ptxBuilderAtomicRMW.newOperand("=" + tyId, /*init=*/true);
        auto *ptrOpr = ptxBuilderAtomicRMW.newAddrOperand(rmwPtr, "l");
        auto *valOpr = ptxBuilderAtomicRMW.newOperand(rmwVal, tyId);

        auto scope = stringifyMemSyncScope(op.getScope()).str();
        auto &atom = ptxBuilderAtomicRMW.create<>("atom")->global().o(scope);
        auto rmwOp = stringifyRMWOp(atomicRmwAttr).str();
        auto sBits = std::to_string(valueElemNBits);
        switch (atomicRmwAttr) {
        case RMWOp::AND:
          sTy = "b" + sBits;
          break;
        case RMWOp::OR:
          sTy = "b" + sBits;
          break;
        case RMWOp::XOR:
          sTy = "b" + sBits;
          break;
        case RMWOp::ADD:
          sTy = "u" + sBits;
          break;
        case RMWOp::FADD:
          rmwOp = "add";
          rmwOp += (valueElemNBits == 16 ? ".noftz" : "");
          sTy = "f" + sBits;
          sTy += (vec == 2 && valueElemNBits == 16) ? "x2" : "";
          break;
        case RMWOp::MAX:
          sTy = "s" + sBits;
          break;
        case RMWOp::MIN:
          sTy = "s" + sBits;
          break;
        case RMWOp::UMAX:
          rmwOp = "max";
          sTy = "u" + sBits;
          break;
        case RMWOp::UMIN:
          rmwOp = "min";
          sTy = "u" + sBits;
          break;
        case RMWOp::XCHG:
          sTy = "b" + sBits;
          break;
        default:
          return failure();
        }
        std::string semStr;
        llvm::raw_string_ostream os(semStr);
        os << op.getSem();
        atom.o(semStr).o(rmwOp).o(sTy);
        if (tensorTy) {
          atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
          auto retType = vec == 1 ? valueElemTy : vecTy;
          auto ret = ptxBuilderAtomicRMW.launch(rewriter, loc, retType);
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
          }
        } else {
          auto ASMReturnTy = void_ty(ctx);
          atom(dstOpr, ptrOpr, valOpr).predicate(rmwMask);
          auto old = ptxBuilderAtomicRMW.launch(rewriter, loc, valueElemTy);
          if (op->user_begin() == op->user_end()) {
            rewriter.replaceOp(op, {old});
            return success();
          }
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter,
                                                    op.getOperation(), target);
          atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
          // Only threads with rmwMask = True store the result
          PTXBuilder ptxBuilderStore;
          auto &storeShared =
              ptxBuilderStore.create<>("st")->shared().o("b" + sBits);
          auto *ptrOpr = ptxBuilderStore.newAddrOperand(atomPtr, "r");
          auto *valOpr = ptxBuilderStore.newOperand(old, tyId);
          storeShared(ptrOpr, valOpr).predicate(rmwMask);
          ptxBuilderStore.launch(rewriter, loc, void_ty(ctx));
          createBarrier(rewriter, loc, numCTAs);
          Value ret = load(valueElemTy, atomPtr);
          createBarrier(rewriter, loc, numCTAs);
          rewriter.replaceOp(op, {ret});
        }
      } break;
      case triton::Target::GENX: {
        assert((valueElemNBits == 16 || valueElemNBits == 32 ||
                valueElemNBits == 64) &&
               "Unexpected width");

        Value zero;
        llvm::TypeSwitch<mlir::Type>(valueElemTy)
            .Case<mlir::IntegerType>(
                [&](auto ty) { zero = int_val(valueElemNBits, 0); })
            .Case<mlir::Float16Type>([&](auto ty) { zero = f16_val(0); })
            .Case<mlir::Float32Type>([&](auto ty) { zero = f32_val(0); })
            .Case<mlir::Float64Type>([&](auto ty) { zero = f64_val(0); });

        Block &endBlock = mlir::LLVM::createPredicatedBlock(
            rewriter, loc, rmwMask, {zero}, [&] {
              mlir::LLVM::AtomicBinOp rmwKind;
              switch (atomicRmwAttr) {
              case RMWOp::AND:
                rmwKind = LLVM::AtomicBinOp::_and;
                break;
              case RMWOp::OR:
                rmwKind = LLVM::AtomicBinOp::_or;
                break;
              case RMWOp::XOR:
                rmwKind = LLVM::AtomicBinOp::_xor;
                break;
              case RMWOp::ADD:
                rmwKind = LLVM::AtomicBinOp::add;
                break;
              case RMWOp::FADD:
                rmwKind = LLVM::AtomicBinOp::fadd;
                break;
              case RMWOp::MAX:
                rmwKind = LLVM::AtomicBinOp::max;
                break;
              case RMWOp::UMAX:
                rmwKind = LLVM::AtomicBinOp::umax;
                break;
              case RMWOp::MIN:
                rmwKind = LLVM::AtomicBinOp::min;
                break;
              case RMWOp::UMIN:
                rmwKind = LLVM::AtomicBinOp::umin;
                break;
              case RMWOp::XCHG:
                rmwKind = LLVM::AtomicBinOp::xchg;
                break;
              }

              rmwVal = bitcast(rmwVal, valueElemTy);
              auto atomRMW = rewriter.create<LLVM::AtomicRMWOp>(
                  loc, rmwKind, rmwPtr, rmwVal, LLVM::AtomicOrdering::acq_rel);
              return SmallVector<Value, 1>{atomRMW.getRes()};
            });

        Value ret = endBlock.getArgument(0);
        Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
        ret = bitcast(ret, retType);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
          }
        } else {
          Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter,
                                                    op.getOperation(), target);
          atomPtr = bitcast(atomPtr, ptr_ty(ctx, 3));
          // Only threads with rmwMask = True store the result
          mlir::LLVM::storeShared(rewriter, loc, atomPtr, ret, rmwMask, target);
          createBarrier(rewriter, loc, numCTAs);
          Value loadVal = load(valueElemTy, atomPtr);
          createBarrier(rewriter, loc, numCTAs);
          rewriter.replaceOp(op, {loadVal});
        }
      } break;
      default:
        llvm_unreachable("Unhandled target");
      }
    }

    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = packLLElements(loc, getTypeConverter(), resultVals,
                                          rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }
};

struct InsertSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<tensor::InsertSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      tensor::InsertSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // This function has been removed upstream and should only exist for genx
    assert(target == triton::Target::GENX &&
           "InsertSliceOpConversion: genx target not supported yet");

    // %dst = insert_slice %src into %dst[%offsets]
    Location loc = op->getLoc();
    Value dst = op.getDest();
    Value src = op.getSource();
    Value res = op.getResult();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcLayout && "Unexpected srcLayout in InsertSliceOpConversion");

    auto dstTy = dst.getType().dyn_cast<RankedTensorType>();
    auto dstLayout = dstTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto llDst = adaptor.getDest();
    assert(dstLayout && "Unexpected dstLayout in InsertSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by InsertSliceOpConversion");

    // newBase = base + offset
    // Triton support either static and dynamic offsets
    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, llDst, dstTy.getElementType(), rewriter);
    SmallVector<Value, 4> offsets;
    SmallVector<Value, 4> srcStrides;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        offsets.emplace_back(adaptor.getOffsets()[i]);
      } else {
        offsets.emplace_back(i32_val(op.getStaticOffset(i)));
      }
      // Like insert_slice_async, we only support slice from one dimension,
      // which has a slice size of 1
      if (op.getStaticSize(i) != 1) {
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }

    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, offsets, smemObj.strides);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    auto smemBase = gep(elemPtrTy, elemTy, smemObj.base, offset);

    auto inVals = unpackLLElements(loc, adaptor.getSource(), rewriter);
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy, true);
    storeDistributedToShared(src, inVals, srcStrides, srcIndices, dst, smemBase,
                             elemTy, loc, rewriter);
    // Barrier is not necessary.
    // The membar pass knows that it writes to shared memory and will handle it
    // properly.
    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct InsertSliceAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::InsertSliceAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  InsertSliceAsyncOpConversion(TritonGPUToLLVMTypeConverter &converter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               Target target, PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::InsertSliceAsyncOp>(
            converter, target, benefit),
        LoadStoreConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::InsertSliceAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // This function should not be called on the genx target since all
    // InsertSliceAsyncOps would be decomposed into InsertSliceOps by the
    // decomposeInsertSliceAsyncOp function.
    // FIXME: remove this assertion once a suitable replacement instruction
    // exists for the generated PTX in this function (cp.async.cg.shared.global)
    assert(target != triton::Target::GENX &&
           "InsertSliceAsyncOpConversion: genx target not supported yet");

    // insert_slice_async %src, %dst, %index, %mask, %other
    auto loc = op.getLoc();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();

    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getDst().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto srcLayout = srcTy.getEncoding();
    assert((srcLayout.isa<BlockedEncodingAttr, SliceEncodingAttr>() &&
            "Unexpected srcLayout in InsertSliceAsyncOpConversion"));
    auto resSharedLayout = dstTy.getEncoding().cast<SharedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert((srcShape.size() <= 3) &&
           "insert_slice_async: Unexpected rank of %src");

    Value llDst = adaptor.getDst();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llIndex = adaptor.getIndex();

    // %src
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);

    // %dst
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    SmallVector<Value, 4> offsetVals;
    SmallVector<Value, 4> srcStrides;
    for (auto i = 0; i < dstTy.getShape().size(); ++i) {
      if (i == axis) {
        offsetVals.emplace_back(llIndex);
      } else {
        offsetVals.emplace_back(i32_val(0));
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }
    // Compute the offset based on the original dimensions of the shared
    // memory object
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value dstPtrBase = gep(dstPtrTy, resElemTy, smemObj.base, dstOffset);

    // %mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (llOther) {
      // FIXME(Keren): always assume other is 0 for now
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      // assert(false && "insert_slice_async: Other value not supported yet");
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    // We don't use getVec() here because we are copying from memory to memory.
    // If contiguity > vector size, we can have one pointer maintaining the
    // start of the vector and the other pointer moving to the next vector.
    unsigned inVec = getContiguity(op.getSrc());
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = inVec;
    if (outVec > 1)
      minVec = std::min(outVec, inVec);
    unsigned numElems = getTotalElemsPerThread(srcTy);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, resSharedLayout, resElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    // A sharedLayout encoding has a "vec" parameter.
    // On the column dimension, if inVec > outVec, it means we have to divide
    // single vector read into multiple ones
    auto numVecCols = std::max<unsigned>(inVec / outVec, 1);

    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // 16 * 8 = 128bits
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto numWords = vecBitWidth / bitWidth;
      auto numWordElems = bitWidth / resElemTy.getIntOrFloatBitWidth();

      // Tune CG and CA here.
      auto byteWidth = bitWidth / 8;
      CacheModifier srcCacheModifier =
          byteWidth == 16 ? CacheModifier::CG : CacheModifier::CA;
      assert(byteWidth == 16 || byteWidth == 8 || byteWidth == 4);
      auto resByteWidth = resElemTy.getIntOrFloatBitWidth() / 8;

      Value basePtr = sharedPtrs[elemIdx];
      for (size_t wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        PTXBuilder ptxBuilder;
        auto wordElemIdx = wordIdx * numWordElems;
        auto &copyAsyncOp =
            *ptxBuilder.create<PTXCpAsyncLoadInstr>(srcCacheModifier);
        auto *dstOperand =
            ptxBuilder.newAddrOperand(basePtr, "r", wordElemIdx * resByteWidth);
        auto *srcOperand =
            ptxBuilder.newAddrOperand(srcElems[elemIdx + wordElemIdx], "l");
        auto *copySize = ptxBuilder.newConstantOperand(byteWidth);
        auto *srcSize = copySize;
        if (op.getMask()) {
          // We don't use predicate in this case, setting src-size to 0
          // if there's any mask. cp.async will automatically fill the
          // remaining slots with 0 if cp-size > src-size.
          // XXX(Keren): Always assume other = 0 for now.
          auto selectOp = select(maskElems[elemIdx + wordElemIdx],
                                 i32_val(byteWidth), i32_val(0));
          srcSize = ptxBuilder.newOperand(selectOp, "r");
        }
        copyAsyncOp(dstOperand, srcOperand, copySize, srcSize);
        ptxBuilder.launch(rewriter, loc, void_ty(getContext()));
      }
    }

    rewriter.replaceOp(op, llDst);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.getSrc().getType();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    auto typeConverter = getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(srcTy.getElementType());

    // newBase = base + offset
    // Triton supports either static and dynamic offsets
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0, j = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        // adaptor.getOffsets() returns list of variable offsets. the size of
        // the list may not be the same as mixedOffsets
        opOffsetVals.emplace_back(adaptor.getOffsets()[j]);
        ++j;
      } else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.getStaticSizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    smemObj =
        SharedMemoryObject(gep(elemPtrTy, llvmElemTy, smemObj.base, offset),
                           llvmElemTy, strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncBulkWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncBulkWaitOp = *ptxBuilder.create<>("cp.async.bulk.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncBulkWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::AsyncBulkCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.bulk.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateLoadStoreOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, Target target,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion>(typeConverter, axisInfoAnalysis, target,
                                 benefit);
  patterns.add<StoreOpConversion>(typeConverter, axisInfoAnalysis, target,
                                  benefit);
  patterns.add<AtomicCASOpConversion>(typeConverter, axisInfoAnalysis, target,
                                      benefit);
  patterns.add<AtomicRMWOpConversion>(typeConverter, axisInfoAnalysis, target,
                                      benefit);
  patterns.add<InsertSliceOpConversion>(typeConverter, target, benefit);
  patterns.add<InsertSliceAsyncOpConversion>(typeConverter, axisInfoAnalysis,
                                             target, benefit);
  patterns.add<ExtractSliceOpConversion>(typeConverter, target, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, target, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, target, benefit);
  patterns.add<AsyncBulkCommitGroupOpConversion>(typeConverter, target,
                                                 benefit);
  patterns.add<AsyncBulkWaitOpConversion>(typeConverter, target, benefit);
}
