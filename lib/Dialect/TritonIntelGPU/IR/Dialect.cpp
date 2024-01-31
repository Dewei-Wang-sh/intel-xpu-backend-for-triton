#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "triton/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::gpu::intel;

// Utility
namespace mlir {
namespace triton {
namespace gpu {
namespace intel {}
} // namespace gpu
} // namespace triton
} // namespace mlir

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = attr.dyn_cast<IntegerAttr>();
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned, 2> &res,
                                       StringRef desc) {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//
SmallVector<unsigned> DpasEncodingAttr::getShapeA() const {
  return {getRepeatCount(), getSystolicDepth() * getOpsPerChan()};
}

SmallVector<unsigned> DpasEncodingAttr::getShapeB() const {
  return {getSystolicDepth() * getOpsPerChan(), getExecutionSize()};
}

SmallVector<unsigned> DpasEncodingAttr::getShapeC() const {
  return {getRepeatCount(), getExecutionSize()};
}

SmallVector<unsigned> DpasEncodingAttr::getSizePerThread() const {
  unsigned threadsPerWarp = getSugGroupSize();
  auto shapeC = getShapeC();
  unsigned elemsNum = product<unsigned>(shapeC);
  unsigned elemsPerThread = elemsNum / threadsPerWarp;
  // The Value is shard per col to threads.
  return {elemsPerThread, 1};
};

SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto shapeC = getShapeC();
  return {shapeC[0] * getWarpsPerCTA()[0], shapeC[1] * getWarpsPerCTA()[1]};
}
SmallVector<unsigned>
DpasEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto shapePerCTATile = getShapePerCTATile(shape);
  unsigned tilesRow = ceil<unsigned>(shape[0], shapePerCTATile[0]);
  unsigned tilesCol = ceil<unsigned>(shape[1], shapePerCTATile[1]);
  auto sizePerThread = getSizePerThread();
  elemsPerThread[0] = sizePerThread[0] * tilesRow;
  elemsPerThread[1] = sizePerThread[1] * tilesCol;

  return elemsPerThread;
}
unsigned DpasEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}
SmallVector<unsigned> DpasEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}
SmallVector<unsigned> DpasEncodingAttr::getCTAOrder() const {
  SmallVector<unsigned> res{1, 0};
  return res;
}
SmallVector<unsigned> DpasEncodingAttr::getCTAsPerCGA() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}
// unsigned DpasEncodingAttr::getNumCTAs() const { return 1; }
SmallVector<int64_t> DpasEncodingAttr::getXMXRep(ArrayRef<int64_t> shape,
                                                 int opIdx) const {
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0) {
    auto shapePerWarp = getShapeA();
    return {std::max<int64_t>(1, shape[0] / (shapePerWarp[0] * warpsPerCTA[0])),
            std::max<int64_t>(1, shape[1] / shapePerWarp[1])};
  } else {
    assert(opIdx == 1);
    auto shapePerWarp = getShapeB();
    return {
        std::max<int64_t>(1, shape[0] / shapePerWarp[0]),
        std::max<int64_t>(1, shape[1] / (shapePerWarp[1] * warpsPerCTA[1]))};
  }
}
unsigned DpasEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, mlir::Type eltTy, int kWidth, int opIdx) const {
  auto shapePerCTA = getShapePerCTA(*this, shape);
  int warpsPerCTAM = getWarpsPerCTA()[0];
  int warpsPerCTAN = getWarpsPerCTA()[1];
  auto rep = getXMXRep(shapePerCTA, opIdx);
  auto threadsPerWar = getSugGroupSize();
  if (opIdx == 0) {
    auto instrShapeA = getShapeA();
    auto totalElem = product<unsigned>(instrShapeA);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * rep[0] * rep[1];
  } else { // if (opIdx == 1)
    auto instrShapeB = getShapeB();
    auto totalElem = product<unsigned>(instrShapeB);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * rep[0] * rep[1];
  }
}
SmallVector<unsigned> DpasEncodingAttr::getWarpOrder() const { return {1, 0}; }
SmallVector<unsigned> DpasEncodingAttr::getThreadOrder() const {
  return {1, 0};
}
SmallVector<unsigned> DpasEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__().begin(),
                               getWarpsPerCTA__().end());
}

SmallVector<unsigned> DpasEncodingAttr::getThreadsPerWarp() const {
  auto executionSize = getExecutionSize();
  auto subGroupSize = getSugGroupSize();
  if (subGroupSize < executionSize) {
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not be "
                             "smaller than the execution size");
  }
  return {subGroupSize / executionSize, executionSize};
}
SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                   int opIdx) const {
  auto parentShapePerCTATile = getShapePerCTATile(shape);
  auto threadsPerWarp = getThreadsPerWarp();
  if (opIdx == 0) {
    auto shapeA = getShapeA();
    return {parentShapePerCTATile[0], shapeA[1]};
  } else if (opIdx == 1) {
    auto shapeB = getShapeB();
    return {shapeB[0], parentShapePerCTATile[1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
}
SmallVector<unsigned>
DpasEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  if (opIdx == 0) {
    auto shapeA = getShapeA();
    auto subGroupSize = getSugGroupSize();
    auto systolicDepth = getSystolicDepth();
    if (subGroupSize < systolicDepth) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the systolic depth");
    }
    SmallVector<unsigned, 2> threadsPerWarp = {subGroupSize / systolicDepth,
                                               systolicDepth};
    return {shapeA[0] / threadsPerWarp[0], shapeA[1] / threadsPerWarp[1]};
  } else if (opIdx == 1) {
    auto shapeB = getShapeB();
    auto subGroupSize = getSugGroupSize();
    auto executionSize = getExecutionSize();
    if (subGroupSize < executionSize) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the execution size");
    }
    SmallVector<unsigned, 2> threadsPerWarp = {subGroupSize / executionSize,
                                               executionSize};
    return {shapeB[0] / threadsPerWarp[0], shapeB[1] / threadsPerWarp[1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
    return {};
  }
}

Attribute DpasEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned, 2> warpsPerCTA;
  unsigned repeatCount;
  unsigned systolicDepth;
  unsigned executionSize;
  unsigned opsPerChan;
  unsigned threadsPerWarp;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "repeatCount") {
      if (parseUInt(parser, attr, repeatCount, "repeatCount").failed())
        return {};
    }
    if (attr.getName() == "systolicDepth") {
      if (parseUInt(parser, attr, systolicDepth, "systolicDepth").failed())
        return {};
    }
    if (attr.getName() == "executionSize") {
      if (parseUInt(parser, attr, executionSize, "executionSize").failed())
        return {};
    }
    if (attr.getName() == "opsPerChan") {
      if (parseUInt(parser, attr, opsPerChan, "opsPerChan").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "threadsPerWarp") {
      if (parseUInt(parser, attr, threadsPerWarp, "threadsPerWarp").failed())
        return {};
    }
  }

  return parser.getChecked<DpasEncodingAttr>(
      parser.getContext(), repeatCount, systolicDepth, executionSize,
      opsPerChan, warpsPerCTA, threadsPerWarp);
}

void DpasEncodingAttr::print(AsmPrinter &printer) const {
  auto shapeA = getShapeA();
  llvm::ArrayRef<unsigned> rA = shapeA;
  auto shapeB = getShapeB();
  llvm::ArrayRef<unsigned> rB = shapeB;
  auto shapeC = getShapeC();
  llvm::ArrayRef<unsigned> rC = shapeC;
  auto warpsPerCTA = getWarpsPerCTA();
  printer << "<{"
          << "repeatCount = " << getRepeatCount() << ", "
          << "systolicDepth = " << getSystolicDepth() << ", "
          << "executionSize = " << getExecutionSize() << ", "
          << "opsPerChan = " << getOpsPerChan() << ", "
          << "threadsPerWarp = " << getSugGroupSize() << ", "
          << "warpsPerCTA = [" << llvm::ArrayRef<unsigned>(warpsPerCTA) << "], "
          << "A = [" << rA << "], "
          << "B = [" << rB << "], "
          << "C = [" << rC << "]"
          << "}>";
}

// Dialect Interface
struct TritonIntelGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding = mlir::triton::gpu::SliceEncodingAttr::get(
        getDialect()->getContext(), axis, operandEncoding);
    return success();
  }

  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     Attribute &resultEncoding) const override {
    mlir::triton::gpu::SharedEncodingAttr sharedEncoding =
        operandEncoding.dyn_cast<mlir::triton::gpu::SharedEncodingAttr>();
    if (!sharedEncoding)
      return failure();
    SmallVector<unsigned> retOrder(sharedEncoding.getOrder().begin(),
                                   sharedEncoding.getOrder().end());
    std::reverse(retOrder.begin(), retOrder.end());
    // TODO(Qingyi): Need to check whether CTAOrder should also be reversed.
    // This is not a problem for tests where numCTAs = 1.
    resultEncoding = mlir::triton::gpu::SharedEncodingAttr::get(
        getDialect()->getContext(), sharedEncoding.getVec(),
        sharedEncoding.getPerPhase(), sharedEncoding.getMaxPhase(), retOrder,
        sharedEncoding.getCTALayout(), sharedEncoding.getHasLeadingOffset());
    return mlir::success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding =
        operandEncoding.dyn_cast<mlir::triton::gpu::SliceEncodingAttr>();
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    auto mmaRetEncoding =
        retEncoding.dyn_cast<mlir::triton::gpu::intel::DpasEncodingAttr>();
    if (mmaRetEncoding) {
      // TODO: support gmma when A/B does not reside in shared memory
      if (!operandEncoding.isa<mlir::triton::gpu::SharedEncodingAttr>())
        return emitOptionalError(
            location, "unexpected operand layout for MmaEncodingAttr v3");
    } else if (auto dotOpEnc =
                   operandEncoding
                       .dyn_cast<mlir::triton::gpu::DotOperandEncodingAttr>()) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(location, "Incompatible parent encoding");
    } else
      return emitOptionalError(
          location, "Dot's a/b's encoding should be of DotOperandEncodingAttr");
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        operandEncodingA.dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    auto bEncoding =
        operandEncodingB.dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!aEncoding && !bEncoding)
      return mlir::success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEncoding.getKWidth() != bEncoding.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }
};

//===----------------------------------------------------------------------===//

void TritonIntelGPUDialect::initialize() {

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"
      >();

  addInterfaces<TritonIntelGPUInferLayoutInterface>();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"
      >();
}

// verify TritonIntelGPU ops
LogicalResult
TritonIntelGPUDialect::verifyOperationAttribute(Operation *op,
                                                NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
