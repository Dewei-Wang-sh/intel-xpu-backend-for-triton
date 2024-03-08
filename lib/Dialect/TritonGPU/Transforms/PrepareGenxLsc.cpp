#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include <memory>

namespace mlir {
// #define GEN_PASS_DEF_TRITONGPUPREPAREGENXLSC
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {
RankedTensorType getCastType(Type type, Type elemType) {
  auto tType = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape{tType.getShape()};
  auto oldBits = tType.getElementType().getIntOrFloatBitWidth();
  auto ratio = elemType.getIntOrFloatBitWidth() / oldBits;
  shape[0] /= ratio;
  auto newType = RankedTensorType::get(shape, elemType);
  return newType;
}

class PrepareGenxLscPass
    : public TritonGPUPrepareGenxLscBase<PrepareGenxLscPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ModuleOp mod = getOperation();

    for (auto func : mod.getOps<tt::FuncOp>()) {
      loadMap.clear();
      // for (auto dot : func.getBody().getOps<tt::DotOp>()) {
      func.walk([&](tt::DotOp dot) {
        auto a = dot.getA();
        auto b = dot.getB();
        for (auto val : {a, b}) {
          if (auto op = val.getDefiningOp()) {
            OpBuilder b(op);
            auto loc = op->getLoc();
            auto elemType = val == a ? b.getI16Type() : b.getI32Type();
            if (auto extract = dyn_cast<tt::ExtractOp>(op)) {
              auto base = extract.getBase();
              auto load = base.getDefiningOp();
              assert(isa<tt::LoadOp>(load));
              Value newBase;
              if (loadMap.count(load)) {
                newBase = loadMap[load];
              } else {
                auto newType = getCastType(base.getType(), elemType);
                newBase = b.create<tt::CastOp>(loc, newType, base);
                loadMap[load] = newBase;
              }
              auto newType = getCastType(extract.getType(), elemType);
              auto newOp = b.create<tt::ExtractOp>(loc, newType, newBase,
                                                   extract.getIdx());
              auto cast = b.create<tt::CastOp>(loc, val.getType(), newOp);
              val.replaceAllUsesWith(cast);
              extract->erase();
            } else if (auto bc = dyn_cast<tt::CastOp>(op)) {
              ;
            } else {
              assert(0 && "fixme");
            }
          }
        }
      });
    }
  }

private:
  DenseMap<Operation *, Value> loadMap;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::triton::gpu::createPrepareGenxLscPass() {
  return std::make_unique<PrepareGenxLscPass>();
}
