//===----- scheduleLoad.cpp -------------------------------------- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// This file implements a naive scheduler for loop with load/dot
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class ScheduleLoadPass
    : public TritonIntelGPUScheduleLoadBase<ScheduleLoadPass> {
public:
  /*
    void markAllUsesAsVisited(tt::DotOp dot) {
      visited.insert(dot);
      // only trace first level for now
      for (auto operand : dot->getOperands()) {
        visited.insert(operand);
      }
    }
    SmallVector<Value> getNotVisitedUses(SmallVectorImpl<tt::DotOp> dots) {
      SmallVector<Value> notVisited;
      for (auto &dot : dots) {
        visited.insert(dot);
        // only trace dotB ...
        // for (auto operand : dot->getOperands())
        auto val = dot.getB();
          if (visited.count(val) != 0)
            continue;
          auto def = val.getDefiningOp();
          if (auto extract = dyn_cast<ttgi::ExtractOp>(def)) {
            if (visited.count(extract.getBase()) == 0)
              notVisited.push_back(extract.getBase());
          }
          if (definedInLoop)
            notVisited.push_back(val);
          visited.insert(val);
      }
      return notVisited;
    }
  */
  // hack!!! only trace dotB ...
  SmallVector<Value> getNotVisitedUses(SmallVector<tt::DotOp> dots) {
    SmallVector<Value> notVisited;
    for (auto &dot : dots) {
      // for (auto operand : dot->getOperands())
      auto val = dot.getB();
      if (visited.count(val) != 0)
        continue;
      auto def = val.getDefiningOp();
      if (auto extract = dyn_cast<ttgi::ExtractOp>(def)) {
        auto base = extract.getBase();
        if (visited.count(base) == 0) {
          notVisited.push_back(base);
          visited.insert(base);
        }
      }
      notVisited.push_back(val);
      visited.insert(val);
    }
    return notVisited;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp m = getOperation();
    m.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      visited.clear();
      unsigned group = -1;
      SmallVector<SmallVector<tt::DotOp>> dotsGroup;
      SmallVector<tt::DotOp> dots;
      for (auto dot : loop.getOps<tt::DotOp>()) {
        auto groupAttr = dot->getAttrOfType<IntegerAttr>("schedule-group");
        unsigned currGroup = groupAttr.getInt();
        if (currGroup != group && !dots.empty()) {
          dotsGroup.push_back(dots);
          dots.clear();
        }
        if (currGroup == 0)
          getNotVisitedUses({dot});
        // markAllUsesAsVisited(dot);
        else
          dots.push_back(dot);
        group = currGroup;
      }
      assert(!dots.empty());
      dotsGroup.push_back(dots);

      for (auto dots : dotsGroup) {
        auto notVisited = getNotVisitedUses(dots);
        for (auto val : notVisited) {
          auto op = val.getDefiningOp();
          op->moveBefore(dots.begin()->getOperation());
        }
      }
    });

    // HOHO, add fastmath for all
    m.walk([&](Operation *op) {
      auto fmIf = dyn_cast<arith::ArithFastMathInterface>(op);
      if (fmIf)
        op->setAttr(
            fmIf.getFastMathAttrName(),
            arith::FastMathFlagsAttr::get(ctx, arith::FastMathFlags::fast));
    });
  }

private:
  DenseSet<Value> visited;
};

std::unique_ptr<Pass> mlir::triton::gpu::intel::createScheduleLoadPass() {
  return std::make_unique<ScheduleLoadPass>();
}
