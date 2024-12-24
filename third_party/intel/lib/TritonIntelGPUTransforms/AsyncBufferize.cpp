//===- AsyncBufferize.cpp ----------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUASYNCBUFFERIZE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#undef DEBUG_TYPE
#define DEBUG_TYPE "ttig-async-bufferize"
#define LDBG(X) (llvm::dbgs() << X << "\n")
// #define LDBG(X) (llvm::errs() << X << "\n")

namespace {
struct DotInfo {
  tt::DotOp dot;
  SmallVector<Value> chainOpsA;
  tt::LoadOp loadA;
  tt::AdvanceOp advanceA;
  ttgi::AllocOp bufA;
  SmallVector<Value> chainOpsB;
  tt::LoadOp loadB;
  tt::AdvanceOp advanceB;
  ttgi::AllocOp bufB;
  SmallVector<Value> chainOpsC;
  ttgi::AllocOp bufC;
  void dump() {
    dot.dump();
    LDBG("***** chain ops of dotA *****");
    for (auto val : chainOpsA)
      val.dump();
    LDBG("***** chain ops end *********");
    if (loadA)
      loadA.dump();
    if (advanceA)
      advanceA.dump();
    LDBG("\n");
    LDBG("***** chain ops of dotB *****");
    for (auto val : chainOpsB)
      val.dump();
    LDBG("***** chain ops end *********");
    if (loadB)
      loadB.dump();
    if (advanceB)
      advanceB.dump();
    LDBG("***** chain ops of dotC *****");
    for (auto val : chainOpsC)
      val.dump();
    LDBG("***** chain ops end *********");
  }
};
// only support at most 2 dot in a loop for now
struct LoopDotInfo {
  DotInfo dotInfo0;
  DotInfo dotInfo1;
  bool connectDotA = false;
  bool connectDotB = false;
  bool connectDotC = false;
  void dump() {
    LDBG("\n");
    LDBG("***** first dot info *****");
    // LLVM_DEBUG(dotInfo0.dump());
    dotInfo0.dump();
    if (dotInfo1.dot) {
      LDBG("\n");
      LDBG("connect to first DotA " << connectDotA << "\n");
      LDBG("connect to first DotB " << connectDotB << "\n");
      LDBG("connect to first DotC " << connectDotC << "\n");
      LDBG("***** second dot info *****");
      // LLVM_DEBUG(dotInfo1.dump());
      dotInfo1.dump();
    }
  }
};

bool isTensorPtrOfAddrSpace(Value ptr, unsigned addrSpace = 1) {
  if (auto ptrTy = dyn_cast<tt::PointerType>(ptr.getType()))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType()))
      return ptrTy.getAddressSpace() == addrSpace;
  return false;
}

Operation *getImmDom(Value ptr, Operation *op) {
  // SmallSetVector<Operation *> otherUsers;
  // for (auto users : ptr.getUsers()) {
  //   if (user != op)
  //     otherUsers.insert(user);
  // }
  DominanceInfo dom(op->getParentOfType<tt::FuncOp>());
  for (auto candidate : ptr.getUsers()) {
    if (dom.properlyDominates(candidate, op)) {
      bool imm = true;
      for (auto other : ptr.getUsers()) {
        if (dom.properlyDominates(other, op) &&
            dom.properlyDominates(candidate, other))
          imm = false;
      }
      if (imm)
        return candidate;
    }
  }
  return nullptr;
}

class StorePattern : public OpRewritePattern<tt::StoreOp> {
public:
  using OpRewritePattern<tt::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::StoreOp store,
                                PatternRewriter &rewriter) const override {
    Value ptr = store.getPtr();
    if (!isTensorPtrOfAddrSpace(ptr, 3))
      return failure();
    Operation *lastUser = getImmDom(ptr, store);
    if (auto load = dyn_cast_or_null<tt::LoadOp>(lastUser)) {
      if (store.getValue() == load.getResult()) {
        // load.erase(); may have other users
        store.erase();
        return success();
      }
    }
    return failure();
  }
};

class AsyncBufferizePass
    : public triton::gpu::intel::impl::TritonIntelGPUAsyncBufferizeBase<
          AsyncBufferizePass> {
private:
  DenseMap<SmallVector<Value> *, Value> bufferMap;
  Dialect *arithDialect = nullptr;
  Dialect *mathDialect = nullptr;

public:
  LogicalResult initialize(MLIRContext *context) override {
    arithDialect = context->getLoadedDialect("arith");
    mathDialect = context->getLoadedDialect("math");
    bufferMap.clear();
    return success();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();
    for (auto func : m.getOps<tt::FuncOp>()) {
      /// collect loop dot info
      llvm::SmallSetVector<scf::ForOp, 2> loops;
      func.walk([&](tt::DotOp dot) {
        if (auto loop = dyn_cast<scf::ForOp>(dot->getParentOp()))
          loops.insert(loop);
      });
      if (loops.size() == 0)
        return;
      assert(loops.size() == 1 && "only support 1 loop for now");
      LoopDotInfo loopDotInfo;
      collectLoopDotInfo(loops.front(), loopDotInfo);
      LLVM_DEBUG(loopDotInfo.dump());

      /// allocate buffer
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      allocateBuffer(loopDotInfo, b);

      /// transform load/store/dot
      SmallVector<std::pair<Value, Value>> toReplace;
      SmallVector<Operation *> toErase;
      func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        OpBuilder b(op);
        Location loc = op->getLoc();
        if (auto load = dyn_cast<tt::LoadOp>(op)) {
          Value ptr = load.getPtr();
          if (!isTensorPtrOfAddrSpace(ptr))
            return WalkResult::advance();
          Value buf = lookupBuffer(load.getResult());
          b.create<ttgi::AsyncCopyGlobalToLocalOp>(loc, ptr, buf);
          auto newLoad =
              b.create<tt::LoadOp>(loc, buf, tt::CacheModifier::NONE,
                                   tt::EvictionPolicy::NORMAL, false);
          toReplace.push_back(
              std::make_pair(load.getResult(), newLoad.getResult()));
          toErase.push_back(op);
        } else if (auto store = dyn_cast<tt::StoreOp>(op)) {
          Value ptr = store.getPtr();
          if (!isTensorPtrOfAddrSpace(ptr))
            return WalkResult::advance();
          Value buf = lookupBuffer(store.getValue());
          b.create<tt::StoreOp>(loc, buf, store.getValue(),
                                tt::CacheModifier::NONE,
                                tt::EvictionPolicy::NORMAL);
          b.create<ttgi::AsyncCopyLocalToGlobalOp>(loc, buf, ptr);
          toErase.push_back(op);
        } else if (auto dot = dyn_cast<tt::DotOp>(op)) {
          Value bufA = lookupBuffer(dot.getA());
          Value bufB = lookupBuffer(dot.getB());
          Value bufC = lookupBuffer(dot.getC());
          b.create<tt::StoreOp>(loc, bufA, dot.getA(), tt::CacheModifier::NONE,
                                tt::EvictionPolicy::NORMAL);
          b.create<tt::StoreOp>(loc, bufB, dot.getB(), tt::CacheModifier::NONE,
                                tt::EvictionPolicy::NORMAL);
          auto [isSimple, loop, argNo] = isSimpleLoopOverDot(dot);
          if (isSimple) {
            {
              OpBuilder::InsertionGuard g(b);
              // b.setInsertionPoint(loop);
              // b.create<tt::StoreOp>(loc, bufC, loop.getInitArgs()[argNo]);
              b.setInsertionPointAfter(loop);
              auto newLoad =
                  b.create<tt::LoadOp>(loc, bufC, tt::CacheModifier::NONE,
                                       tt::EvictionPolicy::NORMAL, false);
              // loop.getResults()[argNo].replaceAllUsesWith(newLoad.getResult());
              toReplace.push_back(std::make_pair(loop.getResults()[argNo],
                                                 newLoad.getResult()));
            }
            Value init = loop.getInitArgs()[argNo];
            b.create<ttgi::AsyncDotOp>(loc, bufA, bufB, bufC, init);
            toReplace.push_back(std::make_pair(dot.getResult(), dot.getC()));
          } else {
            b.create<tt::StoreOp>(loc, bufC, dot.getC(),
                                  tt::CacheModifier::NONE,
                                  tt::EvictionPolicy::NORMAL);
            b.create<ttgi::AsyncDotOp>(loc, bufA, bufB, bufC, nullptr);
            auto newLoad =
                b.create<tt::LoadOp>(loc, bufC, tt::CacheModifier::NONE,
                                     tt::EvictionPolicy::NORMAL, false);
            toReplace.push_back(
                std::make_pair(dot.getResult(), newLoad.getResult()));
          }
          toErase.push_back(op);
        }
        return WalkResult::advance();
      });
      for (auto &[old, newVal] : toReplace)
        old.replaceAllUsesWith(newVal);
      for (auto op : toErase)
        op->erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "Module before canonicalization.\n"
                            << m << "\n\n");
    canonicalize();
  }

  void allocateBuffer(LoopDotInfo &loopDotInfo, OpBuilder b) {
    auto allocBuffer = [&](Value val, SmallVector<Value> &ops) {
      if (val && !ops.empty()) {
        Type slmTy = tt::PointerType::get(val.getType(), 3);
        auto buf = b.create<ttgi::AllocOp>(val.getLoc(), slmTy);
        bufferMap[&ops] = buf;
      }
    };
    allocBuffer(loopDotInfo.dotInfo0.loadA, loopDotInfo.dotInfo0.chainOpsA);
    allocBuffer(loopDotInfo.dotInfo0.loadB, loopDotInfo.dotInfo0.chainOpsB);
    allocBuffer(loopDotInfo.dotInfo0.dot, loopDotInfo.dotInfo0.chainOpsC);
    if (loopDotInfo.dotInfo1.dot) {
      allocBuffer(loopDotInfo.dotInfo1.loadA, loopDotInfo.dotInfo1.chainOpsA);
      allocBuffer(loopDotInfo.dotInfo1.loadB, loopDotInfo.dotInfo1.chainOpsB);
      allocBuffer(loopDotInfo.dotInfo1.dot, loopDotInfo.dotInfo1.chainOpsC);
    }
  }

  void collectLoopDotInfo(scf::ForOp loop, LoopDotInfo &loopDotInfo) {
    auto dots = llvm::to_vector(loop.getOps<tt::DotOp>());
    tt::DotOp dot = dots[0];
    Value a = dot.getA();
    Value b = dot.getB();
    Value c = dot.getC();
    auto &info0 = loopDotInfo.dotInfo0;
    info0.dot = dot;
    expandDefChain(loop, a, info0.chainOpsA, info0.loadA, info0.advanceA);
    expandDefChain(loop, b, info0.chainOpsB, info0.loadB, info0.advanceB);
    expandDotCChain(loop, dot, info0.chainOpsC, loopDotInfo);
  }

  void expandDefChain(scf::ForOp loop, Value val, SmallVector<Value> &ops,
                      tt::LoadOp &load, tt::AdvanceOp &advance) {
    ops.push_back(val);
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
      expandDefChain(loop, loopArg, ops, load, advance);
    } else if (auto op = val.getDefiningOp()) {
      if (auto ld = dyn_cast<tt::LoadOp>(op)) {
        load = ld;
        for (auto user : ld.getPtr().getUsers()) {
          if (user == ld)
            continue;
          else if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user))
            advance = advanceOp;
          else
            assert(0 && "consider more support");
        }
        // block pointer should also be tracked
        expandDefChain(loop, ld.getPtr(), ops, load, advance);
      } else if (auto currAdvance = dyn_cast<tt::AdvanceOp>(op)) {
        expandDefChain(loop, currAdvance.getPtr(), ops, load, advance);
      } else if (op->getDialect() == arithDialect ||
                 op->getDialect() == mathDialect) {
        for (auto operand : op->getOperands()) {
          expandDefChain(loop, operand, ops, load, advance);
        }
      }
    }
    return;
  }

  void expandDotCChain(scf::ForOp loop, tt::DotOp dot, SmallVector<Value> &ops,
                       LoopDotInfo &loopDotInfo) {
    SmallVector<Value> defList;
    tt::LoadOp nullLoad;
    tt::AdvanceOp nullAdv;
    expandDefChain(loop, dot.getC(), defList, nullLoad, nullAdv);
    for (auto op : llvm::reverse(defList))
      ops.push_back(op);
    ops.push_back(dot);
    for (auto it = ++dot->getIterator(); it != loop.end(); it++) {
      auto op = &*it;
      bool inUseChain = llvm::any_of(op->getOperands(), [&](Value val) {
        return std::find(ops.begin(), ops.end(), val) != ops.end();
      });
      if (!inUseChain)
        continue;
      else if (op->getDialect() == arithDialect ||
               op->getDialect() == mathDialect)
        ops.push_back(op->getResults()[0]);
      else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        auto loop = cast<scf::ForOp>(yield->getParentOp());
        // HaHa
        Value res = loop.getResult(0);
        ops.push_back(res);
      }

      else if (isa<tt::ReduceOp, tt::ExpandDimsOp, tt::BroadcastOp>(op))
        ;
      else if (auto dot1 = dyn_cast<tt::DotOp>(op)) {
        auto &info1 = loopDotInfo.dotInfo1;
        info1.dot = dot1;
        auto dotA = dot1.getA();
        auto dotB = dot1.getB();
        auto dotC = dot1.getC();
        if (std::find(ops.begin(), ops.end(), dotA) == ops.end())
          expandDefChain(loop, dotA, info1.chainOpsA, info1.loadA,
                         info1.advanceA);
        else
          loopDotInfo.connectDotA = true;
        if (std::find(ops.begin(), ops.end(), dotB) == ops.end())
          expandDefChain(loop, dotB, info1.chainOpsB, info1.loadB,
                         info1.advanceB);
        else
          loopDotInfo.connectDotB = true;
        if (std::find(ops.begin(), ops.end(), dotC) == ops.end())
          expandDotCChain(loop, dot1, info1.chainOpsC, loopDotInfo);
        else
          loopDotInfo.connectDotC = true;
      }
    }
  }

  Value lookupBuffer(Value val) {
    for (const auto &[ops, buf] : bufferMap) {
      if (llvm::is_contained(*ops, val))
        return buf;
    }
    assert(0 && "buffer not found");
    return nullptr;
  }

  std::tuple<bool, scf::ForOp, unsigned> isSimpleLoopOverDot(tt::DotOp dot) {
    Value c = dot.getC();
    Value d = dot.getD();
    if (auto loop = dyn_cast<scf::ForOp>(dot->getParentOp())) {
      SmallVector<Operation *> users = llvm::to_vector(d.getUsers());
      if (isa<BlockArgument>(c) && users.size() == 1 &&
          isa<scf::YieldOp>(users[0])) {
        unsigned argNo = llvm::cast<BlockArgument>(c).getArgNumber();
        return std::make_tuple(true, loop, argNo - 1);
      }
    }
    return std::make_tuple(false, nullptr, 0);
  }

  void canonicalize() {
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.add<StorePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
