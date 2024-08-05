//===- ExpandSLM.cpp - expand slm related operations -*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUEXPANDSLM
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {} // namespace

class TritonIntelGPUExpandSLMPass
    : public triton::gpu::intel::impl::TritonIntelGPUExpandSLMBase<
          TritonIntelGPUExpandSLMPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    // fixed for now
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 8192));
    for (auto func : mod.getOps<tt::FuncOp>()) {
      SmallVector<Operation *, 4> slmOps;
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<tt::AllocOp, tt::AllReduceOp>(op))
          slmOps.push_back(op);
      });
      if (slmOps.empty())
        continue;

      // Operation *op = slmOps.front();
      // auto type = op->getResultTypes().front();
      // Type elmTy;
      // if (auto ptrTy = dyn_cast<tt::PointerType>(type)) {
      //   auto tType = cast<RankedTensorType>(ptrTy.getPointeeType());
      //   elmTy = tType.getElementType();
      // } else if (auto tType = dyn_cast<RankedTensorType>(type)) {
      //   elmTy = tType.getElementType();
      // } else {
      //   assert(0 && "not expected type");
      // }

      // or just i8 type
      // add kernel arg
      auto i8Ty = IntegerType::get(ctx, 8);
      auto slmTy = tt::PointerType::get(i8Ty, 3);
      func.insertArgument(func.getNumArguments(), slmTy, {}, func.getLoc());

      Block &block = func.front();
      Value base = block.getArgument(block.getNumArguments() - 1);
      // expand operation: tt.alloc, tt.all_reduce
      for (Operation *op : slmOps) {
        OpBuilder b(op);
        Location loc = op->getLoc();
        // FIXME: get it from module attribute
        unsigned numWarps = 16;
        if (auto alloc = dyn_cast<tt::AllocOp>(op)) {
          auto ptr = alloc.getResult();
          auto ptrTy = cast<tt::PointerType>(ptr.getType());
          auto tType = cast<RankedTensorType>(ptrTy.getPointeeType());
          auto baseTy = tt::PointerType::get(tType.getElementType(), 3);
          base = b.create<tt::BitcastOp>(loc, baseTy, base);
          SmallVector<Value> shape;
          shape.push_back(
              b.create<arith::ConstantIntOp>(loc, tType.getShape()[0], 64));
          shape.push_back(
              b.create<arith::ConstantIntOp>(loc, tType.getShape()[1], 64));
          SmallVector<Value> strides;
          strides.push_back(
              b.create<arith::ConstantIntOp>(loc, tType.getShape()[1], 64));
          strides.push_back(b.create<arith::ConstantIntOp>(loc, 1, 64));
          SmallVector<Value> offsets;
          offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
          offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
          auto newPtrTy = tt::PointerType::get(tType, 3);
          auto newPtr = b.create<tt::MakeTensorPtrOp>(
              loc, newPtrTy, base, shape, strides, offsets,
              b.getDenseI32ArrayAttr({1, 0}));
          ptr.replaceAllUsesWith(newPtr);
          // udpate base
          auto num = tType.getNumElements();
          auto size = b.create<arith::ConstantIntOp>(loc, num, 32);
          base = b.create<tt::AddPtrOp>(loc, base.getType(), base, size);
        } else if (auto reduce = dyn_cast<tt::AllReduceOp>(op)) {
          auto result = reduce.getResult();
          auto src = reduce.getSrc();
          auto tType = cast<RankedTensorType>(result.getType());
          auto numElms = tType.getNumElements();
          auto elmTy = tType.getElementType();
          auto baseTy = tt::PointerType::get(elmTy, 3);

          // each warp store value to slm
          base = b.create<tt::BitcastOp>(loc, baseTy, base);
          auto sgid = b.create<gpu::SubgroupIdOp>(loc);
          // Value warpId64 = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
          // sgid);
          Value warpId =
              b.create<arith::IndexCastOp>(loc, b.getI32Type(), sgid);
          SmallVector<int64_t> slmShape;
          SmallVector<unsigned> slmShapeI32;
          // FIXME
          if (numElms == 1) {
            slmShape.push_back(numWarps);
            slmShape.push_back(1);
            slmShapeI32.push_back(numWarps);
            slmShapeI32.push_back(1);
          } else {
            slmShape.push_back(numWarps);
            slmShape.push_back(64);
            slmShapeI32.push_back(numWarps);
            slmShapeI32.push_back(64);
          };
          SmallVector<Value> shape;
          shape.push_back(b.create<arith::ConstantIntOp>(loc, slmShape[0], 64));
          shape.push_back(b.create<arith::ConstantIntOp>(loc, slmShape[1], 64));
          SmallVector<Value> strides;
          strides.push_back(
              b.create<arith::ConstantIntOp>(loc, slmShape[1], 64));
          strides.push_back(b.create<arith::ConstantIntOp>(loc, 1, 64));
          SmallVector<Value> offsets;
          auto cst0 = b.create<arith::ConstantIntOp>(loc, 0, 32);
          offsets.push_back(cst0);
          offsets.push_back(cst0);

          Value warp0Ptr;
          if (numElms == 1) {
            Value warpPtr =
                b.create<tt::AddPtrOp>(loc, base.getType(), base, warpId);
            Value newSrc = b.create<tt::BitcastOp>(loc, elmTy, src);
            b.create<tt::StoreOp>(loc, warpPtr, newSrc, tt::CacheModifier::NONE,
                                  tt::EvictionPolicy::NORMAL);
          } else {
            offsets[0] = warpId;
            auto ptrTy = tt::PointerType::get(tType, 3);
            Value warpPtr = b.create<tt::MakeTensorPtrOp>(
                loc, ptrTy, base, shape, strides, offsets,
                b.getDenseI32ArrayAttr({1, 0}));
            b.create<tt::StoreOp>(loc, warpPtr, src, tt::CacheModifier::NONE,
                                  tt::EvictionPolicy::NORMAL);
            offsets[0] = cst0;
            warp0Ptr = b.create<tt::MakeTensorPtrOp>(
                loc, ptrTy, base, shape, strides, offsets,
                b.getDenseI32ArrayAttr({1, 0}));
          };
          // barrier
          b.create<mlir::gpu::BarrierOp>(loc);
          // warp 0 load values from slm
          auto cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             warpId, cst0);
          scf::IfOp ifOp;
          // fixed 0 for now
          bool broadcast = reduce.getDstWarps().empty();
          if (broadcast)
            ifOp = b.create<scf::IfOp>(loc, cmp, false);
          else
            ifOp = b.create<scf::IfOp>(loc, tType, cmp, true /*hasElse*/);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(ifOp.getBody());
            auto attr =
                ttgi::WarpEncodingAttr::get(ctx, slmShapeI32, {1, 1}, {1, 0});
            auto slmType = RankedTensorType::get(slmShape, elmTy, attr);
            auto newPtrTy = tt::PointerType::get(slmType, 3);
            auto allWarpPtr = b.create<tt::MakeTensorPtrOp>(
                loc, newPtrTy, base, shape, strides, offsets,
                b.getDenseI32ArrayAttr({1, 0}));
            Value load =
                b.create<tt::LoadOp>(loc, allWarpPtr, tt::CacheModifier::NONE,
                                     tt::EvictionPolicy::NORMAL, false);
            // warp 0 doing reduction
            auto subRed = b.create<tt::ReduceOp>(loc, load, 0 /*axis*/);
            mlir::Region &subRegion = subRed.getCombineOp();
            {
              OpBuilder::InsertionGuard guard(b);
              mlir::Block *block = b.createBlock(&subRegion, Region::iterator(),
                                                 {elmTy, elmTy}, {loc, loc});
              // mlir::Block &block = subRegion.front();
              b.setInsertionPointToStart(block);
              Value red;
              if (reduce.getCombine() == tt::RMWOp::MAX)
                red = b.create<arith::MaxNumFOp>(loc, block->getArgument(0),
                                                 block->getArgument(1));
              else if (reduce.getCombine() == tt::RMWOp::FADD)
                red = b.create<arith::AddFOp>(loc, block->getArgument(0),
                                              block->getArgument(1));
              else
                assert(0 && "more support");
              b.create<tt::ReduceReturnOp>(loc, red);
            }
            // warp 0 store value to slm_base
            if (numElms == 1) {
              Value redCast =
                  b.create<tt::BitcastOp>(loc, elmTy, subRed.getResult()[0]);
              b.create<tt::StoreOp>(loc, base, redCast, tt::CacheModifier::NONE,
                                    tt::EvictionPolicy::NORMAL);
            } else {
              Value redCast =
                  b.create<tt::BitcastOp>(loc, tType, subRed.getResult()[0]);
              if (broadcast) {
                b.create<tt::StoreOp>(loc, warp0Ptr, redCast,
                                      tt::CacheModifier::NONE,
                                      tt::EvictionPolicy::NORMAL);
              } else {
                b.create<scf::YieldOp>(loc, redCast);
                b.setInsertionPointToStart(&ifOp.getElseRegion().front());
                mlir::Attribute zeroAttr =
                    b.getZeroAttr(tType.getElementType());
                auto val = DenseElementsAttr::get(cast<mlir::ShapedType>(tType),
                                                  zeroAttr);
                Value zero = b.create<arith::ConstantOp>(loc, tType, val);
                b.create<scf::YieldOp>(loc, zero);
                // auto yield =
                //     cast<scf::YieldOp>(ifOp.getBody()->getTerminator());
                // yield.getResultsMutable().assign({redCast});
              }
            }
          }
          // each warp get the reduction value
          if (broadcast) {
            // barrier
            b.create<mlir::gpu::BarrierOp>(loc);
            // each warp load value from slm_base
            Value load;
            if (numElms == 1) {
              load = b.create<tt::LoadOp>(loc, base, tt::CacheModifier::NONE,
                                          tt::EvictionPolicy::NORMAL, false);
              load = b.create<tt::BitcastOp>(loc, tType, load);
            } else {
              load =
                  b.create<tt::LoadOp>(loc, warp0Ptr, tt::CacheModifier::NONE,
                                       tt::EvictionPolicy::NORMAL, false);
            }
            result.replaceAllUsesWith(load);
          }
          // do not need broadcast to each warp
          else {
            result.replaceAllUsesWith(ifOp.getResults().front());
          }
          // update base
          auto size =
              b.create<arith::ConstantIntOp>(loc, numElms * numWarps, 32);
          base = b.create<tt::AddPtrOp>(loc, base.getType(), base, size);
        }
        op->erase();
      }

      // merge scf.if
      func.walk<WalkOrder::PreOrder>([&](tt::StoreOp op) {
        if (auto parent = dyn_cast<scf::IfOp>(op->getParentOp())) {
          auto def = dyn_cast<scf::IfOp>(op.getValue().getDefiningOp());
          // if (def && parent.getCondition() == def.getCondition())
          if (def) {
            Block *currBody = parent.getBody();
            Block *body = def.getBody();
            Value src =
                cast<scf::YieldOp>(body->getTerminator()).getResults().front();
            body->getTerminator()->erase();
            currBody->getOperations().splice(currBody->begin(),
                                             body->getOperations());
            op.getValue().replaceAllUsesWith(src);
            def->erase();
          }
        }
      });

      // // move load into if
      // func.walk<WalkOrder::PreOrder>([&](tt::StoreOp op) {
      //   if (isa<scf::IfOp>(op->getParentOp())) {
      //     auto def = op.getValue().getDefiningOp();
      //     if (def->getParentOp() != op->getParentOp()) {
      //       def->moveBefore(op);
      //     }
      //   }
      // });
    }
  }
};
