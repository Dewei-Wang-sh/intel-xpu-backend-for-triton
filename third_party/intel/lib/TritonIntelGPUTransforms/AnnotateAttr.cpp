//===- AnnotateAttr.cpp - annotate with layout attribute -*-C++ -*-===//
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

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
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
#define GEN_PASS_DEF_TRITONINTELGPUANNOTATEATTR
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {
// pass named attrs (e.g., tt.contiguity) from Triton to TritonGPU
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}
} // namespace

class TritonIntelGPUAnnotateAttrPass
    : public triton::gpu::intel::impl::TritonIntelGPUAnnotateAttrBase<
          TritonIntelGPUAnnotateAttrPass> {
private:
  DenseMap<Value, Attribute> valueAttrMap;
  Dialect *arithDialect = nullptr;
  Dialect *mathDialect = nullptr;
  MLIRContext *ctx = nullptr;

public:
  // LogicalResult initialize(MLIRContext *context) override {
  //   arithDialect = context->getLoadedDialect("arith");
  //   mathDialect = context->getLoadedDialect("math");
  //   ctx = &getContext();
  //   valueAttrMap.clear();
  //   return success();
  // }

  void runOnOperation() override {
    ctx = &getContext();
    arithDialect = ctx->getLoadedDialect("arith");
    mathDialect = ctx->getLoadedDialect("math");
    valueAttrMap.clear();
    ModuleOp mod = getOperation();

    /// find dot that has tiling attr
    for (auto func : mod.getOps<tt::FuncOp>()) {
      valueAttrMap.clear();
      tt::DotOp dot;
      bool found = false;
      func.walk([&](tt::DotOp op) {
        if (op->hasAttr("tiling") && found == false) {
          dot = op;
          found = true;
        }
      });
      if (!found)
        continue;

      /// get valueAttrMap for all
      // blockedencoding  or warpencoding
      SmallVector<unsigned> shape(dot.getType().getShape());
      auto dotLayout = ttgi::WarpEncodingAttr::get(ctx, shape, {1, 1}, {1, 0});
      expandDefChain(dot, dotLayout);
      expandUseChain(dot, dotLayout);

      /// adding tensor layout attr to related ops
      auto opHasTensorType = [&](Operation *op) {
        auto oprndHasTensorType = llvm::any_of(op->getOperandTypes(),
                                               tt::isTensorOrTensorPointerType);
        auto resultHasTensorType =
            llvm::any_of(op->getResultTypes(), tt::isTensorOrTensorPointerType);
        return oprndHasTensorType || resultHasTensorType;
      };
      func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (!opHasTensorType(op))
          return WalkResult::advance();

        auto numResults = op->getResults().size();
        if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
          transformArithConstantOp(cst, valueAttrMap[cst]);
        } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
          transformScfForOp(loop);
        } else if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
          ;
        } else if (numResults != 0) {
          assert(numResults == 1 && "only support 1 result");
          transformGenericOp(op);
        }
        return WalkResult::advance();
      });
    }

    /// adding module attributes
    // num-warps already got???numWarps.getValue()
    auto i32Ty = IntegerType::get(ctx, 32);
    mod->setAttr(tt::AttrNumWarpsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 16)));
    mod->setAttr(tt::AttrNumThreadsPerWarp,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
    mod->setAttr(tt::AttrNumCTAsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
  }

  Type addAttrToType(Type type, Attribute attr) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return RankedTensorType::get(tensorType.getShape(),
                                   tensorType.getElementType(), attr);
    else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      auto newPointeeType = addAttrToType(ptrType.getPointeeType(), attr);
      return tt::PointerType::get(newPointeeType, ptrType.getAddressSpace());
    }
    return type;
  }

  void transformArithConstantOp(arith::ConstantOp op, Attribute attr) {
    auto newType = addAttrToType(op.getType(), attr);
    auto value = cast<DenseElementsAttr>(op.getValue());
    value = value.reshape(newType.cast<ShapedType>());
    OpBuilder b(op);
    auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);
    addNamedAttrs(newOp, op->getAttrDictionary());
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();
    return;
  }

  void transformScfForOp(scf::ForOp op) {
    auto body = op.getBody();
    for (auto [lhs, rhs] :
         llvm::zip(body->getArguments().drop_front(1), op.getInitArgs()))
      lhs.setType(rhs.getType());
    for (auto i = 0; i < op->getResults().size(); i++) {
      auto init = op.getInitArgs()[i];
      auto type = init.getType();
      op->getResult(i).setType(type);
    }
    return;
  }

  void transformGenericOp(Operation *op) {
    // if already got
    auto result = op->getResults()[0];
    assert(valueAttrMap.count(result) != 0);
    auto newType = addAttrToType(result.getType(), valueAttrMap[result]);
    result.setType(newType);

    // get the attr by propagating
    // else if (op->getDialect() == arithDialect ||
    //          op->getDialect() == mathDialect || isa<tt::BroadcastOp>(op)) {
    //   Attribute attr;
    //   for (auto operand : op->getOperands()) {
    //     if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
    //       if (type.getEncoding())
    //         attr = type.getEncoding();
    //   }
    //   auto newType = addAttrToType(result.getType(), attr);
    //   result.setType(newType);
    // } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(op)) {
    //   auto src = expand.getSrc();
    //   auto attr = cast<ttg::SliceEncodingAttr>(src.getType().getEncoding());
    //   Type newType = addAttrToType(result.getType(), attr.getParent());
    //   result.setType(newType);
    // }
  }

  void wrapUseChain(Value val, Attribute attr) {
    if (valueAttrMap.count(val) == 0) {
      valueAttrMap[val] = attr;
      expandUseChain(val, attr);
    } else {
      assert(valueAttrMap.at(val) == attr);
    }
  }

  void expandUseChain(Value val, Attribute attr) {
    for (auto user : val.getUsers()) {
      if (user->getDialect() == arithDialect ||
          user->getDialect() == mathDialect ||
          isa<tt::AllReduceOp, tt::StoreOp, tt::BroadcastOp, tt::LoadOp>(
              user)) {
        if (user->getResults().size() != 0) {
          auto result = user->getResults().front();
          wrapUseChain(result, attr);
        }
        for (auto operand : user->getOperands()) {
          expandDefChain(operand, attr);
        }
      } else if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        unsigned resNum = -1;
        unsigned i = 0;
        for (auto operand : user->getOperands()) {
          if (operand == val) {
            resNum = i;
            break;
          }
          i++;
        }
        auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
        auto res = loop.getResult(resNum);
        wrapUseChain(res, attr);
      } else if (auto reduce = dyn_cast<tt::ReduceOp>(user)) {
        assert(reduce.getSrcs().size() == 1);
        auto axis = reduce.getAxis();
        auto src = reduce.getSrcs()[0];
        auto sliceAttr = ttg::SliceEncodingAttr::get(ctx, axis, attr);
        auto result = reduce.getResults()[0];
        wrapUseChain(result, sliceAttr);
        // ? expand
      } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(user)) {
        auto sAttr = cast<ttg::SliceEncodingAttr>(attr);
        auto result = expand.getResult();
        wrapUseChain(result, sAttr.getParent());
      } else if (auto dot = dyn_cast<tt::DotOp>(user)) {
        ;
        // } else if (isa< tt::SplatOp>(user)) {
      } else {
        assert(0 && "add more support");
      }
    }
  }

  void expandDefChain(Value val, Attribute attr) {
    if (valueAttrMap.count(val)) {
      assert(valueAttrMap.at(val) == attr);
      return;
    }
    valueAttrMap[val] = attr;
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      assert(loop);
      auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
      expandDefChain(loopArg, attr);
    } else if (auto def = val.getDefiningOp()) {
      // include arith::ConstantOp
      if (def->getDialect() == arithDialect ||
          def->getDialect() == mathDialect ||
          isa<tt::BroadcastOp, tt::AllReduceOp, tt::LoadOp, tt::BroadcastOp>(
              def)) {
        for (auto operand : def->getOperands()) {
          expandDefChain(operand, attr);
          expandUseChain(operand, attr);
        }
      } else if (auto dot = dyn_cast<tt::DotOp>(def)) {
        auto dotALayout = ttg::DotOperandEncodingAttr::get(ctx, 0, attr, 0);
        auto dotBLayout = ttg::DotOperandEncodingAttr::get(ctx, 1, attr, 0);
        expandDefChain(dot.getA(), dotALayout);
        expandUseChain(dot.getA(), dotALayout);
        expandDefChain(dot.getB(), dotBLayout);
        expandUseChain(dot.getB(), dotBLayout);
        expandDefChain(dot.getC(), attr);
        expandUseChain(dot.getC(), attr);
      } else if (auto reduce = dyn_cast<tt::ReduceOp>(def)) {
        assert(reduce.getSrcs().size() == 1);
        auto src = reduce.getSrcs()[0];
        auto sAttr = cast<ttg::SliceEncodingAttr>(attr);
        expandDefChain(src, sAttr.getParent());
        //  or just do nothing about expand??
      } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(def)) {
        auto src = expand.getSrc();
        auto axis = expand.getAxis();
        auto sliceAttr = ttg::SliceEncodingAttr::get(ctx, axis, attr);
        expandDefChain(src, sliceAttr);
      } else if (isa<tt::MakeTensorPtrOp, tt::AllocOp>(def)) {
        ;
        // } else if (isa<tt::SplatOp>(def)) {
        //   ;
      } else {
        assert(0 && "add more support");
      }
    } else {
      assert(0 && "add more support");
    }
  }
};
