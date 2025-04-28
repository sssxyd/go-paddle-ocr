// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"

#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

std::vector<ir::Var> ExprVec2VarVec(const std::vector<ir::Expr>& in) {
  std::vector<ir::Var> out;
  for (auto& expr : in) {
    out.push_back(expr.as_var_ref());
  }
  return out;
}

std::vector<ir::Expr> VarVec2ExprVec(const std::vector<ir::Var>& in) {
  return std::vector<ir::Expr>(in.begin(), in.end());
}

std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Expr& body,
                                            const ir::Tensor& tensor) {
  VLOG(4) << "GetEachTensorLoadExpr: " << tensor;
  std::vector<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesInOrder(
      body, [&tensor](const Expr* expr) {
        return expr->As<ir::Load>() && expr->As<ir::Load>()->is_addr_tensor() &&
               expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                   tensor->name;
      });
  for (auto& t : load_exprs) {
    VLOG(4) << "GetEachTensorLoadExpr Found: " << t << " " << t.ptr();
  }
  return load_exprs;
}

MappingTargetExprToDestExprMutator::MappingTargetExprToDestExprMutator(
    const ir::Expr& source, const ir::Expr& dest)
    : source_(source), dest_(dest) {}

void MappingTargetExprToDestExprMutator::operator()(Expr* expr) {
  IRMutator::Visit(expr, expr);
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Load* load, Expr* op) {
  if (load == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(load, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(
    const ir::ScheduleBlockRealize* r, Expr* op) {
  if (r == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(r, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::For* for_node,
                                               Expr* op) {
  if (for_node == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(for_node, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Block* block_node,
                                               Expr* op) {
  if (block_node == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(block_node, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Store* store,
                                               Expr* op) {
  if (store == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(store, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Reduce* reduce,
                                               Expr* op) {
  if (reduce == source_.ptr()) {
    *op = dest_;
  } else {
    IRMutator::Visit(reduce, op);
  }
}

void MappingTargetExprToDestExprMutator::Visit(const ir::_Var_* var, Expr* op) {
  if (source_.is_var() && source_.as_var_ref()->name == var->name) {
    *op = dest_;
  } else {
    IRMutator::Visit(var, op);
  }
}

bool CheckIterEq(const std::vector<ir::Var>& up_iter,
                 const std::vector<ir::Var>& down_iter) {
  if (up_iter.size() != down_iter.size()) return false;

  for (int i = 0; i < up_iter.size(); ++i) {
    const ir::Var& up_iter_var = up_iter[i];
    const ir::Var& down_iter_var = down_iter[i];

    if (up_iter_var != down_iter_var) return false;
    if (up_iter_var->lower_bound.as_int64() !=
        down_iter_var->lower_bound.as_int64())
      return false;
    if (up_iter_var->upper_bound.as_int64() !=
        down_iter_var->upper_bound.as_int64())
      return false;
  }
  return true;
}

ir::Expr CopiedReplaceExpr(const Expr& source,
                           const std::vector<Var>& replaced,
                           const std::vector<Expr>& candidates) {
  VLOG(4) << "CopiedReplaceExpr Start";
  VLOG(4) << "Replace Body : " << source;
  VLOG(4) << "Replace From : " << cinn::utils::Join(replaced, " ");
  VLOG(4) << "Replace To   : " << cinn::utils::Join(candidates, " ");

  PADDLE_ENFORCE_EQ(
      replaced.size(),
      candidates.size(),
      ::common::errors::InvalidArgument(
          "In ReplaceExpr, the size of Vars to be replaced must be equal to "
          "the size of candidate Exprs! Please check."));
  auto copied_source = ir::ir_utils::IRCopy(source);
  if (replaced.empty()) return copied_source;
  std::map<Var, Expr, ir::CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  ir::MappingVarToExprMutator mapper(replacing_map);
  mapper(&copied_source);
  VLOG(4) << "CopiedReplaceExpr Result: " << copied_source;
  return copied_source;
}

void SubstituteTargetExprWithDestExpr(const ir::Expr& source,
                                      const ir::Expr& dest,
                                      ir::Expr* body) {
  VLOG(4) << "SubstitideExpr Start";
  VLOG(5) << "Substitide Body : " << *body;
  ir::Expr new_dest = dest;
  if (source.type() != dest.type()) {
    VLOG(4) << "Cast the dest" << dest << " to type" << source.type();
    new_dest = ir::Cast::Make(source.type(), dest);
  }
  VLOG(4) << "Substitide From : " << source;
  VLOG(4) << "Substitide To   : " << new_dest;
  MappingTargetExprToDestExprMutator mapper(source, new_dest);
  mapper(body);
  VLOG(5) << "SubstitideExpr Result: " << *body;
}

ir::Expr SubstituteIndexVector(const Expr& source,
                               const std::vector<Var>& load_vars,
                               const std::vector<ir::Expr>& indices) {
  return CopiedReplaceExpr(source, load_vars, indices);
}
}  // namespace ComposeUtils

namespace ExprSetFinderUtils {

using ExprSet = std::vector<ir::Expr>;
using Expr2ExprSet = std::function<ExprSet(const ir::Expr& x)>;
ExprSetFinder::ExprSetFinder(Expr2ExprSet f, std::string s) {
  f_ = f;
  name = s;
}
ExprSet ExprSetFinder::operator()(const ir::Expr& x) const { return f_(x); }
ir::Expr ExprSetFinder::GetSingle(const ir::Expr& x) const {
  ExprSetFinder call = (*this) * ExprSetFinder::GetIdentity();
  const auto& o = call.operator()(x);
  PADDLE_ENFORCE_EQ(o.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "Try to get single result, but we get %d. \nFinder: "
                        "%s. \nRoot:\n%s \nResult:\n%s",
                        o.size(),
                        call.name,
                        x,
                        cinn::utils::Join(o, "\n")));
  return *o.begin();
}

ExprSetFinder ExprSetFinder::operator*(ExprSetFinder x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ExprSet {
    const auto& rs = self.f_(e);
    VLOG(6) << "ExprSetFinder Info : " << self.name;
    VLOG(6) << "        Inputs  :" << e;
    for (const auto& r : rs) {
      VLOG(6) << "      Outputs : \n" << r;
    }
    std::vector<ir::Expr> res;
    for (const auto& r : rs) {
      const auto& x_res = x.f_(r);
      res.insert(res.end(), x_res.begin(), x_res.end());
    }
    return res;
  };
  return ExprSetFinder(std::function(new_f), x.name + " * " + this->name);
}

ExprSetFinder ExprSetFinder::GetIdentity() {
  return ExprSetFinder(
      [](const ir::Expr& e) { return std::vector<ir::Expr>{e}; }, "identity");
}

ExprSetFinder Identity = ExprSetFinder::GetIdentity();

ExprSetFinder Store2Value = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::Store>()) {
        return {e.As<ir::Store>()->value};
      }
      return {};
    },
    "Store2Value");

ExprSetFinder Realizer2ScheduleBlock = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlockRealize>()) {
        return {e.As<ir::ScheduleBlockRealize>()->schedule_block};
      }
      return {};
    },
    "Realizer2ScheduleBlock");

ExprSetFinder Realizer2IterValues = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlockRealize>()) {
        return e.As<ir::ScheduleBlockRealize>()->iter_values;
      }
      return {};
    },
    "Realizer2IterValues");

ExprSetFinder ScheduleBlock2Body = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlock>()) {
        return {e.As<ir::ScheduleBlock>()->body};
      }
      return {};
    },
    "ScheduleBlock2Body");

ExprSetFinder ScheduleBlockRealizeNotRoot = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("root") == std::string::npos);
    },
    "ScheduleBlockRealizeNotRoot");

ExprSetFinder ScheduleBlockRealizeIsRoot = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("root") != std::string::npos);
    },
    "ScheduleBlockRealizeIsRoot");

ExprSetFinder ScheduleBlockRealizeIsNotInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") == std::string::npos);
    },
    "ScheduleBlockRealizeIsNotInit");

ExprSetFinder ScheduleBlockRealizeIsInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") != std::string::npos);
    },
    "ScheduleBlockRealizeIsInit");

ExprSetFinder IsFor = FilterMaker(
    [](const ir::Expr& e) -> bool { return e.As<ir::For>(); }, "IsFor");

ExprSetFinder ChildScheduleBlocks =
    Collector([](const ir::Expr* e) { return e->As<ir::ScheduleBlock>(); },
              "ChildScheduleBlocks");

ExprSetFinder ChildScheduleBlockRealizes =
    Collector(
        [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); },
        "ChildScheduleBlockRealizes") *
    ScheduleBlockRealizeNotRoot;

ExprSetFinder ChildRootScheduleBlockRealizes =
    Collector(
        [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); },
        "ChildScheduleBlockRealizes") *
    ScheduleBlockRealizeIsRoot;

ExprSetFinder IsForIterVar(const ir::Var& var) {
  return FilterMaker(
      [var = var](const ir::Expr& e) -> bool {
        return e.As<ir::For>() && e.As<ir::For>()->loop_var == var;
      },
      "IsForIterVar");
}

ExprSetFinder For2Min = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->min}; },
    "For2Min");

ExprSetFinder For2Max = ExprSetFinder(
    [](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->extent}; },
    "For2Max");

ExprSetFinder ChildStores = Collector(
    [](const ir::Expr* e) { return e->As<ir::Store>(); }, "ChildStores");

ExprSetFinder ChildTensorLoads = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Load>() && e->As<ir::Load>()->is_addr_tensor();
    },
    "ChildLoads");

ExprSetFinder ChildTensorStores = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Store>() && e->As<ir::Store>()->is_addr_tensor();
    },
    "ChildTensorStores");

ExprSetFinder FilterLoadByTensor(const ir::Tensor& tensor) {
  return FilterMaker(
      [tensor = tensor](const ir::Expr& e) -> bool {
        return e.As<ir::Load>() &&
               e.As<ir::Load>()->tensor.as_tensor_ref()->name == tensor->name;
      },
      "FilterLoadByTensor(" + tensor->name + ")");
}

ExprSetFinder ChildFors =
    Collector([](const ir::Expr* e) { return e->As<ir::For>(); }, "ChildFors");

ExprSetFinder ChildIfThenElses =
    Collector([](const ir::Expr* e) { return e->As<ir::IfThenElse>(); },
              "ChildIfThenElses");

ExprSetFinder ChildVars =
    Collector([](const ir::Expr* e) { return e->as_var(); }, "ChildVars");

ExprSetFinder FindFather(const ir::Expr& root) {
  const auto& f = [root](const auto& child) -> ExprSet {
    ExprSetFinder find_child =
        Collector([child](const ir::Expr* e) { return *e == child; });
    const auto& father_collector = Collector([&](const ir::Expr* current) {
      auto res = (*current != child) && !find_child(*current).empty();
      return res;
    });
    return father_collector(root);
  };
  return ExprSetFinder(f, "FindFather");
}

ExprSetFinder DirectlyFather(const ir::Expr& root) {
  const auto& f = [root](const auto& child) -> ExprSet {
    ExprSet result = FindFather(root)(child);
    // VLOG(4) << "Directly Father of \n" << child << "\nIn root: \n" << root <<
    // "\n is : "; for (const auto& r: result){ VLOG(4) << "\n  RESULT: " << r;
    //}
    return {result[result.size() - 1]};
  };
  return ExprSetFinder(f, "DirectlyFather");
}

}  // namespace ExprSetFinderUtils

namespace ExprTransformerUtils {
using ExprTransformFunc = std::function<ir::Expr(ir::Expr)>;

ExprTransformer::ExprTransformer(ExprTransformFunc f) { f_ = f; }
ir::Expr ExprTransformer::operator()(const ir::Expr& x) const { return f_(x); }
ExprTransformer ExprTransformer::operator*(const ExprTransformer& x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ir::Expr {
    const auto& rs = self.f_(e);
    return x.f_(rs);
  };
  return ExprTransformer(std::function(new_f));
}

ExprTransformer Identity = ExprTransformer([](const ir::Expr& e) { return e; });
ExprTransformer WrapForTransformer(const ir::Var& v) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    auto block = e;
    if (!block.As<ir::Block>()) {
      block = ir::Block::Make({e});
    }
    return ir::For::Make(v,
                         v->lower_bound,
                         v->upper_bound,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         block);
  };
  return ExprTransformer(f);
}

ExprTransformer WrapForsTransformer(const std::vector<ir::Var>& vs) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    ExprTransformer t = Identity;
    for (const auto& v : vs) {
      t = WrapForTransformer(v) * t;
    }
    return t(e);
  };
  return ExprTransformer(f);
}

ExprTransformer UnsqueezeForTransformer(
    const ExprSetFinderUtils::ExprSetFinder& followed_finder,
    const ir::Var& to_append_var) {
  const auto& suqueeze_for_func = [&](const ir::Expr& e) -> ir::Expr {
    auto copied_e = ir::ir_utils::IRCopy(e);
    ir::Expr followed_expr = followed_finder.GetSingle(copied_e);
    // (ExprSetFinderUtils::ChildFors *
    // ExprSetFinderUtils::IsForIterVar(following_for_iter_var)).GetSingle(copied_e);
    VLOG(6) << "UnsqueezeForTransformer: for insert after " << followed_expr;
    if (followed_expr.As<ir::For>()) {
      followed_expr.As<ir::For>()->body = ir::Block::Make({WrapForTransformer(
          to_append_var)(followed_expr.As<ir::For>()->body)});
    } else if (followed_expr.As<ir::ScheduleBlockRealize>()) {
      const auto& schedule_block = followed_expr.As<ir::ScheduleBlockRealize>()
                                       ->schedule_block.As<ir::ScheduleBlock>();
      schedule_block->body =
          WrapForTransformer(to_append_var)(schedule_block->body);
    } else {
      PADDLE_THROW(::common::errors::PreconditionNotMet(
          "UnsqueezeForTransformer: only support insert after a (For / "
          "ScheduleBlockRealizer)"));
    }
    VLOG(6) << "UnsqueezeForTransformer: After changed: " << copied_e;
    return copied_e;
  };
  return ExprTransformer(suqueeze_for_func);
}

ExprTransformer ChangeTensorLoadTransformer(const ir::Tensor& tensor,
                                            const ir::Expr& dst_load) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    auto copied_e = ir::ir_utils::IRCopy(e);
    const auto& load = (ExprSetFinderUtils::ChildTensorLoads *
                        ExprSetFinderUtils::FilterLoadByTensor(tensor))
                           .GetSingle(copied_e);
    ComposeUtils::MappingTargetExprToDestExprMutator(load, dst_load)(&copied_e);
    return copied_e;
  };
  return ExprTransformer(f);
}

void ReplaceTarget(ir::Expr* e, const ir::Expr& t, const ir::Expr dst) {
  ComposeUtils::MappingTargetExprToDestExprMutator(t, dst)(e);
}

ExprTransformer WrapStoreTransformer(const ir::Tensor& tensor,
                                     const std::vector<ir::Expr>& indices) {
  const auto& MakeStoreNode = [=](const ir::Expr& e) -> ir::Expr {
    return ir::Store::Make(tensor, e, indices);
  };
  return ExprTransformer(MakeStoreNode);
}

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars) {
  int i = 0;
  std::vector<ir::Var> vars;
  for (const auto& v : block_vars) {
    vars.emplace_back("inner_block_" + std::to_string(i++));
    vars.back()->is_reduce_axis = v->is_reduce_axis;
  }
  return vars;
}

ExprTransformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                     const std::vector<ir::Var>& dest_vars) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopiedReplaceExpr(
        e,
        target_vars,
        std::vector<ir::Expr>(dest_vars.begin(), dest_vars.end()));
  };
  return ExprTransformer(f);
}

ExprTransformer ReplaceVarTransformer(const std::vector<ir::Var>& target_vars,
                                      const std::vector<ir::Expr>& dest_expr) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopiedReplaceExpr(e, target_vars, dest_expr);
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveVarInScheduleBlockRealize(const ir::Var& target_vars,
                                                const ir::Expr& replaced_expr) {
  /*
   * remove var in schedule block realize, replace it with replaced_expr and
   * remove it in axes.bind()
   */
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    PADDLE_ENFORCE_NE(
        e.As<ir::ScheduleBlockRealize>(),
        nullptr,
        ::common::errors::InvalidArgument(
            "RemoveVarInScheduleBlockRealize: input expr is not a "
            "ScheduleBlockRealize."));
    auto copied_ir = ir::ir_utils::IRCopy(e);
    auto schedule_block_iter_vars =
        copied_ir.As<ir::ScheduleBlockRealize>()->iter_values;
    auto block_bound_vars = copied_ir.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars;
    // find replace idx
    int target_idx = -1;
    for (int i = 0; i < schedule_block_iter_vars.size(); ++i) {
      if (schedule_block_iter_vars[i].is_var() &&
          schedule_block_iter_vars[i].as_var()->name == target_vars->name) {
        target_idx = i;
      }
    }
    if (target_idx == -1) {
      return copied_ir;  // do nothing, can't find target vars;
    }
    return ir::ScheduleBlockRealize::Make(
        fusion::GatherVectorExcept<ir::Expr, int>(schedule_block_iter_vars,
                                                  {target_idx}),
        ir::ScheduleBlock::Make(
            fusion::GatherVectorExcept<ir::Var, int>(block_bound_vars,
                                                     {target_idx}),
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->read_buffers,
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->write_buffers,
            copied_ir.As<ir::ScheduleBlockRealize>()
                ->schedule_block.As<ir::ScheduleBlock>()
                ->name,
            ComposeUtils::CopiedReplaceExpr(
                copied_ir.As<ir::ScheduleBlockRealize>()
                    ->schedule_block.As<ir::ScheduleBlock>()
                    ->body,
                {block_bound_vars[target_idx]},
                {replaced_expr})));
  };
  return ExprTransformer(f);
}

bool IsReduceBool(const ir::Expr& lhs, const ir::Expr& rhs) {
  return lhs.type().is_bool() || rhs.type().is_bool();
}

ExprTransformer WrapReduceOperation(const ir::Reduce::ReduceType& reduce_type,
                                    const ir::Tensor& tensor,
                                    const std::vector<ir::Expr>& axis_exprs) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    switch (reduce_type) {
      case ir::Reduce::kSum:
        if (IsReduceBool(tensor(axis_exprs), e)) {
          return ir::Store::Make(tensor, tensor(axis_exprs) || e, axis_exprs);
        }
        return ir::Store::Make(tensor, tensor(axis_exprs) + e, axis_exprs);
      case ir::Reduce::kMul:
        if (IsReduceBool(tensor(axis_exprs), e)) {
          return ir::Store::Make(tensor, tensor(axis_exprs) && e, axis_exprs);
        }
        return ir::Store::Make(tensor, tensor(axis_exprs) * e, axis_exprs);
      case ir::Reduce::kMax:
        return ir::Store::Make(
            tensor, ir::Max::Make(tensor(axis_exprs), e), axis_exprs);
      case ir::Reduce::kMin:
        return ir::Store::Make(
            tensor, ir::Min::Make(tensor(axis_exprs), e), axis_exprs);
      case ir::Reduce::kAll:
        return ir::Store::Make(tensor, tensor(axis_exprs) && e, axis_exprs);
      case ir::Reduce::kAny:
        return ir::Store::Make(tensor, tensor(axis_exprs) || e, axis_exprs);
      default:
        CINN_NOT_IMPLEMENTED
    }
  };
  return ExprTransformer(f);
}

ExprTransformer SubstituteByScheduleBlockRealize(const ir::Expr& realize) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    const auto& iter_values =
        realize.As<ir::ScheduleBlockRealize>()->iter_values;
    const auto& iter_vars = realize.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars;
    return ExprTransformerUtils::ChangeVarTransformer(
        iter_vars, ComposeUtils::ExprVec2VarVec(iter_values))(e);
  };
  return ExprTransformer(f);
}

ExprTransformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                     const std::string& tensor_name) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    PADDLE_ENFORCE_EQ(e.As<ir::ScheduleBlock>(),
                      nullptr,
                      ::common::errors::InvalidArgument(
                          "please input a non-schedule block expr."));
    const auto& inner_block_var = CreateInnerBlockVars(block_vars);
    const auto& replaced_e =
        ChangeVarTransformer(block_vars, inner_block_var)(e);
    const auto& schedule_block = ir::ScheduleBlock::Make(
        inner_block_var, {}, {}, tensor_name, replaced_e);
    const auto& schedule_realizer = ir::ScheduleBlockRealize::Make(
        std::vector<ir::Expr>(block_vars.begin(), block_vars.end()),
        schedule_block);
    return schedule_realizer;
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveForTransformer(int axis) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    ir::Expr copied = ir::ir_utils::IRCopy(root);
    const auto& iters = GetAllLoopVars(copied);
    // Find target expr and replace with for->body.
    const ir::Expr& target_for = (ExprSetFinderUtils::ChildFors *
                                  ExprSetFinderUtils::IsForIterVar(iters[axis]))
                                     .GetSingle(copied);
    const ir::Expr& target_block =
        ExprSetFinderUtils::DirectlyFather(copied).GetSingle(target_for);
    if (target_block.As<ir::ScheduleBlockRealize>() != nullptr) {
      VLOG(4) << "RemoveForsTransformer: father block is root realize";
      ir::Expr schedule_block =
          target_block.As<ir::ScheduleBlockRealize>()->schedule_block;
      PADDLE_ENFORCE_EQ(
          schedule_block.As<ir::ScheduleBlock>()->body,
          target_for,
          ::common::errors::InvalidArgument(
              "Root realize body should be equal to target for."));
      const auto for_body = target_for.As<ir::For>()->body;
      const auto for_body_stmts = for_body.As<ir::Block>()->stmts;
      if (for_body_stmts.size() == 1 &&
          for_body_stmts[0].As<ir::For>() != nullptr) {
        schedule_block.As<ir::ScheduleBlock>()->body = for_body_stmts[0];
      } else {
        schedule_block.As<ir::ScheduleBlock>()->body = for_body;
      }
    } else if (target_block.As<ir::Block>() != nullptr) {
      std::vector<ir::Expr> new_bodies;
      for (const auto& expr : target_block.As<ir::Block>()->stmts) {
        if (expr != target_for) {
          new_bodies.push_back(expr);
        } else {
          for (const auto& origin_for :
               target_for.As<ir::For>()->body.As<ir::Block>()->stmts) {
            new_bodies.push_back(origin_for);
          }
        }
      }
      ir::Expr to_replace_block = ir::Block::Make(new_bodies);
      ComposeUtils::MappingTargetExprToDestExprMutator(
          target_block, to_replace_block)(&copied);
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "RemoveForsTransformer: target for father should be a ir::Block or "
          "ir::ScheduleBlockRealize."));
    }
    // Remove var to 0 in ScheduleBlockRealizer
    InplaceMutateSingleExpr(
        &copied,
        (ExprSetFinderUtils::ChildScheduleBlockRealizes *
         ExprSetFinderUtils::ScheduleBlockRealizeIsNotInit),
        RemoveVarInScheduleBlockRealize(iters[axis], ir::Expr(0)));
    InplaceMutateSingleExpr(
        &copied,
        (ExprSetFinderUtils::ChildScheduleBlockRealizes *
         ExprSetFinderUtils::ScheduleBlockRealizeIsInit),
        RemoveVarInScheduleBlockRealize(iters[axis], ir::Expr(0)));
    // Remove var in append if
    auto loop_var = target_for.As<ir::For>()->loop_var;
    auto realizes = (ExprSetFinderUtils::ChildScheduleBlockRealizes *
                     ExprSetFinderUtils::ScheduleBlockRealizeNotRoot)(copied);
    for (auto& realize : realizes) {
      ComposeUtils::MappingTargetExprToDestExprMutator(loop_var,
                                                       ir::Expr(0))(&realize);
    }
    return copied;
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveForsTransformer(const std::vector<int32_t>& axes) {
  ExprTransformer f = Identity;
  for (const auto& axis : axes) {
    f = RemoveForTransformer(axis) * f;
  }
  return ExprTransformer(f);
}

ExprTransformer TransposeForsTransformer(const std::vector<int32_t>& perm) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    const auto& iters = GetAllLoopVars(root);
    PADDLE_ENFORCE_GE(
        iters.size(),
        perm.size(),
        ::common::errors::InvalidArgument(
            "Transposed iters size and perm size should be equal."));
    for (size_t i = 0; i < perm.size(); ++i) {
      if (iters[i]->is_reduce_axis) {
        PADDLE_ENFORCE_EQ(i,
                          perm[i],
                          ::common::errors::InvalidArgument(
                              "Can only transpose non reduce iters."));
      }
    }
    const auto transposed_iters = cinn::fusion::TransposeVector(iters, perm);
    const auto non_reduce_iters = cinn::fusion::FilterVector(
        transposed_iters, [](const ir::Var& v) { return !v->is_reduce_axis; });
    const auto body_block = GetBodyBlock(root);
    return ir::Block::Make({(WrapForsTransformer(non_reduce_iters) *
                             WrapScheduleRealizer({}, "root"))(body_block)});
  };
  return ExprTransformer(f);
}

ExprTransformer InsertForsTransformer(const std::vector<int32_t>& axis,
                                      const std::vector<ir::Var>& vars) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    auto iters = GetNonReduceLoopVars(root);
    PADDLE_ENFORCE_EQ(
        axis.size(),
        vars.size(),
        ::common::errors::InvalidArgument(
            "The number of axis to insert and vars should be equal."));
    for (size_t i = 0; i < axis.size(); ++i) {
      iters.insert(iters.begin() + axis[i], vars[i]);
    }
    const auto body_block = GetBodyBlock(root);
    return ir::Block::Make({(WrapForsTransformer(iters) *
                             WrapScheduleRealizer({}, "root"))(body_block)});
  };
  return ExprTransformer(f);
}

ExprTransformer InsertIfForAppendVarsTransformer(
    const std::vector<ir::Var>& append_vars) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    std::function<std::string(ir::Var)> var2name =
        [](const ir::Var& v) -> std::string { return v->name; };
    std::set<std::string> var_names =
        cinn::fusion::ToSet(cinn::fusion::MapVector(append_vars, var2name));
    const auto vars = GetNonReduceLoopVars(root);
    std::vector<std::vector<ir::Var>> neighbor_append_vars;
    bool last_var_is_append = false;
    for (const auto& var : vars) {
      if (var_names.count(var->name) != 0 &&
          !(var->upper_bound.is_constant() &&
            var->upper_bound.as_int64() == 1)) {
        if (last_var_is_append) {
          neighbor_append_vars.back().push_back(var);
        } else {
          neighbor_append_vars.push_back({var});
          last_var_is_append = true;
        }
      } else {
        last_var_is_append = false;
      }
    }
    // Merge if for neighbor append vars
    std::vector<ir::Expr> conditions;
    for (const auto& vars : neighbor_append_vars) {
      ir::Expr lhs = Expr(0);
      ir::Expr rhs = Expr(1);
      ir::Expr offset = Expr(1);
      for (int i = vars.size() - 1; i >= 0; --i) {
        lhs = lhs + vars[i] * offset;
        rhs = rhs * vars[i]->lower_bound;
        offset = offset * (vars[i]->upper_bound - vars[i]->lower_bound);
      }
      conditions.push_back(ir::EQ::Make(lhs, rhs));
    }
    std::reverse(conditions.begin(), conditions.end());

    auto realizes = (ExprSetFinderUtils::ChildScheduleBlockRealizes *
                     ExprSetFinderUtils::ScheduleBlockRealizeNotRoot)(root);
    for (auto realize : realizes) {
      auto schedule_block =
          realize.As<ir::ScheduleBlockRealize>()->schedule_block;
      auto new_body = schedule_block.As<ir::ScheduleBlock>()->body;
      for (const auto& cond : conditions) {
        new_body = ir::IfThenElse::Make(cond, new_body, ir::Expr());
      }
      schedule_block.As<ir::ScheduleBlock>()->body = new_body;
    }
    return root;
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveTargetIfTransformer(
    const std::function<bool(const ir::Expr&)>& filter) {
  const auto& f = [=](const ir::Expr& root) -> ir::Expr {
    auto copied = root;
    const auto& iters = GetAllLoopVars(copied);
    // Find target if and replace with if body.
    const std::vector<ir::Expr>& all_if =
        ExprSetFinderUtils::ChildIfThenElses(copied);
    const auto all_target_if = cinn::fusion::FilterVector(all_if, filter);
    for (const auto& target_if : all_target_if) {
      const ir::Expr& target_block =
          ExprSetFinderUtils::DirectlyFather(copied).GetSingle(target_if);
      const auto if_body = target_if.As<ir::IfThenElse>()->true_case;
      const auto if_body_stmts = if_body.As<ir::Block>()->stmts;
      if (target_block.As<ir::ScheduleBlockRealize>()) {
        ir::Expr schedule_block =
            target_block.As<ir::ScheduleBlockRealize>()->schedule_block;
        if (if_body_stmts.size() == 1) {
          schedule_block.As<ir::ScheduleBlock>()->body = if_body_stmts[0];
        } else {
          schedule_block.As<ir::ScheduleBlock>()->body = if_body;
        }
      } else {
        ir::Expr to_replace_block = ir::Block::Make(if_body_stmts);
        ComposeUtils::MappingTargetExprToDestExprMutator(
            target_block, to_replace_block)(&copied);
      }
    }
    return copied;
  };
  return ExprTransformer(f);
}

ExprTransformer RemoveAllAppendIfTransformer() {
  const auto is_appeend_var_in_cond = [](const ir::Expr& if_expr) {
    auto cond = if_expr.As<ir::IfThenElse>()->condition.As<ir::EQ>();
    if (!cond) return false;
    for (const auto& var : ExprSetFinderUtils::ChildVars(cond->a())) {
      if (var.as_var()->name.find("append") != std::string::npos) return true;
    }
    return false;
  };
  return RemoveTargetIfTransformer(is_appeend_var_in_cond);
}

ExprTransformer EliminateUselessIfTransformer() {
  const auto has_useless_cond = [](const ir::Expr& if_expr) {
    auto cond = if_expr.As<ir::IfThenElse>()->condition.As<ir::EQ>();
    if (!cond) return false;
    for (const auto& var : ExprSetFinderUtils::ChildVars(cond->a())) {
      if (!var.as_var()->is_symbolic_constant) return false;
    }
    return true;
  };
  return RemoveTargetIfTransformer(has_useless_cond);
}

int InplaceMutateSingleExpr(ir::Expr* root,
                            const ExprSetFinderUtils::ExprSetFinder& finder,
                            const ExprTransformer& transformer) {
  // NOTE!!!
  // source ir::node type must be supported in
  // MappingTargetExprToDestExprMutator.
  const auto& source = finder(*root);
  if (source.empty()) {
    return 0;
  }
  PADDLE_ENFORCE_EQ(
      source.size(),
      1,
      ::common::errors::InvalidArgument("Only one expr should be found."));
  const auto& target = transformer(source[0]);
  ComposeUtils::MappingTargetExprToDestExprMutator(source[0], target)(root);
  return 1;  // operation number.
}

}  // namespace ExprTransformerUtils

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops) {
  const auto& op_pattern_map =
      Operator::GetAttrs<cinn::hlir::framework::OpPatternKind>("OpPattern");
  std::vector<OpPatternKind> op_patterns;
  const auto ConvertToPattern = [&op_pattern_map](const ::pir::Operation* op) {
    const std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    return op_pattern_map[cinn_op];
  };
  std::transform(ops.begin(),
                 ops.end(),
                 std::back_inserter(op_patterns),
                 ConvertToPattern);
  return op_patterns;
}

bool IsTrivialKind(OpPatternKind kind) {
  return kind == OpPatternKind::kElementWise ||
         kind == OpPatternKind::kBroadcast || kind == OpPatternKind::kInjective;
}

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns) {
  if (VLOG_IS_ON(4)) {
    for (const auto& func : op_compute_bodies) {
      VLOG(4) << "FuncBody is :" << func;
    }
    for (const auto& op_ptn : op_patterns) {
      VLOG(4) << "OpPattern is :" << op_ptn;
    }
  }
  VLOG(4) << "      op_patterns.size() = " << op_compute_bodies.size();
  VLOG(4) << "op_compute_bodies.size() = " << op_patterns.size();
  PADDLE_ENFORCE_EQ(
      op_patterns.size(),
      op_compute_bodies.size(),
      ::common::errors::InvalidArgument(
          "The number of op_compute_bodies and op_patterns should be equal."));
}

bool IsReducePattern(const ir::Expr& root) {
  return !(ExprSetFinderUtils::ChildScheduleBlockRealizes *
           ExprSetFinderUtils::ScheduleBlockRealizeIsInit)(root)
              .empty();
}

std::vector<ir::Var> AppendBound(const std::vector<ir::Var> vars,
                                 const ir::Expr& root) {
  return ExprSetFinderUtils::MapVector<ir::Var>(
      vars, [&](const auto& v) -> ir::Var {
        return ir::Var(
            (ExprSetFinderUtils::ChildFors *
             ExprSetFinderUtils::IsForIterVar(v) * ExprSetFinderUtils::For2Min)
                .GetSingle(root),
            (ExprSetFinderUtils::ChildFors *
             ExprSetFinderUtils::IsForIterVar(v) * ExprSetFinderUtils::For2Max)
                .GetSingle(root),
            v->name,
            v->is_reduce_axis);
      });
}

std::vector<ir::Var> GetNonReduceLoopVars(const ir::Expr& root) {
  const auto& fors_expr = (ExprSetFinderUtils::ChildFors)(root);
  std::vector<ir::Var> loop_vars;
  for (const auto& for_expr : fors_expr) {
    loop_vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  const auto non_reduce_loop_vars =
      IsReducePattern(root)
          ? cinn::fusion::FilterVector(
                loop_vars, [](const ir::Var& v) { return !v->is_reduce_axis; })
          : loop_vars;
  return AppendBound(non_reduce_loop_vars, root);
}

std::vector<ir::Var> GetAllLoopVars(const ir::Expr& root) {
  const auto& fors_expr = (ExprSetFinderUtils::ChildFors)(root);
  std::vector<ir::Var> loop_vars;
  for (const auto& for_expr : fors_expr) {
    loop_vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  return AppendBound(loop_vars, root);
}

std::vector<ir::Var> GetReduceLoopVars(const ir::Expr& root) {
  auto reduce_init = (ExprSetFinderUtils::ChildScheduleBlockRealizes *
                      ExprSetFinderUtils::ScheduleBlockRealizeIsInit)(root);
  if (!reduce_init.empty()) {
    auto father_block = ExprSetFinderUtils::DirectlyFather(root)
                            .GetSingle((reduce_init[0]))
                            .As<ir::Block>();
    PADDLE_ENFORCE(father_block && father_block->stmts.size() == 2,
                   ::common::errors::InvalidArgument(
                       "Father block size of reduce init should be 2."));
    auto reduce_body = father_block->stmts[1];
    return GetAllLoopVars(reduce_body);
  }
  return {};
}

ir::Expr GetBodyBlock(const ir::Expr& root) {
  const auto& iters = GetNonReduceLoopVars(root);
  auto block_realize =
      ExprSetFinderUtils::ChildScheduleBlockRealizes(root).front();
  return ExprSetFinderUtils::DirectlyFather(root).GetSingle(block_realize);
}

ir::Expr ReshapeLoop(const ir::Expr& root,
                     const std::vector<symbol::DimExpr>& input_shape,
                     const std::vector<symbol::DimExpr>& output_shape) {
  auto copied = ir::ir_utils::IRCopy(root);

  ir::ModuleExpr mod_expr({copied});
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, cinn::utils::ErrorMessageLevel::kGeneral, true);

  const auto block_realize =
      (ExprSetFinderUtils::ChildScheduleBlockRealizes)(copied)[0];
  const auto block_name = block_realize.As<ir::ScheduleBlockRealize>()
                              ->schedule_block.As<ir::ScheduleBlock>()
                              ->name;
  auto non_reduce_num = GetNonReduceLoopVars(copied).size();
  bool is_reduce = non_reduce_num < input_shape.size();
  auto in_shape =
      is_reduce ? cinn::fusion::SliceVector(input_shape, 0, non_reduce_num)
                : input_shape;
  auto out_shape =
      is_reduce ? cinn::fusion::SliceVector(output_shape, 0, non_reduce_num)
                : output_shape;

  const auto shape_partition =
      fusion::PartitionReshapeAxes(in_shape, out_shape);

  for (int idx = shape_partition.size() - 1; idx > 0; --idx) {
    const auto& in_s = shape_partition[idx - 1].first;
    const auto& in_e = shape_partition[idx].first;
    const auto& out_s = shape_partition[idx - 1].second;
    const auto& out_e = shape_partition[idx].second;

    std::vector<int> fuse_indices;
    for (int i = in_e - 1; i >= in_s; --i) {
      if (in_shape[i] != symbol::DimExpr(1)) {
        fuse_indices.insert(fuse_indices.begin(), i);
      } else {
        VLOG(4) << "Remove index[" << i << "]: " << in_shape[i]
                << " for expr: \n"
                << copied;
        copied = ExprTransformerUtils::RemoveForTransformer(i)(copied);
        ir_sch.SetExprs({copied});
        for (auto& index : fuse_indices) {
          index--;
        }
      }
    }
    if (fuse_indices.size() > 1) {
      VLOG(4) << "fuse_indices: " << cinn::utils::Join(fuse_indices, ",");
      ir_sch.Fuse(block_name, fuse_indices);
    }

    std::vector<ir::Expr> split_shapes;
    for (int i = out_s; i < out_e; ++i) {
      if (out_shape[i] != symbol::DimExpr(1)) {
        split_shapes.push_back(
            cinn::common::DimExprConverter().ConvertToIrExpr(out_shape[i]));
      }
    }
    if (split_shapes.size() > 1) {
      ir_sch.Split(ir_sch.GetLoops(block_name)[in_s], split_shapes)[0];
    }
  }

  std::vector<int> insert_axis;
  std::vector<ir::Var> ones_var;
  for (int i = 0; i < out_shape.size(); ++i) {
    if (out_shape[i] == symbol::DimExpr(1)) {
      insert_axis.push_back(i);
      ones_var.push_back(ir::Var(1, "one_" + std::to_string(ones_var.size())));
    }
  }
  copied = ExprTransformerUtils::InsertForsTransformer(insert_axis,
                                                       ones_var)(copied);

  return copied;
}

void CheckLoopAlignment(const std::vector<ir::Expr>& roots) {
  if (roots.size() < 2) return;

  auto var_equal = [](const ir::Var& lhs, const ir::Var& rhs) {
    auto index_equal = [](const ir::Expr& lhs, const ir::Expr& rhs) -> bool {
      return lhs.is_index() && rhs.is_index() ? lhs.as_index() == rhs.as_index()
                                              : lhs == rhs;
    };
    return index_equal(lhs->upper_bound, rhs->upper_bound) &&
           index_equal(lhs->lower_bound, rhs->lower_bound);
  };

  int base_loop_idx = -1;
  int base_reduce_idx = -1;
  std::vector<ir::Var> base_loop_vars;
  std::vector<ir::Var> base_reduce_vars;
  for (size_t i = 0; i < roots.size(); ++i) {
    const auto loop_vars = GetAllLoopVars(roots[i]);
    if (base_loop_idx < 0) {
      base_loop_vars = loop_vars;
      base_loop_idx = i;
      continue;
    }
    PADDLE_ENFORCE(fusion::VectorEqual(base_loop_vars, loop_vars, var_equal),
                   ::common::errors::PreconditionNotMet(
                       "CheckLoopAlignment Failed, The loop vars are not equal "
                       "between FusionOps: \n%s\n%s",
                       roots[base_loop_idx],
                       roots[i]));

    const auto reduce_vars = GetReduceLoopVars(roots[i]);
    if (!reduce_vars.empty()) {
      if (base_reduce_idx < 0) {
        base_reduce_vars = reduce_vars;
        base_reduce_idx = i;
        continue;
      }
      PADDLE_ENFORCE(
          fusion::VectorEqual(base_reduce_vars, reduce_vars, var_equal),
          ::common::errors::PreconditionNotMet(
              "CheckLoopAlignment Failed, The reduce vars are not equal "
              "between FusionOps: \n%s\n%s",
              roots[base_reduce_idx],
              roots[i]));
    }
  }
}

ir::Tensor GetOutputTensor(const ir::Expr& root) {
  return (ExprSetFinderUtils::ChildScheduleBlockRealizes *
          ExprSetFinderUtils::ScheduleBlockRealizeIsNotInit *
          ExprSetFinderUtils::ChildStores)
      .GetSingle(root)
      .As<ir::Store>()
      ->tensor.as_tensor_ref();
}

void InlineGlobalVarComputeImpl(const ir::Expr& global_root,
                                const std::vector<ir::Expr>& roots) {
  PADDLE_ENFORCE(!IsReducePattern(global_root),
                 ::common::errors::InvalidArgument(
                     "Can not inline global var compute for reduce pattern."));
  auto store = (ExprSetFinderUtils::ChildScheduleBlockRealizes *
                ExprSetFinderUtils::ScheduleBlockRealizeIsNotInit *
                ExprSetFinderUtils::ChildStores)
                   .GetSingle(global_root);
  auto store_indices = store.As<ir::Store>()->indices;
  std::vector<int> var_indices_pos;
  std::vector<ir::Var> target_indices;
  for (int i = 0; i < store_indices.size(); ++i) {
    if (store_indices[i].is_var()) {
      target_indices.push_back(store_indices[i].as_var_ref());
      var_indices_pos.push_back(i);
    }
  }
  auto target_tensor = store.As<ir::Store>()->tensor.as_tensor_ref();
  auto target_compute_body = (ExprSetFinderUtils::Store2Value).GetSingle(store);

  for (auto root : roots) {
    if (root == global_root) continue;
    SequenceMutator(
        ComposeUtils::GetEachTensorLoadExpr(root, target_tensor),
        &root,
        [&](const ir::Expr& load_expr, ir::Expr* compute_body) {
          ComposeUtils::SubstituteTargetExprWithDestExpr(
              load_expr,
              ComposeUtils::SubstituteIndexVector(
                  target_compute_body,
                  target_indices,
                  cinn::fusion::GatherVector(load_expr.As<ir::Load>()->indices,
                                             var_indices_pos)),
              compute_body);
        });
  }
}

void InlineGlobalVarCompute(const std::vector<ir::Expr>& roots,
                            const std::set<std::string>& global_var_names) {
  for (int i = 0; i < roots.size(); ++i) {
    if (IsReducePattern(roots[i])) continue;
    auto output_tensor_name = GetOutputTensor(roots[i])->name;
    if (global_var_names.count(output_tensor_name)) {
      VLOG(4) << "Inline compute of global var: " << output_tensor_name;
      InlineGlobalVarComputeImpl(roots[i], roots);
    }
  }
}

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
