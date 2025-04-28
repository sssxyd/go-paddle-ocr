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

#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

namespace cinn::fusion {

std::vector<int64_t> GetInt64ArrayAttributeData(
    const ::pir::Attribute& attr_val) {
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::ArrayAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input attribute should be an array."));
  const auto& array_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int64_t> data;
  for (int i = 0; i < array_attr.size(); ++i) {
    const auto& int64_attr = array_attr.at(i).dyn_cast<::pir::Int64Attribute>();
    PADDLE_ENFORCE_NOT_NULL(int64_attr,
                            ::common::errors::InvalidArgument(
                                "The array element should be int64 type."));
    data.push_back(int64_attr.data());
  }
  return data;
}

std::vector<int32_t> GetInt32ArrayAttributeData(
    const ::pir::Attribute& attr_val) {
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::ArrayAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The input attribute should be an array."));
  const auto& array_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  std::vector<int32_t> data;
  for (int i = 0; i < array_attr.size(); ++i) {
    const auto& int32_attr = array_attr.at(i).dyn_cast<::pir::Int32Attribute>();
    PADDLE_ENFORCE_NOT_NULL(int32_attr,
                            ::common::errors::InvalidArgument(
                                "The array element should be int32 type."));
    data.push_back(int32_attr.data());
  }
  return data;
}

std::unordered_set<pir::Operation*> GetGroupOutputOps(
    const std::vector<pir::Operation*>& ops) {
  const auto is_global_inplace_op =
      [](pir::Operation* op,
         const std::unordered_set<pir::Operation*>& ops_set) -> bool {
    if (op->num_results() != 1 ||
        !op->HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
      return false;
    }
    auto op_info =
        op->dyn_cast<paddle::dialect::OpYamlInfoInterface>().GetOpInfo();
    auto input_info_list = std::get<0>(op_info);
    auto output_info_list = std::get<2>(op_info);
    auto inplace_info_map = std::get<3>(op_info).inplace;
    // 1. Find which input is inplace with the output
    std::string output_name = output_info_list.front().name;
    std::string inplace_input_name;
    for (const auto& [out, in] : inplace_info_map) {
      if (out == output_name) inplace_input_name = in;
    }
    if (inplace_input_name.empty()) return false;
    int inplace_input_idx = -1;
    for (int i = 0; i < input_info_list.size(); ++i) {
      if (input_info_list[i].name == inplace_input_name) {
        inplace_input_idx = i;
        break;
      }
    }
    if (inplace_input_idx == -1) return false;
    // 2. Check whether the inplace input is not the output of op in the ops_set
    pir::Value inplace_input_value = op->operand_source(inplace_input_idx);
    return ops_set.find(inplace_input_value.defining_op()) == ops_set.end();
  };

  auto ops_set = ToUnorderedSet(ops);
  std::unordered_set<pir::Operation*> output_ops;
  for (auto* op : ops) {
    if (op->HasTrait<paddle::dialect::InplaceTrait>()) {
      if (is_global_inplace_op(op, ops_set)) output_ops.insert(op);
      continue;
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      if (!result) continue;
      for (auto use_iter = result.use_begin(); use_iter != result.use_end();
           ++use_iter) {
        auto* use_op = use_iter->owner();
        if (ops_set.find(use_op) == ops_set.end()) {
          output_ops.insert(op);
          break;
        }
      }
      if (output_ops.count(op)) break;
    }
  }
  return output_ops;
}

std::vector<int64_t> GetReduceAxisIdx(pir::Operation* reduce_op) {
  const size_t input_rank = GetCompatibleRank(reduce_op->operand_source(0));
  const auto& attr_val = reduce_op->attributes().at("axis");
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::ArrayAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The axis attribute should be an array."));
  const auto& axis_attr = attr_val.dyn_cast<::pir::ArrayAttribute>();
  if (axis_attr.empty()) {
    // dim: [] means reduce_all.
    std::vector<int64_t> all_axis;
    for (int i = 0; i < input_rank; ++i) {
      all_axis.push_back(i);
    }
    return all_axis;
  }
  std::vector<int64_t> reduce_axis_idx;
  for (int i = 0; i < axis_attr.size(); ++i) {
    int64_t axis = axis_attr.at(i).dyn_cast<::pir::Int64Attribute>().data();
    if (axis < 0) {
      axis += input_rank;
    }
    PADDLE_ENFORCE_GE(
        axis,
        0,
        ::common::errors::InvalidArgument(
            "The 'axis' must be greater than or equal to 0, but received %d.",
            axis));

    PADDLE_ENFORCE_LT(axis,
                      input_rank,
                      ::common::errors::InvalidArgument(
                          "The 'axis' must be less than 'input_rank', but "
                          "received axis = %d and input_rank = %d.",
                          axis,
                          input_rank));

    reduce_axis_idx.push_back(axis);
  }
  VLOG(4) << "GetReduceAxisIdx: " << utils::Join(reduce_axis_idx, ",");
  return reduce_axis_idx;
}

bool GetReduceOpKeepDims(pir::Operation* reduce_op) {
  const auto& attr_val = reduce_op->attributes().at("keepdim");
  PADDLE_ENFORCE_EQ(attr_val.isa<::pir::BoolAttribute>(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The keepdim attribute should be a bool."));
  return attr_val.dyn_cast<::pir::BoolAttribute>().data();
}

std::pair<std::vector<int64_t>, bool> GetSliceAxis(pir::Operation* slice_op) {
  std::vector<int64_t> slice_axis =
      GetInt64ArrayAttributeData(slice_op->attributes().at("axes"));
  std::vector<int64_t> decrease_axis =
      GetInt64ArrayAttributeData(slice_op->attributes().at("decrease_axis"));
  PADDLE_ENFORCE_EQ(slice_axis.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The axis attribute should not be empty."));

  bool keepdim = true;
  if (!decrease_axis.empty()) {
    PADDLE_ENFORCE_EQ(
        decrease_axis,
        slice_axis,
        ::common::errors::InvalidArgument(
            "The size of decrease axis should be equal to the size of axis."));
    keepdim = false;
  }
  return std::make_pair(slice_axis, keepdim);
}

std::optional<std::pair<pir::Value, pir::Value>> GetBroadcastOpInputOutputValue(
    pir::Operation* op) {
  auto* mut_op = const_cast<pir::Operation*>(op);
  if (op->isa<paddle::dialect::ExpandOp>()) {
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();
    return std::make_pair(expand_op.x(), expand_op.out());
  } else if (op->isa<cinn::dialect::BroadcastOp>()) {
    auto broadcast_op = mut_op->dyn_cast<cinn::dialect::BroadcastOp>();
    return std::make_pair(broadcast_op.x(), broadcast_op.out());
  } else {
    PADDLE_THROW(::common::errors::Unimplemented("Unsupported broadcast op, %s",
                                                 op->name()));
  }
  return std::nullopt;
}

std::vector<std::pair<size_t, size_t>> GetNonBroadCastDims(pir::Operation* op) {
  std::vector<std::pair<size_t, size_t>> res;
  auto* shape_analysis =
      &pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

  const auto& broad_cast_value = GetBroadcastOpInputOutputValue(op);
  CHECK(broad_cast_value.has_value());

  const auto& [input_value, output_value] = broad_cast_value.value();
  const int input_rank = GetRank(input_value);
  const int output_rank = GetRank(output_value);
  PADDLE_ENFORCE_GE(output_rank,
                    input_rank,
                    ::common::errors::PreconditionNotMet(
                        "[Error info] The output_rank should "
                        "be greater or equal to input_rank."));

  // Compare axis one by one, from back to front.
  // The rule of broadcasting:
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/tensor_cn.html#id7
  for (int i = 1; i <= input_rank; ++i) {
    int input_axis = input_rank - i;
    int output_axis = output_rank - i;
    if (input_axis < 0 || output_axis < 0) break;
    if (shape_analysis->IsProductEqual(
            input_value, {input_axis}, output_value, {output_axis})) {
      res.emplace_back(input_axis, output_axis);
    }
  }

  return res;
}

std::shared_ptr<pir::ShapeConstraintIRAnalysis> GetShapeAnalysisFromValue(
    const pir::Value& value) {
  pir::Operation* related_op = value.defining_op();
  if (value.defining_op() == nullptr) {
    // For inputs of the program, the defining_op is nullptr,
    // we use it's user as the related op.
    PADDLE_ENFORCE_EQ(value.use_empty(),
                      false,
                      ::common::errors::PreconditionNotMet(
                          "Value is an input value, it should have a use."));
    related_op = value.first_use().owner();
  }
  return pir::ShapeAnalysisManager::Instance()
      .Get(related_op->GetParentProgram())
      .shared_from_this();
}

std::vector<symbol::DimExpr> GetValueAllDims(const pir::Value& value) {
  return GetValueDims(value, ArangeVector<int>(0, GetRank(value)));
}

std::vector<symbol::DimExpr> GetCompatibleValueAllDims(
    const pir::Value& value) {
  return GetRank(value) == 0 ? std::vector<symbol::DimExpr>{symbol::DimExpr(1)}
                             : GetValueAllDims(value);
}

symbol::DimExpr GetShapeProduct(const std::vector<symbol::DimExpr>& shape,
                                int start,
                                int end) {
  symbol::DimExpr product(1);
  for (int i = start; i < end; ++i) {
    product = product * shape[i];
  }
  return symbol::SimplifyDimExpr(product);
}

bool ShapeProductEqual(const std::vector<symbol::DimExpr>& in_shape,
                       const std::vector<symbol::DimExpr>& out_shape,
                       int in_start,
                       int in_end,
                       int out_start,
                       int out_end) {
  return GetShapeProduct(in_shape, in_start, in_end) ==
         GetShapeProduct(out_shape, out_start, out_end);
}

bool ShapeProductEqual(const std::vector<symbol::DimExpr>& in_shape,
                       const std::vector<symbol::DimExpr>& out_shape) {
  return ShapeProductEqual(
      in_shape, out_shape, 0, in_shape.size(), 0, out_shape.size());
}

std::vector<std::pair<int, int>> PartitionReshapeAxes(
    const std::vector<symbol::DimExpr>& in_shape,
    const std::vector<symbol::DimExpr>& out_shape) {
  PADDLE_ENFORCE(ShapeProductEqual(in_shape, out_shape),
                 ::common::errors::InvalidArgument(
                     "Shape product should be equal for reshape operation."));

  int input_rank = in_shape.size();
  int output_rank = out_shape.size();
  std::vector<std::pair<int, int>> partition = {{0, 0}};
  for (int i = 1, j = 1; i <= in_shape.size() && j <= out_shape.size();) {
    bool shape_product_equal = ShapeProductEqual(in_shape,
                                                 out_shape,
                                                 partition.back().first,
                                                 i,
                                                 partition.back().second,
                                                 j);
    if (shape_product_equal) {
      partition.emplace_back(i++, j++);
      if (i > input_rank || j > output_rank) {
        // In case of the last few dims are 1
        partition.back().first = input_rank;
        partition.back().second = output_rank;
      }
    } else if (j < output_rank) {
      j++;
    } else if (i < input_rank) {
      i++;
      j = partition.back().second + 1;
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Shape product should be equal for reshape operation."));
    }
  }
  return partition;
}

}  // namespace cinn::fusion
