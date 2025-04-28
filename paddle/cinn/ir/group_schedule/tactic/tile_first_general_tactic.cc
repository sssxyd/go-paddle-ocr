// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/tactic/tile_first_general_tactic.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

namespace cinn {
namespace ir {

using cinn::ir::analyzer::IsReductionSBlock;
using BoundVariableMap = std::unordered_map<std::string, std::vector<Var>>;

bool IsSpatialRegion(const ScheduleConfig& config) {
  if (config.base_info->iter_space_type.size() == 1 &&
      config.base_info->iter_space_type.back().first == "S") {
    return true;
  }
  return false;
}

bool UseContinuousDataTile(const ScheduleConfig& config) {
  // use continuous data tile for [S] and [...R]
  if (config.base_info->iter_space_type.size() == 1 &&
      config.base_info->iter_space_type.back().first == "S") {
    return true;
  }
  if (config.base_info->iter_space_type.back().first == "R") {
    return true;
  }
  return false;
}

bool ScheduleBlockEnableVectorize(const ScheduleConfig& config,
                                  const std::string& block_id) {
  if (!config.base_info->can_apply_vectorize) return false;

  if (!UseContinuousDataTile(config)) return false;
  return true;
}

class TileFirstGeneralTactic final : public ScheduleTactic {
 public:
  void Init(ScheduleContext* context, ir::IRSchedule* sch) override;

  void Apply(ir::IRSchedule* sch, const std::string& block_id) override;
  void ApplyVectorize(ir::IRSchedule* sch, const std::string& block_id);

  std::string TacticName() const override { return "TileFirstGeneralTactic"; }

 private:
  void MergeFlattenAxis(ir::IRSchedule* sch, const std::string& block_id);
  void MergeReduceAxis(ir::IRSchedule* sch, const std::string& block_id);
  void SetBufferType(ir::IRSchedule* sch, const std::string& block_id);
  void SetReduceType(ir::IRSchedule* sch, const std::string& block_id);

 private:
  ScheduleContext* context_;
  bool can_apply_;
  std::vector<int32_t> vec_flatten_axis_;
  std::vector<int32_t> vec_reduce_axis_;
  std::unordered_map<std::string, std::string> map_rf_block_;
  std::unordered_map<std::string, std::string> map_global_rf_block_;
};

void TileFirstGeneralTactic::Init(ScheduleContext* context,
                                  ir::IRSchedule* sch) {
  context_ = context;
  can_apply_ = false;

  // Check whether this group has been tiled by previous tactic.
  ir::Expr module_root = sch->GetModule().GetExprs().front();
  ir::Expr root_block = ir::analyzer::GetRootSBlock(module_root);
  auto* root_node = root_block.As<ir::ScheduleBlockRealize>()
                        ->schedule_block.As<ir::ScheduleBlock>();
  if (root_node->attrs.count(kTileMethod) > 0) {
    return;
  }
  can_apply_ = true;
  root_node->attrs[kTileMethod] = TacticName();

  // reduce axes have been re-ordered to the last
  vec_flatten_axis_.clear();
  vec_reduce_axis_.clear();
  int data_rank = context_->config.base_info->loop_ranges.size();
  int32_t reduce_start_idx =
      data_rank - context_->config.base_info->reduce_axis.size();
  for (int32_t i = 0; i < data_rank; ++i) {
    if (i >= reduce_start_idx) {
      vec_reduce_axis_.push_back(i);
    } else {
      vec_flatten_axis_.push_back(i);
    }
  }
  map_rf_block_.clear();
  map_global_rf_block_.clear();
}

void TileFirstGeneralTactic::Apply(ir::IRSchedule* sch,
                                   const std::string& block_id) {
  if (!can_apply_) return;
  if (ir::IsReduceInitTensorName(block_id)) return;

  // loops tiling with vectorize
  if (ScheduleBlockEnableVectorize(context_->config, block_id)) {
    ApplyVectorize(sch, block_id);
    return;
  }

  VLOG(4) << "Using ApplyContinuousDataTile";
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto sp_loop = context_->config.tile_config.spatial_inner_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  const auto rd_block = context_->config.tile_config.grid_reduce_num;
  const auto rd_loop = context_->config.tile_config.reduce_inner_num;
  VLOG(4) << "ApplyContinuousDataTile sp_thread=" << sp_thread;
  VLOG(4) << "ApplyContinuousDataTile sp_loop=" << sp_loop;
  VLOG(4) << "ApplyContinuousDataTile rd_thread=" << rd_thread;
  VLOG(4) << "ApplyContinuousDataTile rd_block=" << rd_block;
  VLOG(4) << "ApplyContinuousDataTile vec_flatten_axis: "
          << utils::Join(vec_flatten_axis_, ", ");
  VLOG(4) << "ApplyContinuousDataTile vec_reduce_axis: "
          << utils::Join(vec_reduce_axis_, ", ");

  // Merge reduce axes
  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Merge spatial axes
  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split spatial axes -> [sp_block, sp_loop, sp_thread]
  int current_reduce_axis = 0;
  if (vec_flatten_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    if (sp_loop > 1 && sp_thread > 1) {
      // [S, R] => [S(-1), S(inner_loop), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop, sp_thread});
      current_reduce_axis = 3;
    } else if (sp_loop > 1 || sp_thread > 1) {
      // [S, R] => [S(-1), S(thread), R]
      sch->Split(loops[0], {-1, sp_loop > 1 ? sp_loop : sp_thread});
      current_reduce_axis = 2;
    } else {
      // [S, R] => [S, R]
      current_reduce_axis = 1;
    }
  }
  VLOG(4) << "After SplitSptial on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Split reduce axes -> [rd_loop, rd_block, rd_thread]
  if (vec_reduce_axis_.size() > 0) {
    auto loops = sch->GetLoops(block_id);
    sch->Split(loops[current_reduce_axis], {rd_loop, rd_block * rd_thread});

    loops = sch->GetLoops(block_id);
    sch->Reorder({loops[current_reduce_axis + 1], loops[current_reduce_axis]});

    loops = sch->GetLoops(block_id);
    if (IsReductionSBlock(sch->GetBlock(block_id))) {
      ir::Expr rf_tensor =
          sch->FactorizeReduction(loops[current_reduce_axis],
                                  /* rf_axis = */ 0,
                                  /* with_write_back_block_init = */ false);
      map_rf_block_[block_id] = rf_tensor.as_tensor_ref()->name;
    }

    if (rd_block > 1) {
      loops = sch->GetLoops(block_id);
      sch->Split(loops[current_reduce_axis], {rd_block, rd_thread});

      if (IsReductionSBlock(sch->GetBlock(block_id))) {
        loops = sch->GetLoops(map_rf_block_[block_id]);
        sch->Split(loops[current_reduce_axis], {rd_block, rd_thread});

        loops = sch->GetLoops(block_id);
        ir::Expr rf_tensor =
            sch->FactorizeReduction(loops[current_reduce_axis],
                                    /* rf_axis = */ 0,
                                    /* with_write_back_block_init = */ false);
        std::string tensor_name = rf_tensor.as_tensor()->name;
        map_global_rf_block_[block_id] = tensor_name;
        rf_tensor.as_tensor()->WithBuffer("global", "_" + tensor_name);
      }
    }
  }
  VLOG(4) << "After SplitReduce on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Bind CUDA info
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    std::string sp_axis_type = "threadIdx.y";
    std::string rd_axis_type = "threadIdx.x";
    sch->Bind(loops[0], "blockIdx.x");
    if (!vec_flatten_axis_.empty() && sp_thread > 1) {
      if (vec_reduce_axis_.empty()) {
        // [S..S] => [S(blockIdx.x), optional(inner_loop), S(threadIdx.x)]
        sch->Bind(loops[current_reduce_axis - 1], rd_axis_type);
      } else {
        // [S..S, R..R] =>
        // [S(blockIdx.x), optional(inner_loop), S(threadIdx.y), R..R]
        sch->Bind(loops[current_reduce_axis - 1], sp_axis_type);
      }
    }
    if (!vec_reduce_axis_.empty() && current_reduce_axis > 0) {
      if (rd_block > 1) {
        sch->Bind(loops[current_reduce_axis], "blockIdx.y");
        if (loops.size() > current_reduce_axis + 1) {
          sch->Bind(loops[current_reduce_axis + 1], rd_axis_type);
        }
      } else {
        sch->Bind(loops[current_reduce_axis], rd_axis_type);
      }
    }
  };
  DoBind(sch->GetLoops(block_id));
  if (map_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_rf_block_[block_id]));
  }
  if (map_global_rf_block_.count(block_id) > 0) {
    DoBind(sch->GetLoops(map_global_rf_block_[block_id]));
  }
  VLOG(4) << "After BindCudaInfo on block: [" << block_id << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  SetBufferType(sch, block_id);
  SetReduceType(sch, block_id);
}

void TileFirstGeneralTactic::MergeFlattenAxis(ir::IRSchedule* sch,
                                              const std::string& block_id) {
  if (vec_flatten_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_flatten_axis_);
  }
}

void TileFirstGeneralTactic::MergeReduceAxis(ir::IRSchedule* sch,
                                             const std::string& block_id) {
  std::vector<ir::Expr> loops = sch->GetLoops(block_id);
  int32_t max_loop_idx = 0;
  for (int32_t idx : vec_reduce_axis_) {
    max_loop_idx = std::max(max_loop_idx, idx);
    PADDLE_ENFORCE_EQ(idx < loops.size() || loops.size() == 1,
                      true,
                      ::common::errors::InvalidArgument(
                          "The reduce axis should meet: axis's idx < "
                          "loops.size() or loops.size() == 1, but received "
                          "idx= %d ,loops.size() = %d",
                          idx,
                          loops.size()));
  }
  if (max_loop_idx < loops.size() && vec_reduce_axis_.size() >= 2) {
    sch->Fuse(block_id, vec_reduce_axis_);
  }
}

void TileFirstGeneralTactic::SetBufferType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  auto block = sch->GetBlock(block_id);
  if (context_->output_names.count(block_id) > 0) {
    sch->SetBuffer(block, "global");
  } else {
    sch->SetBuffer(block, "local");
  }

  if (map_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_rf_block_[block_id]);
    sch->SetBuffer(block, "local");
  }
}

void TileFirstGeneralTactic::SetReduceType(ir::IRSchedule* sch,
                                           const std::string& block_id) {
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    auto block = sch->GetBlock(block_id)
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
  if (map_global_rf_block_.count(block_id) > 0) {
    auto block = sch->GetBlock(map_global_rf_block_[block_id])
                     .As<ir::ScheduleBlockRealize>()
                     ->schedule_block.As<ir::ScheduleBlock>();
    block->reduce_method = context_->config.tile_config.reduce_method;
  }
}

namespace {

void ProcessScheduleBlockRealize(
    const ir::ScheduleBlockRealize* block_realize,
    const std::string& loop_var_name,
    BoundVariableMap& bound_variable_map) {  // NOLINT
  auto* schedule_block = block_realize->schedule_block.As<ScheduleBlock>();
  auto iter_values = block_realize->iter_values;
  auto iter_vars = schedule_block->iter_vars;

  for (std::size_t i = 0; i < iter_values.size(); ++i) {
    const auto& iter_value = iter_values[i];
    if (iter_value.is_var() &&
        iter_value.as_var()->name.find(loop_var_name) != std::string::npos) {
      bound_variable_map[loop_var_name].emplace_back(iter_vars[i]);
    } else if (iter_value.is_index()) {
      ir::ir_utils::CollectIRNodes(
          iter_value,
          [&loop_var_name, &bound_variable_map, &iter_vars, i](const Expr* x) {
            if (const auto* var = x->As<ir::_Var_>()) {
              if (var->name == loop_var_name) {
                bound_variable_map[loop_var_name].emplace_back(iter_vars[i]);
              }
            }
            return false;
          });
    }
  }
}

BoundVariableMap GetBoundVariables(const Expr& expr, const Expr& loop_var) {
  BoundVariableMap bound_variable_map;
  auto loop_var_name = loop_var.as_var()->name;

  ir::ir_utils::CollectIRNodes(
      expr,
      [&loop_var_name, &bound_variable_map](const Expr* x) {
        if (const auto block_realize = x->As<ir::ScheduleBlockRealize>()) {
          ProcessScheduleBlockRealize(
              block_realize, loop_var_name, bound_variable_map);
        }
        return false;
      },
      true);

  return bound_variable_map;
}

/*
 * Check if the current loop variable containing the vectorize axis
 * is present in the iter values of the axis bind within the loop body.
 * If it is present, the loop cannot be vectorized.
 * For example, the following loop cannot be vectorized:
 *
 * serial for (j, 0, 4)
 * {
 *   ScheduleBlock(var_2) {
 *     i0 = axis.bind(i)
 *     var_2[i0] = (var_1[i0] + var[0, i0, 0, 0])
 *   }
 * }
 */
bool ContainsVectorizableAxis(const ir::IRSchedule* sch,
                              const size_t vectorize_axis,
                              const std::string& block_id) {
  auto loops = sch->GetLoops(block_id);
  auto vectorize_expr = loops[vectorize_axis];

  VLOG(4) << "Checking ContainsVectorizableAxis on block: [" << block_id
          << "], loop:\n"
          << sch->GetModule().GetExprs().front() << "\n vectorize expr:\n"
          << vectorize_expr;

  // Get all the lter values in the axis bind that contain a loop var and the
  // corresponding iter var.
  auto bound_variable_map =
      GetBoundVariables(vectorize_expr, vectorize_expr.As<ir::For>()->loop_var);
  return !bound_variable_map.empty();
}

void SpatialRegionVectorizeTilingSchedule(ir::IRSchedule* sch,
                                          const std::string& block_id,
                                          const int sp_thread,
                                          const int vectorize_factor) {
  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[1], "threadIdx.x");
  };

  auto loops = sch->GetLoops(block_id);
  // The iter_value bound by axis_bind must contain the loop_var of the axis
  // to be vectorized.
  if (ContainsVectorizableAxis(sch, loops.size() - 1, block_id)) {
    sch->Split(loops[0], std::vector<int>{-1, sp_thread, vectorize_factor});

    // set vectorize schedule primitives
    loops = sch->GetLoops(block_id);
    auto vectorize_axis = loops.size() - 1;
    sch->Vectorize(loops[vectorize_axis], vectorize_factor);
  } else {
    sch->Split(loops[0], std::vector<int>{-1, sp_thread});
  }

  loops = sch->GetLoops(block_id);
  DoBind(loops);
  return;
}

void ReduceRegionWithReduceBlockVectorizeTilingSchedule(
    ir::IRSchedule* sch,
    std::unordered_map<std::string, std::string>* map_rf_block,
    const std::string& block_id,
    const int rd_thread,
    const int vectorize_factor) {
  int loop_axis = 1;
  int threads_axis = 2;
  int vectorize_axis = 3;
  auto loops = sch->GetLoops(block_id);
  if (ContainsVectorizableAxis(sch, loops.size() - 1, block_id)) {
    sch->Split(loops[1], {-1, rd_thread, vectorize_factor});
    loops = sch->GetLoops(block_id);
    sch->Vectorize(loops[vectorize_axis], vectorize_factor);
  } else {
    sch->Split(loops[1], {-1, rd_thread});
    loops = sch->GetLoops(block_id);
  }

  loops = sch->GetLoops(block_id);
  sch->Reorder({loops[threads_axis], loops[loop_axis]});
  threads_axis = 1;
  loops = sch->GetLoops(block_id);
  if (IsReductionSBlock(sch->GetBlock(block_id))) {
    ir::Expr rf_tensor =
        sch->FactorizeReduction(loops[threads_axis],
                                /* rf_axis = */ 0,
                                /* with_write_back_block_init = */ false);
    (*map_rf_block)[block_id] = rf_tensor.as_tensor_ref()->name;
  }

  const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
    sch->Bind(loops[0], "blockIdx.x");
    sch->Bind(loops[threads_axis], "threadIdx.x");
  };

  DoBind(sch->GetLoops(block_id));
  if (map_rf_block->count(block_id) > 0) {
    DoBind(sch->GetLoops((*map_rf_block)[block_id]));
  }
}

void ReduceRegionWithSpatialBlockVectorizeTilingSchedule(
    ir::IRSchedule* sch,
    const std::string& block_id,
    const int rd_thread,
    const int vectorize_factor) {
  auto loops = sch->GetLoops(block_id);
  if (ContainsVectorizableAxis(sch, loops.size() - 1, block_id)) {
    sch->Split(loops[1], std::vector<int>{-1, rd_thread, vectorize_factor});

    // set vectorize schedule primitives
    loops = sch->GetLoops(block_id);
    auto vectorize_axis = loops.size() - 1;
    sch->Vectorize(loops[vectorize_axis], vectorize_factor);
    const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
      sch->Bind(loops[0], "blockIdx.x");
      auto threadsIdx_x_axis = vectorize_axis - 1;
      sch->Bind(loops[threadsIdx_x_axis], "threadIdx.x");
    };
    loops = sch->GetLoops(block_id);
    DoBind(loops);
  } else {
    sch->Split(loops[1], std::vector<int>{-1, rd_thread});
    const auto DoBind = [&](const std::vector<ir::Expr>& loops) {
      sch->Bind(loops[0], "blockIdx.x");
      auto threadsIdx_x_axis = loops.size() - 1;
      sch->Bind(loops[threadsIdx_x_axis], "threadIdx.x");
    };
    loops = sch->GetLoops(block_id);
    DoBind(loops);
  }
}

void ReduceRegionVectorizeTilingSchedule(
    ir::IRSchedule* sch,
    std::unordered_map<std::string, std::string>* map_rf_block,
    const std::string& block_id,
    const int rd_thread,
    const int vectorize_factor) {
  auto loops = sch->GetLoops(block_id);
  if (IsReductionSBlock(sch->GetBlock(block_id))) {  // deal with reduce block
    ReduceRegionWithReduceBlockVectorizeTilingSchedule(
        sch, map_rf_block, block_id, rd_thread, vectorize_factor);
  } else {  // deal with spatial block
    ReduceRegionWithSpatialBlockVectorizeTilingSchedule(
        sch, block_id, rd_thread, vectorize_factor);
  }
  return;
}

}  // namespace

void TileFirstGeneralTactic::ApplyVectorize(ir::IRSchedule* sch,
                                            const std::string& block_id) {
  const auto sp_thread = context_->config.tile_config.warp_num * 32 /
                         context_->config.tile_config.tree_reduce_num;
  const auto rd_thread = context_->config.tile_config.tree_reduce_num;
  const auto vectorize_factor = context_->config.tile_config.vectorize_factor;
  VLOG(4) << "ApplyApplyVectorize sp_thread=" << sp_thread;
  VLOG(4) << "ApplyApplyVectorize rd_thread=" << rd_thread;
  VLOG(4) << "ApplyApplyVectorize vectorize_factor=" << vectorize_factor;
  // Merge reduce axes
  MergeReduceAxis(sch, block_id);
  VLOG(4) << "After MergeReduceAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Merge spatial axes
  MergeFlattenAxis(sch, block_id);
  VLOG(4) << "After MergeFlattenAxis on block: [" << block_id
          << "], loop nest:\n"
          << sch->GetModule().GetExprs().front();

  // Spatial situation
  if (IsSpatialRegion(context_->config)) {
    SpatialRegionVectorizeTilingSchedule(
        sch, block_id, sp_thread, vectorize_factor);
  } else {  // Reduce situation
    ReduceRegionVectorizeTilingSchedule(
        sch, &map_rf_block_, block_id, rd_thread, vectorize_factor);
  }

  SetBufferType(sch, block_id);
  SetReduceType(sch, block_id);
  return;
}

std::unique_ptr<ScheduleTactic> CreateTileFirstGeneralTactic() {
  return std::make_unique<TileFirstGeneralTactic>();
}

}  // namespace ir
}  // namespace cinn
