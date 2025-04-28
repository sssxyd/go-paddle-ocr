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

#include "paddle/fluid/pir/transforms/general/auto_layout_pass.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/common/layout.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/auto_layout_insert_pass.h"
#include "paddle/fluid/pir/transforms/general/auto_layout_simplify_pass.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pass/utils.h"

#ifdef PADDLE_WITH_CINN
COMMON_DECLARE_bool(cinn_debug);
#endif

namespace {
void PrintProgram(const pir::Program& prog, const std::string& stage) {
  bool print_flag = VLOG_IS_ON(1);
#ifdef PADDLE_WITH_CINN
  print_flag &= FLAGS_cinn_debug;
#endif
  if (print_flag) {
    std::cout << "===================== [AutoLayoutPass] " << stage
              << " =====================\n"
              << prog << std::endl;
  }
}
void PrintConvTransposeInfo(int conv_num, int transpose_num, int scale) {
  bool print_flag = true;
#ifdef PADDLE_WITH_CINN
  print_flag &= FLAGS_cinn_debug;
#endif
  if (print_flag) {
    LOG(INFO) << "end IsNeedAllTranspose"
              << " conv_count_: " << conv_num
              << " transpose_count_: " << transpose_num
              << " transpose_scale_ * transpose_count_: "
              << scale * transpose_num;
  }
}
class AutoLayoutPass : public pir::Pass {
 public:
  AutoLayoutPass() : pir::Pass("auto_layout_pass", 2) {}
  void Run(pir::Operation* op) override {
    auto program = op->GetParentProgram();
    PrintProgram(*program, "Before pass");
    ::pir::IrMapping ir_mapping;
    auto program_clone = program->Clone(ir_mapping);

    pir::PassManager pm(::pir::IrContext::Instance(), 2);

    pm.AddPass(pir::CreateAutoLayoutInsertPass({"pd_op.fused_conv2d_add_act",
                                                "pd_op.conv2d",
                                                "pd_op.conv2d_transpose"}));
    pm.AddPass(pir::CreateAutoLayoutSimplifyPass());
    pm.Run(program_clone.get());

    PrintProgram(*program, "Middle");
    if (IsNeedAllTranspose(program_clone->module_op())) {
      pir::PassManager pm_(::pir::IrContext::Instance(), 2);
      pm_.AddPass(pir::CreateAutoLayoutInsertPass({"pd_op.fused_conv2d_add_act",
                                                   "pd_op.conv2d",
                                                   "pd_op.conv2d_transpose"}));
      pm_.AddPass(pir::CreateAutoLayoutSimplifyPass());
      pm_.Run(program);
    } else {
      // Same as TransferLayoutPass, only transpose fused_conv2d_add_act
      pir::PassManager pm_(::pir::IrContext::Instance(), 2);
      pm_.AddPass(
          pir::CreateAutoLayoutInsertPass({"pd_op.fused_conv2d_add_act"}));
      pm_.AddPass(pir::CreateAutoLayoutSimplifyPass());
      pm_.Run(program);
    }
    PrintProgram(*program, "After pass");
  }

  // Check whether all conv2d, conv2d_transpose and fused_conv2d_add_act ops
  // need to be transposed.
  bool IsNeedAllTranspose(pir::Operation* op) {
    VLOG(4) << "enter IsNeedAllTranspose";
    for (size_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        for (auto&& op : block) {
          if (op.isa<paddle::dialect::TransposeOp>()) {
            if (!op.HasAttribute("source")) continue;
            auto source = op.attribute<pir::StrAttribute>("source").AsString();
            if (source == "auto_layout_pass") {
              transpose_count_++;
            } else {
              // The original transpose should not be counted
              continue;
            }
          } else if (op.isa<paddle::dialect::Conv2dOp>() ||
                     op.isa<paddle::dialect::Conv2dTransposeOp>() ||
                     op.isa<paddle::dialect::FusedConv2dAddActOp>()) {
            auto layout_interface =
                op.dyn_cast<paddle::dialect::LayoutTransformationInterface>();
            if (layout_interface.PreferLayout(&op) != common::DataLayout::NHWC)
              continue;
            op.isa<paddle::dialect::FusedConv2dAddActOp>() ? conv_count_ += 3
                                                           : conv_count_ += 1.5;
          } else {
            // Other op
            continue;
          }
        }
      }
    }

    PrintConvTransposeInfo(conv_count_, transpose_count_, transpose_scale_);
    return conv_count_ > transpose_scale_ * transpose_count_;
  }

 private:
  int conv_count_ = 0;
  int transpose_count_ = 0;
  // Due to the current transpose performance issues, our single explicit
  // transpose does not perform better than cudnn, and there are interruptions
  // in the network due to operators that do not support NHWC. So we set the
  // scale to 1.3 temporarily.
  float transpose_scale_ = 1.3;
};
}  // namespace
namespace pir {

std::unique_ptr<Pass> CreateAutoLayoutPass() {
  return std::make_unique<AutoLayoutPass>();
}

}  // namespace pir

REGISTER_IR_PASS(auto_layout_pass, AutoLayoutPass);
