// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/send_recv_functor.h"

#include "glog/logging.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/p_send_kernel.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

template <typename T, typename Context>
void PRecvKernel(const Context& dev_ctx,
                 int peer,
                 DataType dtype,
                 const std::vector<int>& out_shape,
                 bool dynamic_shape,
                 DenseTensor* out) {
#if defined(PADDLE_WITH_XPU_BKCL)

  auto comm_ctx =
      GetCommContext<Context, distributed::BKCLCommContext>(dev_ctx, peer);
  XPUStream stream = dev_ctx.stream();

  // auto data_type = phi::TransToPhiDataType(dtype);
  if (dynamic_shape) {
    DDim new_dim =
        recv_shape_info<Context, distributed::BKCLCommContext, XPUStream>(
            dev_ctx, out, comm_ctx, peer);
    out->Resize(new_dim);
  }
  dev_ctx.Alloc(out, dtype);
  comm_ctx->Recv(out, out->numel(), peer, stream);
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with XPU."));
#endif
}

template <typename T, typename Context>
void PRecvArrayKernel(const Context& dev_ctx,
                      int peer,
                      DataType dtype,
                      const std::vector<int>& out_shape,
                      TensorArray* out_array) {
#if defined(PADDLE_WITH_XPU_BKCL)

  auto comm_ctx =
      GetCommContext<Context, distributed::BKCLCommContext>(dev_ctx, peer);
  XPUStream stream = dev_ctx.stream();
  for (size_t idx = 0; idx < out_shape.size(); ++idx) {
    VLOG(3) << "DenseTensorArray: idx(" << idx << ")";
    auto out = out_array->at(idx);
    auto out_dims = out.dims();
    dev_ctx.Alloc(&out, dtype);
    comm_ctx->Recv(&out, out.numel(), peer, stream);
    VLOG(3) << "rank " << comm_ctx->GetRank() << " recv "
            << common::product(out_dims) << " from " << peer;
  }
#else
  PADDLE_THROW(
      errors::PreconditionNotMet("PaddlePaddle should compile with XPU."));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(p_recv,
                   XPU,
                   ALL_LAYOUT,
                   phi::PRecvKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(p_recv_array,
                   XPU,
                   ALL_LAYOUT,
                   phi::PRecvArrayKernel,
                   float,
                   double,
                   uint8_t,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
