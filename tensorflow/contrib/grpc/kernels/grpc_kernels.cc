// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
// =============================================================================

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
// #include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/macros.h"

#include "grpc++/grpc++.h"
#include <memory>
#include <iostream>
// #include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"

namespace tensorflow {
/*
class ChannelInterface : public ResourceBase {
 public:
  // Returns a debug string for *this.
  virtual string DebugString() {
    return "Channel";
  }

  // Returns memory used by this resource.
  virtual int64 MemoryUsed() const { return 0; }
};*/

class GetBytesKernel : public OpKernel {
 public:
  explicit GetBytesKernel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    std::cout << "We got called." << std::endl;

    const Tensor* input;
    OP_REQUIRES_OK(context, context->input("request", &input));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                 errors::InvalidArgument(
                     "Input message tensor must be scalar, but had shape: ",
                     input->shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("reply",
                                                       TensorShape({}), &output));

    // Get the message and response as string scalars
    const auto &request = input->scalar<string>()();
    auto &reply = output->scalar<string>()();

    reply.resize(request.size());
    memmove(&reply[0], &request[0], request.size());
  }
};

REGISTER_KERNEL_BUILDER(Name("GetBytes").Device(DEVICE_CPU), GetBytesKernel);

} // namespace tensorflow
