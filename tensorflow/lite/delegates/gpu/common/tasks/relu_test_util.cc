/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/tasks/relu_test_util.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"

namespace tflite {
namespace gpu {

void ReLUNoClipNoAlphaTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.0f;
  attr.clip = 0.0f;

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateReLU(op_def, attr);
      ASSERT_TRUE(env->ExecuteGPUOperation(
                         src_tensor,
                         absl::make_unique<GPUOperation>(std::move(operation)),
                         BHWC(1, 2, 1, 2), &dst_tensor)
                      .ok());
      EXPECT_THAT(dst_tensor.data,
                  testing::Pointwise(testing::FloatNear(eps),
                                     {0.0f, 0.8f, 0.0f, 3.2f}));
    }
  }
}

void ReLUClipTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.0f;
  attr.clip = 0.9f;

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateReLU(op_def, attr);
      ASSERT_TRUE(env->ExecuteGPUOperation(
                         src_tensor,
                         absl::make_unique<GPUOperation>(std::move(operation)),
                         BHWC(1, 2, 1, 2), &dst_tensor)
                      .ok());
      EXPECT_THAT(dst_tensor.data,
                  testing::Pointwise(testing::FloatNear(eps),
                                     {0.0f, 0.8f, 0.0f, 0.9f}));
    }
  }
}

void ReLUAlphaTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.5f;
  attr.clip = 0.0f;

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateReLU(op_def, attr);
      ASSERT_TRUE(env->ExecuteGPUOperation(
                         src_tensor,
                         absl::make_unique<GPUOperation>(std::move(operation)),
                         BHWC(1, 2, 1, 2), &dst_tensor)
                      .ok());
      EXPECT_THAT(dst_tensor.data,
                  testing::Pointwise(testing::FloatNear(eps),
                                     {-0.25f, 0.8f, -0.3f, 3.2f}));
    }
  }
}

void ReLUAlphaClipTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {-0.5f, 0.8f, -0.6f, 3.2f};

  ReLUAttributes attr;
  attr.alpha = 0.5f;
  attr.clip = 0.5f;

  for (auto storage : env->GetSupportedStorages()) {
    for (auto precision : env->GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      GPUOperation operation = CreateReLU(op_def, attr);
      ASSERT_TRUE(env->ExecuteGPUOperation(
                         src_tensor,
                         absl::make_unique<GPUOperation>(std::move(operation)),
                         BHWC(1, 2, 1, 2), &dst_tensor)
                      .ok());
      EXPECT_THAT(dst_tensor.data,
                  testing::Pointwise(testing::FloatNear(eps),
                                     {-0.25f, 0.5f, -0.3f, 0.5f}));
    }
  }
}

}  // namespace gpu
}  // namespace tflite