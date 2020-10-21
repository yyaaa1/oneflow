/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/data/onerec_data_reader.h"
#include <nvToolsExt.h> 
#include <sys/syscall.h>
#include <unistd.h>

namespace oneflow {

namespace {

class OneRecReaderWrapper final : public user_op::OpKernelState {
 public:
  explicit OneRecReaderWrapper(user_op::KernelInitContext* ctx) : reader_(ctx) {}
  ~OneRecReaderWrapper() = default;

  void Read(user_op::KernelComputeContext* ctx) { 
    LOG(INFO)<<"OneRecReaderWrapper start time ";
    reader_.Read(ctx); 
    LOG(INFO)<<"OneRecReaderWrapper end time ";
    }

 private:
  data::OneRecDataReader reader_;
};

}  // namespace

class OneRecReaderKernel final : public user_op::OpKernel {
 public:
  OneRecReaderKernel() = default;
  ~OneRecReaderKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<OneRecReaderWrapper> reader(new OneRecReaderWrapper(ctx));
    return reader;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    //nvtxRangePush("reader");
    LOG(INFO)<<"OneRecReaderKernel start time ";
    double start_time = GetCurTime();
    auto* reader = dynamic_cast<OneRecReaderWrapper*>(state);
    double start_time1 = GetCurTime();
    reader->Read(ctx);
    LOG(INFO)<<"OneRecReaderKernel time  "<<(GetCurTime() - start_time)/1e6;
    LOG(INFO)<<"OneRecReaderKernel time1  "<<(GetCurTime() - start_time1)/1e6;
    LOG(INFO)<<"OneRecReaderKernel end time  ";
    //nvtxRangePop();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("OneRecReader")
    .SetCreateFn<OneRecReaderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace oneflow
