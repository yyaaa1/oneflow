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
#ifndef ONEFLOW_USER_DATA_DATA_READER_H_
#define ONEFLOW_USER_DATA_DATA_READER_H_

#include "oneflow/core/common/buffer.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/data/dataset.h"
#include "oneflow/user/data/parser.h"

namespace oneflow {
namespace data {

static const int32_t kDataReaderBatchBufferSize = 4;

template<typename LoadTarget>
class DataReader {
 public:
  using LoadTargetPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  DataReader(user_op::KernelInitContext* ctx)
      : is_closed_(false), batch_buffer_(kDataReaderBatchBufferSize) {}
  virtual ~DataReader() {
    Close();
    if (load_thrd_.joinable()) { load_thrd_.join(); }
  }

  void Read(user_op::KernelComputeContext* ctx) {
    double start_time1=GetCurTime();
    CHECK(load_thrd_.joinable()) << "You should call StartLoadThread before read data";
    double start_time=GetCurTime();
    LOG(INFO)<<"FetchBatchData start time  ";
    auto batch_data = FetchBatchData();
    LOG(INFO)<<"FetchBatchData time  "<<(GetCurTime() - start_time)/1e6;
    LOG(INFO)<<"FetchBatchData end time  ";

    start_time=GetCurTime();
    LOG(INFO)<<"Parse start time  ";
    parser_->Parse(batch_data, ctx);
    LOG(INFO)<<"Parse end time  ";
    
    LOG(INFO)<<"Read time  "<<(GetCurTime() - start_time1)/1e6;
  }

  void Close() {
    is_closed_.store(true);
    bool buffer_drained = false;
    while (!buffer_drained) {
      std::shared_ptr<LoadTargetPtrList> abandoned_batch_data(nullptr);
      auto status = batch_buffer_.TryReceive(&abandoned_batch_data);
      CHECK_NE(status, BufferStatus::kBufferStatusErrorClosed);
      buffer_drained = (status == BufferStatus::kBufferStatusEmpty);
    }
    batch_buffer_.Close();
  }

 protected:
  void StartLoadThread() {
    if (load_thrd_.joinable()) { return; }
    load_thrd_ = std::thread([this] {
      while (!is_closed_.load() && LoadBatch()) {}
    });
  }

  std::unique_ptr<Dataset<LoadTarget>> loader_;
  std::unique_ptr<Parser<LoadTarget>> parser_;

 private:
  std::shared_ptr<LoadTargetPtrList> FetchBatchData() {
    std::shared_ptr<LoadTargetPtrList> batch_data(nullptr);
    CHECK_EQ(batch_buffer_.Receive(&batch_data), BufferStatus::kBufferStatusSuccess);
    return batch_data;
  }

  bool LoadBatch() {
    LOG(INFO)<<"LoadBatch start time  ";
    std::shared_ptr<LoadTargetPtrList> batch_data =
        std::make_shared<LoadTargetPtrList>(std::move(loader_->Next()));
    auto return_val =  batch_buffer_.Send(batch_data) == BufferStatus::kBufferStatusSuccess;
    LOG(INFO)<<"LoadBatch end time  ";
    return return_val;
  }

  std::atomic<bool> is_closed_;
  Buffer<std::shared_ptr<LoadTargetPtrList>> batch_buffer_;
  std::thread load_thrd_;
};

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_DATA_READER_H_
