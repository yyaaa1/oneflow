#include "oneflow/core/kernel/multinomial_logistic_loss_kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
void MultinomialLogisticLossKernel<device_type, PredType, LabelType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* prediction = BnInOp2Blob("prediction");
  const Blob* label = BnInOp2Blob("label");
  Blob* loss = BnInOp2Blob("loss");
  Blob* loss_buff = BnInOp2Blob("loss_buffer");

  MultinomialLogisticLossKernelUtil<device_type, PredType, LabelType>::Forward(
      ctx.device_ctx, prediction->shape().At(0), prediction->shape().At(1),
      prediction->dptr<PredType>(), label->dptr<LabelType>(),
      loss->mut_dptr<PredType>(), loss_buff->mut_dptr<PredType>());

  Blob* prediction_diff = BnInOp2Blob(GenDiffBn("prediction"));
  if (prediction_diff != nullptr) {
    Memset<device_type>(ctx.device_ctx, prediction_diff->mut_dptr<PredType>(),
                        0, prediction_diff->TotalByteSize());
    MultinomialLogisticLossKernelUtil<device_type, PredType, LabelType>::
        Backward(ctx.device_ctx, prediction->shape().At(0),
                 prediction->shape().At(1), prediction->dptr<PredType>(),
                 label->dptr<LabelType>(),
                 prediction_diff->mut_dptr<PredType>());
  }
}

template<typename PredType, typename LabelType>
class MultinomialLogisticLossKernelUtil<DeviceType::kCPU, PredType, LabelType>
    final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultinomialLogisticLossKernelUtil);
  MultinomialLogisticLossKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const int64_t instance_num,
                      const int64_t num_of_classes, const PredType* prediction,
                      const LabelType* labels, PredType* loss,
                      PredType* loss_buff) {
    ctx->cpu_stream()->SendWork([=]() {
      loss[0] = 0;
      for (int64_t i = 0; i < instance_num; ++i) {
        PredType prob =
            prediction[i * num_of_classes + static_cast<int64_t>(labels[i])];
        loss[0] -= SAFE_LOG(prob);
      }
    });
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num,
                       const int64_t num_of_classes, const PredType* prediction,
                       const LabelType* labels, PredType* prediction_diff) {
    ctx->cpu_stream()->SendWork([=]() {
      for (int64_t i = 0; i < instance_num; ++i) {
        int64_t label = static_cast<int64_t>(labels[i]);
        PredType prob =
            MAX_WITH_LOG_THRESHOLD(prediction[i * num_of_classes + label]);
        prediction_diff[i * num_of_classes + label] = -1 / prob;
      }
    });
  }
};

Kernel* CreateMultinomialLogisticLossKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define MULTI_LOG_LOSS_KERNEL_ENTRY(device_type, data_type_pair, \
                                    label_type_pair)             \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair),    \
              OF_PP_PAIR_SECOND(label_type_pair)),               \
   []() {                                                        \
     return new MultinomialLogisticLossKernel<                   \
         device_type, OF_PP_PAIR_FIRST(data_type_pair),          \
         OF_PP_PAIR_FIRST(label_type_pair)>;                     \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MULTI_LOG_LOSS_KERNEL_ENTRY,
                                       DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ,
                                       INT_DATA_TYPE_SEQ)};

  return creators.at(GetHashKey(op_ctx.device_type(),
                                op_ctx.bn_in_op2data_type().at("prediction"),
                                op_ctx.bn_in_op2data_type().at("label")))();
}

COMMAND(AddKernelCreator(OperatorConf::kMultinomialLogisticLossConf,
                         CreateMultinomialLogisticLossKernel));

}  // namespace oneflow
