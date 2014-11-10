#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_loss(const int count, const Dtype *data,
                            const Dtype *label, Dtype *out) {
  CUDA_KERNEL_LOOP(index, count) {
    const int l = (int)label[index];
    const Dtype p = data[index];
    out[index] = -(p * (l - (p >= 0)) -
                   log(1 + exp(p - 2 * p * (p >= 0))));
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  // Stable version of loss computation from input data
  const Dtype *data = prob_.gpu_data();
  const Dtype *label = bottom[1]->gpu_data();
  Dtype *loss_data = loss_.mutable_gpu_data();
  caffe_copy(count, data, loss_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_loss<Dtype>
  <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, label, loss_data);
  Dtype loss = loss_.asum_data();
  top[0]->mutable_cpu_data()[0] = loss / num / dim;
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const Dtype *data = prob_.gpu_data();
    const Dtype *label = bottom[1]->gpu_data();
    Dtype *diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, data, diff);
    caffe_gpu_axpy(count, Dtype(-1), label, diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num / dim, diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe
