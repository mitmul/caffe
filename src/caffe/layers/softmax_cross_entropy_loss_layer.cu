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
    out[index] = -(l * log(max(p, Dtype(kLOG_THRESHOLD)))
                   + (1 - l) * log(max(1 - p, Dtype(kLOG_THRESHOLD))));
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype *data = prob_.gpu_data();
  const Dtype *label = bottom[1]->gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  Dtype *loss_data = loss_.mutable_gpu_data();
  caffe_copy(count, data, loss_data);

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_loss<Dtype>
  <<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, data, label, loss_data);
  Dtype loss = loss_.asum_data();
  top[0]->mutable_cpu_data()[0] = loss / num / channels / spatial_dim;
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
    const int channels = bottom[0]->channels();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    const Dtype *data = prob_.gpu_data();
    const Dtype *label = bottom[1]->gpu_data();
    Dtype *diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, data, diff);
    caffe_gpu_axpy(count, Dtype(-1), label, diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num / channels / spatial_dim, diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe