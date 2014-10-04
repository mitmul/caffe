#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_loss(const int num, const int spatial_dim, const int dim,
                            const Dtype *label, const Dtype *prob, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int i = index / spatial_dim;
    int j = index % spatial_dim;
    int pos = label[i * spatial_dim + j];
    out[index] = -log(max(prob[i * dim + pos * spatial_dim + j],
                          Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
__global__ void kernel_diff(const int num, const int spatial_dim, const int dim,
                            const Dtype *label, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int i = index / spatial_dim;
    int j = index % spatial_dim;
    int pos = label[i * spatial_dim + j];
    out[i * dim + pos * spatial_dim + j] -= 1;
  }
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype *prob_data = prob_.gpu_data();
  const Dtype *bottom_label = bottom[1]->gpu_data();
  const int num = prob_.num();
  const int dim = prob_.count() / num;
  const int spatial_dim = prob_.height() * prob_.width();
  Dtype *loss_vec = loss_.mutable_gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_loss<Dtype> <<< CAFFE_GET_BLOCKS(num *spatial_dim),
              CAFFE_CUDA_NUM_THREADS>>>(num, spatial_dim, dim,
                                        bottom_label, prob_data, loss_vec);
  Dtype loss = 0;
  caffe_gpu_asum(num * spatial_dim, loss_vec, &loss);
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *prob_data = prob_.gpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype *bottom_label = bottom[1]->gpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_diff<Dtype> <<< CAFFE_GET_BLOCKS(num *spatial_dim),
                CAFFE_CUDA_NUM_THREADS>>>(num, spatial_dim, dim,
                                          bottom_label, bottom_diff);
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}

INSTANTIATE_CLASS(LabelingLossLayer);

}  // namespace caffe
