#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_loss(
  const int num, const int spatial_dim, const int dim,
  const Dtype *label, const Dtype *prob, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    const int i = index / spatial_dim;
    const int j = index % spatial_dim;
    const int l = (int)label[i * spatial_dim + j];
    out[index] = -log(max(prob[i * dim + l * spatial_dim + j],
                          Dtype(FLT_MIN)));
  }
}

template <typename Dtype>
__global__ void kernel_diff(
  const int num, const int spatial_dim, const int dim,
  const Dtype *label, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    const int i = index / spatial_dim;
    const int j = index % spatial_dim;
    const int l = (int)label[i * spatial_dim + j];
    out[i * dim + l * spatial_dim + j] -= 1;
  }
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *bottom_label = bottom[1]->gpu_data();
  const Dtype *prob_data = bottom[0]->gpu_data();
  Dtype *loss_data = loss_.mutable_gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_loss<Dtype>
  <<<CAFFE_GET_BLOCKS(num * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
  (num, spatial_dim, dim, bottom_label, prob_data, loss_data);
  Dtype loss = loss_.asum_data();
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
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), bottom_diff);
    const Dtype *bottom_label = bottom[1]->gpu_data();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_diff<Dtype>
    <<<CAFFE_GET_BLOCKS(num * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
    (num, spatial_dim, dim, bottom_label, bottom_diff);

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(bottom[0]->count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelingLossLayer);

}  // namespace caffe
