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
  const int num, const int dim, const int channels, const int spatial_dim,
  const Dtype *label, const Dtype *prob, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    const int i = index / dim; // num
    const int j = index % dim; // dim
    const int c = j / spatial_dim; // channel
    const int k = j % spatial_dim; // pos
    const int l = (int)label[i * spatial_dim + k];
    const Dtype p = prob[i * dim + c * spatial_dim + k];
    if (c + 1 == l) {
      out[index] = -log(max(p, Dtype(kLOG_THRESHOLD)));
    } else {
      out[index] = -log(max(Dtype(1) - p, Dtype(kLOG_THRESHOLD)));
    }
  }
}

template <typename Dtype>
__global__ void kernel_diff(
  const int num, const int dim, const int channels, const int spatial_dim,
  const Dtype *label, const Dtype *prob, Dtype *out) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    const int i = index / dim; // num
    const int j = index % dim; // dim
    const int c = j / spatial_dim; // channel
    const int k = j % spatial_dim; // pos
    const int l = (int)label[i * spatial_dim + k];
    const Dtype p = prob[i * dim + c * spatial_dim + k];
    if (c + 1 == l) {
      out[index] = -Dtype(1) / max(p, Dtype(kLOG_THRESHOLD));
    } else {
      out[index] = Dtype(1) / max(Dtype(1) - p, Dtype(kLOG_THRESHOLD));
    }
  }
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *data = bottom[0]->gpu_data();
  const Dtype *label = bottom[1]->gpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  Dtype *loss_data = loss_.mutable_gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_loss<Dtype>
  <<<CAFFE_GET_BLOCKS(num * channels * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
  (num, dim, channels, spatial_dim, label, data, loss_data);
  Dtype loss = loss_.asum_data();
  top[0]->mutable_cpu_data()[0] = loss / num / channels / spatial_dim;
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
    const Dtype *bottom_data = bottom[0]->gpu_data();
    const Dtype *bottom_label = bottom[1]->gpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int channels = bottom[0]->channels();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();

    // NOLINT_NEXT_LINE(whitespace/operators)
    kernel_diff<Dtype>
    <<<CAFFE_GET_BLOCKS(num * channels * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
    (num, dim, channels, spatial_dim, bottom_label, bottom_data, bottom_diff);

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(bottom[0]->count(),
                   loss_weight / num / channels / spatial_dim, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelingLossLayer);

}  // namespace caffe