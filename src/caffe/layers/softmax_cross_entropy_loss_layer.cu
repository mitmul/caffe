#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
  Forward_cpu(bottom, top);
}

template<typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype> *>& top,
  const vector<bool>         & propagate_down,
  const vector<Blob<Dtype> *>& bottom) {
  Backward_gpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);
} // namespace caffe
