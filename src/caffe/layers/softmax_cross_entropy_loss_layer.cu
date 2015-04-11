#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {

  // input details
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int spatial_dim = bottom[0]->width() * bottom[0]->height();
  const int channels = bottom[0]->channels();

  // all units in this channel goes to zero
  const int zero_channel =
    this->layer_param_.softmax_cross_entropy_loss_param().zero_channel();
  if (zero_channel >= 0) {
    Dtype *data = softmax_bottom_vec_[0]->mutable_cpu_data();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int index = i * dim + zero_channel * spatial_dim + j;
        data[index] = 0.0;
      }
    }
  }

  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // Stable version of loss computation from input data
  const Dtype *data = prob_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss = 0;

  // Compute the loss (negative log likelihood)
  const google::protobuf::RepeatedField<float> weights =
    this->layer_param_.softmax_cross_entropy_loss_param().weights();
  if (weights.size() > 0) {
    CHECK_EQ(weights.size(), bottom[0]->channels());
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        for (int c = 0; c < channels; ++c) {
          const int index = i * dim + c * spatial_dim + j;
          loss -= weights.Get(c) * label[index] *
                  log(std::max(data[index], Dtype(kLOG_THRESHOLD)));
        }
      }
    }
  } else {
    for (int i = 0; i < count; ++i) {
      loss -= label[i] * log(std::max(data[i], Dtype(kLOG_THRESHOLD)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / dim;
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int spatial_dim = bottom[0]->width() * bottom[0]->height();
    const int channels = bottom[0]->channels();
    const Dtype *data = prob_.cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    Dtype *diff = bottom[0]->mutable_cpu_diff();
    const google::protobuf::RepeatedField<float> weights =
      this->layer_param_.softmax_cross_entropy_loss_param().weights();
    if (weights.size() > 0) {
      CHECK_EQ(weights.size(), bottom[0]->channels());
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          for (int c = 0; c < channels; ++c) {
            const int index = i * dim + c * spatial_dim + j;
            diff[index] = weights.Get(c) * (data[index] - label[index]);
          }
        }
      }
    } else {
      caffe_sub(count, data, label, diff);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num / dim, diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe
