#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  loss_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());

  // Check the shapes of data and label
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The number of num of data and label should be same.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
      << "The number of channels of data and label should be same.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The heights of data and label should be same.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "The width of data and label should be same.";
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  // Stable version of loss computation from input data
  const Dtype *data = prob_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= label[i] * log(std::max(data[i], Dtype(kLOG_THRESHOLD)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num / dim;
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
    const Dtype *data = prob_.cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    Dtype *diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, data, label, diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num / dim, diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_CROSS_ENTROPY_LOSS, SoftmaxCrossEntropyLossLayer);

}  // namespace caffe