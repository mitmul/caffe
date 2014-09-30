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
void LabelingLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  // Reshape data
  label_num_ = this->layer_param_.labeling_loss_param().label_num();
  label_height_ = this->layer_param_.labeling_loss_param().label_height();
  label_width_ = this->layer_param_.labeling_loss_param().label_width();
  spatial_dim_ = label_height_ * label_width_;
  bottom[0]->Reshape(bottom[0]->num(), label_num_, label_height_, label_width_);
  bottom[1]->Reshape(bottom[1]->num(), 1, label_height_, label_width_);

  // Check the shapes of data and label
  CHECK_EQ(bottom[0]->channels(), label_num_) << "The number of channels of data should be " << label_num_ << ".";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "The heights of data and label should be same.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "The width of data and label should be same.";
  CHECK_EQ(bottom[1]->channels(), 1) << "The number of channels of label should be one.";
  CHECK_EQ(bottom[1]->height(), label_height_) << "The label height should be " << label_height_ << ".";
  CHECK_EQ(bottom[1]->width(), label_width_) << "The label width should be " << label_width_ << ".";
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype *prob_data = prob_.cpu_data();
  const Dtype *bottom_label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
      int index = i * dim + label * spatial_dim + j;
      loss -= log(std::max(prob_data[index], Dtype(FLT_MIN)));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype *bottom_label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
        bottom_diff[i * dim + label * spatial_dim + j] -= 1;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelingLossLayer);
#endif

INSTANTIATE_CLASS(LabelingLossLayer);

}  // namespace caffe
