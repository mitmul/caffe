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
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&sigmoid_output_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&sigmoid_output_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&softmax_output_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  loss_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[1]->height(), bottom[1]->width());

  // Check the shapes of data and label
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->channels(), 1)
      << "The number of channels of label should be one.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
      << "The heights of data and label should be same.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
      << "The width of data and label should be same.";
}

template <typename Dtype>
void LabelingLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype *prob_data = softmax_output_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  const int num = softmax_output_.num();
  const int dim = softmax_output_.count() / num;
  const int channels = softmax_output_.channels();
  const int spatial_dim = softmax_output_.height() * softmax_output_.width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      for (int c = 0; c < channels; ++c) {
        const Dtype prob = prob_data[i * dim + c * spatial_dim + j];
        CHECK_GE(prob, 0.0);
        CHECK_LE(prob, 1.0);
        if (c == label_value) {
          loss -= log(std::max(prob, Dtype(kLOG_THRESHOLD)));
        } else {
          loss -= log(std::max(Dtype(1) - prob, Dtype(kLOG_THRESHOLD)));
        }
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num / channels / spatial_dim;
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
    const Dtype *bottom_data = softmax_output_.cpu_data();
    const Dtype *bottom_label = bottom[1]->cpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(softmax_output_.count(), bottom_data, bottom_diff);
    const int num = softmax_output_.num();
    const int dim = softmax_output_.count() / num;
    const int channels = softmax_output_.channels();
    const int spatial_dim = softmax_output_.height() * softmax_output_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
        for (int c = 0; c < channels; ++c) {
          const Dtype data = bottom_data[i * dim + c * spatial_dim + j];
          CHECK_GE(data, 0.0);
          CHECK_LE(data, 1.0);

          Dtype diff = 0;
          if (c == label) {
            diff = Dtype(1) / std::max(data, Dtype(FLT_MIN));
          } else {
            diff = -Dtype(1) / std::max(Dtype(1) - data, Dtype(FLT_MIN));
          }
          bottom_diff[i * dim + c * spatial_dim + j] = diff;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(softmax_output_.count(),
               loss_weight / num / channels / spatial_dim, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelingLossLayer);
#endif

INSTANTIATE_CLASS(LabelingLossLayer);
REGISTER_LAYER_CLASS(LABELING_LOSS, LabelingLossLayer);

}  // namespace caffe
