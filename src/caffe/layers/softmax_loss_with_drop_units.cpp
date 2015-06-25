#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void SoftmaxLossWithDropUnitsLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype> *>& bottom, const vector<Blob<Dtype> *>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  random_drop_ =
    this->layer_param_.softmax_loss_with_drop_units_param().random_drop();
  drop_channel_ =
    this->layer_param_.softmax_loss_with_drop_units_param().drop_channel();
  drop_ratio_ =
    this->layer_param_.softmax_loss_with_drop_units_param().drop_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
}

template<typename Dtype>
void SoftmaxLossWithDropUnitsLayer<Dtype>::Reshape(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  loss_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                    bottom[0]->height(), bottom[0]->width());

  // Check the shapes of data and label
  CHECK_EQ(bottom[0]->num(),      bottom[1]->num())
    << "The number of num of data and label should be same.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The number of channels of data and label should be same.";
  CHECK_EQ(bottom[0]->height(),   bottom[1]->height())
    << "The heights of data and label should be same.";
  CHECK_EQ(bottom[0]->width(),    bottom[1]->width())
    << "The width of data and label should be same.";
}

template<typename Dtype>
void SoftmaxLossWithDropUnitsLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype> *>& bottom,
  const vector<Blob<Dtype> *>& top) {
  softmax_bottom_vec_[0] = bottom[0];

  // input details
  const int count       = bottom[0]->count();
  const int num         = bottom[0]->num();
  const int dim         = bottom[0]->count() / num;
  const int spatial_dim = bottom[0]->width() * bottom[0]->height();
  const int channels    = bottom[0]->channels();

  // drop specific channel
  if (drop_channel_ >= 0) {
    Dtype *data = softmax_bottom_vec_[0]->mutable_cpu_data();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int index = i * dim + drop_channel_ * spatial_dim + j;
        data[index] = 0.0;
      }
    }
  }

  // randomly drop units
  else if (random_drop_) {
    unsigned int *mask  = rand_vec_.mutable_cpu_data();
    const int     count = bottom[0]->count();

    if (this->phase_ == TRAIN) {
      // Create random numbers
      caffe_rng_bernoulli(count, 1. - drop_ratio_, mask);

      for (int i = 0; i < count; ++i) {
        top_data[i] = bottom_data[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
  }

  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // Stable version of loss computation from input data
  const Dtype *data  = prob_.cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype loss         = 0;

  for (int i = 0; i < count; ++i) {
    CHECK_GE(label[i], 0);
    CHECK_LE(label[i], 1);
    CHECK_GE(data[i], 0);
    CHECK_LE(data[i], 1);
    loss -= label[i] * log(std::max(data[i], Dtype(kLOG_THRESHOLD)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template<typename Dtype>
void SoftmaxLossWithDropUnitsLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype> *>& top,
  const vector<bool>         & propagate_down,
  const vector<Blob<Dtype> *>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    // First, compute the diff
    const int count       = bottom[0]->count();
    const int num         = bottom[0]->num();
    const int dim         = bottom[0]->count() / num;
    const int spatial_dim = bottom[0]->width() *
                            bottom[0]->height();
    const int channels = bottom[0]->channels();
    const Dtype *data  = prob_.cpu_data();
    const Dtype *label = bottom[1]->cpu_data();
    Dtype *diff        = bottom[0]->mutable_cpu_diff();

    if (drop_channel_ >= 0) {
      for (int i = 0; i < num; ++i) {
        for (int j = 0; j < spatial_dim; ++j) {
          for (int c = 0; c < channels; ++c) {
            const int index = i * dim + c * spatial_dim + j;

            if (c == drop_channel_) diff[index] = 0;
            else diff[index] = data[index] - label[index];
          }
        }
      }
  } else if (random_drop_) {
      if (this->phase_ == TRAIN) {
        const unsigned int* mask = rand_vec_.cpu_data();
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i) {
          bottom_diff[i] = top_diff[i] * mask[i] * scale_;
        }
      } else {
        caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }
      caffe_sub(count, data, label, diff);
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxLossWithDropUnitsLayer);
#endif // ifdef CPU_ONLY

INSTANTIATE_CLASS(SoftmaxLossWithDropUnitsLayer);
REGISTER_LAYER_CLASS(SoftmaxLossWithDropUnits);
} // namespace caffe
