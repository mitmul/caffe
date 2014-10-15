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
void LabelingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
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
  const Dtype *data = bottom[0]->cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      for (int c = 0; c < channels; ++c) {
        const Dtype p = data[i * dim + c * spatial_dim + j];
        if (c == label_value) {
          loss -= log(std::max(p, Dtype(kLOG_THRESHOLD)));
        } else {
          loss -= log(std::max(Dtype(1) - p, Dtype(kLOG_THRESHOLD)));
        }
      }
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
    const Dtype *bottom_data = bottom[0]->cpu_data();
    const Dtype *bottom_label = bottom[1]->cpu_data();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int channels = bottom[0]->channels();
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
        for (int c = 0; c < channels; ++c) {
          const Dtype data = bottom_data[i * dim + c * spatial_dim + j];
          Dtype diff = 0;
          if (c == label) {
            diff = -Dtype(1) / std::max(data, Dtype(FLT_MIN));
          } else {
            diff = Dtype(1) / std::max(Dtype(1) - data, Dtype(FLT_MIN));
          }
          bottom_diff[i * dim + c * spatial_dim + j] = diff;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(),
               loss_weight / num / channels / spatial_dim, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelingLossLayer);
#endif

INSTANTIATE_CLASS(LabelingLossLayer);
REGISTER_LAYER_CLASS(LABELING_LOSS, LabelingLossLayer);

}  // namespace caffe
