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

  // Reshape loss for gpu
  loss_.Reshape(bottom[0]->num(), 1, bottom[1]->height(), bottom[1]->width());

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
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();

  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      loss -= log(std::max(data[i * dim + label_value * spatial_dim + j],
                           Dtype(FLT_MIN)));
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
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), bottom_diff);
    const Dtype *bottom_label = bottom[1]->cpu_data();
    const int num = bottom[0]->num();
    const int dim = bottom[0]->count() / num;
    const int spatial_dim = bottom[0]->height() * bottom[0]->width();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
        bottom_diff[i * dim + label * spatial_dim + j] -= 1;
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / num / spatial_dim,
               bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LabelingLossLayer);
#endif

INSTANTIATE_CLASS(LabelingLossLayer);
REGISTER_LAYER_CLASS(LABELING_LOSS, LabelingLossLayer);

}  // namespace caffe
