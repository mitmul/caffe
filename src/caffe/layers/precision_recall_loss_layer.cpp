#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void PrecisionRecallLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void PrecisionRecallLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*> &bottom,
  const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::Reshape(bottom, top);
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
void PrecisionRecallLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {
  const Dtype *data = bottom[0]->cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int pnum = this->layer_param_.precision_recall_loss_param().point_num();
  Dtype auprc = 0.0;
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int p = 0; p <= pnum; ++p) {
        // compute precision and recall at below threshold
        const Dtype thresh = 1.0 / pnum * p;
        int true_positive = 0;
        int false_positive = 0;
        int false_negative = 0;
        int true_negative = 0;
        for (int j = 0; j < spatial_dim; ++j) {
          const Dtype data_value = data[i * dim + c * spatial_dim + j];
          const int label_value = (int)label[i * dim + c * spatial_dim + j];
          if (label_value == 1 && data_value >= thresh) {
            ++true_positive;
          }
          if (label_value == 0 && data_value >= thresh) {
            ++false_positive;
          }
          if (label_value == 1 && data_value < thresh) {
            ++false_negative;
          }
          if (label_value == 0 && data_value < thresh) {
            ++true_negative;
          }
        }
        Dtype precision = 0.0;
        Dtype recall = 0.0;
        if (true_positive > 0) {
          precision =
            (Dtype)true_positive / (Dtype)(true_positive + false_positive);
          recall =
            (Dtype)true_positive / (Dtype)(true_positive + false_negative);
        }
        auprc += precision * recall;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = auprc / num;
}
template <typename Dtype>
void PrecisionRecallLossLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*> &top,
  const vector<bool> &propagate_down,
  const vector<Blob<Dtype>*> &bottom) {
  const Dtype *data = bottom[0]->cpu_data();
  const Dtype *label = bottom[1]->cpu_data();
  Dtype *diff = bottom[0]->mutable_cpu_diff();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count() / num;
  const int channels = bottom[0]->channels();
  const int spatial_dim = bottom[0]->height() * bottom[0]->width();
  const int pnum = this->layer_param_.precision_recall_loss_param().point_num();
  Dtype auprc = 0.0;
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int p = 0; p <= pnum; ++p) {
        // compute precision and recall at below threshold
        const Dtype thresh = 1.0 / pnum * p;
        int true_positive = 0;
        int false_positive = 0;
        int false_negative = 0;
        int true_negative = 0;
        for (int j = 0; j < spatial_dim; ++j) {
          const Dtype data_value = data[i * dim + c * spatial_dim + j];
          const int label_value = (int)label[i * dim + c * spatial_dim + j];
          if (label_value == 1 && data_value >= thresh) {
            ++true_positive;
          }
          if (label_value == 0 && data_value >= thresh) {
            ++false_positive;
          }
          if (label_value == 1 && data_value < thresh) {
            ++false_negative;
          }
          if (label_value == 0 && data_value < thresh) {
            ++true_negative;
          }
        }
        Dtype precision = 0.0;
        Dtype recall = 0.0;
        if (true_positive > 0) {
          precision =
            (Dtype)true_positive / (Dtype)(true_positive + false_positive);
          recall =
            (Dtype)true_positive / (Dtype)(true_positive + false_negative);
        }
        auprc += precision * recall;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = auprc / num;

}

INSTANTIATE_CLASS(PrecisionRecallLossLayer);
REGISTER_LAYER_CLASS(PRECISION_RECALL_LOSS, PrecisionRecallLossLayer);

}  // namespace caffe