#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LabelingLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LabelingLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(2, 3, 5, 5)),
      blob_bottom_label_(new Blob<Dtype>(2, 1, 5, 5)),
      blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(25364);

    // fill values
    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    caffe_rng_uniform<Dtype>(blob_bottom_data_->count(), 0.2, 0.8, data);
    const int dim = blob_bottom_data_->count() / blob_bottom_data_->num();
    const int channels = blob_bottom_data_->channels();
    const int spatial_dim = dim / channels;
    for (int i = 0; i < blob_bottom_data_->num(); ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        Dtype sum = 0;
        for (int k = 0; k < channels; ++k) {
          sum += data[i * dim + k * spatial_dim + j];
        }
        for (int k = 0; k < channels; ++k) {
          data[i * dim + k * spatial_dim + j] /= sum;
        }
      }
    }

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 3;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~LabelingLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LabelingLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelingLossLayerTest, TestSoftmax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Softmax loss "
            << layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const int num = this->blob_bottom_vec_[0]->num();
  const int channels = this->blob_bottom_vec_[0]->channels();
  const int height = this->blob_bottom_vec_[0]->height();
  const int width = this->blob_bottom_vec_[0]->width();
  const int dim = this->blob_bottom_vec_[0]->count() / num;
  const int spatial_dim = height * width;
  EXPECT_EQ(num, 2);
  EXPECT_EQ(channels, 3);
  EXPECT_EQ(height, 5);
  EXPECT_EQ(width, 5);

  const Dtype kErrorMargin = 1e-5;
  const Dtype *softmax_output_data = this->blob_bottom_vec_[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Dtype value = 0;
        for (int c = 0; c < channels; ++c) {
          const int idx = i * dim + c * spatial_dim + y * width + x;
          value += softmax_output_data[idx];
        }
        EXPECT_NEAR(value, 1.0, kErrorMargin);
      }
    }
  }
}

TYPED_TEST(LabelingLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LabelingLossLayer<Dtype> layer_weight_1(layer_param);
  layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype loss_weight_1 = this->blob_top_vec_[0]->cpu_data()[0];
  LOG(INFO) << "Forward loss 1: " << loss_weight_1;

  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LabelingLossLayer<Dtype> layer_weight_2(layer_param);
  layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss_weight_2 = layer_weight_2.Forward(
                                this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss 2: " << loss_weight_2;

  const Dtype kErrorMargin = 1e-5;
  EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

  const Dtype kNonTrivialAbsThresh = 1e-1;
  EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
}

TYPED_TEST(LabelingLossLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss: " << this->blob_top_vec_[0]->cpu_data()[0];

  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  const Dtype *diff = this->blob_bottom_vec_[0]->cpu_diff();
  const Dtype *label = this->blob_bottom_vec_[1]->cpu_data();

  const int num = this->blob_bottom_vec_[0]->num();
  const int channels = this->blob_bottom_vec_[0]->channels();
  const int height = this->blob_bottom_vec_[0]->height();
  const int width = this->blob_bottom_vec_[0]->width();
  const int dim = this->blob_bottom_vec_[0]->count() / num;
  const int spatial_dim = height * width;
  const int label_dim = this->blob_bottom_vec_[1]->count() / num;
  EXPECT_EQ(num, 2);
  EXPECT_EQ(channels, 3);
  EXPECT_EQ(height, 5);
  EXPECT_EQ(width, 5);

  const Dtype kErrorMargin = 1e-3;
  const Dtype *prob_data = this->blob_bottom_vec_[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      const int label_ans = label[i * label_dim + j];
      for (int c = 0; c < channels; ++c) {
        const int prob_idx = i * dim + c * spatial_dim + j;
        if (c == label_ans) {
          EXPECT_NEAR(-1 / std::max(prob_data[prob_idx], Dtype(FLT_MIN)),
                      diff[prob_idx] * num * channels * spatial_dim,
                      kErrorMargin);
        } else {
          EXPECT_NEAR(1 / std::max(1 - prob_data[prob_idx], Dtype(FLT_MIN)),
                      diff[prob_idx] * num * channels * spatial_dim,
                      kErrorMargin);
        }
      }
    }
  }
}

TYPED_TEST(LabelingLossLayerTest, TestDiff) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  const int feature_id = 120;
  Dtype diff = this->blob_bottom_vec_[0]->cpu_diff()[feature_id];

  Dtype step = 1e-2;
  this->blob_bottom_vec_[0]->mutable_cpu_data()[feature_id] += step;
  Dtype positive_objective =
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  LOG(INFO) << "positive_objective: " << positive_objective;

  this->blob_bottom_vec_[0]->mutable_cpu_data()[feature_id] -= step * 2;
  Dtype negative_objective =
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  LOG(INFO) << "negative_objective: " << negative_objective;
  this->blob_bottom_vec_[0]->mutable_cpu_data()[feature_id] = step;

  const int num = this->blob_bottom_vec_[0]->num();
  const int channels = this->blob_bottom_vec_[0]->channels();
  const int height = this->blob_bottom_vec_[0]->height();
  const int width = this->blob_bottom_vec_[0]->width();
  const int spatial_dim = height * width;

  LOG(INFO) << "expected diff: "
            << (positive_objective - negative_objective) / (2 * step);
  LOG(INFO) << "actual diff: "
            << diff;

  EXPECT_NEAR(diff,
              (positive_objective - negative_objective) / (2 * step), 1e-2);
}

TYPED_TEST(LabelingLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
