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
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_data_);

    // perform softmax to bottom data
    vector<Blob<Dtype>*> softmax_vec_;
    softmax_vec_.push_back(blob_bottom_data_);
    LayerParameter param;
    SoftmaxLayer<Dtype> softmax(param);
    softmax.SetUp(softmax_vec_, softmax_vec_);
    softmax.Forward(softmax_vec_, softmax_vec_);

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
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Softmax loss " << this->blob_top_vec_[0]->cpu_data()[0];

  const int num = this->blob_bottom_vec_[0]->num();
  const int channels = this->blob_bottom_vec_[0]->channels();
  const int height = this->blob_bottom_vec_[0]->height();
  const int width = this->blob_bottom_vec_[0]->width();
  const int dim = this->blob_bottom_vec_[0]->count() / num;
  const int spatial_dim = height * width;
  const Dtype *prob_data = this->blob_bottom_vec_[0]->cpu_data();
  EXPECT_EQ(num, 2);
  EXPECT_EQ(channels, 3);
  EXPECT_EQ(height, 5);
  EXPECT_EQ(width, 5);

  const Dtype kErrorMargin = 1e-5;
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Dtype value = 0;
        for (int c = 0; c < channels; ++c) {
          const int idx = i * dim + c * spatial_dim + y * width + x;
          value += prob_data[idx];
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

  const int num = this->blob_bottom_vec_[0]->num();
  const int channels = this->blob_bottom_vec_[0]->channels();
  const int height = this->blob_bottom_vec_[0]->height();
  const int width = this->blob_bottom_vec_[0]->width();
  const int spatial_dim = height * width;
  const int dim = channels * spatial_dim;
  EXPECT_EQ(num, 2);
  EXPECT_EQ(channels, 3);
  EXPECT_EQ(height, 5);
  EXPECT_EQ(width, 5);

  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer_weight_2.Backward(this->blob_top_vec_,
                          propagate_down,
                          this->blob_bottom_vec_);

  const Dtype *bottom_label = this->blob_bottom_vec_[1]->cpu_data();
  const Dtype *bottom_data = this->blob_bottom_vec_[0]->cpu_data();
  const Dtype *bottom_diff = this->blob_bottom_vec_[0]->cpu_diff();
  const Dtype loss_weight = this->blob_top_vec_[0]->cpu_diff()[0];

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      const int label = static_cast<int>(bottom_label[i * spatial_dim + j]);
      EXPECT_NEAR(bottom_data[i * dim + 0 * spatial_dim + j]
                  + bottom_data[i * dim + 1 * spatial_dim + j]
                  + bottom_data[i * dim + 2 * spatial_dim + j], 1, 1e-5);
      EXPECT_EQ((bottom_data[i * dim + label * spatial_dim + j] - 1)
                * (loss_weight / num / spatial_dim),
                bottom_diff[i * dim + label * spatial_dim + j]);
    }
  }
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
  layer.Backward(this->blob_top_vec_,
                 propagate_down,
                 this->blob_bottom_vec_);

  const Dtype *diff = this->blob_bottom_vec_[0]->cpu_diff();
  const Dtype *label = this->blob_bottom_vec_[1]->cpu_data();
  const Dtype loss_weight = this->blob_top_vec_[0]->cpu_diff()[0];
  LOG(INFO) << "Loss: " << this->blob_top_vec_[0]->cpu_data()[0];
  LOG(INFO) << "Loss weight: " << this->blob_top_vec_[0]->cpu_diff()[0];

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

  const Dtype kErrorMargin = 1e-5;
  const Dtype *prob_data = this->blob_bottom_vec_[0]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Dtype value = 0;
        Dtype diff_value = 0;
        for (int c = 0; c < channels; ++c) {
          const int idx = i * dim + c * spatial_dim + y * width + x;
          value += prob_data[idx];
          diff_value += diff[idx];
        }
        EXPECT_NEAR(value, 1.0, kErrorMargin);
        EXPECT_NEAR(diff_value, 0.0, kErrorMargin);

        const int label_idx = i * label_dim + y * width + x;
        const int label_ans = label[label_idx];
        const int prob_idx = i * dim + label_ans * spatial_dim + y * width + x;

        EXPECT_NEAR((prob_data[prob_idx] - 1)
                    * (loss_weight / num / spatial_dim),
                    diff[prob_idx], kErrorMargin);
      }
    }
  }
}

TYPED_TEST(LabelingLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss: " << this->blob_top_vec_[0]->cpu_data()[0];

  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
