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
class SoftmaxCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxCrossEntropyLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(2, 3, 5, 5)),
      blob_bottom_label_(new Blob<Dtype>(2, 3, 5, 5)),
      blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(25364);

    // Fill the data and labelvector
    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    Dtype *label = blob_bottom_label_->mutable_cpu_data();
    caffe_rng_uniform<Dtype>(blob_bottom_data_->count(), 0.01, 9.99, data);
    const int num = blob_bottom_data_->num();
    const int dim = blob_bottom_data_->count() / num;
    const int channels = blob_bottom_data_->channels();
    const int spatial_dim = dim / channels;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        for (int k = 0; k < channels; ++k) {
          label[i * dim + k * spatial_dim + j] = 0;
        }
        const int c = caffe_rng_rand() % channels;
        label[i * dim + c * spatial_dim + j] = 1;
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~SoftmaxCrossEntropyLossLayerTest() {
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

TYPED_TEST_CASE(SoftmaxCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestSoftmax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "cross entropy loss "
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

  // confirm that prepared input data is surely softmaxed
  const Dtype kErrorMargin = 1e-5;
  const Dtype *data = layer.prob_.cpu_data();
  const Dtype *label = this->blob_bottom_vec_[1]->cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Dtype data_value = 0;
        Dtype label_value = 0;
        for (int c = 0; c < channels; ++c) {
          const int idx = i * dim + c * spatial_dim + y * width + x;
          data_value += data[idx];
          label_value += label[idx];
        }
        EXPECT_NEAR(data_value, 1.0, kErrorMargin);
        EXPECT_NEAR(label_value, 1.0, kErrorMargin);
      }
    }
  }
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxCrossEntropyLossLayer<Dtype> layer_weight_1(layer_param);
  layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype loss_weight_1 = this->blob_top_vec_[0]->cpu_data()[0];
  LOG(INFO) << "Forward loss 1: " << loss_weight_1;

  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SoftmaxCrossEntropyLossLayer<Dtype> layer_weight_2(layer_param);
  layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss_weight_2 = layer_weight_2.Forward(
                                this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss 2: " << loss_weight_2;

  const Dtype kErrorMargin = 1e-5;
  EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

  const Dtype kNonTrivialAbsThresh = 1e-1;
  EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss: " << this->blob_top_vec_[0]->cpu_data()[0];

  vector<bool> propagate_down;
  propagate_down.push_back(true);
  propagate_down.push_back(false);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  const Dtype *data = layer.prob_.cpu_data();
  const Dtype *diff = this->blob_bottom_vec_[0]->cpu_diff();
  const Dtype *label = this->blob_bottom_vec_[1]->cpu_data();

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

  const Dtype kErrorMargin = 1e-3;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; ++j) {
      for (int c = 0; c < channels; ++c) {
        const Dtype label_value = label[i * dim + c * spatial_dim + j];
        const Dtype predict = data[i * dim + c * spatial_dim + j];
        EXPECT_GE(predict, 0);
        EXPECT_LE(predict, 1);
        EXPECT_NEAR(diff[i * dim + c * spatial_dim + j] * num * spatial_dim,
                    -(label_value * (1 - predict)),
                    kErrorMargin);
      }
    }
  }
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-3, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe