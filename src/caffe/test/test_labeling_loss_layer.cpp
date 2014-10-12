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
    : blob_bottom_data_(new Blob<Dtype>(63, 3, 5, 5)),
      blob_bottom_label_(new Blob<Dtype>(63, 1, 5, 5)),
      blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 2;
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
  LabelingLossParameter *label_param =
    layer_param.mutable_labeling_loss_param();
  label_param->set_label_num(3);
  label_param->set_label_height(5);
  label_param->set_label_width(5);
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Softmax loss "
            << layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(layer.prob_.num(), 63);
  EXPECT_EQ(layer.prob_.channels(), 3);
  EXPECT_EQ(layer.prob_.height(), 5);
  EXPECT_EQ(layer.prob_.width(), 5);

  int num = layer.prob_.num();
  int channels = layer.prob_.channels();
  int height = layer.prob_.height();
  int width = layer.prob_.width();
  int dim = layer.prob_.count() / num;
  int spatial_dim = height * width;

  const Dtype kErrorMargin = 1e-5;
  const Dtype *prob_data = layer.prob_.cpu_data();
  for (int i = 0; i < num; ++i) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        Dtype value = 0;
        for (int c = 0; c < channels; ++c) {
          int idx = i * dim + c * spatial_dim + y * width + x;
          value += prob_data[idx];
        }
        EXPECT_NEAR(value, 1.0, kErrorMargin);
      }
    }
  }
}

TYPED_TEST(LabelingLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  // Get the loss without a specified objective weight -- should be
  // equivalent to explicitly specifiying a weight of 1.
  LayerParameter layer_param;
  LabelingLossParameter *label_param =
    layer_param.mutable_labeling_loss_param();
  label_param->set_label_num(3);
  label_param->set_label_height(5);
  label_param->set_label_width(5);

  LabelingLossLayer<Dtype> layer_weight_1(layer_param);
  layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype kErrorMargin = 1e-5;
  const Dtype loss_weight_1 =
    layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss 1: " << loss_weight_1;

  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LabelingLossLayer<Dtype> layer_weight_2(layer_param);
  layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss_weight_2 =
    layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss 2: " << loss_weight_2;
  EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

  const Dtype kNonTrivialAbsThresh = 1e-1;
  EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
}

TYPED_TEST(LabelingLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LabelingLossParameter *label_param =
    layer_param.mutable_labeling_loss_param();
  label_param->set_label_num(3);
  label_param->set_label_height(5);
  label_param->set_label_width(5);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LabelingLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Backward loss: " << loss;

  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
