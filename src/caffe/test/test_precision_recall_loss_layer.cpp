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
class PrecisionRecallLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  PrecisionRecallLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(2, 3, 5, 5)),
      blob_bottom_label_(new Blob<Dtype>(2, 3, 5, 5)),
      blob_top_loss_(new Blob<Dtype>()) {
    Caffe::set_random_seed(25364);

    // Fill the data and labelvector
    FillerParameter filler_param;
    filler_param.set_min(0.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    Dtype *label = blob_bottom_label_->mutable_cpu_data();
    const int num = blob_bottom_data_->num();
    const int dim = blob_bottom_data_->count() / num;
    const int channels = blob_bottom_data_->channels();
    const int spatial_dim = dim / channels;
    // set label to be 1-of-K coding form
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        Dtype data_value = 0.0;
        for (int k = 0; k < channels; ++k) {
          data_value += exp(data[i * dim + k * spatial_dim + j]);
        }
        for (int k = 0; k < channels; ++k) {
          data[i * dim + k * spatial_dim + j] =
            exp(data[i * dim + k * spatial_dim + j]) / data_value;
        }
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

  virtual ~PrecisionRecallLossLayerTest() {
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

TYPED_TEST_CASE(PrecisionRecallLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(PrecisionRecallLossLayerTest, TestSoftmax) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PrecisionRecallLossLayer<Dtype> layer(layer_param);
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
  const Dtype *data = this->blob_bottom_vec_[0]->cpu_data();
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

TYPED_TEST(PrecisionRecallLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PrecisionRecallLossLayer<Dtype> layer_weight_1(layer_param);
  layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype loss_weight_1 = this->blob_top_vec_[0]->cpu_data()[0];
  LOG(INFO) << "Forward loss 1: " << loss_weight_1;

  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  PrecisionRecallLossLayer<Dtype> layer_weight_2(layer_param);
  layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype loss_weight_2 = layer_weight_2.Forward(
                                this->blob_bottom_vec_, this->blob_top_vec_);
  LOG(INFO) << "Forward loss 2: " << loss_weight_2;

  const Dtype kErrorMargin = 1e-5;
  EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);

  const Dtype kNonTrivialAbsThresh = 1e-1;
  EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
}

// TYPED_TEST(PrecisionRecallLossLayerTest, TestDiff) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   PrecisionRecallLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//   vector<bool> propagate_down;
//   propagate_down.push_back(true);
//   propagate_down.push_back(false);
//   layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

//   const Dtype step = 1e-2;
//   const Dtype threshold = 1e-2;
//   const int num = this->blob_bottom_vec_[0]->num();
//   const int channels = this->blob_bottom_vec_[0]->channels();
//   const int dim = this->blob_bottom_vec_[0]->count() / num;
//   const int height = this->blob_bottom_vec_[0]->height();
//   const int width = this->blob_bottom_vec_[0]->width();
//   const int spatial_dim = height * width;
//   for (int i = 0; i < num; ++i) {
//     for (int h = 0; h < height; ++h) {
//       for (int w = 0; w < width; ++w) {
//         for (int c = 0; c < channels; ++c) {
//           const int index = i * dim + c * spatial_dim + h * width + w;
//           const Dtype diff = this->blob_bottom_vec_[0]->cpu_diff()[index];
//           const Dtype feature = this->blob_bottom_vec_[0]->cpu_data()[index];
//           const Dtype label = this->blob_bottom_vec_[1]->cpu_data()[index];

//           this->blob_bottom_vec_[0]->mutable_cpu_data()[index] += step;
//           const Dtype positive_objective =
//             layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//           layer.Backward(this->blob_top_vec_, propagate_down,
//                          this->blob_bottom_vec_);

//           this->blob_bottom_vec_[0]->mutable_cpu_data()[index] -= step * 2;
//           const Dtype negative_objective =
//             layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//           layer.Backward(this->blob_top_vec_, propagate_down,
//                          this->blob_bottom_vec_);

//           // reverse data value
//           this->blob_bottom_vec_[0]->mutable_cpu_data()[index] += step;

//           EXPECT_NE(positive_objective, negative_objective);

//           const Dtype expected_diff =
//             (positive_objective - negative_objective) / (2 * step);

//           if (fabs(diff - expected_diff) > threshold) {
//             LOG(INFO) << "num: " << i;
//             LOG(INFO) << "channels: " << c;
//             LOG(INFO) << "height: " << h;
//             LOG(INFO) << "width: " << w;
//             LOG(INFO) << "feature: " << feature;
//             LOG(INFO) << "diff: " << diff;
//             LOG(INFO) << "label: " << label;
//             LOG(INFO) << "positive_objective: " << positive_objective;
//             LOG(INFO) << "negative_objective: " << negative_objective;
//             LOG(INFO) << "delta: " << positive_objective - negative_objective;
//             LOG(INFO) << "expected diff: " << expected_diff;
//             LOG(INFO) << "computed diff: " << diff;
//             LOG(INFO) << "difference: " << diff - expected_diff;
//           }
//           EXPECT_NEAR(diff, expected_diff, threshold);
//         }
//       }
//     }
//   }
// }

// TYPED_TEST(PrecisionRecallLossLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   const Dtype kLossWeight = 3.7;
//   layer_param.add_loss_weight(kLossWeight);
//   PrecisionRecallLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//                                   this->blob_top_vec_, 0);
// }

}  // namespace caffe