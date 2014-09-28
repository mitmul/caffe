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
    : blob_bottom_data_(new Blob<Dtype>(2, 3, 4, 4)),
      blob_bottom_label_(new Blob<Dtype>(2, 1, 4, 4)),
      blob_top_loss_(new Blob<Dtype>()) {
    Dtype *bottom_data = blob_bottom_data_->mutable_cpu_data();
    Dtype *bottom_label = blob_bottom_label_->mutable_cpu_data();

    // fill the values
    /**
     * data = [[[[0, 0, 0, 0],
     *           [0, 0, 0, 0],
     *           [0, 0, 0, 0],
     *           [0, 0, 0, 0]],
     *          [[0, 1, 2, 3],
     *           [1, 2, 3, 4],
     *           [2, 3, 4, 5],
     *           [3, 4, 5, 6]], // sum: 48
     *          [[0, 2, 4, 6],
     *           [2, 4, 6, 8],
     *           [4, 6, 8,10],
     *           [6, 8,10,12]]], ...] // sum: 96
     *
     * softmax(data) =
     * [[[  3.33333330e-01,   9.00305700e-02,   1.58762400e-02,   2.35563000e-03],
     *   [  9.00305700e-02,   1.58762400e-02,   2.35563000e-03,   3.29320439e-04],
     *   [  1.58762400e-02,   2.35563000e-03,   3.29320439e-04,   4.50940412e-05],
     *   [  2.35563000e-03,   3.29320439e-04,   4.50940412e-05,   6.12898247e-06]],
     *  [[  3.33333330e-01,   2.44728470e-01,   1.17310430e-01,   4.73141600e-02],
     *   [  2.44728470e-01,   1.17310430e-01,   4.73141600e-02,   1.79802867e-02],
     *   [  1.17310430e-01,   4.73141600e-02,   1.79802867e-02,   6.69254912e-03],
     *   [  4.73141600e-02,   1.79802867e-02,   6.69254912e-03,   2.47260800e-03]],
     *  [[  3.33333330e-01,   6.65240960e-01,   8.66813330e-01,   9.50330210e-01],
     *   [  6.65240960e-01,   8.66813330e-01,   9.50330210e-01,   9.81690393e-01],
     *   [  8.66813330e-01,   9.50330210e-01,   9.81690393e-01,   9.93262357e-01],
     *   [  9.50330210e-01,   9.81690393e-01,   9.93262357e-01,   9.97521263e-01]]]
     *
     * label = [[[[1, 1, 1, 1],
     *            [1, 1, 1, 1],
     *            [1, 1, 1, 1],
     *            [1, 1, 1, 1]]],
     *          [[[2, 2, 2, 2],
     *            [2, 2, 2, 2],
     *            [2, 2, 2, 2],
     *            [2, 2, 2, 2]]]]
     *
     * loss ~= 1.6636151596351503
     */
    int num = blob_bottom_data_->num();
    int channels = blob_bottom_data_->channels();
    int height = blob_bottom_data_->height();
    int width = blob_bottom_data_->width();
    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            int d_index = i * channels * height * width
                          + c * height * width
                          + y * width + x;
            bottom_data[d_index] = c * (x + y);
            int l_index = i * height * width + y * width + x;
            bottom_label[l_index] = i + 1;
          }
        }
      }
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

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    LabelingLossParameter *label_param = layer_param.mutable_labeling_loss_param();
    label_param->set_label_num(3);
    label_param->set_label_height(4);
    label_param->set_label_width(4);
    LabelingLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
      layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    LOG(INFO) << loss_weight_1;
    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    // const Dtype kLossWeight = 3.7;
    // layer_param.add_loss_weight(kLossWeight);
    // LabelingLossLayer<Dtype> layer_weight_2(layer_param);
    // layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    // const Dtype loss_weight_2 =
    //   layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // const Dtype kErrorMargin = 1e-5;
    // EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // // Make sure the loss is non-trivial.
    // const Dtype kNonTrivialAbsThresh = 1e-1;
    // EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LabelingLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelingLossLayerTest, TestForward) {
  this->TestForward();
}

// TYPED_TEST(LabelingLossLayerTest, TestGradient) {
//   typedef typename TypeParam::Dtype Dtype;
//   LayerParameter layer_param;
//   const Dtype kLossWeight = 3.7;
//   layer_param.add_loss_weight(kLossWeight);
//   LabelingLossLayer<Dtype> layer(layer_param);
//   layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//   GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
//   checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//                                   this->blob_top_vec_);
// }

}  // namespace caffe
