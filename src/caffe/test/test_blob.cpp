#include <cstring>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobSimpleTest : public ::testing::Test {
 protected:
  BlobSimpleTest()
    : blob_(new Blob<Dtype>()),
      blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobSimpleTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype> *const blob_;
  Blob<Dtype> *const blob_preshaped_;
};

TYPED_TEST_CASE(BlobSimpleTest, TestDtypes);

TYPED_TEST(BlobSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_TRUE(this->blob_preshaped_);
  EXPECT_EQ(this->blob_preshaped_->num(), 2);
  EXPECT_EQ(this->blob_preshaped_->channels(), 3);
  EXPECT_EQ(this->blob_preshaped_->height(), 4);
  EXPECT_EQ(this->blob_preshaped_->width(), 5);
  EXPECT_EQ(this->blob_preshaped_->count(), 120);
  EXPECT_EQ(this->blob_->num(), 0);
  EXPECT_EQ(this->blob_->channels(), 0);
  EXPECT_EQ(this->blob_->height(), 0);
  EXPECT_EQ(this->blob_->width(), 0);
  EXPECT_EQ(this->blob_->count(), 0);
}

TYPED_TEST(BlobSimpleTest, TestPointersCPUGPU) {
  EXPECT_TRUE(this->blob_preshaped_->gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->cpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_gpu_data());
  EXPECT_TRUE(this->blob_preshaped_->mutable_cpu_data());
}

TYPED_TEST(BlobSimpleTest, TestReshape) {
  this->blob_->Reshape(2, 3, 4, 5);
  EXPECT_EQ(this->blob_->num(), 2);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 4);
  EXPECT_EQ(this->blob_->width(), 5);
  EXPECT_EQ(this->blob_->count(), 120);
}

template <typename TypeParam>
class BlobMathTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  BlobMathTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)) {}
  virtual ~BlobMathTest() { delete blob_; }
  Blob<Dtype>* const blob_;
};

TYPED_TEST_CASE(BlobMathTest, TestDtypesAndDevices);

TYPED_TEST(BlobMathTest, TestSumOfSquares) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Blob should have sum of squares == 0.
  EXPECT_EQ(0, this->blob_->sumsq_data());
  EXPECT_EQ(0, this->blob_->sumsq_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_sumsq = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_sumsq += data[i] * data[i];
  }
  // Do a mutable access on the current device,
  // so that the sumsq computation is done on that device.
  // (Otherwise, this would only check the CPU sumsq implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_FLOAT_EQ(expected_sumsq, this->blob_->sumsq_data());
  EXPECT_EQ(0, this->blob_->sumsq_diff());

  // Check sumsq_diff too.
  const Dtype kDiffScaleFactor = 7;
  caffe_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_FLOAT_EQ(expected_sumsq, this->blob_->sumsq_data());
  EXPECT_FLOAT_EQ(expected_sumsq * kDiffScaleFactor * kDiffScaleFactor,
                  this->blob_->sumsq_diff());
}

TYPED_TEST(BlobMathTest, TestAsum) {
  typedef typename TypeParam::Dtype Dtype;

  // Uninitialized Blob should have sum of squares == 0.
  EXPECT_EQ(0, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());
  FillerParameter filler_param;
  filler_param.set_min(-3);
  filler_param.set_max(3);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_);
  Dtype expected_asum = 0;
  const Dtype* data = this->blob_->cpu_data();
  for (int i = 0; i < this->blob_->count(); ++i) {
    expected_asum += std::fabs(data[i]);
  }
  // Do a mutable access on the current device,
  // so that the asum computation is done on that device.
  // (Otherwise, this would only check the CPU asum implementation.)
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_data();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_data();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_FLOAT_EQ(expected_asum, this->blob_->asum_data());
  EXPECT_EQ(0, this->blob_->asum_diff());

  // Check asum_diff too.
  const Dtype kDiffScaleFactor = 7;
  caffe_cpu_scale(this->blob_->count(), kDiffScaleFactor, data,
                  this->blob_->mutable_cpu_diff());
  switch (TypeParam::device) {
  case Caffe::CPU:
    this->blob_->mutable_cpu_diff();
    break;
  case Caffe::GPU:
    this->blob_->mutable_gpu_diff();
    break;
  default:
    LOG(FATAL) << "Unknown device: " << TypeParam::device;
  }
  EXPECT_FLOAT_EQ(expected_asum, this->blob_->asum_data());
  EXPECT_FLOAT_EQ(expected_asum * kDiffScaleFactor, this->blob_->asum_diff());
}

}  // namespace caffe
