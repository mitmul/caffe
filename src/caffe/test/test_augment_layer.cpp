#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class AugmentLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  AugmentLayerTest()
    : blob_bottom_data_(new Blob<Dtype>()),
      blob_bottom_label_(new Blob<Dtype>()),
      blob_top_data_(new Blob<Dtype>()),
      blob_top_label_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);

    // load data
    string data_db_file =
      CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data.lmdb" CMAKE_EXT;

    LayerParameter data_param;
    DataParameter *data_dparam = data_param.mutable_data_param();
    data_dparam->set_batch_size(10);
    data_dparam->set_source(data_db_file.c_str());
    data_dparam->set_backend(DataParameter_DB_LMDB);
    data_layer_ = new DataLayer<Dtype>(data_param);
    vector<Blob<Dtype>*> blob_data_layer_vec;
    blob_data_layer_vec.push_back(blob_bottom_data_);
    data_layer_->SetUp(blob_bottom_vec_, blob_data_layer_vec);
    data_layer_->Forward(blob_bottom_vec_, blob_data_layer_vec);

    LOG(INFO) << "data num: " << blob_bottom_data_->num();
    LOG(INFO) << "data channels: " << blob_bottom_data_->channels();
    LOG(INFO) << "data height: " << blob_bottom_data_->height();
    LOG(INFO) << "data width: " << blob_bottom_data_->width();

    // load label
    string label_db_file =
      CMAKE_SOURCE_DIR "caffe/test/test_data/sample_label.lmdb" CMAKE_EXT;

    LayerParameter label_param;
    DataParameter *label_dparam = label_param.mutable_data_param();
    label_dparam->set_batch_size(10);
    label_dparam->set_source(label_db_file.c_str());
    label_dparam->set_backend(DataParameter_DB_LMDB);
    label_layer_ = new DataLayer<Dtype>(label_param);
    vector<Blob<Dtype>*> blob_label_layer_vec;
    blob_label_layer_vec.push_back(blob_bottom_label_);
    label_layer_->SetUp(blob_bottom_vec_, blob_label_layer_vec);
    label_layer_->Forward(blob_bottom_vec_, blob_label_layer_vec);

    LOG(INFO) << "label num: " << blob_bottom_label_->num();
    LOG(INFO) << "label channels: " << blob_bottom_label_->channels();
    LOG(INFO) << "label height: " << blob_bottom_label_->height();
    LOG(INFO) << "label width: " << blob_bottom_label_->width();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);

    string out_dir = CMAKE_SOURCE_DIR "caffe/test/test_data" CMAKE_EXT;
    SaveData(*blob_bottom_data_, out_dir, "setup_data");
    SaveData(*blob_bottom_label_, out_dir, "setup_label");
  }

  void SaveData(const Blob<Dtype> &blob, const string &out_dir,
                const string &prefix) {
    // save data
    const int num = blob.num();
    const int channels = blob.channels();
    const int height = blob.height();
    const int width = blob.width();
    LOG(INFO) << "SaveData: " << num << " x " << channels
              << " x " << height << " x " << width;
    for (int i = 0; i < num; ++i) {
      const Dtype *data = blob.cpu_data() + blob.offset(i);
      cv::Mat img(height, width, CV_32FC(channels));
      for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            const Dtype pix = data[c * height * width + y * width + x];
            const int pos = y * width * channels + x * channels + c;
            reinterpret_cast<float *>(img.data)[pos] = static_cast<float>(pix);
          }
        }
      }

      cv::Scalar mean, stddev;
      cv::meanStdDev(img, mean, stddev);
      for (int c = 0; c < channels; ++c) {
        LOG(INFO) << "mean: " << mean[c] << "\tstddev: " << stddev[c];
      }

      cv::Mat dst;
      cv::normalize(img, dst, 0, 255, cv::NORM_MINMAX);
      dst.convertTo(dst, CV_8U);

      stringstream ss;
      ss << out_dir << "/test_data_" << prefix << "_" << i << "_"
         << channels << "_" << height << "_" << width << ".png";
      cv::imwrite(ss.str(), dst);
    }
  }

  virtual ~AugmentLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_data_;
    delete blob_top_label_;
    delete data_layer_;
    delete label_layer_;
  }

  Blob<Dtype> *const blob_bottom_data_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype> *const blob_top_data_;
  Blob<Dtype> *const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  DataLayer<Dtype> *data_layer_;
  DataLayer<Dtype> *label_layer_;
};

TYPED_TEST_CASE(AugmentLayerTest, TestDtypesAndDevices);

TYPED_TEST(AugmentLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  AugmentParameter *augment_param = param.mutable_augment_param();
  augment_param->set_data_crop_size(64);
  augment_param->set_label_crop_size(16);
  augment_param->set_rotate(true);
  augment_param->set_normalize(true);
  // google::protobuf::RepeatedField<float> *mean =
  //   augment_param->mutable_mean();
  // mean->Add(75.871057942708333);
  // mean->Add(84.302632378472225);
  // mean->Add(82.718964843750001);
  // google::protobuf::RepeatedField<float> *stddev =
  //   augment_param->mutable_stddev();
  // stddev->Add(50.142589887293951);
  // stddev->Add(48.17382927556428);
  // stddev->Add(50.290859541028532);

  // save image dir
  string out_dir = CMAKE_SOURCE_DIR "caffe/test/test_data" CMAKE_EXT;

  AugmentLayer<Dtype> augment(param);
  augment.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  augment.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  this->SaveData(*this->blob_top_data_, out_dir, "data");
  this->SaveData(*this->blob_top_label_, out_dir, "label");
}

}  // namespace caffe
