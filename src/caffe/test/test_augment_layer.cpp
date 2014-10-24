#include <string>
#include <vector>

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
    : blob_top_data_(new Blob<Dtype>()),
      blob_top_label_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);

    // load data
    string data_db_file =
      CMAKE_SOURCE_DIR "caffe/test/test_data/sample_data.lmdb" CMAKE_EXT;

    LayerParameter data_param;
    DataParameter *data_dparam = data_param.mutable_data_param();
    data_dparam->set_batch_size(5);
    data_dparam->set_source(data_db_file.c_str());
    data_dparam->set_backend(DataParameter_DB_LMDB);
    data_layer_ = new DataLayer<Dtype>(data_param);

    vector<Blob<Dtype>*> blob_data_layer_vec;
    blob_data_layer_vec.push_back(blob_top_data_);
    data_layer_->SetUp(blob_bottom_vec_, blob_data_layer_vec);
    data_layer_->Forward(blob_bottom_vec_, blob_data_layer_vec);

    LOG(INFO) << "data num: " << blob_top_data_->num();

    // load label
    string label_db_file =
      CMAKE_SOURCE_DIR "caffe/test/test_data/sample_label.lmdb" CMAKE_EXT;

    LayerParameter label_param;
    DataParameter *label_dparam = label_param.mutable_data_param();
    label_dparam->set_batch_size(5);
    label_dparam->set_source(label_db_file.c_str());
    label_dparam->set_backend(DataParameter_DB_LMDB);
    label_layer_ = new DataLayer<Dtype>(label_param);

    vector<Blob<Dtype>*> blob_label_layer_vec;
    blob_label_layer_vec.push_back(blob_top_label_);
    label_layer_->SetUp(blob_bottom_vec_, blob_label_layer_vec);
    label_layer_->Forward(blob_bottom_vec_, blob_label_layer_vec);

    LOG(INFO) << "label num: " << blob_top_label_->num();

    blob_bottom_vec_.push_back(blob_top_data_);
    blob_bottom_vec_.push_back(blob_top_label_);
  }

  virtual ~AugmentLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete data_layer_;
    delete label_layer_;
  }

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
  augment_param->set_label_crop_size(64);
  augment_param->set_rotate(true);
  google::protobuf::RepeatedField<float> *mean =
    augment_param->mutable_mean();
  mean->Add(77.196875000000006);
  mean->Add(87.893381076388891);
  mean->Add(85.800737847222223);
  google::protobuf::RepeatedField<float> *stddev =
    augment_param->mutable_stddev();
  stddev->Add(41.398123906821638);
  stddev->Add(39.564715010611259);
  stddev->Add(41.093476599359363);

  AugmentLayer<Dtype> augment(param);
  augment.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  augment.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

}  // namespace caffe
