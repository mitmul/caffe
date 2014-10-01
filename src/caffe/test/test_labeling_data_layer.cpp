#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LabelingDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LabelingDataLayerTest()
    : blob_top_data_(new Blob<Dtype>()),
      blob_top_label_(new Blob<Dtype>()),
      seed_(1701) {}

  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LMDB with data
  void FillLMDB() {
    LOG(INFO) << "Using temporary lmdb " << *filename_;
    CHECK_EQ(mkdir(filename_->c_str(), 0744), 0) << "mkdir " << filename_ << "failed";

    MDB_env *env;
    MDB_dbi dbi;
    MDB_val mdbkey, mdbdata;
    MDB_txn *txn;
    CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS) << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(env, filename_->c_str(), 0, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(env, NULL, 0, &txn), MDB_SUCCESS) << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS) << "mdb_open failed";

    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(0);
      datum.set_channels(3);
      datum.set_height(2);
      datum.set_width(3);
      std::string *data = datum.mutable_data();
      google::protobuf::RepeatedField<float> *label = datum.mutable_float_data();
      for (int j = 0; j < 18; ++j) {
        data->push_back(static_cast<uint8_t>(j));
      }
      label->Add(1); // label_channels
      label->Add(2); // label_height
      label->Add(3); // label_width
      for (int j = 0; j < 6; ++j) {
        label->Add(static_cast<float>(j));
      }
      stringstream ss;
      ss << i;

      string value;
      datum.SerializeToString(&value);
      mdbdata.mv_size = value.size();
      mdbdata.mv_data = reinterpret_cast<void *>(&value[0]);
      string keystr = ss.str();
      mdbkey.mv_size = keystr.size();
      mdbkey.mv_data = reinterpret_cast<void *>(&keystr[0]);
      CHECK_EQ(mdb_put(txn, dbi, &mdbkey, &mdbdata, 0), MDB_SUCCESS) << "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(env, dbi);
    mdb_env_close(env);
  }

  void TestRead() {
    LayerParameter param;
    LabelingDataParameter *labeling_data_param = param.mutable_labeling_data_param();
    labeling_data_param->set_batch_size(5);
    labeling_data_param->set_source(filename_->c_str());

    LabelingDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 3);
    EXPECT_EQ(blob_top_data_->height(), 2);
    EXPECT_EQ(blob_top_data_->width(), 3);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 2);
    EXPECT_EQ(blob_top_label_->width(), 3);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j) {
        EXPECT_EQ(j, blob_top_label_->cpu_data()[i * 6 + j]);
      }
    }
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 18; ++j) {
        EXPECT_EQ(j, blob_top_data_->cpu_data()[i * 18 + j]);
      }
    }
  }

  virtual ~LabelingDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  shared_ptr<string> filename_;
  Blob<Dtype> *const blob_top_data_;
  Blob<Dtype> *const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(LabelingDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelingDataLayerTest, TestRead) {
  this->FillLMDB();
  this->TestRead();
}

}  // namespace caffe